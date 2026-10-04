#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Launch ONE g5.8xlarge (1x A10G 24 GB) to run the Qwen3 single-GPU stack
# validation, then tear it down with ./g5/terminate-instance.sh.
#
# Security posture (deliberate):
#   * the security group has NO ingress rules at all -- egress only, which is
#     what the container pull from nvcr.io and the tokenizer fetch need
#   * no SSH key pair and no port 22; all access is via SSM Session Manager,
#     which is an outbound-initiated channel
#   * no public listener is created by the workload (see g5/run.sh)
#
# Idempotent: re-running reuses the IAM role, instance profile and security
# group if they already exist.
set -euo pipefail

AWS_PROFILE="${AWS_PROFILE:-compute-sa-team-Administrator}"
export AWS_PROFILE
REGION="${REGION:-us-west-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.8xlarge}"
NAME="${NAME:-qwen3-g5-validation}"
ROOT_VOLUME_GB="${ROOT_VOLUME_GB:-300}"

# How many instances. A g5.8xlarge has exactly ONE A10G (verified:
# ec2 describe-instance-types reports GpuInfo.Gpus[0].Count = 1), so the
# instance count IS the data-parallel degree.
#
#   NODES=1 (default) : zero-ingress security group. Nothing can reach the
#                       instance; access is SSM only.
#   NODES=2           : the torchrun rendezvous must cross between the two
#                       instances, so ONE ingress rule is added on
#                       MASTER_PORT whose source is the security group
#                       ITSELF. That means only instances in this group can
#                       reach it -- not the VPC, and not the internet.
NODES="${NODES:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
if [[ "${NODES}" != "1" && "${NODES}" != "2" ]]; then
  echo "FATAL: NODES=${NODES}; this script supports 1 or 2." >&2
  exit 1
fi

# ------------------------------------------------------------------------ EFA
# Attach an Elastic Fabric Adapter instead of a plain ENA.
#
# WHY THIS DEFAULTS ON AT 2 NODES
# The first completed 2-node run (2026-10-04) measured 8,086 tok/s against
# 24,727 tok/s on ONE node -- a 3x SLOWDOWN -- because every step was dominated
# by the gradient exchange: ~1,033 MB per step moving at 510 MB/s, so the
# 2.026s step time WAS the transfer. That 510 MB/s is 4.08 Gbit/s on a 25 Gbit
# link because the nodes had no EFA device, so NCCL's libfabric plugin logged
# "No eligible providers were found" and fell back to TCP.
#
# An EFA interface cannot be added to a RUNNING instance, so this is a launch
# -time decision. At NODES=1 it changes nothing (no inter-node traffic), so it
# is forced off there rather than paying for an interface nothing uses.
#
# USE_EFA=0 reproduces the slow TCP baseline deliberately.
USE_EFA="${USE_EFA:-$([[ "${NODES}" -gt 1 ]] && echo 1 || echo 0)}"
if [[ "${USE_EFA}" != "0" && "${USE_EFA}" != "1" ]]; then
  echo "FATAL: USE_EFA=${USE_EFA}; must be 0 or 1." >&2
  exit 1
fi
if [[ "${USE_EFA}" -eq 1 && "${NODES}" -eq 1 ]]; then
  echo "NOTE: USE_EFA=1 with NODES=1 has no effect (no inter-node traffic)."
  echo "      Disabling it so the launch does not request a device nothing uses."
  USE_EFA=0
fi
# Deep Learning Base OSS Nvidia Driver GPU AMI: ships the NVIDIA driver, Docker
# and the NVIDIA container toolkit. The NeMo container brings its own
# PyTorch/CUDA userspace, so the "base" variant is all we need.
SSM_AMI_PARAM="${SSM_AMI_PARAM:-/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id}"

aws() { command aws --region "${REGION}" "$@"; }

echo "==> account / identity"
aws sts get-caller-identity --output table

# Verify the instance type can actually take an EFA before requesting one, so a
# wrong INSTANCE_TYPE fails here with the reason rather than at run-instances
# with "InterfaceType efa is not supported". Checked rather than hardcoded,
# because INSTANCE_TYPE is overridable: g5.8xlarge reports EfaSupported=true
# with MaximumEfaInterfaces=1, but g5.xlarge/2xlarge/4xlarge do NOT.
if [[ "${USE_EFA}" -eq 1 ]]; then
  echo "==> EFA requested: checking ${INSTANCE_TYPE} supports it"
  read -r EFA_OK EFA_MAX <<EOF
$(aws ec2 describe-instance-types --instance-types "${INSTANCE_TYPE}" \
  --query 'InstanceTypes[0].NetworkInfo.[EfaSupported,EfaInfo.MaximumEfaInterfaces]' \
  --output text)
EOF
  echo "    EfaSupported=${EFA_OK}, MaximumEfaInterfaces=${EFA_MAX}"
  if [[ "${EFA_OK}" != "True" ]]; then
    echo "FATAL: ${INSTANCE_TYPE} does not support EFA (EfaSupported=${EFA_OK})." >&2
    echo "       Re-run with USE_EFA=0 to launch with a plain ENA, or pick an" >&2
    echo "       EFA-capable type. NOTE that without EFA a 2-node run of this" >&2
    echo "       model is MEASURED 3x SLOWER than a single node, so USE_EFA=0" >&2
    echo "       at NODES=2 is a deliberate slow baseline, not a default." >&2
    exit 1
  fi
  if [[ "${EFA_MAX}" == "None" || -z "${EFA_MAX}" || "${EFA_MAX}" -lt 1 ]]; then
    echo "FATAL: ${INSTANCE_TYPE} reports EfaSupported=True but" >&2
    echo "       MaximumEfaInterfaces=${EFA_MAX}, so no interface can be attached." >&2
    exit 1
  fi
fi

echo "==> resolving AMI from ${SSM_AMI_PARAM}"
AMI_ID="$(aws ssm get-parameter --name "${SSM_AMI_PARAM}" --query 'Parameter.Value' --output text)"
if [[ -z "${AMI_ID}" || "${AMI_ID}" == "None" ]]; then
  echo "FATAL: could not resolve AMI id" >&2; exit 1
fi
echo "    AMI_ID=${AMI_ID}"

# ---------------------------------------------------------------- IAM for SSM
ROLE_NAME="${NAME}-ssm"
echo "==> IAM role ${ROLE_NAME} (for SSM Session Manager)"
if ! aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  aws iam create-role --role-name "${ROLE_NAME}" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
    }' >/dev/null
  echo "    created"
else
  echo "    exists, reusing"
fi
aws iam attach-role-policy --role-name "${ROLE_NAME}" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore >/dev/null

if ! aws iam get-instance-profile --instance-profile-name "${ROLE_NAME}" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "${ROLE_NAME}" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "${ROLE_NAME}" \
    --role-name "${ROLE_NAME}" >/dev/null
  echo "    instance profile created; waiting for IAM propagation"
  sleep 15
fi

# ------------------------------------------------------------ VPC / subnet / SG
VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text)"
if [[ -z "${VPC_ID}" || "${VPC_ID}" == "None" ]]; then
  echo "FATAL: no default VPC in ${REGION}; set SUBNET_ID and VPC_ID manually" >&2; exit 1
fi

# Build the candidate subnet list: every public subnet in an AZ that offers the
# instance type. GPU capacity is frequently exhausted in a given AZ, so we try
# them in turn rather than failing on the first InsufficientInstanceCapacity.
CANDIDATE_SUBNETS=()
if [[ -n "${SUBNET_ID:-}" ]]; then
  CANDIDATE_SUBNETS=("${SUBNET_ID}")
else
  OFFERED_AZS="$(aws ec2 describe-instance-type-offerings \
    --location-type availability-zone \
    --filters Name=instance-type,Values="${INSTANCE_TYPE}" \
    --query 'InstanceTypeOfferings[].Location' --output text)"
  for az in ${OFFERED_AZS}; do
    # Skip Local Zones (e.g. us-west-2-lax-1a): different capacity pool and no
    # default-VPC subnet.
    case "${az}" in *-[a-z][a-z][a-z]-[0-9][a-z]) continue ;; esac
    candidate="$(aws ec2 describe-subnets \
      --filters Name=vpc-id,Values="${VPC_ID}" \
                Name=availability-zone,Values="${az}" \
                Name=map-public-ip-on-launch,Values=true \
      --query 'Subnets[0].SubnetId' --output text 2>/dev/null || true)"
    if [[ -n "${candidate}" && "${candidate}" != "None" ]]; then
      CANDIDATE_SUBNETS+=("${candidate}")
      echo "    candidate ${az} -> ${candidate}"
    fi
  done
fi
if [[ ${#CANDIDATE_SUBNETS[@]} -eq 0 ]]; then
  echo "FATAL: no public subnet found in any AZ offering ${INSTANCE_TYPE}" >&2; exit 1
fi

SG_NAME="${NAME}-sg"
echo "==> security group ${SG_NAME} (egress only, zero ingress rules)"
SG_ID="$(aws ec2 describe-security-groups \
  --filters Name=group-name,Values="${SG_NAME}" Name=vpc-id,Values="${VPC_ID}" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
if [[ -z "${SG_ID}" || "${SG_ID}" == "None" ]]; then
  SG_ID="$(aws ec2 create-security-group --group-name "${SG_NAME}" \
    --vpc-id "${VPC_ID}" \
    --description "Qwen3 g5 validation: egress only, no ingress, SSM access only" \
    --query 'GroupId' --output text)"
  echo "    created ${SG_ID} with no ingress rules"
else
  echo "    exists, reusing ${SG_ID}"
fi

# For 2 nodes the traffic has to cross between instances. The rule's source is
# the security group ITSELF (UserIdGroupPairs), not a CIDR -- so the only
# things that can reach these ports are instances in this same group. A CIDR
# would widen it to the whole subnet or VPC; 0.0.0.0/0 would expose an
# unauthenticated PyTorch rendezvous store to the internet. Neither is used.
#
# WHY THE WHOLE TCP RANGE AND NOT JUST MASTER_PORT:
# Opening only MASTER_PORT looks tighter and does not work. torchrun's TCPStore
# rendezvous does use MASTER_PORT, but NCCL then builds its communicator over
# its OWN sockets on EPHEMERAL ports: each rank listens on a kernel-assigned
# port and the peer opens a NEW INBOUND connection to it. Security groups are
# stateful only for return traffic on an already-established flow, so those
# fresh inbound connections are dropped and NCCL's bootstrap hangs forever --
# with the rendezvous having succeeded, so the symptom is a silent stall right
# after "NCCL version ..." and not an error. Measured: 55 minutes of hang with
# ~140 bytes/sec of inter-node traffic.
# This is AWS's documented requirement, not a workaround:
#   https://docs.aws.amazon.com/us_en/AWSEC2/latest/UserGuide/efa-start-nccl.html
#   https://docs.aws.amazon.com/pcs/latest/userguide/working-with_networking_sg.html
# The exposure is unchanged in kind: still self-referencing, still no CIDR, so
# the only hosts that can reach any of it are the cluster's own nodes.
# NOTE: enabling EFA additionally needs IpProtocol=-1 (EFA is not TCP).
# NOTE: EFA is NOT TCP -- it is its own protocol over the same NIC -- so a
# tcp/1-65535 rule does not carry it. AWS's own guide calls the
# self-referencing ALL-traffic rule "mandatory for EFA to function. Without
# these rules, EFA traffic between instances will be blocked and NCCL
# communication will fail." So the rule shape depends on USE_EFA:
#   USE_EFA=0 -> tcp/1-65535 from the group itself
#   USE_EFA=1 -> ALL protocols from the group itself (IpProtocol=-1)
# Both are self-referencing with no CIDR, so in each case the only hosts that
# can reach anything are the cluster's own nodes: IpProtocol=-1 widens the
# protocol set, NOT the source set.
RDZV_DESC="NCCL + torchrun between cluster nodes only (self-referencing)"
if [[ "${USE_EFA}" -eq 1 ]]; then
  WANT_PERM="IpProtocol=-1,UserIdGroupPairs=[{GroupId=${SG_ID},Description=\"${RDZV_DESC}\"}]"
  WANT_DESC="all protocols (EFA is not TCP)"
else
  WANT_PERM="IpProtocol=tcp,FromPort=1,ToPort=65535,UserIdGroupPairs=[{GroupId=${SG_ID},Description=\"${RDZV_DESC}\"}]"
  WANT_DESC="tcp/1-65535"
fi
if [[ "${NODES}" -gt 1 ]]; then
  echo "==> 2 nodes: granting ${WANT_DESC} from ${SG_ID} to itself"

  # Converge a group created by an earlier version of this script, or by a run
  # with a different USE_EFA. Earlier shapes: a single-port rule (too narrow to
  # carry NCCL's ephemeral ports) and a tcp/1-65535 rule (too narrow to carry
  # EFA). Leaving either in place would also break the count assertion below.
  # Revoking only ever REDUCES exposure.
  if aws ec2 revoke-security-group-ingress \
      --group-id "${SG_ID}" \
      --ip-permissions "IpProtocol=tcp,FromPort=${MASTER_PORT},ToPort=${MASTER_PORT},UserIdGroupPairs=[{GroupId=${SG_ID}}]" \
      >/dev/null 2>&1; then
    echo "    revoked the legacy single-port rule (tcp/${MASTER_PORT} only)"
  fi
  if [[ "${USE_EFA}" -eq 1 ]]; then
    if aws ec2 revoke-security-group-ingress \
        --group-id "${SG_ID}" \
        --ip-permissions "IpProtocol=tcp,FromPort=1,ToPort=65535,UserIdGroupPairs=[{GroupId=${SG_ID}}]" \
        >/dev/null 2>&1; then
      echo "    revoked the TCP-only rule (it does not carry EFA traffic)"
    fi
  else
    if aws ec2 revoke-security-group-ingress \
        --group-id "${SG_ID}" \
        --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=${SG_ID}}]" \
        >/dev/null 2>&1; then
      echo "    revoked the all-protocol rule (USE_EFA=0 needs only TCP)"
    fi
  fi

  if aws ec2 authorize-security-group-ingress \
      --group-id "${SG_ID}" \
      --ip-permissions "${WANT_PERM}" \
      >/dev/null 2>&1; then
    echo "    added (source = the group itself, NOT a CIDR)"
  else
    echo "    already present, or add failed; the assertion below will decide"
  fi
fi

# Assert the invariant rather than trusting it: a stale SG could carry ingress
# that nothing in this script put there.
#   NODES=1 -> exactly 0 ingress rules
#   NODES=2 -> exactly 1, and it must be the self-referencing rendezvous rule
EXPECT_INGRESS=$(( NODES > 1 ? 1 : 0 ))
INGRESS_COUNT="$(aws ec2 describe-security-groups --group-ids "${SG_ID}" \
  --query 'length(SecurityGroups[0].IpPermissions)' --output text)"
if [[ "${INGRESS_COUNT}" != "${EXPECT_INGRESS}" ]]; then
  echo "FATAL: security group ${SG_ID} has ${INGRESS_COUNT} ingress rule(s); expected ${EXPECT_INGRESS}." >&2
  echo "       Inspect it, or delete it and re-run to get a clean group." >&2
  aws ec2 describe-security-groups --group-ids "${SG_ID}" \
    --query 'SecurityGroups[0].IpPermissions' --output json >&2 || true
  exit 1
fi

if [[ "${NODES}" -gt 1 ]]; then
  # Verify the shape, not just the count: a rule on the right port that is
  # open to a CIDR would pass a count check while being the exact thing this
  # is meant to prevent.
  CIDRS="$(aws ec2 describe-security-groups --group-ids "${SG_ID}" \
    --query 'SecurityGroups[0].IpPermissions[].IpRanges[].CidrIp' --output text)"
  if [[ -n "${CIDRS}" ]]; then
    echo "FATAL: the rendezvous rule has CIDR source(s): ${CIDRS}" >&2
    echo "       It must be sourced from the security group itself. Refusing" >&2
    echo "       to launch with a CIDR-scoped rendezvous port." >&2
    exit 1
  fi
  PEER_SG="$(aws ec2 describe-security-groups --group-ids "${SG_ID}" \
    --query 'SecurityGroups[0].IpPermissions[0].UserIdGroupPairs[0].GroupId' --output text)"
  if [[ "${PEER_SG}" != "${SG_ID}" ]]; then
    echo "FATAL: rendezvous rule source is ${PEER_SG}, expected ${SG_ID}" >&2
    exit 1
  fi
  # Assert the rule actually carries what the run needs. This is the assertion
  # that catches the real defects: a single-port rule satisfies every check
  # above and still hangs the NCCL bootstrap for as long as you let it bill,
  # and a tcp/1-65535 rule looks correct while silently blocking EFA, which
  # degrades to TCP and costs a measured 3x in throughput.
  #
  # An IpProtocol=-1 rule reports FromPort/ToPort as None, not 1/65535, so the
  # two shapes cannot share one numeric comparison -- under `set -u` an
  # arithmetic test against "None" would abort here instead of reporting.
  read -r RULE_PROTO RULE_FROM RULE_TO <<EOF
$(aws ec2 describe-security-groups --group-ids "${SG_ID}" \
  --query 'SecurityGroups[0].IpPermissions[0].[IpProtocol,FromPort,ToPort]' --output text)
EOF
  if [[ "${USE_EFA}" -eq 1 ]]; then
    # EFA: the protocol must be ALL. "-1" is what the API returns for it.
    if [[ "${RULE_PROTO}" != "-1" ]]; then
      echo "FATAL: USE_EFA=1 but the ingress rule is ${RULE_PROTO}/${RULE_FROM}-${RULE_TO}." >&2
      echo "       EFA is not TCP, so a TCP rule -- even tcp/1-65535 -- does not" >&2
      echo "       carry it. NCCL would silently fall back to TCP and the run" >&2
      echo "       would be ~3x SLOWER than a single node (measured 2026-10-04)." >&2
      echo "       AWS requires a self-referencing ALL-traffic rule:" >&2
      echo "         https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html" >&2
      exit 1
    fi
    echo "    verified: 1 ingress rule, ALL protocols, source = ${SG_ID} (self), no CIDRs"
  else
    if [[ "${RULE_PROTO}" != "tcp" || "${RULE_FROM}" -gt 1024 || "${RULE_TO}" -lt 65535 ]]; then
      echo "FATAL: ingress rule is ${RULE_PROTO}/${RULE_FROM}-${RULE_TO}; it must span" >&2
      echo "       the ephemeral range (tcp/1-65535). NCCL opens NEW inbound" >&2
      echo "       connections on kernel-assigned ports, so a rule covering only" >&2
      echo "       the rendezvous port lets the rendezvous succeed and then hangs" >&2
      echo "       the NCCL bootstrap indefinitely. See the comment above." >&2
      exit 1
    fi
    echo "    verified: 1 ingress rule, ${RULE_PROTO}/${RULE_FROM}-${RULE_TO}, source = ${SG_ID} (self), no CIDRs"
  fi
else
  echo "    verified: 0 ingress rules"
fi

# ---------------------------------------------------------------------- launch
echo "==> launching ${NODES}x ${INSTANCE_TYPE}"
INSTANCE_IDS=()
LAUNCH_ERR="$(mktemp)"
trap 'rm -f "${LAUNCH_ERR}"' EXIT

# All nodes go in ONE subnet, so one AZ. Cross-AZ would add latency to every
# gradient all-reduce, and the all-reduce is already the limiting factor on a
# 25 Gbit link (see the scaling table in g5/README.md). --count ${NODES} also
# means EC2 either places them all or fails, rather than leaving one orphan.
for subnet in "${CANDIDATE_SUBNETS[@]}"; do
  az="$(aws ec2 describe-subnets --subnet-ids "${subnet}" \
    --query 'Subnets[0].AvailabilityZone' --output text)"
  echo "    trying ${az} (${subnet}) for ${NODES} instance(s)"

  # An EFA has to be requested as a network INTERFACE (InterfaceType=efa),
  # which cannot be combined with --subnet-id / --security-group-ids: those
  # describe the interface EC2 would otherwise build for us, so the subnet and
  # group move into the interface spec instead. Everything else is identical.
  #
  # AssociatePublicIpAddress is set EXPLICITLY here. The subnet has
  # MapPublicIpOnLaunch=true, but specifying --network-interfaces overrides
  # that default, and these nodes need egress to pull the ~77 GB NeMo image
  # and stream c4 -- without it the bootstrap hangs on a network that looks
  # up but reaches nothing.
  #
  # NetworkCardIndex=0, DeviceIndex=0 per AWS's own example for a primary EFA:
  #   https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/create-efa.html
  if [[ "${USE_EFA}" -eq 1 ]]; then
    PLACEMENT_ARGS=(--network-interfaces
      "NetworkCardIndex=0,DeviceIndex=0,InterfaceType=efa,Groups=${SG_ID},SubnetId=${subnet},AssociatePublicIpAddress=true,DeleteOnTermination=true")
  else
    PLACEMENT_ARGS=(--subnet-id "${subnet}" --security-group-ids "${SG_ID}")
  fi

  if IDS="$(aws ec2 run-instances \
      --image-id "${AMI_ID}" \
      --instance-type "${INSTANCE_TYPE}" \
      "${PLACEMENT_ARGS[@]}" \
      --iam-instance-profile "Name=${ROLE_NAME}" \
      --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
      --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${ROOT_VOLUME_GB},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true,\"Encrypted\":true}}]" \
      --tag-specifications \
          "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}},{Key=Purpose,Value=qwen3-single-gpu-stack-validation},{Key=Ephemeral,Value=true}]" \
      --count "${NODES}" \
      --query 'Instances[].InstanceId' --output text 2>"${LAUNCH_ERR}")"; then
    SUBNET_ID="${subnet}"
    # shellcheck disable=SC2206
    INSTANCE_IDS=(${IDS})
    echo "    launched in ${az}: ${INSTANCE_IDS[*]}"
    break
  fi

  # Capacity is the one error worth trying another AZ for. Anything else
  # (quota, permissions, bad AMI) will fail identically everywhere, so stop.
  if grep -q "InsufficientInstanceCapacity\|Unsupported" "${LAUNCH_ERR}"; then
    echo "    no capacity for ${NODES}x in ${az}, trying the next AZ"
    INSTANCE_IDS=()
    continue
  fi
  echo "FATAL: run-instances failed for a reason unrelated to capacity:" >&2
  cat "${LAUNCH_ERR}" >&2
  exit 1
done

if [[ "${#INSTANCE_IDS[@]}" -ne "${NODES}" ]]; then
  echo "FATAL: no ${INSTANCE_TYPE} capacity for ${NODES} instance(s) in any" >&2
  echo "       candidate AZ. Retry later, or try another region with REGION=..." >&2
  exit 1
fi
echo "    INSTANCE_IDS=${INSTANCE_IDS[*]}"

echo "==> waiting for ${NODES} instance(s) to reach running + status ok"
aws ec2 wait instance-running --instance-ids "${INSTANCE_IDS[@]}"
aws ec2 wait instance-status-ok --instance-ids "${INSTANCE_IDS[@]}"

# Verify the EFA actually attached. This is the check whose absence cost a whole
# run: on 2026-10-04 the nodes launched fine, reported InterfaceType "interface"
# rather than "efa", and the only symptom was a 2-node job running 3x SLOWER
# than one node after 35 minutes of training. Requesting an interface is not
# evidence of getting one, so assert it here while nothing has been spent yet.
if [[ "${USE_EFA}" -eq 1 ]]; then
  echo "==> verifying the EFA interface attached on every node"
  EFA_TYPES="$(aws ec2 describe-instances --instance-ids "${INSTANCE_IDS[@]}" \
    --query 'Reservations[].Instances[].NetworkInterfaces[].InterfaceType' --output text)"
  EFA_N=0
  for t in ${EFA_TYPES}; do
    [[ "${t}" == "efa" ]] && EFA_N=$(( EFA_N + 1 ))
  done
  echo "    interface types across all nodes: ${EFA_TYPES}"
  if [[ "${EFA_N}" -ne "${NODES}" ]]; then
    echo "FATAL: ${EFA_N} of ${NODES} node(s) have an 'efa' interface." >&2
    echo "       The instances launched but WITHOUT the fabric, so NCCL will" >&2
    echo "       fall back to TCP and a 2-node run will be slower than one" >&2
    echo "       node. Terminate them rather than training on this:" >&2
    echo "         ./g5/terminate-instance.sh ${INSTANCE_IDS[*]}" >&2
    exit 1
  fi
  echo "    verified: ${EFA_N}/${NODES} nodes have an EFA interface"
fi

echo "==> waiting for the SSM agent to register on every node (up to 5 min)"
for i in $(seq 1 30); do
  ONLINE_COUNT="$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$(IFS=,; echo "${INSTANCE_IDS[*]}")" \
    --query 'length(InstanceInformationList[?PingStatus==`Online`])' \
    --output text 2>/dev/null || echo 0)"
  if [[ "${ONLINE_COUNT}" == "${NODES}" ]]; then
    echo "    SSM Online on all ${NODES}"
    ONLINE="Online"
    break
  fi
  sleep 10   # linear poll; SSM registration is seconds-to-minutes, not hours
done
if [[ "${ONLINE:-}" != "Online" ]]; then
  echo "FATAL: SSM agent did not come Online on all ${NODES} node(s)." >&2
  echo "       The instance(s) are RUNNING and still billing. Terminate with:" >&2
  echo "       ./g5/terminate-instance.sh ${INSTANCE_IDS[*]}" >&2
  exit 1
fi

# Private IPv4 of each node. Node 0's address is what the other node needs as
# MASTER_ADDR, and g5/run.sh refuses anything that is not an RFC1918 address.
PRIVATE_IPS=()
for id in "${INSTANCE_IDS[@]}"; do
  PRIVATE_IPS+=("$(aws ec2 describe-instances --instance-ids "${id}" \
    --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)")
done

echo
echo "================================================================"
echo " nodes    : ${NODES}x ${INSTANCE_TYPE} in ${REGION}"
for i in "${!INSTANCE_IDS[@]}"; do
  echo "   rank ${i} : ${INSTANCE_IDS[$i]}  private ${PRIVATE_IPS[$i]}"
done
echo " ami      : ${AMI_ID}"
if [[ "${NODES}" -gt 1 ]]; then
  echo " sg       : ${SG_ID}  (1 ingress rule: tcp/1-65535 from itself, no CIDR)"
else
  echo " sg       : ${SG_ID}  (0 ingress rules)"
fi
echo " connect  : aws ssm start-session --target ${INSTANCE_IDS[0]} --region ${REGION}"
echo " TEARDOWN : ./g5/terminate-instance.sh ${INSTANCE_IDS[*]}"
echo "================================================================"
if [[ "${NODES}" -gt 1 ]]; then
  echo
  echo "To start the 2-node run, on EACH node (same command except NODE_RANK):"
  echo
  echo "  # rank 0 (${INSTANCE_IDS[0]})"
  echo "  NNODES=2 NODE_RANK=0 MASTER_ADDR=${PRIVATE_IPS[0]} \\"
  echo "    GLOBAL_BATCH_SIZE=16 DATA_PATH=/workspace/run/datasets/c4_qwen3 \\"
  echo "    TRAIN_ITERS=1000 ./g5/run.sh"
  echo
  echo "  # rank 1 (${INSTANCE_IDS[1]})"
  echo "  NNODES=2 NODE_RANK=1 MASTER_ADDR=${PRIVATE_IPS[0]} \\"
  echo "    GLOBAL_BATCH_SIZE=16 DATA_PATH=/workspace/run/datasets/c4_qwen3 \\"
  echo "    TRAIN_ITERS=1000 ./g5/run.sh"
  echo
  echo "MASTER_ADDR is rank 0's PRIVATE address on both. GLOBAL_BATCH_SIZE=16"
  echo "(not 8) is what makes the second node worth paying for -- at a fixed"
  echo "batch size the gradient all-reduce does not shrink and you gain ~5%."
  echo
  echo "Each node needs its OWN copy of the dataset: run ./g5/prepare-c4.sh on"
  echo "both, since there is no shared filesystem here. (The EKS path uses one"
  echo "ReadWriteMany volume instead -- see g5/eks/pretrain.yaml.)"
fi
# Record the REGION as well as the ids: the instances are not necessarily in
# the default region (GPU capacity often forces another), and a teardown
# pointed at the wrong region silently leaves billing instances running.
#
# INSTANCE_IDS MUST BE QUOTED. Unquoted, a multi-id value parses as
# "VAR=first second" -- an assignment prefixing the command `second` -- so any
# reader that sources this file dies with "command not found" and, under
# `set -e`, aborts. That broke both the teardown and the 2-node driver.
cat > "$(dirname "${BASH_SOURCE[0]}")/.last-instance-id" <<EOF
INSTANCE_ID="${INSTANCE_IDS[0]}"
INSTANCE_IDS="${INSTANCE_IDS[*]}"
NODES="${NODES}"
REGION="${REGION}"
EOF
