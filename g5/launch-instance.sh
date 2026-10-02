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
# Deep Learning Base OSS Nvidia Driver GPU AMI: ships the NVIDIA driver, Docker
# and the NVIDIA container toolkit. The NeMo container brings its own
# PyTorch/CUDA userspace, so the "base" variant is all we need.
SSM_AMI_PARAM="${SSM_AMI_PARAM:-/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id}"

aws() { command aws --region "${REGION}" "$@"; }

echo "==> account / identity"
aws sts get-caller-identity --output table

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

# Assert the invariant rather than trusting it: a stale SG could carry ingress.
INGRESS_COUNT="$(aws ec2 describe-security-groups --group-ids "${SG_ID}" \
  --query 'length(SecurityGroups[0].IpPermissions)' --output text)"
if [[ "${INGRESS_COUNT}" != "0" ]]; then
  echo "FATAL: security group ${SG_ID} has ${INGRESS_COUNT} ingress rule(s); expected 0." >&2
  echo "       Inspect it, or delete it and re-run to get a clean egress-only group." >&2
  exit 1
fi
echo "    verified: 0 ingress rules"

# ---------------------------------------------------------------------- launch
echo "==> launching 1x ${INSTANCE_TYPE}"
INSTANCE_ID=""
LAUNCH_ERR="$(mktemp)"
trap 'rm -f "${LAUNCH_ERR}"' EXIT

for subnet in "${CANDIDATE_SUBNETS[@]}"; do
  az="$(aws ec2 describe-subnets --subnet-ids "${subnet}" \
    --query 'Subnets[0].AvailabilityZone' --output text)"
  echo "    trying ${az} (${subnet})"
  if INSTANCE_ID="$(aws ec2 run-instances \
      --image-id "${AMI_ID}" \
      --instance-type "${INSTANCE_TYPE}" \
      --subnet-id "${subnet}" \
      --security-group-ids "${SG_ID}" \
      --iam-instance-profile "Name=${ROLE_NAME}" \
      --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
      --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${ROOT_VOLUME_GB},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true,\"Encrypted\":true}}]" \
      --tag-specifications \
          "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}},{Key=Purpose,Value=qwen3-single-gpu-stack-validation},{Key=Ephemeral,Value=true}]" \
      --count 1 \
      --query 'Instances[0].InstanceId' --output text 2>"${LAUNCH_ERR}")"; then
    SUBNET_ID="${subnet}"
    echo "    launched in ${az}: ${INSTANCE_ID}"
    break
  fi

  # Capacity is the one error worth trying another AZ for. Anything else
  # (quota, permissions, bad AMI) will fail identically everywhere, so stop.
  if grep -q "InsufficientInstanceCapacity\|Unsupported" "${LAUNCH_ERR}"; then
    echo "    no capacity in ${az}, trying the next AZ"
    INSTANCE_ID=""
    continue
  fi
  echo "FATAL: run-instances failed for a reason unrelated to capacity:" >&2
  cat "${LAUNCH_ERR}" >&2
  exit 1
done

if [[ -z "${INSTANCE_ID}" || "${INSTANCE_ID}" == "None" ]]; then
  echo "FATAL: no ${INSTANCE_TYPE} capacity in any candidate AZ." >&2
  echo "       Retry later, or try another region with REGION=... ." >&2
  exit 1
fi
echo "    INSTANCE_ID=${INSTANCE_ID}"

echo "==> waiting for instance to reach running + status ok"
aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}"
aws ec2 wait instance-status-ok --instance-ids "${INSTANCE_ID}"

echo "==> waiting for the SSM agent to register (up to 5 min)"
for i in $(seq 1 30); do
  ONLINE="$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=${INSTANCE_ID}" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)"
  if [[ "${ONLINE}" == "Online" ]]; then echo "    SSM Online"; break; fi
  sleep 10   # linear poll; SSM registration is seconds-to-minutes, not hours
done
if [[ "${ONLINE:-}" != "Online" ]]; then
  echo "FATAL: SSM agent did not come Online for ${INSTANCE_ID}." >&2
  echo "       The instance is RUNNING and still billing. Terminate it with:" >&2
  echo "       ./g5/terminate-instance.sh ${INSTANCE_ID}" >&2
  exit 1
fi

echo
echo "================================================================"
echo " instance : ${INSTANCE_ID}  (${INSTANCE_TYPE}, ${REGION})"
echo " ami      : ${AMI_ID}"
echo " sg       : ${SG_ID}  (0 ingress rules)"
echo " connect  : aws ssm start-session --target ${INSTANCE_ID} --region ${REGION}"
echo " TEARDOWN : ./g5/terminate-instance.sh ${INSTANCE_ID}"
echo "================================================================"
# Record the REGION as well as the id: the instance is not necessarily in the
# default region (GPU capacity often forces another), and a teardown pointed at
# the wrong region silently leaves a billing instance running.
cat > "$(dirname "${BASH_SOURCE[0]}")/.last-instance-id" <<EOF
INSTANCE_ID=${INSTANCE_ID}
REGION=${REGION}
EOF
