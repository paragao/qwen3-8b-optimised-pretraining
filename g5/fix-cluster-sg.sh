#!/usr/bin/env bash
# Converge the cluster security group so multi-node NCCL can actually connect,
# WITHOUT launching anything.
#
# WHY THIS EXISTS
# Two different too-narrow shapes have each cost a run on this project.
#
# 1. tcp/MASTER_PORT only. Enough for torchrun's TCPStore rendezvous and NOT
#    enough for NCCL, which builds its communicator over its own sockets on
#    EPHEMERAL ports, where each rank listens and the peer opens a NEW INBOUND
#    connection. Security groups are stateful only for return traffic on an
#    already-established flow, so those fresh inbound connections are dropped.
#    Symptom: a SILENT hang, measured at 55 minutes with ~140 bytes/sec of
#    inter-node traffic while both nodes billed.
#
# 2. tcp/1-65535 INGRESS ONLY, with the default 0.0.0.0/0 egress. Enough for
#    TCP NCCL and NOT enough for EFA, which is not IP: a CIDR-based rule cannot
#    match EFA traffic, so the default egress rule does not carry it. AWS is
#    explicit -- "the self-referencing inbound AND OUTBOUND rules (allowing all
#    traffic to and from the security group itself) are mandatory for EFA to
#    function. Without these rules, EFA traffic between instances will be
#    blocked and NCCL communication will fail."
#    Symptom: libfabric selects the efa provider and the handshake then fails:
#      NET/OFI Request ... completed with error. RC: 103. Error: 4126
#      (Unresponsive receiver (reachable by EFA device but handshake failed)
#    i.e. the device is addressable but its packets are dropped.
#
# So USE_EFA selects the shape:
#   USE_EFA=1 (default) : ingress ALL protocols self-referencing
#                         PLUS egress ALL protocols self-referencing
#   USE_EFA=0           : ingress tcp/1-65535 self-referencing
#
# UNLIKE an EFA interface, a security group rule applies to RUNNING instances
# immediately -- so this script repairs a live cluster with no relaunch.
#
# EXPOSURE IS UNCHANGED IN KIND. Every rule this script adds is
# self-referencing: its source or destination is the security group itself,
# never a CIDR and never 0.0.0.0/0. The only hosts that can reach any of it are
# the cluster's own nodes. IpProtocol=-1 widens the PROTOCOL set, not the source
# set. This script REFUSES to proceed if it would leave a CIDR-sourced INGRESS
# rule in place.
#
# The pre-existing 0.0.0.0/0 EGRESS rule is deliberately LEFT ALONE: the nodes
# need outbound internet to pull the ~77 GB container image and stream c4.
# Revoking it would break the bootstrap. The self-referencing egress rule is
# ADDED alongside it, not instead of it.
#
#   https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html
#   https://docs.aws.amazon.com/pcs/latest/userguide/working-with_networking_sg.html
#
# Usage:
#   ./g5/fix-cluster-sg.sh                 # EFA shape (default)
#   USE_EFA=0 ./g5/fix-cluster-sg.sh       # TCP-only shape
#   REGION=us-west-2 ./g5/fix-cluster-sg.sh
set -euo pipefail

REGION="${REGION:-us-west-2}"
NAME="${NAME:-qwen3-g5-validation}"
SG_NAME="${NAME}-sg"
MASTER_PORT="${MASTER_PORT:-29500}"
USE_EFA="${USE_EFA:-1}"
DESC="NCCL + torchrun between cluster nodes only (self-referencing)"

if [[ "${USE_EFA}" != "0" && "${USE_EFA}" != "1" ]]; then
  echo "FATAL: USE_EFA=${USE_EFA}; must be 0 or 1." >&2
  exit 1
fi

echo "==> resolving ${SG_NAME} in ${REGION}"
SG_ID="$(aws ec2 describe-security-groups --region "${REGION}" \
  --filters "Name=group-name,Values=${SG_NAME}" \
  --query 'SecurityGroups[0].GroupId' --output text)"
if [[ -z "${SG_ID}" || "${SG_ID}" == "None" ]]; then
  echo "FATAL: no security group named ${SG_NAME} in ${REGION}." >&2
  echo "       Set REGION= or NAME= if the cluster lives elsewhere." >&2
  exit 1
fi
echo "    ${SG_ID}  (target shape: $( [[ "${USE_EFA}" -eq 1 ]] && echo 'EFA -- all protocols, in AND out' || echo 'TCP -- tcp/1-65535 inbound' ))"

echo "==> current rules"
echo "    ingress:"
aws ec2 describe-security-groups --region "${REGION}" --group-ids "${SG_ID}" \
  --query 'SecurityGroups[0].IpPermissions[].[IpProtocol,FromPort,ToPort,UserIdGroupPairs[0].GroupId,IpRanges[0].CidrIp]' \
  --output text | sed 's/^/      /'
echo "    egress:"
aws ec2 describe-security-groups --region "${REGION}" --group-ids "${SG_ID}" \
  --query 'SecurityGroups[0].IpPermissionsEgress[].[IpProtocol,FromPort,ToPort,UserIdGroupPairs[0].GroupId,IpRanges[0].CidrIp]' \
  --output text | sed 's/^/      /'

# ----------------------------------------------------------------- ingress
if [[ "${USE_EFA}" -eq 1 ]]; then
  WANT_IN="IpProtocol=-1,UserIdGroupPairs=[{GroupId=${SG_ID},Description=\"${DESC}\"}]"
  DROP_IN="IpProtocol=tcp,FromPort=1,ToPort=65535,UserIdGroupPairs=[{GroupId=${SG_ID}}]"
  DROP_WHY="the TCP-only rule (it does not carry EFA traffic)"
else
  WANT_IN="IpProtocol=tcp,FromPort=1,ToPort=65535,UserIdGroupPairs=[{GroupId=${SG_ID},Description=\"${DESC}\"}]"
  DROP_IN="IpProtocol=-1,UserIdGroupPairs=[{GroupId=${SG_ID}}]"
  DROP_WHY="the all-protocol rule (USE_EFA=0 needs only TCP)"
fi

# Drop the legacy single-port rule, whichever shape we are converging to.
# Revoking only ever REDUCES exposure.
if aws ec2 revoke-security-group-ingress --region "${REGION}" \
    --group-id "${SG_ID}" \
    --ip-permissions "IpProtocol=tcp,FromPort=${MASTER_PORT},ToPort=${MASTER_PORT},UserIdGroupPairs=[{GroupId=${SG_ID}}]" \
    >/dev/null 2>&1; then
  echo "==> revoked the legacy tcp/${MASTER_PORT}-only rule"
fi
if aws ec2 revoke-security-group-ingress --region "${REGION}" \
    --group-id "${SG_ID}" --ip-permissions "${DROP_IN}" >/dev/null 2>&1; then
  echo "==> revoked ${DROP_WHY}"
fi

if aws ec2 authorize-security-group-ingress --region "${REGION}" \
    --group-id "${SG_ID}" --ip-permissions "${WANT_IN}" >/dev/null 2>&1; then
  echo "==> granted the inbound rule from ${SG_ID} to itself"
else
  echo "==> inbound rule already present"
fi

# ------------------------------------------------------------------ egress
# ADDITIVE ONLY. The pre-existing 0.0.0.0/0 egress stays: the nodes need
# outbound internet for the image pull and the c4 download. This adds the
# self-referencing rule AWS requires for EFA alongside it.
if [[ "${USE_EFA}" -eq 1 ]]; then
  if aws ec2 authorize-security-group-egress --region "${REGION}" \
      --group-id "${SG_ID}" \
      --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=${SG_ID},Description=\"${DESC}\"}]" \
      >/dev/null 2>&1; then
    echo "==> granted the OUTBOUND rule to ${SG_ID} itself (required for EFA)"
  else
    echo "==> outbound self-referencing rule already present"
  fi
fi

# ------------------------------------------------------------------ verify
# Verify the shape rather than trusting the calls above.
echo "==> verifying"
CIDRS="$(aws ec2 describe-security-groups --region "${REGION}" --group-ids "${SG_ID}" \
  --query 'SecurityGroups[0].IpPermissions[].IpRanges[].CidrIp' --output text)"
if [[ -n "${CIDRS}" ]]; then
  echo "FATAL: a CIDR-sourced INGRESS rule is present: ${CIDRS}" >&2
  echo "       Every inbound rule must be sourced from the group itself." >&2
  exit 1
fi

read -r PROTO FROM TO <<EOF
$(aws ec2 describe-security-groups --region "${REGION}" --group-ids "${SG_ID}" \
  --query 'SecurityGroups[0].IpPermissions[0].[IpProtocol,FromPort,ToPort]' --output text)
EOF
PEER="$(aws ec2 describe-security-groups --region "${REGION}" --group-ids "${SG_ID}" \
  --query 'SecurityGroups[0].IpPermissions[0].UserIdGroupPairs[0].GroupId' --output text)"

# An IpProtocol=-1 rule reports FromPort/ToPort as None, not 1/65535, so the
# two shapes cannot share one numeric comparison -- under `set -u` an
# arithmetic test against "None" would abort here instead of reporting.
if [[ "${USE_EFA}" -eq 1 ]]; then
  if [[ "${PROTO}" != "-1" ]]; then
    echo "FATAL: ingress is ${PROTO}/${FROM}-${TO}; EFA needs ALL protocols." >&2
    echo "       EFA is not TCP, so even tcp/1-65535 blocks it." >&2
    exit 1
  fi
  echo "    ingress OK: ALL protocols, source = ${SG_ID} (self), no CIDRs"
else
  if [[ "${PROTO}" != "tcp" || "${FROM}" -gt 1024 || "${TO}" -lt 65535 ]]; then
    echo "FATAL: ingress is ${PROTO}/${FROM}-${TO}; NCCL needs the ephemeral range." >&2
    exit 1
  fi
  echo "    ingress OK: ${PROTO}/${FROM}-${TO}, source = ${SG_ID} (self), no CIDRs"
fi
if [[ "${PEER}" != "${SG_ID}" ]]; then
  echo "FATAL: rule source is ${PEER}, expected ${SG_ID} (itself)." >&2
  exit 1
fi

# The egress assertion is the one that catches the 2026-10-04 EFA handshake
# failure: ingress was already correct and the run still could not connect.
if [[ "${USE_EFA}" -eq 1 ]]; then
  SELF_EGRESS="$(aws ec2 describe-security-groups --region "${REGION}" --group-ids "${SG_ID}" \
    --query "length(SecurityGroups[0].IpPermissionsEgress[?IpProtocol=='-1'] | [?UserIdGroupPairs[?GroupId=='${SG_ID}']])" \
    --output text)"
  if [[ "${SELF_EGRESS}" -lt 1 ]]; then
    echo "FATAL: no self-referencing ALL-protocol EGRESS rule." >&2
    echo "       The default 0.0.0.0/0 egress does NOT carry EFA traffic: EFA" >&2
    echo "       is not IP, so a CIDR rule cannot match it. Without this the" >&2
    echo "       efa provider is selected and the handshake then fails with" >&2
    echo "       'Unresponsive receiver (reachable by EFA device but" >&2
    echo "       handshake failed)'." >&2
    exit 1
  fi
  echo "    egress  OK: self-referencing ALL-protocol rule present"
fi

echo
echo "Security group rules apply to RUNNING instances immediately, so no"
echo "relaunch is needed. Re-run the training:"
echo "    TRAIN_ITERS=1000 ./g5/finish-run-2node.sh"
