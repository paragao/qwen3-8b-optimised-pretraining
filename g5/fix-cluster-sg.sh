#!/usr/bin/env bash
# Converge the cluster security group so multi-node NCCL can actually connect,
# WITHOUT launching anything.
#
# WHY THIS EXISTS
# An earlier version of g5/launch-instance.sh opened only tcp/MASTER_PORT
# between the nodes. That is enough for torchrun's TCPStore rendezvous and NOT
# enough for NCCL: NCCL builds its communicator over its own sockets on
# EPHEMERAL ports, where each rank listens and the peer opens a NEW INBOUND
# connection. Security groups are stateful only for return traffic on an
# already-established flow, so those fresh inbound connections are dropped.
#
# The observed symptom is a SILENT hang, not an error: the rendezvous succeeds,
# rank 0 prints "NCCL version ...", and then nothing -- measured at 55 minutes
# with ~140 bytes/sec of inter-node traffic while both nodes billed.
#
# AWS documents the required shape:
#   https://docs.aws.amazon.com/us_en/AWSEC2/latest/UserGuide/efa-start-nccl.html
#   https://docs.aws.amazon.com/pcs/latest/userguide/working-with_networking_sg.html
#
# EXPOSURE IS UNCHANGED IN KIND. The rule stays self-referencing: its source is
# the security group itself, never a CIDR and never 0.0.0.0/0. The only hosts
# that can reach any of these ports are the cluster's own nodes. This script
# REFUSES to proceed if it would leave a CIDR-sourced rule in place.
set -euo pipefail

REGION="${REGION:-us-west-2}"
NAME="${NAME:-qwen3-g5-validation}"
SG_NAME="${NAME}-sg"
MASTER_PORT="${MASTER_PORT:-29500}"
DESC="NCCL + torchrun between cluster nodes only (self-referencing)"

echo "==> resolving ${SG_NAME} in ${REGION}"
SG_ID="$(aws ec2 describe-security-groups --region "${REGION}" \
  --filters "Name=group-name,Values=${SG_NAME}" \
  --query 'SecurityGroups[0].GroupId' --output text)"
if [[ -z "${SG_ID}" || "${SG_ID}" == "None" ]]; then
  echo "FATAL: no security group named ${SG_NAME} in ${REGION}." >&2
  echo "       Set REGION= or NAME= if the cluster lives elsewhere." >&2
  exit 1
fi
echo "    ${SG_ID}"

echo "==> current ingress"
aws ec2 describe-security-groups --region "${REGION}" --group-ids "${SG_ID}" \
  --query 'SecurityGroups[0].IpPermissions[].[IpProtocol,FromPort,ToPort]' \
  --output text | sed 's/^/    /'

# Drop the too-narrow legacy rule. Revoking only ever REDUCES exposure, and
# leaving it behind would also make the rule count ambiguous.
if aws ec2 revoke-security-group-ingress --region "${REGION}" \
    --group-id "${SG_ID}" \
    --ip-permissions "IpProtocol=tcp,FromPort=${MASTER_PORT},ToPort=${MASTER_PORT},UserIdGroupPairs=[{GroupId=${SG_ID}}]" \
    >/dev/null 2>&1; then
  echo "==> revoked the legacy tcp/${MASTER_PORT}-only rule"
fi

if aws ec2 authorize-security-group-ingress --region "${REGION}" \
    --group-id "${SG_ID}" \
    --ip-permissions "IpProtocol=tcp,FromPort=1,ToPort=65535,UserIdGroupPairs=[{GroupId=${SG_ID},Description=\"${DESC}\"}]" \
    >/dev/null 2>&1; then
  echo "==> granted tcp/1-65535 from ${SG_ID} to itself"
else
  echo "==> rule already present"
fi

# Verify the shape rather than trusting the calls above.
echo "==> verifying"
CIDRS="$(aws ec2 describe-security-groups --region "${REGION}" --group-ids "${SG_ID}" \
  --query 'SecurityGroups[0].IpPermissions[].IpRanges[].CidrIp' --output text)"
if [[ -n "${CIDRS}" ]]; then
  echo "FATAL: a CIDR-sourced ingress rule is present: ${CIDRS}" >&2
  echo "       Every rule must be sourced from the group itself. Remove it." >&2
  exit 1
fi

read -r PROTO FROM TO <<EOF
$(aws ec2 describe-security-groups --region "${REGION}" --group-ids "${SG_ID}" \
  --query 'SecurityGroups[0].IpPermissions[0].[IpProtocol,FromPort,ToPort]' --output text)
EOF
PEER="$(aws ec2 describe-security-groups --region "${REGION}" --group-ids "${SG_ID}" \
  --query 'SecurityGroups[0].IpPermissions[0].UserIdGroupPairs[0].GroupId' --output text)"

if [[ "${PROTO}" != "tcp" || "${FROM}" -gt 1024 || "${TO}" -lt 65535 ]]; then
  echo "FATAL: ingress is ${PROTO}/${FROM}-${TO}; NCCL needs the ephemeral range." >&2
  exit 1
fi
if [[ "${PEER}" != "${SG_ID}" ]]; then
  echo "FATAL: rule source is ${PEER}, expected ${SG_ID} (itself)." >&2
  exit 1
fi

echo "    OK: ${PROTO}/${FROM}-${TO}, source = ${SG_ID} (self), no CIDRs"
echo
echo "Now re-run the training:"
echo "    TRAIN_ITERS=1000 ./g5/finish-run-2node.sh"
