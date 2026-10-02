#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Terminate the g5 validation instance. g5.8xlarge bills ~$2.45/hr on-demand
# in us-west-2, so this is not optional housekeeping.
#
# Usage:
#   ./g5/terminate-instance.sh                        # uses g5/.last-instance-id
#   ./g5/terminate-instance.sh i-0123456789           # explicit, one node
#   ./g5/terminate-instance.sh i-0123456789 i-0abc    # explicit, two nodes
set -euo pipefail

AWS_PROFILE="${AWS_PROFILE:-compute-sa-team-Administrator}"
export AWS_PROFILE
REGION="${REGION:-us-west-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Every argument is an instance id, so a 2-node cluster is torn down in one
# call. Taking only $1 would terminate node 0 and leave node 1 billing
# silently, which is the exact failure this script exists to prevent.
IDS=("$@")
if [[ "${#IDS[@]}" -eq 0 ]]; then
  if [[ -f "${HERE}/.last-instance-id" ]]; then
    # Written by launch-instance.sh as INSTANCE_IDS=... / REGION=... . The
    # region matters: GPU capacity often forces a region other than the
    # default, and a teardown aimed at the wrong one leaves instances running
    # and billing.
    # shellcheck disable=SC1091
    source "${HERE}/.last-instance-id"
    REGION="${REGION:-us-west-2}"
    # INSTANCE_IDS (plural) is written by current launch-instance.sh;
    # INSTANCE_ID is kept for a record file written by an older version.
    if [[ -n "${INSTANCE_IDS:-}" ]]; then
      # shellcheck disable=SC2206
      IDS=(${INSTANCE_IDS})
    elif [[ -n "${INSTANCE_ID:-}" ]]; then
      IDS=("${INSTANCE_ID}")
    fi
    echo "Using ${HERE}/.last-instance-id: ${IDS[*]} in ${REGION}"
  fi
fi

if [[ "${#IDS[@]}" -eq 0 ]]; then
  echo "FATAL: no instance id given and ${HERE}/.last-instance-id is absent" >&2
  echo "       or empty." >&2
  echo "       Sweep every region for orphaned validation instances with:" >&2
  echo "       for r in \$(aws ec2 describe-regions --query 'Regions[].RegionName' --output text); do \\" >&2
  echo "         aws ec2 describe-instances --region \$r \\" >&2
  echo "           --filters Name=tag:Purpose,Values=qwen3-single-gpu-stack-validation \\" >&2
  echo "                     Name=instance-state-name,Values=pending,running,stopping,stopped \\" >&2
  echo "           --query \"Reservations[].Instances[].[InstanceId,'\$r']\" --output text; done" >&2
  exit 1
fi

# Classify every id BEFORE terminating anything, so a typo in the second
# argument does not leave the first half-processed.
LIVE=()
for id in "${IDS[@]}"; do
  state="$(aws ec2 describe-instances --region "${REGION}" --instance-ids "${id}" \
    --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null \
    || echo "missing")"
  echo "  ${id}: ${state}"
  case "${state}" in
    terminated) ;;                       # nothing to do
    missing)
      echo "FATAL: instance ${id} not found in ${REGION}. Terminating nothing." >&2
      echo "       Check the region: the cluster may be elsewhere." >&2
      exit 1
      ;;
    *) LIVE+=("${id}") ;;
  esac
done

if [[ "${#LIVE[@]}" -eq 0 ]]; then
  echo "All ${#IDS[@]} instance(s) already terminated; nothing to do."
  rm -f "${HERE}/.last-instance-id"
  exit 0
fi

aws ec2 terminate-instances --region "${REGION}" --instance-ids "${LIVE[@]}" \
  --query 'TerminatingInstances[].{Id:InstanceId,From:PreviousState.Name,To:CurrentState.Name}' \
  --output table

echo "Waiting for termination to complete on ${#LIVE[@]} instance(s)..."
aws ec2 wait instance-terminated --region "${REGION}" --instance-ids "${LIVE[@]}"

# Confirm from the API rather than trusting the waiter's exit status, so the
# "terminated" claim is backed by observed state.
REMAINING="$(aws ec2 describe-instances --region "${REGION}" \
  --instance-ids "${LIVE[@]}" \
  --query 'length(Reservations[].Instances[?State.Name!=`terminated`][])' \
  --output text)"
if [[ "${REMAINING}" != "0" ]]; then
  echo "FATAL: ${REMAINING} instance(s) are still not terminated and may be" >&2
  echo "       BILLING. Check manually:" >&2
  echo "       aws ec2 describe-instances --region ${REGION} --instance-ids ${LIVE[*]}" >&2
  exit 1
fi
echo "Verified: all ${#LIVE[@]} instance(s) terminated. Root EBS volumes were"
echo "DeleteOnTermination=true."
rm -f "${HERE}/.last-instance-id"

echo
echo "Note: the IAM role/instance profile and the egress-only security group are"
echo "left in place for re-runs. They cost nothing. Remove them with:"
echo "  aws iam remove-role-from-instance-profile --instance-profile-name qwen3-g5-validation-ssm --role-name qwen3-g5-validation-ssm"
echo "  aws iam delete-instance-profile --instance-profile-name qwen3-g5-validation-ssm"
echo "  aws iam detach-role-policy --role-name qwen3-g5-validation-ssm --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
echo "  aws iam delete-role --role-name qwen3-g5-validation-ssm"
echo "  aws ec2 delete-security-group --region ${REGION} --group-name qwen3-g5-validation-sg"
