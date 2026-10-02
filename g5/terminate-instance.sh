#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Terminate the g5 validation instance. g5.8xlarge bills ~$2.45/hr on-demand
# in us-west-2, so this is not optional housekeeping.
#
# Usage:
#   ./g5/terminate-instance.sh                 # uses g5/.last-instance-id
#   ./g5/terminate-instance.sh i-0123456789    # explicit
set -euo pipefail

AWS_PROFILE="${AWS_PROFILE:-compute-sa-team-Administrator}"
export AWS_PROFILE
REGION="${REGION:-us-west-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INSTANCE_ID="${1:-}"
if [[ -z "${INSTANCE_ID}" ]]; then
  if [[ -f "${HERE}/.last-instance-id" ]]; then
    # Written by launch-instance.sh as INSTANCE_ID=... / REGION=... . The region
    # matters: GPU capacity often forces a region other than the default, and a
    # teardown aimed at the wrong one leaves the instance running and billing.
    # shellcheck disable=SC1091
    source "${HERE}/.last-instance-id"
    REGION="${REGION:-us-west-2}"
    echo "Using ${HERE}/.last-instance-id: ${INSTANCE_ID} in ${REGION}"
  else
    echo "FATAL: no instance id given and ${HERE}/.last-instance-id is absent." >&2
    echo "       Sweep every region for orphaned validation instances with:" >&2
    echo "       for r in \$(aws ec2 describe-regions --query 'Regions[].RegionName' --output text); do \\" >&2
    echo "         aws ec2 describe-instances --region \$r \\" >&2
    echo "           --filters Name=tag:Purpose,Values=qwen3-single-gpu-stack-validation \\" >&2
    echo "                     Name=instance-state-name,Values=pending,running,stopping,stopped \\" >&2
    echo "           --query \"Reservations[].Instances[].[InstanceId,'\$r']\" --output text; done" >&2
    exit 1
  fi
fi

STATE="$(aws ec2 describe-instances --region "${REGION}" --instance-ids "${INSTANCE_ID}" \
  --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "missing")"
echo "Instance ${INSTANCE_ID} is currently: ${STATE}"

if [[ "${STATE}" == "terminated" ]]; then
  echo "Already terminated; nothing to do."
  exit 0
fi
if [[ "${STATE}" == "missing" ]]; then
  echo "FATAL: instance ${INSTANCE_ID} not found in ${REGION}." >&2
  exit 1
fi

aws ec2 terminate-instances --region "${REGION}" --instance-ids "${INSTANCE_ID}" \
  --query 'TerminatingInstances[0].{Id:InstanceId,From:PreviousState.Name,To:CurrentState.Name}' \
  --output table

echo "Waiting for termination to complete..."
aws ec2 wait instance-terminated --region "${REGION}" --instance-ids "${INSTANCE_ID}"
echo "Instance ${INSTANCE_ID} terminated. Root EBS volume was DeleteOnTermination=true."
rm -f "${HERE}/.last-instance-id"

echo
echo "Note: the IAM role/instance profile and the egress-only security group are"
echo "left in place for re-runs. They cost nothing. Remove them with:"
echo "  aws iam remove-role-from-instance-profile --instance-profile-name qwen3-g5-validation-ssm --role-name qwen3-g5-validation-ssm"
echo "  aws iam delete-instance-profile --instance-profile-name qwen3-g5-validation-ssm"
echo "  aws iam detach-role-policy --role-name qwen3-g5-validation-ssm --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
echo "  aws iam delete-role --role-name qwen3-g5-validation-ssm"
echo "  aws ec2 delete-security-group --region ${REGION} --group-name qwen3-g5-validation-sg"
