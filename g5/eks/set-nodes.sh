#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Switch g5/eks/pretrain.yaml between 1 and 2 g5.8xlarge nodes.
#
# Three values must move together, and a mismatch fails in an unhelpful way:
# if parallelism is 2 but NNODES is 1, each pod runs a WORLD_SIZE=1 job and you
# get two independent runs that look superficially fine. If NNODES is 2 but
# parallelism is 1, torchrun waits forever for a rank that will never join.
#
#   parallelism / completions  the number of pods
#   NNODES                     what torchrun is told, so WORLD_SIZE
#   GLOBAL_BATCH_SIZE          8 x nodes, so per-rank work stays constant
#
# The last one is the point of adding a node at all: at a FIXED
# GLOBAL_BATCH_SIZE, two nodes is only ~1.05-1.07x on the smoke/wider profiles
# because the gradient all-reduce does not shrink. Scaling it was derived to
# give ~1.8-2.0x -- and that derivation is now REFUTED by measurement on the
# direct-EC2 path at this geometry:
#
#   1 node   GBS 8  : 24,727 tok/s
#   2 node   GBS 16 : 8,086 tok/s over TCP (0.33x) -- no EFA device attached
#   2 node   GBS 16 : 25,056 tok/s over EFA (1.01x) -- EFA working
#
# So EFA is worth 3.1x against TCP and the second node is still worth only
# +1.3%. The step is bandwidth-bound either way: 1,033 MB moves per step and
# per-rank compute is ~0.331s, so communication is ~2x compute and overlap
# cannot hide it. Even at 100% of the 25 Gbit line rate the transfer costs
# 0.33s, equal to the compute, so no fabric makes a second node pay HERE.
#
# ~1.8-2.0x needs more TOKENS PER STEP PER RANK (compute scales, the 1,033 MB
# does not), not a faster link. Untested prediction: at 32,768 tok/step a
# 2-node run should beat its 1-node counterpart by >1.4x.
#
# EFA remains mandatory if you run 2 nodes at all -- without it you lose 3x.
# Run g5/eks/set-efa.sh and confirm NCCL does not log
# "NET/OFI No eligible providers were found". See the table in g5/README.md.
#
# Usage:
#   ./g5/eks/set-nodes.sh 1
#   ./g5/eks/set-nodes.sh 2
#   ./g5/eks/set-nodes.sh 2 --dry-run
set -euo pipefail

MANIFEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pretrain.yaml"
NODES="${1:-}"
DRY_RUN=0
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=1

if [[ "${NODES}" != "1" && "${NODES}" != "2" ]]; then
  echo "Usage: $0 <1|2> [--dry-run]" >&2
  echo "  A g5.8xlarge has exactly one A10G, so nodes == data-parallel size." >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "FATAL: ${MANIFEST} not found" >&2
  exit 1
fi

GBS=$(( 8 * NODES ))

echo "Target: ${NODES} node(s), GLOBAL_BATCH_SIZE=${GBS}"
echo "Manifest: ${MANIFEST}"
echo

before=$(shasum -a 256 "${MANIFEST}" | cut -d' ' -f1)

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT
cp "${MANIFEST}" "${tmp}"

# Anchored substitutions. Each pattern must match exactly once; the count check
# below fails loudly if the manifest's shape changed and a pattern went stale.
sed -E \
  -e "s/^  parallelism: [0-9]+$/  parallelism: ${NODES}/" \
  -e "s/^  completions: [0-9]+$/  completions: ${NODES}/" \
  -e "s/^  NNODES: \"[0-9]+\"$/  NNODES: \"${NODES}\"/" \
  -e "s/^  GLOBAL_BATCH_SIZE: \"[0-9]+\"$/  GLOBAL_BATCH_SIZE: \"${GBS}\"/" \
  "${tmp}" > "${MANIFEST}.new"

# Verify every field landed, rather than trusting sed's exit status (which is 0
# even when no pattern matches).
fail=0
expect() { # pattern description
  local n
  n=$(grep -cE "$1" "${MANIFEST}.new" || true)
  if [[ "${n}" -eq 1 ]]; then
    printf "  OK   %-46s %s\n" "$2" "1 match"
  else
    printf "  FAIL %-46s %s matches (want 1)\n" "$2" "${n}"
    fail=1
  fi
}
expect "^  parallelism: ${NODES}$"            "parallelism: ${NODES}"
expect "^  completions: ${NODES}$"            "completions: ${NODES}"
expect "^  NNODES: \"${NODES}\"$"             "NNODES: \"${NODES}\""
expect "^  GLOBAL_BATCH_SIZE: \"${GBS}\"$"    "GLOBAL_BATCH_SIZE: \"${GBS}\""

if [[ "${fail}" -ne 0 ]]; then
  echo
  echo "FATAL: not every field was updated. The manifest is UNCHANGED." >&2
  rm -f "${MANIFEST}.new"
  exit 1
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo
  echo "--dry-run: showing the diff, not writing."
  diff -u "${MANIFEST}" "${MANIFEST}.new" || true
  rm -f "${MANIFEST}.new"
  exit 0
fi

mv "${MANIFEST}.new" "${MANIFEST}"
after=$(shasum -a 256 "${MANIFEST}" | cut -d' ' -f1)

echo
if [[ "${before}" == "${after}" ]]; then
  echo "No change: the manifest was already set to ${NODES} node(s)."
else
  echo "Updated. sha256 ${before:0:12} -> ${after:0:12}"
fi
echo
echo "Apply with:"
echo "    kubectl apply -f g5/eks/pretrain.yaml"
if [[ "${NODES}" -eq 2 ]]; then
  echo
  echo "Reminder for 2 nodes:"
  echo "  * the dataset PVC must be ReadWriteMany (EFS). An EBS volume cannot"
  echo "    be mounted by two nodes and the second pod will hang Pending."
  echo "  * the node group needs 2 schedulable g5.8xlarge nodes."
fi
