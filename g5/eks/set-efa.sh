#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Turn EFA on or off in g5/eks/pretrain.yaml.
#
# Two things must change together and only one of them is an env var:
#   USE_EFA in the ConfigMap          - selects the libfabric provider at run time
#   vpc.amazonaws.com/efa in resources - makes the device visible to the pod
#
# A Kubernetes resource request cannot be driven by an env var, so enabling EFA
# is a real edit to the manifest. Setting USE_EFA=1 WITHOUT the resource gets
# you a pod with no EFA device: NCCL falls back to TCP and the run is simply
# slower, which the training container detects and reports rather than hiding.
#
# Prerequisite, and this is the part that bites: the EFA device plugin must be
# installed, or an EFA-requesting pod stays Pending with
#   0/N nodes are available: Insufficient vpc.amazonaws.com/efa
#
#   helm repo add eks https://aws.github.io/eks-charts
#   helm install aws-efa-k8s-device-plugin --namespace kube-system \
#     eks/aws-efa-k8s-device-plugin
#
# The node group must also have been created with EFA enabled on the launch
# template; enabling it here does not add an interface to a running node.
#
# Usage:
#   ./g5/eks/set-efa.sh on
#   ./g5/eks/set-efa.sh off
#   ./g5/eks/set-efa.sh on --dry-run
set -euo pipefail

MANIFEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pretrain.yaml"
MODE="${1:-}"
DRY_RUN=0
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=1

if [[ "${MODE}" != "on" && "${MODE}" != "off" ]]; then
  echo "Usage: $0 <on|off> [--dry-run]" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "FATAL: ${MANIFEST} not found" >&2
  exit 1
fi

before=$(shasum -a 256 "${MANIFEST}" | cut -d' ' -f1)
work="${MANIFEST}.new"

# NOTE: POSIX character classes, not \s. BSD sed and BSD grep (macOS) do not
# support \s, and they fail SILENTLY by simply not matching -- which is how the
# first version of this script reported "0 matches (want 2)" and refused to
# write. [[:space:]] works on both BSD and GNU.
if [[ "${MODE}" == "on" ]]; then
  sed -E \
    -e 's/^  USE_EFA: "0"$/  USE_EFA: "1"/' \
    -e 's/^([[:space:]]*)# (vpc\.amazonaws\.com\/efa: 1)$/\1\2/' \
    "${MANIFEST}" > "${work}"
  WANT_USE='^  USE_EFA: "1"$'
  WANT_RES='^[[:space:]]+vpc\.amazonaws\.com/efa: 1$'
  WANT_RES_N=2
else
  sed -E \
    -e 's/^  USE_EFA: "1"$/  USE_EFA: "0"/' \
    -e 's/^([[:space:]]*)(vpc\.amazonaws\.com\/efa: 1)$/\1# \2/' \
    "${MANIFEST}" > "${work}"
  WANT_USE='^  USE_EFA: "0"$'
  WANT_RES='^[[:space:]]+# vpc\.amazonaws\.com/efa: 1$'
  WANT_RES_N=2
fi

# Verify both halves landed. sed exits 0 even when nothing matched, so the
# counts are the only real check.
fail=0
n_use=$(grep -cE "${WANT_USE}" "${work}" || true)
n_res=$(grep -cE "${WANT_RES}" "${work}" || true)
printf "  %-44s %s match (want 1)\n" "USE_EFA set for '${MODE}'" "${n_use}"
printf "  %-44s %s matches (want %s)\n" "efa resource lines" "${n_res}" "${WANT_RES_N}"
[[ "${n_use}" -eq 1 ]] || fail=1
[[ "${n_res}" -eq "${WANT_RES_N}" ]] || fail=1

if [[ "${fail}" -ne 0 ]]; then
  echo >&2
  echo "FATAL: not every field was updated; the manifest is UNCHANGED." >&2
  echo "       Expected 1 USE_EFA line and ${WANT_RES_N} efa resource lines" >&2
  echo "       (one under requests, one under limits)." >&2
  rm -f "${work}"
  exit 1
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo
  echo "--dry-run: showing the diff, not writing."
  diff -u "${MANIFEST}" "${work}" || true
  rm -f "${work}"
  exit 0
fi

mv "${work}" "${MANIFEST}"
after=$(shasum -a 256 "${MANIFEST}" | cut -d' ' -f1)
echo
if [[ "${before}" == "${after}" ]]; then
  echo "No change: EFA was already ${MODE}."
else
  echo "EFA ${MODE}. sha256 ${before:0:12} -> ${after:0:12}"
fi

if [[ "${MODE}" == "on" ]]; then
  cat <<'NOTE'

Before applying, confirm the cluster side:

  # 1. the EFA device plugin is installed
  kubectl -n kube-system get ds aws-efa-k8s-device-plugin

  # 2. nodes actually advertise the resource
  kubectl get nodes -o custom-columns=\
NAME:.metadata.name,EFA:.status.allocatable.'vpc\.amazonaws\.com/efa'

If that column is empty the pod will stay Pending with
"Insufficient vpc.amazonaws.com/efa". The node group must have been created
with EFA on its launch template -- this script cannot add an interface to a
running node.

EFA only matters at 2 nodes. At 1 node there is no inter-node traffic, so it
changes nothing.
NOTE
fi
