#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Assert each scenario's two reproduction paths describe the SAME run.
#
# Every scenario is defined twice by necessity: scenario.env drives the
# direct-EC2 path (g5/finish-run.sh, g5/finish-run-2node.sh) and
# kustomization.yaml drives the Kubernetes path. Two copies of a geometry drift,
# and the drift is invisible -- both files stay valid YAML and valid shell, both
# paths keep running, and they quietly measure different models. This compares
# them key by key and fails on the first disagreement.
#
# It also asserts the couplings that are not a single value: NNODES must equal
# the training Job's parallelism and completions, because NNODES=2 with
# parallelism=1 hangs at the rendezvous waiting for a pod that was never
# created.
#
# Usage:
#   ./g5/scenario-check.sh                 # both scenarios
#   ./g5/scenario-check.sh single-node     # one
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIOS=("$@")
if [[ "${#SCENARIOS[@]}" -eq 0 ]]; then
  SCENARIOS=(single-node multi-node)
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "FATAL: kubectl not found; it provides the built-in kustomize this needs." >&2
  exit 1
fi

# Read a KEY=VALUE out of scenario.env WITHOUT sourcing it. Sourcing a data
# file executes its contents, so a malformed line both breaks the reader and
# can run arbitrary text -- and this file is read by a verification script,
# which is the worst place to hand over control.
env_get() {   # key file
  sed -n "s/^$1=//p" "$2" 2>/dev/null | tail -1 \
    | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'\$//"
}

fail=0
for s in "${SCENARIOS[@]}"; do
  dir="${HERE}/${s}"
  envf="${dir}/scenario.env"
  echo "==> ${s}"
  for f in "${envf}" "${dir}/kustomization.yaml"; do
    [[ -f "${f}" ]] || { echo "FATAL: ${f} missing" >&2; exit 1; }
  done

  built="$(mktemp)"
  if ! kubectl kustomize "${dir}" > "${built}" 2>"${built}.err"; then
    echo "    FAIL: kubectl kustomize ${dir} does not build:" >&2
    sed 's/^/      /' "${built}.err" >&2
    rm -f "${built}" "${built}.err"
    fail=1
    continue
  fi
  echo "    manifest builds"

  # Compare every key scenario.env declares against the built ConfigMap, and
  # check the Job couplings. Done in python because the comparison needs the
  # RESOLVED manifest, not the patch text: a patch that silently fails to apply
  # leaves the base value in place and a textual grep would not notice.
  if ! python3 - "${built}" "${envf}" "${s}" <<'PY'
import sys, yaml, re

built, envf, scenario = sys.argv[1], sys.argv[2], sys.argv[3]

docs = [d for d in yaml.safe_load_all(open(built)) if d]
cms = [d for d in docs if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "qwen3-config"]
jobs = [d for d in docs if d["kind"] == "Job" and d["metadata"]["name"] == "qwen3-pretrain"]
if len(cms) != 1 or len(jobs) != 1:
    print(f"    FAIL: expected 1 qwen3-config and 1 qwen3-pretrain Job, got "
          f"{len(cms)} and {len(jobs)}")
    sys.exit(1)
cm, job = cms[0]["data"], jobs[0]["spec"]

want = {}
for line in open(envf):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    m = re.match(r'^([A-Z][A-Z0-9_]*)=(.*)$', line)
    if m:
        want[m.group(1)] = m.group(2).strip().strip('"').strip("'")

if not want:
    print(f"    FAIL: no KEY=VALUE pairs parsed out of {envf}")
    sys.exit(1)

bad = []
for k, v in sorted(want.items()):
    got = cm.get(k)
    if got is None:
        bad.append(f"{k}: in scenario.env ({v!r}) but ABSENT from the manifest")
    elif str(got) != str(v):
        bad.append(f"{k}: scenario.env {v!r} != manifest {got!r}")

# The coupling. NNODES is one value in three places.
n = want.get("NNODES")
if n is not None:
    for field in ("parallelism", "completions"):
        if str(job.get(field)) != str(n):
            bad.append(f"Job {field}={job.get(field)!r} != NNODES={n!r} "
                       f"-- a mismatch hangs at the rendezvous")

# EFA is two settings that must agree, and either alone is a silent fault.
res = jobs[0]["spec"]["template"]["spec"]["containers"][0]["resources"]
has_efa = any("vpc.amazonaws.com/efa" in res.get(b, {}) for b in ("requests", "limits"))
use_efa = str(cm.get("USE_EFA", "0")) == "1"
if use_efa and not has_efa:
    bad.append("USE_EFA=1 but no vpc.amazonaws.com/efa resource -- the pod gets "
               "no device and NCCL falls back to TCP (measured 3x slower)")
if has_efa and not use_efa:
    bad.append("vpc.amazonaws.com/efa requested but USE_EFA != 1 -- the device "
               "is reserved and never used, and the pod may stay Pending")
if use_efa and str(cm.get("FI_EFA_USE_DEVICE_RDMA", "")) != "":
    bad.append("FI_EFA_USE_DEVICE_RDMA is set -- g5's EFA has no rdma-read and "
               "this aborts both ranks on startup")

if bad:
    print(f"    FAIL: {len(bad)} disagreement(s):")
    for b in bad:
        print(f"      - {b}")
    sys.exit(1)

print(f"    {len(want)} key(s) agree between scenario.env and the manifest")
print(f"    NNODES={n} matches Job parallelism and completions")
print(f"    EFA consistent (USE_EFA={cm.get('USE_EFA')!r}, device "
      f"{'requested' if has_efa else 'not requested'})")
PY
  then
    fail=1
  fi
  rm -f "${built}" "${built}.err"
done

echo
if [[ "${fail}" -ne 0 ]]; then
  echo "RESULT: FAIL -- a scenario's two paths do not describe the same run." >&2
  exit 1
fi
echo "RESULT: clean -- every scenario's EC2 and Kubernetes paths agree."
