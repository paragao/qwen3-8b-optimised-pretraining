#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Drive a 2-node c4 training run on two g5.8xlarge instances, from a laptop.
#
# WHY A SEPARATE SCRIPT
# g5/finish-run.sh is hardcoded to one instance and has driven three recorded
# measurements. Rather than restructure it, this is a sibling for the 2-node
# case. The single-node path stays byte-for-byte as validated.
#
# WHAT IT DOES
#   1. resolves both instance ids and rank 0's PRIVATE IPv4
#   2. pushes a 60-second ephemeral SSH key to each, over SSM (no SG rule, no
#      persisted authorized_keys)
#   3. copies the changed g5/ files to both
#   4. builds the c4 dataset on BOTH nodes -- there is no shared filesystem on
#      this path, so each needs its own copy (the EKS path uses one RWX volume)
#   5. starts rank 1 first, then rank 0, so the rendezvous has a listener
#   6. waits for both, then parses rank 0's throughput and retrieves its log
#
# NETWORK
# The torchrun rendezvous crosses between the two instances on MASTER_PORT.
# That traffic is permitted by ONE security-group rule whose source is the
# security group itself (added by g5/launch-instance.sh NODES=2), so only
# instances in that group can reach it. Nothing is exposed to the VPC or the
# internet, and this script does not modify any security group.
#
# USAGE
#   ./g5/finish-run-2node.sh                          # reads g5/.last-instance-id
#   ./g5/finish-run-2node.sh i-0aaa i-0bbb            # explicit
#   TRAIN_ITERS=1000 ./g5/finish-run-2node.sh
#   GLOBAL_BATCH_SIZE=16 ./g5/finish-run-2node.sh     # the default; see below
#
# GLOBAL_BATCH_SIZE defaults to 16, not 8. tokens/step is GBS x seq regardless
# of node count, so at GBS=8 a second node halves per-rank compute without
# shrinking the gradient all-reduce and buys ~5% on the smoke/wider profiles.
# At 16 it is ~1.8x. See the tables in g5/README.md.
set -euo pipefail

REGION="${REGION:-us-east-1}"
PROFILE="${AWS_PROFILE_NAME:-compute-sa-team-Administrator}"
OS_USER="${OS_USER:-ubuntu}"
REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/qwen3-g5/qwen3-8b-optimised-pretraining}"
MASTER_PORT="${MASTER_PORT:-29500}"
TRAIN_ITERS="${TRAIN_ITERS:-1000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
NUM_TOKENS="${NUM_TOKENS:-50000000}"
DATA_PATH="${DATA_PATH:-/workspace/run/datasets/c4_qwen3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n==> %s\n' "$*"; }

# ---------------------------------------------------------------- instance ids
IDS=("$@")
if [[ "${#IDS[@]}" -eq 0 && -f "${HERE}/.last-instance-id" ]]; then
  # shellcheck disable=SC1091
  source "${HERE}/.last-instance-id"
  REGION="${REGION:-us-east-1}"
  if [[ -n "${INSTANCE_IDS:-}" ]]; then
    # shellcheck disable=SC2206
    IDS=(${INSTANCE_IDS})
  fi
fi
if [[ "${#IDS[@]}" -ne 2 ]]; then
  echo "FATAL: need exactly 2 instance ids, got ${#IDS[@]}." >&2
  echo "       Launch a 2-node cluster first:" >&2
  echo "         NODES=2 ./g5/launch-instance.sh" >&2
  echo "       or pass both ids explicitly:" >&2
  echo "         ./g5/finish-run-2node.sh i-0aaa i-0bbb" >&2
  exit 1
fi
echo "Nodes: rank 0 = ${IDS[0]}, rank 1 = ${IDS[1]}  (region ${REGION})"

# ------------------------------------------------------------- repo preflight
say "Checking repo state"
for f in g5/train.py g5/run.sh g5/throughput.py g5/prepare_c4.py g5/prepare-c4.sh; do
  [[ -f "$f" ]] || { echo "FATAL: $f missing; run from the repo root." >&2; exit 1; }
done
# train.py must be the DP-aware version, or WORLD_SIZE=2 silently trains with
# the single-node optimizer settings and no gradient-reduce overlap.
if ! grep -q "if WORLD_SIZE > 1:" g5/train.py; then
  echo "FATAL: g5/train.py is not DP-aware (no 'if WORLD_SIZE > 1' branch)." >&2
  echo "       A 2-node run needs it for overlap_grad_reduce and the" >&2
  echo "       distributed optimizer. Check out the branch that adds it." >&2
  exit 1
fi
if ! grep -q 'NNODES="${NNODES:-1}"' g5/run.sh; then
  echo "FATAL: g5/run.sh does not support NNODES." >&2
  exit 1
fi
echo "    train.py is DP-aware and run.sh supports NNODES."

# ------------------------------------------------------- state and addressing
say "Confirming both instances are running with SSM Online"
declare -a AZS=() PRIVATE=()
for id in "${IDS[@]}"; do
  read -r st az ip < <(aws --profile "${PROFILE}" --region "${REGION}" \
    ec2 describe-instances --instance-ids "${id}" \
    --query "Reservations[0].Instances[0].[State.Name,Placement.AvailabilityZone,PrivateIpAddress]" \
    --output text)
  ping=$(aws --profile "${PROFILE}" --region "${REGION}" \
    ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=${id}" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || echo None)
  echo "    ${id}: ${st}, ${az}, private ${ip}, SSM ${ping}"
  [[ "${st}" == "running" ]] || { echo "FATAL: ${id} is ${st}" >&2; exit 1; }
  [[ "${ping}" == "Online" ]] || { echo "FATAL: ${id} SSM is ${ping}" >&2; exit 1; }
  AZS+=("${az}")
  PRIVATE+=("${ip}")
done

if [[ "${AZS[0]}" != "${AZS[1]}" ]]; then
  echo "WARNING: nodes are in different AZs (${AZS[0]} vs ${AZS[1]})." >&2
  echo "         Every gradient all-reduce now crosses an AZ boundary, and the" >&2
  echo "         all-reduce is already the limiting factor. Expect worse than" >&2
  echo "         the scaling tables in g5/README.md." >&2
fi

MASTER_ADDR="${PRIVATE[0]}"
case "${MASTER_ADDR}" in
  10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[0-1].*) ;;
  *) echo "FATAL: rank 0's address ${MASTER_ADDR} is not RFC1918. Refusing to" >&2
     echo "       use a non-private rendezvous address." >&2; exit 1 ;;
esac
echo "    rendezvous will be ${MASTER_ADDR}:${MASTER_PORT} (private)"

# ------------------------------------------------------------------- ssh setup
KEYDIR="$(mktemp -d)"
chmod 700 "${KEYDIR}"
KEY="${KEYDIR}/id_ed25519"
cleanup() {
  for i in 0 1; do
    ssh -O exit -o ControlPath="${KEYDIR}/cm-${i}" \
      "${OS_USER}@${IDS[$i]}" 2>/dev/null || true
  done
  rm -rf "${KEYDIR}"
}
trap cleanup EXIT

say "Generating a one-shot ed25519 key (never persisted on either instance)"
ssh-keygen -t ed25519 -N "" -C "g5-2node-ephemeral" -f "${KEY}" >/dev/null

PROXY="aws --profile ${PROFILE} --region ${REGION} ssm start-session \
--target %h --document-name AWS-StartSSHSession --parameters portNumber=%p"

ssh_opts_for() {   # index -> prints the opts array for that node
  local i="$1"
  printf '%s\n' \
    -i "${KEY}" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -o IdentitiesOnly=yes \
    -o ConnectTimeout=60 \
    -o ControlMaster=auto \
    -o ControlPath="${KEYDIR}/cm-${i}" \
    -o ControlPersist=8h \
    -o ProxyCommand="${PROXY}"
}

push_key() {       # index
  local i="$1"
  aws --profile "${PROFILE}" --region "${REGION}" \
    ec2-instance-connect send-ssh-public-key \
    --instance-id "${IDS[$i]}" \
    --instance-os-user "${OS_USER}" \
    --availability-zone "${AZS[$i]}" \
    --ssh-public-key "$(cat "${KEY}.pub")" \
    --output text >/dev/null
}

say "Opening an SSH-over-SSM session to each node (5 attempts, backoff)"
for i in 0 1; do
  mapfile -t OPTS < <(ssh_opts_for "${i}")
  delay=4
  ok=0
  for attempt in 1 2 3 4 5; do
    # The pushed key is valid for 60s, so re-push before each attempt.
    if push_key "${i}" && ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" true 2>/dev/null; then
      ok=1; break
    fi
    echo "    node ${i} attempt ${attempt} failed; retrying in ${delay}s"
    sleep "${delay}"
    delay=$(( delay * 2 ))
  done
  [[ "${ok}" -eq 1 ]] || { echo "FATAL: could not reach node ${i}" >&2; exit 1; }
  ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
    'echo "    connected $(whoami)@$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed "s/^/      GPU: /"'
done

# ------------------------------------------------------------- copy the payload
say "Copying the changed g5/ files to both nodes"
PAYLOAD=(g5/train.py g5/run.sh g5/throughput.py g5/prepare_c4.py g5/prepare-c4.sh)
for i in 0 1; do
  mapfile -t OPTS < <(ssh_opts_for "${i}")
  scp "${OPTS[@]}" -q "${PAYLOAD[@]}" "${OS_USER}@${IDS[$i]}:${REMOTE_REPO}/g5/"
  ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
    "chmod +x ${REMOTE_REPO}/g5/run.sh ${REMOTE_REPO}/g5/prepare-c4.sh"
  echo "    node ${i}: copied ${#PAYLOAD[@]} files"
done
echo "    local checksums for comparison:"
md5sum "${PAYLOAD[@]}" 2>/dev/null | sed 's/^/      /' || \
  md5 -r "${PAYLOAD[@]}" | sed 's/^/      /'

# ------------------------------------------------------------- build c4 on both
# No shared filesystem on this path, so each node needs its own copy. Both
# builds run concurrently; prepare_c4.py reuses an existing verified dataset,
# so this is a no-op on a node that already has it.
say "Building the c4 dataset on BOTH nodes (concurrent, ~${NUM_TOKENS} tokens each)"
for i in 0 1; do
  mapfile -t OPTS < <(ssh_opts_for "${i}")
  ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
    "cd ${REMOTE_REPO} && NUM_TOKENS='${NUM_TOKENS}' ./g5/prepare-c4.sh" \
    > "${KEYDIR}/prep-${i}.log" 2>&1 &
done
prep_fail=0
wait -n || prep_fail=1
wait || prep_fail=1
for i in 0 1; do
  echo "    --- node ${i} prep (tail) ---"
  tail -6 "${KEYDIR}/prep-${i}.log" | sed 's/^/      /'
done
if [[ "${prep_fail}" -ne 0 ]]; then
  echo "FATAL: dataset build failed on at least one node; see the tails above." >&2
  exit 1
fi

say "Verifying the dataset exists on both nodes before spending a run"
for i in 0 1; do
  mapfile -t OPTS < <(ssh_opts_for "${i}")
  host_data="/home/ubuntu/qwen3-g5/run${DATA_PATH#/workspace/run}"
  if ! ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
      "test -f '${host_data}.bin' && test -f '${host_data}.idx'"; then
    echo "FATAL: node ${i} has no dataset at ${host_data}.{bin,idx}." >&2
    echo "       Training would silently fall back to the MOCK dataset." >&2
    exit 1
  fi
  echo "    node ${i}: ${host_data}.{bin,idx} present"
done

# ----------------------------------------------------------------- the run
# Rank 1 starts FIRST. torchrun's static rendezvous has rank 0 host the store,
# and rank 1 retries until it is up -- but starting rank 1 first means neither
# side is waiting on a process that has not been launched yet.
say "Starting the 2-node run (GBS=${GLOBAL_BATCH_SIZE}, ${TRAIN_ITERS} iters)"
echo "    rendezvous ${MASTER_ADDR}:${MASTER_PORT}, DATA_PATH=${DATA_PATH}"
echo "    rank 1 starts first so the rendezvous has both ends present"

RUN_ENV="NNODES=2 MASTER_ADDR='${MASTER_ADDR}' MASTER_PORT='${MASTER_PORT}' \
GLOBAL_BATCH_SIZE='${GLOBAL_BATCH_SIZE}' TRAIN_ITERS='${TRAIN_ITERS}' \
LOG_INTERVAL='${LOG_INTERVAL:-10}' DATA_PATH='${DATA_PATH}' \
NUM_LAYERS='${NUM_LAYERS:-}' HIDDEN_SIZE='${HIDDEN_SIZE:-}' \
FFN_HIDDEN_SIZE='${FFN_HIDDEN_SIZE:-}' \
NUM_ATTENTION_HEADS='${NUM_ATTENTION_HEADS:-}' \
NUM_QUERY_GROUPS='${NUM_QUERY_GROUPS:-}' SEQ_LENGTH='${SEQ_LENGTH:-}' \
MICRO_BATCH_SIZE='${MICRO_BATCH_SIZE:-}'"

for i in 1 0; do
  mapfile -t OPTS < <(ssh_opts_for "${i}")
  ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
    "cd ${REMOTE_REPO} && ${RUN_ENV} NODE_RANK=${i} ./g5/run.sh" \
    > "${KEYDIR}/run-${i}.log" 2>&1 &
  echo "    launched rank ${i} (${IDS[$i]})"
  [[ "${i}" -eq 1 ]] && sleep 5   # let rank 1 get as far as the rendezvous
done

run_fail=0
wait || run_fail=1
say "Both ranks exited (failure flag: ${run_fail})"
for i in 0 1; do
  echo "    --- rank ${i} (last 15 lines) ---"
  tail -15 "${KEYDIR}/run-${i}.log" | sed 's/^/      /'
done

# --------------------------------------------------------------- results
say "Parsing rank 0's throughput"
mapfile -t OPTS < <(ssh_opts_for 0)
ssh "${OPTS[@]}" "${OS_USER}@${IDS[0]}" bash -s <<REMOTE || true
set -u
log=\$(ls -t /home/ubuntu/qwen3-g5/run/logs/*.log 2>/dev/null | head -1)
[ -n "\${log}" ] || { echo "FATAL: no log on rank 0" >&2; exit 1; }
echo "    log: \${log}"
echo
echo "    --- resolved geometry and data source ---"
grep -E "num_layers +:|hidden_size +:|TOTAL |data parallel size|grad accum|dataset:|tokens per step" "\${log}" || true
echo
grep -E "VALIDATION COMPLETE|peak allocated|peak reserved|end-to-end|wall clock" "\${log}" || true
echo
python3 ${REMOTE_REPO}/g5/throughput.py "\${log}" || true
REMOTE

say "Retrieving rank 0's log"
remote_log=$(ssh "${OPTS[@]}" "${OS_USER}@${IDS[0]}" \
  'ls -t /home/ubuntu/qwen3-g5/run/logs/*.log 2>/dev/null | head -1')
if [[ -n "${remote_log}" ]]; then
  local_log="g5/results/run-2node-$(date +%Y%m%d-%H%M%S).log"
  scp "${OPTS[@]}" -q "${OS_USER}@${IDS[0]}:${remote_log}" "${local_log}"
  echo "    saved: ${local_log}"
  if git check-ignore -q "${local_log}" 2>/dev/null; then
    echo "    WARNING: ${local_log} is gitignored." >&2
  fi
fi

say "Done"
cat <<NOTE
    No security group was modified by this script. The ephemeral SSH keys
    expired after 60s and were never persisted on either instance; the local
    copies are destroyed on exit.

    BOTH INSTANCES ARE STILL RUNNING AND BILLING (~\$4.90/hr combined).
    Tear them down with:
        ./g5/terminate-instance.sh ${IDS[0]} ${IDS[1]}
NOTE
exit "${run_fail}"
