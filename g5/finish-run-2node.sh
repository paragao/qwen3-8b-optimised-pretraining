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
# Defined here rather than only inside the bootstrap heredoc, because the
# preflight and the cleanup trap both need it locally and `set -u` would
# otherwise abort on an unbound reference.
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/nemo:26.04}"
TRAIN_ITERS="${TRAIN_ITERS:-1000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
NUM_TOKENS="${NUM_TOKENS:-50000000}"
DATA_PATH="${DATA_PATH:-/workspace/run/datasets/c4_qwen3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n==> %s\n' "$*"; }

# ---------------------------------------------------------------- instance ids
# Parsed, not sourced: see the note in g5/terminate-instance.sh. An unquoted
# multi-id line in the record file made `source` abort under `set -e`, which is
# exactly how the first 2-node attempt failed.
_record_get() {   # key file
  sed -n "s/^$1=//p" "$2" 2>/dev/null | tail -1 \
    | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'\$//"
}

IDS=("$@")
if [[ "${#IDS[@]}" -eq 0 && -f "${HERE}/.last-instance-id" ]]; then
  _rec_region="$(_record_get REGION "${HERE}/.last-instance-id")"
  [[ -n "${_rec_region}" ]] && REGION="${_rec_region}"
  _rec_ids="$(_record_get INSTANCE_IDS "${HERE}/.last-instance-id")"
  if [[ -n "${_rec_ids}" ]]; then
    # shellcheck disable=SC2206
    IDS=(${_rec_ids})
  fi
  echo "Read ${HERE}/.last-instance-id: ${IDS[*]:-<none>} in ${REGION}"
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
  if [[ "${st}" != "running" ]]; then
    echo "FATAL: ${id} is '${st}' in ${REGION}." >&2
    echo "       If that reads empty or 'None', the instance is probably in a" >&2
    echo "       DIFFERENT region. GPU capacity often forces one. Retry with" >&2
    echo "       REGION=<region> ./g5/finish-run-2node.sh ${IDS[*]}" >&2
    exit 1
  fi
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
RUN_LAUNCHED=0   # set to 1 once containers exist on the nodes

cleanup() {
  # Stop the REMOTE containers first. Killing the local ssh client does NOT
  # stop them, so a Ctrl-C on a hung run used to leave both nodes with a
  # training container holding MASTER_PORT (host networking) and the GPU --
  # which made the NEXT run die with EADDRINUSE on the rendezvous port. This
  # must happen BEFORE the control masters close, since it needs them.
  # Guarded on RUN_LAUNCHED so an early failure (bad args, SSM offline) does
  # not try to ssh to nodes that were never reached.
  if [[ "${RUN_LAUNCHED}" -eq 1 ]]; then
    for i in 0 1; do
      ssh -o ControlPath="${KEYDIR}/cm-${i}" -o ConnectTimeout=10 \
        "${OS_USER}@${IDS[$i]}" \
        "docker ps -q --filter ancestor='${CONTAINER_IMAGE}' | xargs -r docker kill" \
        >/dev/null 2>&1 || true
    done
  fi
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

set_opts() {       # index -> fills the global OPTS array
  # macOS ships bash 3.2, which has no `mapfile`/`readarray`, so read the
  # option list with a plain loop. The redirection MUST be a process
  # substitution rather than a pipe: a pipe would run the loop in a subshell
  # and OPTS would be empty back here.
  local i="$1" line
  OPTS=()
  while IFS= read -r line; do
    OPTS+=("${line}")
  done < <(ssh_opts_for "${i}")
  # Under `set -u` bash 3.2 treats "${OPTS[@]}" on an empty array as an unbound
  # variable, which would fail far from the cause. Fail here instead.
  [[ "${#OPTS[@]}" -ge 2 ]] || {
    echo "FATAL: ssh option list for node ${i} came back empty" >&2; exit 1; }
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
  set_opts "${i}"
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

# ------------------------------------------------------------------ bootstrap
# g5/launch-instance.sh deliberately passes no user-data, so a freshly launched
# instance has neither the repo nor the ~77 GB NeMo image. Without this step
# the scp below fails on a path that does not exist. Both nodes are done
# concurrently; each check is a no-op on an instance that already has them.
say "Bootstrapping both nodes (repo clone + container image)"
echo "    the image is ~77 GB, so a cold node takes ~20-30 min here"
BOOT_PIDS=()
for i in 0 1; do
  set_opts "${i}"
  ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" bash -s <<BOOT \
    > "${KEYDIR}/boot-${i}.log" 2>&1 &
set -euo pipefail
REPO_DIR="${REMOTE_REPO}"
IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/nemo:26.04}"
if [ ! -d "\${REPO_DIR}/.git" ]; then
  echo "cloning ${REPO_URL:-https://github.com/paragao/qwen3-8b-optimised-pretraining.git}"
  mkdir -p "\$(dirname "\${REPO_DIR}")"
  git clone --depth 1 \
    "${REPO_URL:-https://github.com/paragao/qwen3-8b-optimised-pretraining.git}" \
    "\${REPO_DIR}"
else
  echo "repo already present at \${REPO_DIR}"
fi
# preprocess.py must exist: prepare_c4.py lifts write_idx_file out of it.
test -f "\${REPO_DIR}/preprocessing/preprocess.py" \
  || { echo "FATAL: preprocessing/preprocess.py missing after clone" >&2; exit 1; }
mkdir -p /home/ubuntu/qwen3-g5/run/logs /home/ubuntu/qwen3-g5/run/datasets
if docker image inspect "\${IMAGE}" >/dev/null 2>&1; then
  echo "image already present: \${IMAGE}"
else
  echo "pulling \${IMAGE} (this is the long pole)"
  docker pull "\${IMAGE}"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
BOOT
  BOOT_PIDS+=($!)
  echo "    node ${i} bootstrap started"
done
boot_fail=0
# `wait` with NO ARGUMENTS returns 0 even when a background job failed, so it
# cannot detect failure (verified on bash 3.2 and 5.x). `wait PID` does return
# that job's real status, so wait on each one individually.
for pid in "${BOOT_PIDS[@]}"; do
  wait "${pid}" || boot_fail=1
done
for i in 0 1; do
  echo "    --- node ${i} bootstrap (tail) ---"
  tail -5 "${KEYDIR}/boot-${i}.log" | sed 's/^/      /'
done
if [[ "${boot_fail}" -ne 0 ]]; then
  echo "FATAL: bootstrap failed on at least one node; see the tails above." >&2
  exit 1
fi

# ------------------------------------------------------------- copy the payload
say "Copying the changed g5/ files to both nodes"
PAYLOAD=(g5/train.py g5/run.sh g5/throughput.py g5/prepare_c4.py g5/prepare-c4.sh)
for i in 0 1; do
  set_opts "${i}"
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
PREP_PIDS=()
for i in 0 1; do
  set_opts "${i}"
  ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
    "cd ${REMOTE_REPO} && NUM_TOKENS='${NUM_TOKENS}' ./g5/prepare-c4.sh" \
    > "${KEYDIR}/prep-${i}.log" 2>&1 &
  PREP_PIDS+=($!)
done
prep_fail=0
# `wait -n` is bash 4.3+ and on bash 3.2 fails with a usage error, which would
# have set prep_fail=1 on EVERY run and aborted a healthy build.
for pid in "${PREP_PIDS[@]}"; do
  wait "${pid}" || prep_fail=1
done
for i in 0 1; do
  echo "    --- node ${i} prep (tail) ---"
  tail -6 "${KEYDIR}/prep-${i}.log" | sed 's/^/      /'
done
if [[ "${prep_fail}" -ne 0 ]]; then
  echo "FATAL: dataset build failed on at least one node; see the tails above." >&2
  exit 1
fi

say "Verifying both nodes hold the SAME dataset before spending a run"
# Presence is not enough. There is no shared filesystem here, so each node
# builds c4 independently by streaming from the Hub, and two independent
# streaming builds CAN diverge (a truncated shard, an unauthenticated-request
# rate limit, a different stopping point against the token budget).
#
# Non-identical data across data-parallel ranks is not a cosmetic problem.
# Each rank derives its own sample and shuffle index from its own .bin/.idx,
# so differing files give differing sample counts, differing microbatch
# counts, and therefore DIFFERENT SEQUENCES OF COLLECTIVES. The ranks then
# desync and one waits on a collective the other never issues, which NCCL
# reports 600s later as an opaque "WorkNCCL(SeqNum=N) timed out" -- with no
# hint that the datasets were the cause. Catch it here instead, for the price
# of two checksums, because the alternative is 10 minutes of billed silence
# followed by a misleading error.
ref_fp=""
for i in 0 1; do
  set_opts "${i}"
  host_data="/home/ubuntu/qwen3-g5/run${DATA_PATH#/workspace/run}"
  fp=$(ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" bash -s <<REMOTE || true
set -u
b='${host_data}.bin'
x='${host_data}.idx'
test -f "\$b" && test -f "\$x" || { echo MISSING; exit 0; }
printf '%s %s %s %s' \
  "\$(stat -c %s "\$b")" "\$(stat -c %s "\$x")" \
  "\$(md5sum "\$b" | cut -d' ' -f1)" "\$(md5sum "\$x" | cut -d' ' -f1)"
REMOTE
)
  if [[ -z "${fp}" || "${fp}" == "MISSING" ]]; then
    echo "FATAL: node ${i} has no dataset at ${host_data}.{bin,idx}." >&2
    echo "       Training would silently fall back to the MOCK dataset." >&2
    exit 1
  fi
  # read, not `set --`: `set --` would clobber the script's own positional
  # parameters, which the instance-id arguments are still read from.
  bin_sz=""; idx_sz=""; bin_md5=""; idx_md5=""
  read -r bin_sz idx_sz bin_md5 idx_md5 <<EOF
${fp}
EOF
  echo "    node ${i}: .bin ${bin_sz} bytes (md5 ${bin_md5:0:12}), .idx ${idx_sz} bytes (md5 ${idx_md5:0:12})"
  if [[ -z "${ref_fp}" ]]; then
    ref_fp="${fp}"
  elif [[ "${ref_fp}" != "${fp}" ]]; then
    echo "" >&2
    echo "FATAL: the two nodes hold DIFFERENT datasets." >&2
    echo "       node 0 : ${ref_fp}" >&2
    echo "       node ${i} : ${fp}" >&2
    echo "       Data-parallel ranks must index byte-identical data, or they" >&2
    echo "       compute different sample counts, issue different sequences of" >&2
    echo "       collectives, and hang in NCCL ~600s later with an error that" >&2
    echo "       says nothing about the dataset." >&2
    echo "       Rebuild on both nodes so the checksums match:" >&2
    echo "         ./g5/prepare-c4.sh      # on each node" >&2
    exit 1
  fi
done
echo "    both nodes: checksums MATCH -- data-parallel ranks will index identical data"

# ----------------------------------------------------------------- the run
# Rank 1 starts FIRST. torchrun's static rendezvous has rank 0 host the store,
# and rank 1 retries until it is up -- but starting rank 1 first means neither
# side is waiting on a process that has not been launched yet.
# ------------------------------------------------- preflight: clear stale state
# An interrupted previous run leaves its CONTAINERS RUNNING on both nodes. The
# local cleanup only closes the ssh control masters, and killing an ssh client
# does not stop the remote container, so Ctrl-C on a hung run leaves behind:
#   * rank 0's container holding MASTER_PORT (host networking) -> the next run
#     dies with "DistNetworkError ... EADDRINUSE ... port: 29500", and rank 1
#     then fails too because its rendezvous has no server
#   * both containers holding their GPU, so even a free port would OOM
# Clear it rather than reporting it: the stale container is never wanted, and
# leaving the operator to hand-run docker kill on two nodes is the step that
# gets skipped. Only this image is targeted, so nothing else on the box is hit.
say "Preflight: clearing any stale run on both nodes"
for i in 0 1; do
  set_opts "${i}"
  stale="$(ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
    "docker ps -q --filter ancestor='${CONTAINER_IMAGE}' | wc -l" 2>/dev/null || echo 0)"
  stale="$(echo "${stale}" | tr -d '[:space:]')"
  if [[ "${stale}" -gt 0 ]]; then
    echo "    node ${i}: ${stale} stale container(s) from a previous run -- killing"
    ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
      "docker ps -q --filter ancestor='${CONTAINER_IMAGE}' | xargs -r docker kill" \
      >/dev/null 2>&1 || true
  else
    echo "    node ${i}: no stale containers"
  fi
done

# Assert the port is actually free rather than assuming the kill worked: a
# container in a wedged state can survive `docker kill`, and the failure mode
# we are preventing is precisely a still-bound MASTER_PORT.
for i in 0 1; do
  set_opts "${i}"
  for attempt in 1 2 3 4 5; do
    in_use="$(ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
      "ss -ltn 2>/dev/null | grep -c ':${MASTER_PORT} ' || true" 2>/dev/null || echo 0)"
    in_use="$(echo "${in_use}" | tr -d '[:space:]')"
    [[ -z "${in_use}" ]] && in_use=0
    [[ "${in_use}" -eq 0 ]] && break
    echo "    node ${i}: ${MASTER_PORT} still bound, waiting (${attempt}/5)"
    sleep 5
  done
  if [[ "${in_use}" -ne 0 ]]; then
    echo "FATAL: node ${i} still has a listener on ${MASTER_PORT}." >&2
    echo "       A previous run's container did not die. Inspect it with:" >&2
    echo "         aws ssm start-session --target ${IDS[$i]} --region ${REGION}" >&2
    echo "         docker ps ; sudo ss -ltnp | grep ${MASTER_PORT}" >&2
    exit 1
  fi
  echo "    node ${i}: ${MASTER_PORT} is free"
done

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

RUN_PIDS=()
RUN_LAUNCHED=1   # from here on, cleanup must stop the remote containers
for i in 1 0; do
  set_opts "${i}"
  ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
    "cd ${REMOTE_REPO} && ${RUN_ENV} NODE_RANK=${i} ./g5/run.sh" \
    > "${KEYDIR}/run-${i}.log" 2>&1 &
  RUN_PIDS[$i]=$!
  echo "    launched rank ${i} (${IDS[$i]})"
  [[ "${i}" -eq 1 ]] && sleep 5   # let rank 1 get as far as the rendezvous
done

# Indexed by rank, so a failure can be ATTRIBUTED rather than just flagged.
# A hung run produces NO output and `wait` blocks forever, so the nodes bill at
# ~$4.90/hr until a human notices. That cost 55 minutes on a NCCL bootstrap
# hang. Watch the two rank logs for growth instead, and abort if both go quiet.
# Growth is the right signal rather than elapsed time: a legitimate long step
# still logs, while a network-level hang produces nothing at all.
# PyTorch's ProcessGroupNCCL watchdog aborts an unmatched collective after its
# own timeout (600s by default, and NOT settable by environment variable -- it
# comes from the process group options). When this driver's STALL_TIMEOUT was
# also 600 the two timers were a dead heat, NCCL won, and the run died with a
# SIGABRT backtrace instead of this script's diagnosis. So keep a deliberate
# margin and ASSERT it, rather than leaving the ordering to chance.
NCCL_COLLECTIVE_TIMEOUT_SEC=600         # torch default; mirrored here, not set by us
STALL_TIMEOUT="${STALL_TIMEOUT:-420}"   # seconds of total silence before abort
STALL_POLL=30
if [[ "${STALL_TIMEOUT}" -ge "${NCCL_COLLECTIVE_TIMEOUT_SEC}" ]]; then
  echo "FATAL: STALL_TIMEOUT=${STALL_TIMEOUT}s must be BELOW NCCL's collective" >&2
  echo "       timeout of ${NCCL_COLLECTIVE_TIMEOUT_SEC}s, or NCCL aborts first and" >&2
  echo "       this driver never gets to explain why the run hung." >&2
  exit 1
fi
stall_abort=0
last_size=-1
quiet_for=0
while :; do
  any_alive=0
  for i in 0 1; do
    kill -0 "${RUN_PIDS[$i]}" 2>/dev/null && any_alive=1
  done
  [[ "${any_alive}" -eq 0 ]] && break

  size=0
  for i in 0 1; do
    s="$(wc -c < "${KEYDIR}/run-${i}.log" 2>/dev/null || echo 0)"
    size=$(( size + s ))
  done

  if [[ "${size}" -eq "${last_size}" ]]; then
    quiet_for=$(( quiet_for + STALL_POLL ))
  else
    quiet_for=0
    last_size="${size}"
  fi

  if [[ "${quiet_for}" -ge "${STALL_TIMEOUT}" ]]; then
    stall_abort=1
    echo "" >&2
    echo "FATAL: both ranks produced no output for ${quiet_for}s -- treating this" >&2
    echo "       as a hang and aborting so the nodes stop billing." >&2
    echo "" >&2
    # Two very different failures look identical from the outside, so decide
    # between them from the logs rather than always blaming the network.
    # If NCCL never reached "Init COMPLETE" the ranks never connected; if it
    # did, the transport works and a rank went missing from a later collective.
    if grep -q "Init COMPLETE" "${KEYDIR}/run-0.log" 2>/dev/null; then
      nccl_up=yes
    else
      nccl_up=no
    fi
    if [[ "${nccl_up}" == "no" ]]; then
      echo "       NCCL never reached 'Init COMPLETE', so the ranks never" >&2
      echo "       connected. Most likely the security group does not permit" >&2
      echo "       NCCL's ephemeral inbound ports between the nodes: the" >&2
      echo "       torchrun rendezvous on ${MASTER_PORT} can succeed while NCCL's" >&2
      echo "       own bootstrap sockets are dropped, which stalls silently." >&2
      echo "       Check it with:" >&2
      echo "         aws ec2 describe-security-groups --group-ids <sg> \\" >&2
      echo "           --query 'SecurityGroups[0].IpPermissions'" >&2
      echo "       It must span tcp/1-65535 sourced from the group itself," >&2
      echo "       which ./g5/fix-cluster-sg.sh converges." >&2
    else
      echo "       NCCL DID reach 'Init COMPLETE', so the transport is fine and" >&2
      echo "       the network is NOT the problem. One rank failed to arrive at" >&2
      echo "       a collective the others entered -- a rank-side crash or stall." >&2
      echo "       Look at the OTHER rank's log, not rank 0's: rank 0 only" >&2
      echo "       records that it waited. Both are saved under g5/results/." >&2
      echo "       Common causes: the two nodes indexing non-identical datasets," >&2
      echo "       an out-of-memory kill, or a full disk on one node." >&2
    fi
    echo "       Re-run with NCCL_DEBUG=INFO to see the transport NCCL picks." >&2
    for i in 0 1; do
      kill "${RUN_PIDS[$i]}" 2>/dev/null || true
    done
    # Free the GPUs too: killing the local ssh client does not stop the remote
    # container, which would keep holding the device on a still-billing node.
    for i in 0 1; do
      set_opts "${i}"
      ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
        'docker ps -q --filter ancestor=nvcr.io/nvidia/nemo:26.04 | xargs -r docker kill' \
        >/dev/null 2>&1 || true
    done
    break
  fi

  sleep "${STALL_POLL}"
done

run_fail=0
RUN_STATUS=()
for i in 0 1; do
  st=0
  wait "${RUN_PIDS[$i]}" || st=$?
  RUN_STATUS[$i]="${st}"
  [[ "${st}" -eq 0 ]] || run_fail=1
done
say "Rank exit status: rank 0 = ${RUN_STATUS[0]}, rank 1 = ${RUN_STATUS[1]}"
for i in 0 1; do
  echo "    --- rank ${i} (last 15 lines) ---"
  tail -15 "${KEYDIR}/run-${i}.log" | sed 's/^/      /'
done

# --------------------------------------------------------------- results
say "Parsing rank 0's throughput"
set_opts 0
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

# Retrieve EVERY rank's log, not just rank 0's. In a collective timeout the
# rank that FAILED TO ARRIVE holds the cause, and rank 0 only ever records
# that it waited. Keeping rank 0 alone is keeping the wrong half: a
# "SeqNum=N ALLREDUCE timed out" line on rank 0 is a symptom whose
# explanation is always in another rank's log.
say "Retrieving BOTH ranks' logs"
stamp="$(date +%Y%m%d-%H%M%S)"
for i in 0 1; do
  set_opts "${i}"
  remote_log=$(ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
    'ls -t /home/ubuntu/qwen3-g5/run/logs/*.log 2>/dev/null | head -1' || true)
  if [[ -z "${remote_log}" ]]; then
    echo "    WARNING: no log found on rank ${i} (${IDS[$i]})" >&2
    continue
  fi
  local_log="g5/results/run-2node-${stamp}-rank${i}.log"
  if scp "${OPTS[@]}" -q "${OS_USER}@${IDS[$i]}:${remote_log}" "${local_log}"; then
    echo "    rank ${i}: saved ${local_log} ($(wc -l < "${local_log}" | tr -d ' ') lines)"
    if git check-ignore -q "${local_log}" 2>/dev/null; then
      echo "    WARNING: ${local_log} is gitignored." >&2
    fi
  else
    echo "    WARNING: could not copy rank ${i}'s log from ${remote_log}" >&2
  fi
done

# The driver streams each rank's stdout into KEYDIR, which the cleanup trap
# deletes. That stream is the ONLY record of a rank that died before writing
# its own logfile, so preserve both copies when the run failed.
if [[ "${run_fail}" -ne 0 ]]; then
  for i in 0 1; do
    [[ -f "${KEYDIR}/run-${i}.log" ]] || continue
    streamed="g5/results/run-2node-${stamp}-rank${i}-stdout.log"
    cp "${KEYDIR}/run-${i}.log" "${streamed}"
    echo "    rank ${i}: preserved driver-streamed stdout at ${streamed}"
  done
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
