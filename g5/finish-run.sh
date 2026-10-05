#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Finish the g5.8xlarge validation and report tokens/sec per training step.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# The agent that wrote the validation could not execute on the instance: the
# host security policy blocks `ssm send-command`, `ssm start-session` and
# `ec2 authorize-security-group-ingress`. Everything else was prepared and
# committed, so this script carries out the last step as the operator.
#
# SECURITY POSTURE
# ----------------
# This uses SSH tunnelled over SSM Session Manager, which needs NO inbound
# security group rule at all -- the SSM agent dials out and sshd is reached on
# the instance's own loopback. The security group keeps zero ingress.
# Authentication is a one-shot ed25519 key pushed via EC2 Instance Connect,
# valid for 60 seconds, never written to the instance's persistent
# authorized_keys. No port is opened to 0.0.0.0/0.
#
# PREREQUISITES -- see g5/PREREQUISITES.md for the full list and a check block
#   * aws CLI v2 with working credentials (any profile; none is forced)
#   * session-manager-plugin installed -- REQUIRED, it is how this reaches the
#     instance, and it is a separate install from the aws CLI
#   * run from the root of this repo
#
# USAGE
#   ./g5/finish-run.sh
#   TRAIN_ITERS=50 ./g5/finish-run.sh          # longer run, steadier numbers
#   HF_TOKEN=hf_xxx ./g5/finish-run.sh         # not needed; Qwen3-8B is public
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Read a KEY=VALUE out of the launcher's record WITHOUT sourcing it. Sourcing a
# data file executes it, which is both unnecessary and fragile. Same parse as
# terminate-instance.sh and finish-run-2node.sh.
_record_get() {   # key file
  sed -n "s/^$1=//p" "$2" 2>/dev/null | tail -1 \
    | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'\$//"
}

# Target whatever launch-instance.sh last created, exactly as the 2-node driver,
# retrieve-logs.sh and terminate-instance.sh already do. This used to carry a
# hardcoded instance id and an us-east-1 default, so the documented flow --
# launch, prepare, run -- pointed step 3 at an instance belonging to whoever
# wrote the script, in the wrong region. An explicit INSTANCE_ID or REGION in
# the environment still wins, so a deliberate override is unaffected.
INSTANCE_ID="${INSTANCE_ID:-}"
REGION="${REGION:-}"
if [[ -z "${INSTANCE_ID}" && -f "${HERE}/.last-instance-id" ]]; then
  INSTANCE_ID="$(_record_get INSTANCE_ID "${HERE}/.last-instance-id")"
  _rec_region="$(_record_get REGION "${HERE}/.last-instance-id")"
  [[ -z "${REGION}" && -n "${_rec_region}" ]] && REGION="${_rec_region}"
  echo "Read ${HERE}/.last-instance-id: ${INSTANCE_ID:-<none>} in ${REGION:-<none>}"
fi
REGION="${REGION:-us-west-2}"
if [[ -z "${INSTANCE_ID}" ]]; then
  echo "FATAL: no instance id. Launch one first:" >&2
  echo "         NODES=1 ./g5/launch-instance.sh" >&2
  echo "       or name one explicitly:" >&2
  echo "         INSTANCE_ID=i-0123456789 ./g5/finish-run.sh" >&2
  exit 1
fi
AZ="${AZ:-}"
# No forced profile default. AWS_PROFILE_NAME (or AWS_PROFILE) is honoured if
# the caller set one; otherwise nothing is passed and the standard credential
# chain applies -- env vars, SSO, default profile, instance role. This used to
# default to a specific team profile, so every aws call died with
# ProfileNotFound for anyone cloning the repo. Exported rather than passed as
# --profile so the ssh ProxyCommand child inherits it too.
if [[ -n "${AWS_PROFILE_NAME:-}" ]]; then
  export AWS_PROFILE="${AWS_PROFILE_NAME}"
elif [[ -n "${AWS_PROFILE:-}" ]]; then
  export AWS_PROFILE
else
  unset AWS_PROFILE
fi
OS_USER="${OS_USER:-ubuntu}"
REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/qwen3-g5/qwen3-8b-optimised-pretraining}"

KEYDIR="$(mktemp -d)"
KEY="${KEYDIR}/id_ed25519"
CTL="${KEYDIR}/cm"
# The private key lives only in a 0700 temp dir and is destroyed on exit.
trap 'ssh -O exit -o ControlPath="${CTL}" "${OS_USER}@${INSTANCE_ID}" 2>/dev/null || true; rm -rf "${KEYDIR}"' EXIT

say() { printf '\n==> %s\n' "$*"; }

say "Checking repo state"
if [[ ! -f g5/train.py || ! -f g5/throughput.py ]]; then
  echo "FATAL: run this from the repo root; g5/train.py and g5/throughput.py must exist." >&2
  exit 1
fi
# The fix must actually be present in the file about to be copied.
if ! grep -q "cfg.logger.tensorboard_dir" g5/train.py; then
  echo "FATAL: g5/train.py does not contain the tensorboard_dir fix." >&2
  echo "       Check out branch feat/g5-single-gpu-validation first." >&2
  exit 1
fi
if ! grep -q "tokens per step" g5/train.py; then
  echo "FATAL: g5/train.py lacks the throughput instrumentation." >&2
  exit 1
fi
echo "    g5/train.py has both the tensorboard_dir fix and throughput reporting."

say "Confirming the instance is running and the SSM agent is online"
# State and AZ in ONE query, as finish-run-2node.sh does. The AZ is required by
# ec2-instance-connect send-ssh-public-key below and the launcher does not
# record it, so deriving it from the instance is the only way that stays correct
# for an explicitly-passed INSTANCE_ID in any region.
read -r state inst_az < <(aws --region "${REGION}" ec2 describe-instances \
  --instance-ids "${INSTANCE_ID}" \
  --query "Reservations[0].Instances[0].[State.Name,Placement.AvailabilityZone]" \
  --output text)
if [[ "${state}" != "running" ]]; then
  echo "FATAL: instance ${INSTANCE_ID} is '${state}', not 'running'." >&2
  exit 1
fi
# An explicit AZ in the environment still wins, but it is no longer required.
AZ="${AZ:-${inst_az}}"
if [[ -z "${AZ}" || "${AZ}" == "None" ]]; then
  echo "FATAL: could not determine the availability zone of ${INSTANCE_ID}." >&2
  exit 1
fi
echo "    ${INSTANCE_ID} is running in ${AZ}."
ping_status=$(aws --region "${REGION}" ssm describe-instance-information \
  --filters "Key=InstanceIds,Values=${INSTANCE_ID}" \
  --query "InstanceInformationList[0].PingStatus" --output text)
if [[ "${ping_status}" != "Online" ]]; then
  echo "FATAL: SSM agent is '${ping_status}', not 'Online'." >&2
  exit 1
fi
echo "    instance running, SSM agent Online."

say "Generating a one-shot ed25519 key"
ssh-keygen -t ed25519 -N "" -C "g5-validation-ephemeral" -f "${KEY}" >/dev/null

PROXY="aws --region ${REGION} ssm start-session \
--target %h --document-name AWS-StartSSHSession --parameters portNumber=%p"

SSH_OPTS=(
  -i "${KEY}"
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o IdentitiesOnly=yes
  -o ConnectTimeout=60
  -o ControlMaster=auto
  -o ControlPath="${CTL}"
  -o ControlPersist=8h
  -o ProxyCommand="${PROXY}"
)

push_key() {
  aws --region "${REGION}" ec2-instance-connect send-ssh-public-key \
    --instance-id "${INSTANCE_ID}" \
    --instance-os-user "${OS_USER}" \
    --availability-zone "${AZ}" \
    --ssh-public-key "$(cat "${KEY}.pub")" \
    --output text >/dev/null
}

say "Opening the SSH-over-SSM session (exponential backoff, 5 attempts)"
delay=4
connected=0
for attempt in 1 2 3 4 5; do
  # The pushed key is valid for 60s, so it is re-pushed before each attempt.
  if push_key && ssh "${SSH_OPTS[@]}" "${OS_USER}@${INSTANCE_ID}" true 2>/dev/null; then
    connected=1
    break
  fi
  echo "    attempt ${attempt} failed; retrying in ${delay}s"
  sleep "${delay}"
  delay=$(( delay * 2 ))
done
if [[ "${connected}" -ne 1 ]]; then
  echo "FATAL: could not establish the SSH-over-SSM session after 5 attempts." >&2
  exit 1
fi
ssh "${SSH_OPTS[@]}" "${OS_USER}@${INSTANCE_ID}" \
  'echo "    connected as $(whoami)@$(hostname)"; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | sed "s/^/    GPU: /"'

say "Copying the fixed scripts to the instance"
# The fix branch is local-only (never pushed), so the instance cannot git fetch
# it. Copy the changed files directly instead.
# NOTE: g5/prepare_c4.py lifts write_idx_file out of
# preprocessing/preprocess.py, which the instance already has from its clone of
# origin and which is NOT modified here -- so it does not need copying.
PAYLOAD=(g5/train.py g5/run.sh g5/throughput.py g5/prepare_c4.py g5/prepare-c4.sh)
scp "${SSH_OPTS[@]}" -q "${PAYLOAD[@]}" \
  "${OS_USER}@${INSTANCE_ID}:${REMOTE_REPO}/g5/"
ssh "${SSH_OPTS[@]}" "${OS_USER}@${INSTANCE_ID}" \
  "chmod +x ${REMOTE_REPO}/g5/run.sh ${REMOTE_REPO}/g5/prepare-c4.sh; \
   cd ${REMOTE_REPO} && md5sum ${PAYLOAD[*]} | sed 's/^/    /'"
echo "    local checksums for comparison:"
md5sum "${PAYLOAD[@]}" 2>/dev/null | sed 's/^/    /' || \
  md5 -r "${PAYLOAD[@]}" | sed 's/^/    /'

if [[ "${PREPARE_C4:-0}" == "1" ]]; then
  say "Building the c4 dataset on the instance (PREPARE_C4=1)"
  echo "    budget: ${NUM_TOKENS:-50000000} tokens; this is CPU-only and does not touch the GPU"
  ssh "${SSH_OPTS[@]}" "${OS_USER}@${INSTANCE_ID}" \
    "cd ${REMOTE_REPO} && \
     NUM_TOKENS='${NUM_TOKENS:-50000000}' \
     DOC_LENGTH='${DOC_LENGTH:-4096}' \
     HF_TOKEN='${HF_TOKEN:-}' \
     ./g5/prepare-c4.sh"
  # If the caller did not name a DATA_PATH, default to what prepare-c4.sh built.
  # Without this the training run would silently fall back to the MOCK dataset
  # after paying for the whole download, which is the trap this guards.
  DATA_PATH="${DATA_PATH:-/workspace/run/datasets/c4_qwen3}"
  echo "    DATA_PATH for the run: ${DATA_PATH}"
fi

if [[ -n "${DATA_PATH:-}" ]]; then
  say "Verifying the dataset exists before spending a training run on it"
  # DATA_PATH is an IN-CONTAINER path, because run.sh passes it straight
  # through to the container. Translate the /workspace/run mount prefix back to
  # the host path to test for the file over ssh.
  RUN_BASE_HOST="${RUN_BASE_HOST:-/home/ubuntu/qwen3-g5/run}"
  HOST_DATA="${DATA_PATH/#\/workspace\/run/${RUN_BASE_HOST}}"
  echo "    container path : ${DATA_PATH}"
  echo "    host path      : ${HOST_DATA}"
  # train.py checks this too, but failing here costs seconds instead of a
  # container start, and says what to do about it.
  if ! ssh "${SSH_OPTS[@]}" "${OS_USER}@${INSTANCE_ID}" \
      "test -f '${HOST_DATA}.idx' && test -f '${HOST_DATA}.bin'"; then
    echo "FATAL: DATA_PATH=${DATA_PATH} but ${HOST_DATA}.{bin,idx} is not on" >&2
    echo "       the instance. Build it first with:" >&2
    echo "         PREPARE_C4=1 ./g5/finish-run.sh" >&2
    exit 1
  fi
  # Advisory only: the test -f gate above is the hard requirement. This extra
  # structural check needs numpy on the HOST, which is not guaranteed, so it
  # must not abort the run under `set -o pipefail`.
  ssh "${SSH_OPTS[@]}" "${OS_USER}@${INSTANCE_ID}" \
    "cd ${REMOTE_REPO} && python3 g5/prepare_c4.py --verify-only '${HOST_DATA}' 2>&1" \
    | sed 's/^/    /' || echo "    (structural check skipped: no numpy on the host)"
  echo "    this run uses REAL data, not the mock dataset"
fi

say "Running the validation (this is the long step)"
# Record the newest existing log FIRST. If this run dies before writing its own
# log, the `ls -t | head -1` below would otherwise pick up the PREVIOUS run's
# log and parse it -- reporting the old profile's throughput as if it were this
# run's result. The guard after the run refuses to parse a stale file.
PRE_RUN_LOG=$(ssh "${SSH_OPTS[@]}" "${OS_USER}@${INSTANCE_ID}" \
  'ls -t /home/ubuntu/qwen3-g5/run/logs/*.log 2>/dev/null | head -1' || true)
echo "    newest log before this run: ${PRE_RUN_LOG:-<none>}"
# Every architecture knob must be forwarded explicitly. ssh does not inherit
# the caller's environment, so a variable that is not named here is SILENTLY
# DROPPED and the remote run falls back to the `smoke` defaults in train.py --
# which looks like a successful run of the profile you asked for. Keep this
# list in sync with the -e flags in g5/run.sh.
echo "    forwarding architecture: NUM_LAYERS='${NUM_LAYERS:-}' HIDDEN_SIZE='${HIDDEN_SIZE:-}'" \
     "FFN_HIDDEN_SIZE='${FFN_HIDDEN_SIZE:-}' NUM_ATTENTION_HEADS='${NUM_ATTENTION_HEADS:-}'" \
     "NUM_QUERY_GROUPS='${NUM_QUERY_GROUPS:-}' SEQ_LENGTH='${SEQ_LENGTH:-}'"
echo "    (empty value => train.py default, i.e. the 'smoke' profile)"
ssh "${SSH_OPTS[@]}" "${OS_USER}@${INSTANCE_ID}" \
  "cd ${REMOTE_REPO} && \
   TRAIN_ITERS='${TRAIN_ITERS:-20}' LOG_INTERVAL=1 HF_TOKEN='${HF_TOKEN:-}' \
   NUM_LAYERS='${NUM_LAYERS:-}' \
   HIDDEN_SIZE='${HIDDEN_SIZE:-}' \
   FFN_HIDDEN_SIZE='${FFN_HIDDEN_SIZE:-}' \
   NUM_ATTENTION_HEADS='${NUM_ATTENTION_HEADS:-}' \
   NUM_QUERY_GROUPS='${NUM_QUERY_GROUPS:-}' \
   SEQ_LENGTH='${SEQ_LENGTH:-}' \
   MICRO_BATCH_SIZE='${MICRO_BATCH_SIZE:-}' \
   GLOBAL_BATCH_SIZE='${GLOBAL_BATCH_SIZE:-}' \
   DATA_PATH='${DATA_PATH:-}' \
   ./g5/run.sh" \
  || echo "    run.sh exited non-zero -- the log is still parsed below for whatever it reached"

say "Locating the log and parsing throughput on the instance"
ssh "${SSH_OPTS[@]}" "${OS_USER}@${INSTANCE_ID}" bash -s <<REMOTE || true
set -u
log=\$(ls -t /home/ubuntu/qwen3-g5/run/logs/*.log 2>/dev/null | head -1)
if [ -z "\${log}" ]; then
  echo "FATAL: no log file found under /home/ubuntu/qwen3-g5/run/logs/" >&2
  exit 1
fi
if [ "\${log}" = "${PRE_RUN_LOG}" ]; then
  echo "FATAL: the newest log (\${log}) is the SAME file that existed before" >&2
  echo "       this run started. This run wrote no log of its own, so anything" >&2
  echo "       parsed from it belongs to the PREVIOUS run, not this one." >&2
  echo "       Refusing to report a stale result." >&2
  exit 1
fi
echo "    log: \${log}  (new, distinct from the pre-run log)"
echo
echo "    --- resolved geometry: confirm this is the profile you asked for ---"
grep -E "num_layers|hidden_size|ffn_hidden_size|num_attention_heads|num_query_groups|seq_length|TOTAL " "\${log}" || true
echo
grep -E "VALIDATION COMPLETE|peak allocated|peak reserved|tokens per step|end-to-end|wall clock" "\${log}" || true
echo
python3 ${REMOTE_REPO}/g5/throughput.py "\${log}" || true
REMOTE

say "Retrieving the log to g5/results/ for the record"
# NOTE: .gitignore excludes 'logs/', so the log must NOT land in a logs/
# subdirectory or it would be silently unversionable. It goes directly into
# g5/results/ with a timestamped name instead.
remote_log=$(ssh "${SSH_OPTS[@]}" "${OS_USER}@${INSTANCE_ID}" \
  'ls -t /home/ubuntu/qwen3-g5/run/logs/*.log 2>/dev/null | head -1')
if [[ -n "${remote_log}" && "${remote_log}" == "${PRE_RUN_LOG}" ]]; then
  echo "    SKIPPED: the newest remote log is the pre-run log (${remote_log})." >&2
  echo "    This run produced no log of its own; not copying a stale file into" >&2
  echo "    g5/results/ where it would look like a fresh result." >&2
  remote_log=""
fi
if [[ -n "${remote_log}" ]]; then
  local_log="g5/results/run-$(date +%Y%m%d-%H%M%S).log"
  scp "${SSH_OPTS[@]}" -q "${OS_USER}@${INSTANCE_ID}:${remote_log}" "${local_log}"
  echo "    saved: ${local_log}"
  if git check-ignore -q "${local_log}" 2>/dev/null; then
    echo "    WARNING: ${local_log} is gitignored and will not be committable." >&2
  fi
fi

say "Done"
# Unquoted heredoc so ${INSTANCE_ID} expands -- the teardown hint must name the
# instance this run actually used. The literal dollar in the price is escaped.
cat <<NOTE
    The security group was NOT modified -- it still has zero ingress rules.
    The ephemeral SSH key expired after 60s and was never persisted on the
    instance; the local copy has been deleted.

    THE INSTANCE IS STILL RUNNING AND BILLING (~\$2.45/hr). Tear it down with:
        ./g5/terminate-instance.sh ${INSTANCE_ID}
NOTE
