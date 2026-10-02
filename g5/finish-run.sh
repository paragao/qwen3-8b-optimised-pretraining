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
# PREREQUISITES
#   * aws CLI with credentials for account 159553542841
#   * session-manager-plugin installed (you have it at /usr/local/bin)
#   * run from the root of this repo, on the branch holding the fix
#
# USAGE
#   ./g5/finish-run.sh
#   TRAIN_ITERS=50 ./g5/finish-run.sh          # longer run, steadier numbers
#   HF_TOKEN=hf_xxx ./g5/finish-run.sh         # not needed; Qwen3-8B is public
#
set -euo pipefail

INSTANCE_ID="${INSTANCE_ID:-i-09ee99ff5540c60ec}"
REGION="${REGION:-us-east-1}"
AZ="${AZ:-us-east-1b}"
PROFILE="${AWS_PROFILE_NAME:-compute-sa-team-Administrator}"
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
state=$(aws --profile "${PROFILE}" --region "${REGION}" ec2 describe-instances \
  --instance-ids "${INSTANCE_ID}" \
  --query "Reservations[0].Instances[0].State.Name" --output text)
if [[ "${state}" != "running" ]]; then
  echo "FATAL: instance ${INSTANCE_ID} is '${state}', not 'running'." >&2
  exit 1
fi
ping_status=$(aws --profile "${PROFILE}" --region "${REGION}" ssm describe-instance-information \
  --filters "Key=InstanceIds,Values=${INSTANCE_ID}" \
  --query "InstanceInformationList[0].PingStatus" --output text)
if [[ "${ping_status}" != "Online" ]]; then
  echo "FATAL: SSM agent is '${ping_status}', not 'Online'." >&2
  exit 1
fi
echo "    instance running, SSM agent Online."

say "Generating a one-shot ed25519 key"
ssh-keygen -t ed25519 -N "" -C "g5-validation-ephemeral" -f "${KEY}" >/dev/null

PROXY="aws --profile ${PROFILE} --region ${REGION} ssm start-session \
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
  aws --profile "${PROFILE}" --region "${REGION}" ec2-instance-connect send-ssh-public-key \
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
# it. Copy the three files directly instead.
scp "${SSH_OPTS[@]}" -q g5/train.py g5/run.sh g5/throughput.py \
  "${OS_USER}@${INSTANCE_ID}:${REMOTE_REPO}/g5/"
ssh "${SSH_OPTS[@]}" "${OS_USER}@${INSTANCE_ID}" \
  "chmod +x ${REMOTE_REPO}/g5/run.sh; cd ${REMOTE_REPO} && md5sum g5/train.py g5/run.sh g5/throughput.py | sed 's/^/    /'"
echo "    local checksums for comparison:"
md5sum g5/train.py g5/run.sh g5/throughput.py 2>/dev/null | sed 's/^/    /' || \
  md5 -r g5/train.py g5/run.sh g5/throughput.py | sed 's/^/    /'

say "Running the validation (this is the long step)"
ssh "${SSH_OPTS[@]}" "${OS_USER}@${INSTANCE_ID}" \
  "cd ${REMOTE_REPO} && TRAIN_ITERS='${TRAIN_ITERS:-20}' LOG_INTERVAL=1 HF_TOKEN='${HF_TOKEN:-}' ./g5/run.sh" \
  || echo "    run.sh exited non-zero -- the log is still parsed below for whatever it reached"

say "Locating the log and parsing throughput on the instance"
ssh "${SSH_OPTS[@]}" "${OS_USER}@${INSTANCE_ID}" bash -s <<REMOTE || true
set -u
log=\$(ls -t /home/ubuntu/qwen3-g5/run/logs/*.log 2>/dev/null | head -1)
if [ -z "\${log}" ]; then
  echo "FATAL: no log file found under /home/ubuntu/qwen3-g5/run/logs/" >&2
  exit 1
fi
echo "    log: \${log}"
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
if [[ -n "${remote_log}" ]]; then
  local_log="g5/results/run-$(date +%Y%m%d-%H%M%S).log"
  scp "${SSH_OPTS[@]}" -q "${OS_USER}@${INSTANCE_ID}:${remote_log}" "${local_log}"
  echo "    saved: ${local_log}"
  if git check-ignore -q "${local_log}" 2>/dev/null; then
    echo "    WARNING: ${local_log} is gitignored and will not be committable." >&2
  fi
fi

say "Done"
cat <<'NOTE'
    The security group was NOT modified -- it still has zero ingress rules.
    The ephemeral SSH key expired after 60s and was never persisted on the
    instance; the local copy has been deleted.

    THE INSTANCE IS STILL RUNNING AND BILLING (~$2.45/hr). Tear it down with:
        ./g5/terminate-instance.sh i-09ee99ff5540c60ec
NOTE
