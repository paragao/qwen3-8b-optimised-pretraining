#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Retrieve training logs from the 2-node cluster WITHOUT launching anything.
#
# WHY THIS EXISTS
# g5/finish-run-2node.sh retrieves both ranks' logs as its last step, so a run
# it drives to completion leaves the logs in g5/results/. But the driver runs on
# a laptop, and the remote training does NOT die with it: `g5/run.sh` is started
# over ssh and the container keeps running if the local process is killed (a
# closed terminal, a SIGHUP, a lid shut, a `kill -9`). The cleanup trap handles
# EXIT/INT/TERM, and none of those fire on SIGKILL. In that case the only copy
# of the result is the log on the node, and before this script the only way to
# fetch it was to re-run the whole job -- which would overwrite it.
#
# NOT what happened on 2026-10-04. That run LOOKED like a dead driver for 40
# minutes -- local stream frozen at iteration 20, stale ControlMaster sockets,
# no matching process -- and the driver was in fact alive and finished normally
# at 19:28, retrieving both logs itself. The silence was block buffering
# (see the TEE_CMD note in g5/run.sh), not a death. This script is justified by
# the SIGKILL path the trap genuinely cannot cover, and by a run that completes
# but whose retrieval step fails, not by that incident.
#
# So: this script connects, copies the newest log off each node, parses rank 0's
# throughput, and reports whether a container is still holding the GPUs. It
# starts no training, builds no dataset, and writes nothing on either node.
#
# USAGE
#   ./g5/retrieve-logs.sh                     # reads g5/.last-instance-id
#   ./g5/retrieve-logs.sh i-0aaa i-0bbb       # explicit
#   REGION=us-west-2 ./g5/retrieve-logs.sh i-0aaa i-0bbb
set -euo pipefail

REGION="${REGION:-us-east-1}"
PROFILE="${AWS_PROFILE_NAME:-compute-sa-team-Administrator}"
OS_USER="${OS_USER:-ubuntu}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/nemo:26.04}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/home/ubuntu/qwen3-g5/run/logs}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n==> %s\n' "$*"; }

# Parsed, not sourced: an unquoted multi-id line would make `source` abort.
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
if [[ "${#IDS[@]}" -lt 1 ]]; then
  echo "FATAL: need at least 1 instance id, got ${#IDS[@]}." >&2
  echo "       ./g5/retrieve-logs.sh i-0aaa i-0bbb" >&2
  exit 1
fi
echo "Nodes: ${IDS[*]}  (region ${REGION})"

# ------------------------------------------------------- state and addressing
say "Confirming the instances are running with SSM Online"
declare -a AZS=()
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
    echo "FATAL: ${id} is '${st}' in ${REGION}. If that reads empty or 'None'," >&2
    echo "       the instance is probably in a different region; retry with" >&2
    echo "       REGION=<region> ./g5/retrieve-logs.sh ${IDS[*]}" >&2
    exit 1
  fi
  [[ "${ping}" == "Online" ]] || { echo "FATAL: ${id} SSM is ${ping}" >&2; exit 1; }
  AZS+=("${az}")
done

# ------------------------------------------------------------------ ssh setup
KEYDIR="$(mktemp -d)"
KEY="${KEYDIR}/id_ed25519"

cleanup() {
  local n="${#IDS[@]}" i=0
  while [[ "${i}" -lt "${n}" ]]; do
    ssh -O exit -o ControlPath="${KEYDIR}/cm-${i}" \
      "${OS_USER}@${IDS[$i]}" 2>/dev/null || true
    i=$(( i + 1 ))
  done
  rm -rf "${KEYDIR}"
}
trap cleanup EXIT

say "Generating a one-shot ed25519 key (never persisted on any instance)"
ssh-keygen -t ed25519 -N "" -C "g5-retrieve-ephemeral" -f "${KEY}" >/dev/null

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
    -o ControlPersist=10m \
    -o ProxyCommand="${PROXY}"
}

# bash 3.2 has no mapfile; the redirection MUST be a process substitution,
# because a pipe would run the loop in a subshell and OPTS would be empty here.
set_opts() {       # index -> fills the global OPTS array
  local i="$1" line
  OPTS=()
  while IFS= read -r line; do
    OPTS+=("${line}")
  done < <(ssh_opts_for "${i}")
  [[ "${#OPTS[@]}" -ge 2 ]] || {
    echo "FATAL: ssh option list for node ${i} came back empty" >&2; exit 1; }
}

push_key() {       # index -- the pushed key is valid for 60s
  local i="$1"
  aws --profile "${PROFILE}" --region "${REGION}" \
    ec2-instance-connect send-ssh-public-key \
    --instance-id "${IDS[$i]}" \
    --instance-os-user "${OS_USER}" \
    --availability-zone "${AZS[$i]}" \
    --ssh-public-key "$(cat "${KEY}.pub")" \
    --output text >/dev/null
}

connect() {        # index -- re-pushes the key, then proves the hop works
  local i="$1" delay=4 attempt ok=0
  set_opts "${i}"
  for attempt in 1 2 3 4 5; do
    if push_key "${i}" && ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" true 2>/dev/null; then
      ok=1; break
    fi
    echo "    node ${i} attempt ${attempt} failed; retrying in ${delay}s"
    sleep "${delay}"
    delay=$(( delay * 2 ))
  done
  [[ "${ok}" -eq 1 ]] || { echo "FATAL: could not reach node ${i}" >&2; exit 1; }
}

# ------------------------------------------------- what is still on the nodes
# A driver killed mid-run leaves its containers running. That matters twice:
# the GPUs stay held so the next run cannot start, and a container still
# training means the log being copied is INCOMPLETE.
say "Checking what is still running on each node"
STILL_RUNNING=0
n="${#IDS[@]}"; i=0
while [[ "${i}" -lt "${n}" ]]; do
  connect "${i}"
  running=$(ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
    "sudo docker ps -q --filter ancestor=${CONTAINER_IMAGE} 2>/dev/null | wc -l" \
    | tr -d ' ') || running=0
  gpu=$(ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
    'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' 2>/dev/null \
    || echo "unavailable")
  echo "    node ${i} (${IDS[$i]}): ${running} container(s) from ${CONTAINER_IMAGE}, GPU ${gpu}"
  if [[ "${running}" -gt 0 ]]; then
    STILL_RUNNING=$(( STILL_RUNNING + 1 ))
  fi
  i=$(( i + 1 ))
done

if [[ "${STILL_RUNNING}" -gt 0 ]]; then
  echo
  echo "    NOTE: ${STILL_RUNNING} node(s) still have a training container. The log"
  echo "          copied below is a SNAPSHOT of a run still in progress, and the"
  echo "          GPUs are held, so a new run would fail to allocate. Re-run this"
  echo "          script once the containers exit, or stop them with:"
  i=0
  while [[ "${i}" -lt "${n}" ]]; do
    echo "            # node ${i}: sudo docker kill \$(sudo docker ps -q --filter ancestor=${CONTAINER_IMAGE})"
    i=$(( i + 1 ))
  done
fi

# --------------------------------------------------------------- retrieve logs
say "Retrieving the newest log from each node"
mkdir -p "${HERE}/results"
stamp="$(date +%Y%m%d-%H%M%S)"
i=0
while [[ "${i}" -lt "${n}" ]]; do
  set_opts "${i}"
  remote_log=$(ssh "${OPTS[@]}" "${OS_USER}@${IDS[$i]}" \
    "ls -t ${REMOTE_LOG_DIR}/*.log 2>/dev/null | head -1" || true)
  if [[ -z "${remote_log}" ]]; then
    echo "    WARNING: no log found on node ${i} (${IDS[$i]}) in ${REMOTE_LOG_DIR}" >&2
    i=$(( i + 1 )); continue
  fi
  local_log="${HERE}/results/retrieved-${stamp}-rank${i}.log"
  # One explicit remote path, never a glob: scp has used SFTP by default since
  # OpenSSH 9.0 and no longer expands a remote wildcard through the shell.
  if scp "${OPTS[@]}" -q "${OS_USER}@${IDS[$i]}:${remote_log}" "${local_log}"; then
    echo "    node ${i}: ${remote_log}"
    echo "              -> ${local_log} ($(wc -l < "${local_log}" | tr -d ' ') lines)"
  else
    echo "    WARNING: could not copy node ${i}'s log from ${remote_log}" >&2
  fi
  i=$(( i + 1 ))
done

# ------------------------------------------------------------------ throughput
# throughput.py takes ONE positional logfile, so parse each rank separately.
# Every rank is parsed, not just rank 0: Megatron prints the per-iteration
# record with print_rank_last, so on a 2-node run the timing lines are on the
# LAST rank and rank 0's log has none at all.
if [[ -f "${HERE}/throughput.py" ]]; then
  i=0
  while [[ "${i}" -lt "${n}" ]]; do
    one="${HERE}/results/retrieved-${stamp}-rank${i}.log"
    if [[ -f "${one}" ]]; then
      say "Parsing throughput from rank ${i}"
      python3 "${HERE}/throughput.py" "${one}" --seq-len "${SEQ_LENGTH:-1024}" || true
    fi
    i=$(( i + 1 ))
  done
fi

say "Done"
