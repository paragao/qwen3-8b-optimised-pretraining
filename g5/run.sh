#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Qwen3 single-GPU stack validation on one g5.8xlarge (1x A10G 24 GB).
#
# Unlike h200/slurm/run.sh and b300/slurm/run.sh this is NOT a Slurm job and
# needs no PyXis/Enroot, no EFA and no FSx:
#   * one node, one GPU  -> no srun, no multi-node rendezvous
#   * no inter-node NCCL -> the EFA layer in the repo root Dockerfile is dead
#                           weight here, so we run the upstream NeMo image
#                           directly instead of building it
#   * no shared filesystem -> everything lives on the instance's local disk
#
# Usage:
#   ./g5/run.sh                 # smoke profile, mock data
#   TRAIN_ITERS=50 ./g5/run.sh  # override any knob from g5/train.py
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/nemo:26.04}"
RUN_BASE="${RUN_BASE:-${HOME}/qwen3-g5/run}"
LOG_DIR="${RUN_BASE}/logs"
mkdir -p "${LOG_DIR}"

# --- Distributed layout -----------------------------------------------------
# Each g5.8xlarge has exactly ONE A10G (verified: ec2 describe-instance-types
# reports GpuInfo.Gpus[0].Count = 1), so nproc_per_node is always 1 and the
# node count IS the data-parallel degree.
#
#   NNODES=1 (default) : single instance, rendezvous on loopback, nothing
#                        reachable from off-host. This is the layout all three
#                        recorded measurements used.
#   NNODES=2           : two instances, DP=2. The rendezvous must be reachable
#                        from the peer, so it binds a PRIVATE VPC address and
#                        the security group is the boundary -- see the guard
#                        below and g5/launch-instance.sh.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
# Defaults to loopback deliberately. A host/bind variable defaults to
# 127.0.0.1 even though an override exists; the override is opt-in and
# validated.
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

_is_loopback() {
  case "$1" in
    127.*|localhost|::1) return 0 ;;
    *) return 1 ;;
  esac
}

# Accept only addresses inside a private network: RFC1918 ranges, or a
# cluster-internal DNS name. Anything else -- a public IP, or a wildcard --
# is rejected on sight rather than warned about.
_is_private() {
  local a="$1"
  case "$a" in
    0.0.0.0|::|'*') return 1 ;;                        # wildcard: never
    10.*|192.168.*) return 0 ;;
    172.1[6-9].*|172.2[0-9].*|172.3[0-1].*) return 0 ;;
    169.254.*) return 1 ;;                             # link-local/IMDS: no
    *.internal|*.svc.cluster.local|*.cluster.local) return 0 ;;
    *[!0-9.]*) return 0 ;;   # a hostname, not an IPv4 literal: resolved in-VPC
    *) return 1 ;;                                     # numeric and not RFC1918
  esac
}

if [[ "${NNODES}" -lt 1 ]]; then
  echo "FATAL: NNODES=${NNODES} must be >= 1" >&2
  exit 1
fi
if [[ "${NODE_RANK}" -ge "${NNODES}" ]]; then
  echo "FATAL: NODE_RANK=${NODE_RANK} must be < NNODES=${NNODES}" >&2
  exit 1
fi

if [[ "${NNODES}" -eq 1 ]]; then
  if ! _is_loopback "${MASTER_ADDR}"; then
    echo "FATAL: NNODES=1 but MASTER_ADDR=${MASTER_ADDR} is not loopback." >&2
    echo "       A single-node job must not bind a routable interface. Unset" >&2
    echo "       MASTER_ADDR, or set NNODES=2 if you really mean two nodes." >&2
    exit 1
  fi
else
  if _is_loopback "${MASTER_ADDR}"; then
    echo "FATAL: NNODES=${NNODES} requires MASTER_ADDR to be the PRIVATE VPC" >&2
    echo "       address (or internal DNS name) of the NODE_RANK=0 instance." >&2
    echo "       It is currently loopback, which the peer node cannot reach." >&2
    echo "       Find it with:  ec2-metadata --local-ipv4   (on node 0)" >&2
    exit 1
  fi
  if ! _is_private "${MASTER_ADDR}"; then
    echo "FATAL: MASTER_ADDR=${MASTER_ADDR} is not a private address." >&2
    echo "       The rendezvous must stay inside the VPC. Use the instance's" >&2
    echo "       private IPv4 (10.x, 172.16-31.x, 192.168.x) or internal DNS." >&2
    echo "       Refusing to bind a rendezvous reachable from the internet." >&2
    exit 1
  fi
fi

LOG_FILE="${LOG_DIR}/g5-validation-$(date +%Y%m%d-%H%M%S).log"

# --- Preflight: fail loudly and early, not halfway into a container pull ---
if ! command -v docker >/dev/null 2>&1; then
  echo "FATAL: docker is not installed or not on PATH" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: nvidia-smi not found. This must run on a GPU instance with the" \
       "NVIDIA driver installed (use a Deep Learning AMI)." >&2
  exit 1
fi
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "FATAL: nvidia-smi reports no GPUs" >&2
  exit 1
fi
echo "Detected ${GPU_COUNT} GPU(s):"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# Verify docker can actually see the GPU before pulling ~30 GB of image.
if ! docker run --rm --gpus all "${CONTAINER_IMAGE}" true 2>/dev/null; then
  echo "Image not present locally or GPU passthrough unverified; pulling ${CONTAINER_IMAGE}..."
  if ! docker pull "${CONTAINER_IMAGE}"; then
    echo "FATAL: failed to pull ${CONTAINER_IMAGE}." >&2
    echo "       If this is an authentication error, log in to NGC first:" >&2
    echo "       docker login nvcr.io -u '\$oauthtoken' -p <NGC_API_KEY>" >&2
    exit 1
  fi
fi

echo "Logging to ${LOG_FILE}"

# --gpus all            : the single A10G on this instance
# --ipc=host + shm-size : dataloader workers need shared memory
# --ulimit memlock=-1   : pinned host memory for H2D transfers
#
# Networking depends on the node count, and this is the security boundary:
#
#   NNODES=1 : default BRIDGE network. Egress (nvcr.io, huggingface.co) works,
#              NO port is published, and the rendezvous is on 127.0.0.1 -- so
#              nothing in this container is reachable from off-host at all.
#
#   NNODES>1 : host networking, because a bridged container's rendezvous port
#              is not reachable by the peer node. The rendezvous then listens
#              on the instance's PRIVATE VPC address, and the boundary becomes
#              the security group: g5/launch-instance.sh grants tcp/1-65535
#              ONLY from the cluster's own security group (a self-referencing
#              rule), never from 0.0.0.0/0. Who can reach it: the other
#              instance in the same security group, and nothing else.
#              The range is the whole TCP span rather than just MASTER_PORT
#              because NCCL's communicator uses EPHEMERAL ports that the peer
#              connects INTO; a MASTER_PORT-only rule lets the rendezvous
#              succeed and then hangs the NCCL bootstrap with no error.
NET_ARGS=(--network bridge)

# Single node keeps WARN so its recorded measurements stay comparable. Multi
# node defaults to INFO: a transport-level hang prints NOTHING at WARN, which
# is exactly the case where the log is the only evidence available.
NCCL_DEBUG_DEFAULT=WARN

# ------------------------------------------------------------------------ EFA
# The container needs the fabric device passed in; the host having an EFA is not
# enough. The stock NeMo image already carries the software half -- the
# 2026-10-04 run logged "Initializing aws-ofi-nccl 1.17.3" and "Using Libfabric
# version 2.3" -- and still fell back to TCP with "No eligible providers were
# found", because there was no device for libfabric to open. So the only thing
# missing is /dev/infiniband and the provider selection.
#
# AUTODETECTED rather than configured: the device is either present or it is
# not, and a USE_EFA flag here could only ever disagree with the hardware. The
# same detection is what lets this script stay correct on a node launched with
# USE_EFA=0, where it must NOT pass a device that does not exist.
EFA_ARGS=()
EFA_STATUS="absent"
if [[ -d /dev/infiniband ]]; then
  EFA_STATUS="present"
  # FI_PROVIDER=efa selects the fabric provider; the ulimits below already lift
  # memlock, which EFA needs for its registered memory.
  #
  # FI_EFA_USE_DEVICE_RDMA IS DELIBERATELY NOT SET. Setting it to 1 made both
  # ranks SIGABRT instantly on 2026-10-04 with:
  #   "FI_EFA_USE_DEVICE_RDMA=1 was set by user, but EFA device has no
  #    rdma-read capability.  Application will abort()."
  # g5's EFA has no GPUDirect RDMA -- the earlier TCP run's own log already said
  # "Need to force simple protocol: GDR not supported", which is this same
  # capability. Unset, libfabric uses the device's real capability instead of
  # asserting one it lacks, so EFA still works; only zero-copy GPU reads are
  # unavailable, and they were never available on this hardware.
  #
  # NCCL_PROTO is likewise not set: aws-ofi-nccl derives it from the hardware
  # ("Adding NCCL_PROTO=simple to environment"), so a value here would be a
  # guess overriding a measurement.
  EFA_ARGS=(
    --device /dev/infiniband
    -e FI_PROVIDER=efa
  )
fi
if [[ "${NNODES}" -gt 1 ]]; then
  NET_ARGS=(--network host)
  NCCL_DEBUG_DEFAULT=INFO
  echo "Multi-node: ${NNODES} nodes, this is NODE_RANK=${NODE_RANK}"
  echo "  rendezvous : ${MASTER_ADDR}:${MASTER_PORT} (private VPC address)"
  echo "  reachable by: instances in the same security group only"
  echo "  EFA device : ${EFA_STATUS} (/dev/infiniband)"
  if [[ "${EFA_STATUS}" == "present" ]]; then
    echo "        Passing the device into the container and selecting the efa"
    echo "        libfabric provider. Confirm it took effect: the NCCL log must"
    echo "        NOT contain 'No eligible providers were found'."
  else
    echo "  WARNING: no EFA device on this host, so NCCL will fall back to TCP."
    echo "        MEASURED consequence (2026-10-04, 1000 iterations, GBS=16):"
    echo "        8,086 tok/s against 24,727 tok/s on a SINGLE node at the same"
    echo "        geometry -- 0.33x, i.e. a 3x SLOWDOWN from adding a node. The"
    echo "        ~1 GB per-step gradient exchange crosses TCP at ~4 Gbit/s of"
    echo "        a 25 Gbit link, and the 2.0s step time IS that transfer, so"
    echo "        raising GLOBAL_BATCH_SIZE cannot fix it."
    echo "        An EFA cannot be attached to a RUNNING instance. Relaunch:"
    echo "          NODES=2 USE_EFA=1 ./g5/launch-instance.sh"
    echo "        Use 1 node until then. See g5/README.md."
  fi
fi

# Prompt output is operationally load-bearing, not cosmetic. g5/run.sh's stdout
# is what the 2-node driver streams back to the laptop, and on 2026-10-04 a
# HEALTHY 1000-iteration run looked locally FROZEN for 34 minutes: torchrun's
# stdout is a pipe into tee, so Python block-buffered it. LOG_FILE on the node
# stayed current -- the retrieved log had all 100 iteration records -- so the
# only things blinded were the operator and the driver's stall watchdog, which
# treats local log growth as liveness and would have aborted a successful run.
# PYTHONUNBUFFERED (below, in the container env) fixes the dominant layer;
# this unbuffers tee's OWN stdout, which stdio buffers when it is not a tty.
TEE_CMD=(tee "${LOG_FILE}")
if command -v stdbuf >/dev/null 2>&1; then
  TEE_CMD=(stdbuf -oL tee "${LOG_FILE}")
fi

docker run --rm \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${NET_ARGS[@]}" \
  "${EFA_ARGS[@]+${EFA_ARGS[@]}}" \
  -v "${REPO_DIR}:/workspace/repo:ro" \
  -v "${RUN_BASE}:/workspace/run" \
  -e RUN_BASE=/workspace/run \
  -e CKPT_PATH="${CKPT_PATH:-/workspace/run/checkpoints/g5}" \
  -e DATA_PATH="${DATA_PATH:-}" \
  -e NUM_LAYERS="${NUM_LAYERS:-}" \
  -e HIDDEN_SIZE="${HIDDEN_SIZE:-}" \
  -e FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-}" \
  -e NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-}" \
  -e NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-}" \
  -e SEQ_LENGTH="${SEQ_LENGTH:-}" \
  -e MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-}" \
  -e GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}" \
  -e TRAIN_ITERS="${TRAIN_ITERS:-}" \
  -e LOG_INTERVAL="${LOG_INTERVAL:-}" \
  -e DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-}" \
  -e SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}" \
  -e TENSORBOARD_DIR="${TENSORBOARD_DIR:-/workspace/run/tb_logs}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME=/workspace/run/hf \
  -e TORCH_COMPILE_DISABLE=1 \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e NCCL_DEBUG="${NCCL_DEBUG:-${NCCL_DEBUG_DEFAULT}}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}" \
  -w /workspace/run \
  "${CONTAINER_IMAGE}" \
  torchrun --nproc_per_node=1 --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    /workspace/repo/g5/train.py 2>&1 | "${TEE_CMD[@]}"

echo "Log saved to ${LOG_FILE}"
