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

# --gpus all            : single A10G
# --ipc=host + shm-size : dataloader workers need shared memory
# --ulimit memlock=-1   : pinned host memory for H2D transfers
#
# Networking: default bridge. Egress (nvcr.io, huggingface.co) works, and NO
# port is published, so nothing in this container is reachable from off-host.
# The torchrun rendezvous below is pinned to 127.0.0.1 because this is a
# single-process, single-node job: it must not bind a routable interface.
docker run --rm \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
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
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -w /workspace/run \
  "${CONTAINER_IMAGE}" \
  torchrun --nproc_per_node=1 --nnodes=1 \
    --master_addr=127.0.0.1 --master_port=29500 \
    /workspace/repo/g5/train.py 2>&1 | tee "${LOG_FILE}"

echo "Log saved to ${LOG_FILE}"
