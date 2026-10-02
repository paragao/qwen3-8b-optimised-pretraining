#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Build a bounded c4 dataset in Megatron indexed format on the g5 instance,
# so g5/train.py can be run against REAL data instead of the mock dataset.
#
# This runs g5/prepare_c4.py inside the same NeMo container used for training
# (it already has `datasets` and `transformers`), with NO --gpus: tokenising is
# pure CPU, and the GPU should stay free.
#
# Networking: default bridge, egress only (huggingface.co). No port is
# published, so nothing here is reachable from off-host.
#
# Usage:
#   ./g5/prepare-c4.sh                     # 50M tokens, the default budget
#   NUM_TOKENS=200000000 ./g5/prepare-c4.sh
#   ./g5/prepare-c4.sh --verify-only       # just re-check an existing build
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/nemo:26.04}"
RUN_BASE="${RUN_BASE:-${HOME}/qwen3-g5/run}"
NUM_TOKENS="${NUM_TOKENS:-50000000}"
DOC_LENGTH="${DOC_LENGTH:-4096}"
# Path INSIDE the container. RUN_BASE is mounted at /workspace/run.
DATA_PREFIX_IN="${DATA_PREFIX_IN:-/workspace/run/datasets/c4_qwen3}"

mkdir -p "${RUN_BASE}/datasets" "${RUN_BASE}/hf"

if ! command -v docker >/dev/null 2>&1; then
  echo "FATAL: docker is not installed or not on PATH" >&2
  exit 1
fi

# Fail early if the token budget cannot fit on disk. int32 per token, plus
# roughly 3x headroom for the streaming Arrow cache.
NEED_GB=$(( (NUM_TOKENS * 4 / 1000000000) + 1 ))
FREE_GB=$(df -BG --output=avail "${RUN_BASE}" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
echo "Disk: ${FREE_GB} GB free at ${RUN_BASE}; dataset needs ~${NEED_GB} GB plus cache"
if [[ "${FREE_GB}" -gt 0 && "${FREE_GB}" -lt $(( NEED_GB * 4 )) ]]; then
  echo "FATAL: only ${FREE_GB} GB free; want >= $(( NEED_GB * 4 )) GB for" \
       "${NUM_TOKENS} tokens plus the streaming cache. Lower NUM_TOKENS." >&2
  exit 1
fi

echo "Running the c4 preparation in ${CONTAINER_IMAGE} (no GPU)"
# HF_TOKEN is forwarded only if the caller set it. allenai/c4 is public and
# prepare_c4.py does not require it.
docker run --rm \
  --ipc=host \
  --shm-size=8g \
  -v "${REPO_DIR}:/workspace/repo:ro" \
  -v "${RUN_BASE}:/workspace/run" \
  -e HF_HOME=/workspace/run/hf \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -w /workspace/run \
  "${CONTAINER_IMAGE}" \
  python3 /workspace/repo/g5/prepare_c4.py \
    --output-prefix "${DATA_PREFIX_IN}" \
    --num-tokens "${NUM_TOKENS}" \
    --doc-length "${DOC_LENGTH}" \
    "$@"

# Translate the in-container path back to a host path for the operator.
HOST_PREFIX="${RUN_BASE}${DATA_PREFIX_IN#/workspace/run}"
echo
echo "Dataset built. Host path : ${HOST_PREFIX}.{bin,idx}"
ls -la "${HOST_PREFIX}.bin" "${HOST_PREFIX}.idx" 2>/dev/null || true
echo
echo "Train against it with:"
echo "    DATA_PATH=${DATA_PREFIX_IN} TRAIN_ITERS=500 ./g5/run.sh"
echo
echo "NOTE: DATA_PATH is the IN-CONTAINER path (${DATA_PREFIX_IN}), not the"
echo "      host path, because g5/run.sh passes it straight into the container."
