#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#SBATCH --job-name=qwen3-8b-h200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --output=/fsx/paragao/new-cluster-test/run/logs/%j.out
#SBATCH --error=/fsx/paragao/new-cluster-test/run/logs/%j.err

# Base run dir + container image (override via env). Must match the --output dir above.
RUN_BASE="${RUN_BASE:-/fsx/paragao/new-cluster-test/run}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${RUN_BASE}/containers/nemo-efa-26.04.sqsh}"
# train.py resolved from the repo checkout on /fsx (mounted into the container).
# Use SLURM_SUBMIT_DIR when available (sbatch copies this script to spool, so $0
# is not the repo path); fall back to the known repo location.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-/fsx/paragao/new-cluster-test/qwen3-8b-optimised-pretraining/h200}}"
TRAIN_PY="${TRAIN_PY:-${REPO_DIR}/train.py}"
mkdir -p "${RUN_BASE}/logs"
export RUN_BASE

# EFA / NCCL environment
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo,veth
export NCCL_DEBUG=WARN
# Tuner plugin lives INSIDE the container (built with the EFA installer). Override
# with NCCL_TUNER_PLUGIN=... if the container ships a different filename; leave the
# plugin unset to fall back to NCCL's built-in tuner.
export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-/opt/amazon/ofi-nccl/lib/libnccl-tuner-ofi.so}"
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Launch - Megatron uses SLURM env vars (SLURM_PROCID, SLURM_LOCALID) for distributed init
/opt/slurm/bin/srun --mpi=pmix \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts=/fsx:/fsx,/opt/slurm:/opt/slurm \
    --container-env=FI_PROVIDER,NCCL_SOCKET_IFNAME,NCCL_DEBUG,NCCL_TUNER_PLUGIN,LD_LIBRARY_PATH,TORCH_COMPILE_DISABLE,RUN_BASE \
    python "${TRAIN_PY}"
