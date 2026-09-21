#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#SBATCH --job-name=preprocess-c4
#SBATCH --partition=p5en
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/fsx/paragao/new-cluster-test/run/logs/preprocess-%j.out
#SBATCH --export=ALL

# --- HF_TOKEN must be set before submitting ---
# export HF_TOKEN=<your token>
# sbatch preprocess.sh
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set. Export it before submitting:"
    echo "  export HF_TOKEN=<your token> && sbatch preprocess.sh"
    exit 1
fi

# Base directory for all run artifacts (datasets, venv, cache, logs). Override with
# RUN_BASE=/some/path sbatch preprocess.sh   (must match the --output dir above).
RUN_BASE="${RUN_BASE:-/fsx/paragao/new-cluster-test/run}"

export HF_HOME="${RUN_BASE}/.cache/huggingface"

mkdir -p "${RUN_BASE}/logs"
mkdir -p "${RUN_BASE}/datasets"

# Create a virtual environment for the preprocessing
python3 -m venv "${RUN_BASE}/venv"
source "${RUN_BASE}/venv/bin/activate"

PYTHON="${RUN_BASE}/venv/bin/python"
SCRIPT_DIR=$SLURM_SUBMIT_DIR/preprocessing/
SCRIPT="${SCRIPT_DIR}/preprocess.py"

pip install -r $SCRIPT_DIR/requirements.txt

echo "=== C4 Preprocessing ==="
echo "Node: $(hostname) | CPUs: $(nproc) | Base: ${RUN_BASE} | Start: $(date)"

$PYTHON $SCRIPT \
    --output-prefix "${RUN_BASE}/datasets/c4_qwen3_8b" \
    --tokenizer Qwen/Qwen3-8B \
    --num-tokens 1000000000 \
    --workers $(nproc) \
    --cache-dir "${RUN_BASE}/cache/c4"

echo "Finished: $(date)"

echo "Finished: $(date)"
