#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Qwen3-8B Pre-Training on H200 — Megatron-Bridge (NeMo 26.04)

Uses the Megatron-Bridge recipe API with config objects.
No gradient checkpointing needed (distributed optimizer keeps memory at ~114 GB).
Uses allenai/c4 pre-tokenized with Qwen3-8B tokenizer in Megatron indexed format.

Best config: TP=1, PP=1, DP=16, MBS=2, GBS=128, seq=4096, BF16
Result: 497 TFLOP/s/GPU, 162K tok/s on 16x H200
"""
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from megatron.bridge.recipes.qwen.qwen3 import qwen3_8b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

# Base run directory (override with RUN_BASE env var). Datasets and checkpoints
# live under it. Defaults to the new-cluster-test layout on FSx.
RUN_BASE = os.environ.get("RUN_BASE", "/fsx/paragao/new-cluster-test/run")

# Path to Megatron-indexed c4 dataset (prefix without .bin/.idx extension).
DATA_PATH = os.environ.get("DATA_PATH", f"{RUN_BASE}/datasets/c4_qwen3_8b")
CKPT_PATH = os.environ.get("CKPT_PATH", f"{RUN_BASE}/checkpoints/h200")

def main():
    cfg = qwen3_8b_pretrain_config()

    # Parallelism: DP=16 (model fits on one H200 GPU)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1

    # Batch config
    cfg.train.micro_batch_size = 2
    cfg.train.global_batch_size = 128
    cfg.model.seq_length = 4096

    # Training schedule
    cfg.train.train_iters = 100
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100

    # Dataset: use real c4 data instead of mock
    cfg.dataset.data_path = DATA_PATH
    cfg.dataset.seq_length = 4096
    cfg.dataset.split = "9999,8,2"
    cfg.dataset.num_workers = 8

    # Optimizer
    cfg.optimizer.lr = 3e-4
    cfg.optimizer.min_lr = 3e-5
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.clip_grad = 1.0
    cfg.optimizer.overlap_grad_reduce = True
    cfg.optimizer.overlap_param_gather = True

    # Logging and checkpoints
    cfg.logger.log_interval = 5
    cfg.validation.eval_interval = 1000
    cfg.validation.eval_iters = 0
    cfg.checkpoint.save = CKPT_PATH
    cfg.checkpoint.load = CKPT_PATH
    cfg.checkpoint.save_interval = 1000

    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
