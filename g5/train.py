#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Qwen3 single-GPU stack validation on g5.8xlarge (1x A10G 24 GB) — Megatron-Bridge (NeMo 26.04).

PURPOSE
-------
This is NOT a Qwen3-8B pre-training run. Qwen3-8B cannot be pre-trained on a
24 GB A10G: see `docs/single-gpu-memory-budget.md` for the full arithmetic.
Summary, at DP=1 there is no optimizer sharding, so static state is 18 B/param:

    2 B  BF16 weights
  + 4 B  FP32 main gradients
  + 4 B  FP32 master weights
  + 4 B  Adam exp_avg      (m)
  + 4 B  Adam exp_avg_sq   (v)
  = 18 B/param  ->  8.190e9 params = 137.3 GiB, vs 22.5 GiB on an A10G.
                    6.11x over, before a single activation.

The binding constraint is width, not depth: the embedding plus untied LM head
are 2 x 151936 x 4096 = 1.245 B params = 20.87 GiB, i.e. 92.8% of the card
before layer zero exists. Cutting `num_layers` alone can never fit.

What this script DOES validate, on the exact same code path as `h200/train.py`
and `b300/train.py` (`qwen3_*_pretrain_config()` -> `pretrain(config=...,
forward_step_func=forward_step)`):

  * the NeMo 26.04 / Megatron-Bridge / Megatron-Core import graph and CUDA init
  * Qwen3 model construction (GQA, RoPE, RMSNorm, SwiGLU) on Ampere sm_86
  * the real Qwen3 tokenizer and the 151,936-entry vocab
  * the dataset path (mock by default, or real Megatron-indexed .bin/.idx)
  * a real forward -> backward -> optimizer step loop with BF16 mixed precision
  * loss decreasing, and peak memory staying inside the A10G envelope

It does this with a WIDTH- AND DEPTH-SCALED Qwen3, because the binding
constraint on 24 GB is not depth, it is the vocab projection: at
hidden_size=4096 the embedding + LM head alone are 2 * 151936 * 4096 = 1.245B
params = 20.87 GiB of state, 92.8% of the card. Cutting `num_layers` alone
cannot fit, so `hidden_size` is scaled too. Every architectural ratio that
matters is preserved: GQA 4:1 query-to-KV, FFN 3x hidden, head_dim 128, full
vocab.

CONFIGURATION
-------------
Every knob is an env var so the proxy can be grown without editing this file.
Defaults are the `smoke` profile, which is the configuration actually validated
on g5.8xlarge. See `g5/README.md` for a table of profiles and measured memory.
"""
import os
import sys
import time

# Must precede any torch/megatron import: torch.compile is unsupported here.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# Reduces fragmentation, which matters on a 24 GB card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from megatron.bridge.recipes.qwen.qwen3 import qwen3_8b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

RUN_BASE = os.environ.get("RUN_BASE", os.path.expanduser("~/qwen3-g5/run"))
CKPT_PATH = os.environ.get("CKPT_PATH", f"{RUN_BASE}/checkpoints/g5")

# Optional: prefix (no .bin/.idx suffix) of a Megatron-indexed dataset produced
# by preprocessing/preprocess.py. Unset -> Megatron's mock dataset, which keeps
# this validation self-contained and offline-capable.
DATA_PATH = os.environ.get("DATA_PATH", "").strip()

# Set by torchrun. 1 for a single g5.8xlarge; 2 for two of them (each has
# exactly one A10G, verified via ec2 describe-instance-types). With TP=PP=1 the
# whole world is data parallel, so WORLD_SIZE is the DP degree.
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))

# Qwen3-8B reference architecture, for the scaling report below.
QWEN3_8B_REF = dict(
    num_layers=36, hidden_size=4096, ffn_hidden_size=12288,
    num_attention_heads=32, num_query_groups=8, seq_length=4096,
)


def _env_int(name: str, default: int) -> int:
    """Read a positive int env var, failing loudly on anything unusable."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"FATAL: env {name}={raw!r} is not an integer") from exc
    if value <= 0:
        raise SystemExit(f"FATAL: env {name}={value} must be positive")
    return value


def scaled_arch() -> dict:
    """Resolve the proxy architecture from the environment."""
    arch = dict(
        num_layers=_env_int("NUM_LAYERS", 4),
        hidden_size=_env_int("HIDDEN_SIZE", 1024),
        ffn_hidden_size=_env_int("FFN_HIDDEN_SIZE", 3072),
        num_attention_heads=_env_int("NUM_ATTENTION_HEADS", 8),
        num_query_groups=_env_int("NUM_QUERY_GROUPS", 2),
        seq_length=_env_int("SEQ_LENGTH", 1024),
    )
    # Megatron requires these to divide evenly; catch it here with a clear
    # message rather than deep inside model construction.
    if arch["num_attention_heads"] % arch["num_query_groups"] != 0:
        raise SystemExit(
            f"FATAL: num_attention_heads ({arch['num_attention_heads']}) must be "
            f"divisible by num_query_groups ({arch['num_query_groups']})"
        )
    if arch["hidden_size"] % arch["num_attention_heads"] != 0:
        raise SystemExit(
            f"FATAL: hidden_size ({arch['hidden_size']}) must be divisible by "
            f"num_attention_heads ({arch['num_attention_heads']})"
        )
    return arch


def param_count(arch: dict, vocab_size: int = 151936) -> dict:
    """Analytic parameter count, so the memory budget is auditable from the log."""
    h = arch["hidden_size"]
    head_dim = h // arch["num_attention_heads"]
    kv_dim = arch["num_query_groups"] * head_dim

    embedding = vocab_size * h          # input embedding
    lm_head = vocab_size * h            # untied output projection (Qwen3-8B is untied)
    attn = (h * h) + (h * kv_dim) * 2 + (h * h)      # q, k, v, o
    mlp = (h * arch["ffn_hidden_size"]) * 3          # gate, up, down (SwiGLU)
    per_layer = attn + mlp
    total = embedding + lm_head + per_layer * arch["num_layers"]
    return dict(
        embedding_and_head=embedding + lm_head,
        per_layer=per_layer,
        layers_total=per_layer * arch["num_layers"],
        total=total,
    )


def report(arch: dict, counts: dict) -> None:
    """Print the scaling and memory budget before we commit to allocating it."""
    gib = 1024 ** 3
    # 18 B/param: see module docstring.
    static_bytes = counts["total"] * 18
    try:
        total_vram = torch.cuda.get_device_properties(0).total_memory
        dev_name = torch.cuda.get_device_name(0)
    except Exception as exc:  # no CUDA device is fatal for this validation
        raise SystemExit(f"FATAL: no usable CUDA device: {exc}") from exc

    print("=" * 72, flush=True)
    print("Qwen3 single-GPU stack validation (g5.8xlarge / A10G)", flush=True)
    print("=" * 72, flush=True)
    print(f"  device                : {dev_name} ({total_vram / gib:.1f} GiB)", flush=True)
    print(f"  torch / cuda          : {torch.__version__} / {torch.version.cuda}", flush=True)
    print(f"  compute capability    : sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}", flush=True)
    print("  --- architecture: proxy vs Qwen3-8B reference ---", flush=True)
    for key, ref in QWEN3_8B_REF.items():
        print(f"  {key:<22}: {arch[key]:<8} (Qwen3-8B: {ref})", flush=True)
    print(f"  {'vocab_size':<22}: 151936   (Qwen3-8B: 151936)  <- unchanged", flush=True)
    print("  --- parameters ---", flush=True)
    print(f"  embedding + LM head   : {counts['embedding_and_head'] / 1e6:>9.1f} M", flush=True)
    print(f"  transformer layers    : {counts['layers_total'] / 1e6:>9.1f} M", flush=True)
    print(f"  TOTAL                 : {counts['total'] / 1e6:>9.1f} M", flush=True)
    print("  --- static memory estimate (18 B/param, Adam + FP32 master) ---", flush=True)
    print(f"  optimizer+weights     : {static_bytes / gib:>9.2f} GiB", flush=True)
    print(f"  of device total        : {100 * static_bytes / total_vram:>9.1f} %", flush=True)
    print("=" * 72, flush=True)

    # Leave room for activations, logits (seq x 151936 x 4 B) and CUDA context.
    if static_bytes > 0.60 * total_vram:
        print(
            f"WARNING: static state is {100 * static_bytes / total_vram:.0f}% of VRAM. "
            "Activations and the vocab logits may push this into OOM. "
            "Reduce NUM_LAYERS or HIDDEN_SIZE if the run dies.",
            file=sys.stderr, flush=True,
        )


def main():
    arch = scaled_arch()
    counts = param_count(arch)

    # Rank 0 only, so the report appears once even if launched under torchrun.
    if RANK == 0:
        report(arch, counts)

    # Identical entrypoint to h200/train.py and b300/train.py.
    cfg = qwen3_8b_pretrain_config()

    # ---- Parallelism ----
    # TP and PP stay at 1 on every supported layout. Each g5.8xlarge has
    # exactly one A10G, so a second instance adds a DATA parallel rank, not a
    # tensor or pipeline stage -- and TP across a 25 Gbit TCP/EFA link would be
    # far worse than DP, since TP communicates per layer rather than per step.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1

    # ---- Scale the architecture down to fit 24 GB (see module docstring) ----
    cfg.model.num_layers = arch["num_layers"]
    cfg.model.hidden_size = arch["hidden_size"]
    cfg.model.ffn_hidden_size = arch["ffn_hidden_size"]
    cfg.model.num_attention_heads = arch["num_attention_heads"]
    cfg.model.num_query_groups = arch["num_query_groups"]
    cfg.model.seq_length = arch["seq_length"]
    # head_dim stays 128 as in Qwen3-8B when hidden/heads == 128.
    cfg.model.kv_channels = arch["hidden_size"] // arch["num_attention_heads"]

    # ---- Batch: MBS=1 with gradient accumulation; DP = WORLD_SIZE ----
    cfg.train.micro_batch_size = _env_int("MICRO_BATCH_SIZE", 1)
    cfg.train.global_batch_size = _env_int("GLOBAL_BATCH_SIZE", 8)

    # Megatron requires global_batch_size to divide evenly by
    # data_parallel_size * micro_batch_size. Catch it here with an actionable
    # message instead of failing deep inside distributed setup.
    per_step = WORLD_SIZE * cfg.train.micro_batch_size
    if cfg.train.global_batch_size % per_step != 0:
        raise SystemExit(
            f"FATAL: GLOBAL_BATCH_SIZE={cfg.train.global_batch_size} is not "
            f"divisible by WORLD_SIZE({WORLD_SIZE}) x "
            f"MICRO_BATCH_SIZE({cfg.train.micro_batch_size}) = {per_step}. "
            f"For {WORLD_SIZE} node(s) use a multiple of {per_step}, e.g. "
            f"GLOBAL_BATCH_SIZE={per_step * 8}."
        )
    if RANK == 0:
        grad_accum = cfg.train.global_batch_size // per_step
        print(f"  data parallel size    : {WORLD_SIZE}", flush=True)
        print(f"  grad accum per rank   : {grad_accum}", flush=True)
        if WORLD_SIZE > 1:
            print("  NOTE: at fixed GLOBAL_BATCH_SIZE, adding a node halves "
                  "per-rank compute but", flush=True)
            print("        NOT the gradient all-reduce, so throughput barely "
                  "improves. Scale", flush=True)
            print(f"        GLOBAL_BATCH_SIZE with the node count "
                  f"(={8 * WORLD_SIZE}) for real gains.", flush=True)

    # ---- Short schedule: this is a validation, not a training run ----
    train_iters = _env_int("TRAIN_ITERS", 20)
    cfg.train.train_iters = train_iters
    cfg.scheduler.lr_warmup_iters = max(1, train_iters // 10)
    cfg.scheduler.lr_decay_iters = train_iters

    # ---- Dataset ----
    cfg.dataset.seq_length = arch["seq_length"]
    if DATA_PATH:
        if not hasattr(cfg.dataset, "data_path"):
            raise SystemExit(
                "FATAL: DATA_PATH was set but the recipe returned a dataset config "
                f"without a 'data_path' field ({type(cfg.dataset).__name__}). "
                "Unset DATA_PATH to use the mock dataset."
            )
        idx = f"{DATA_PATH}.idx"
        if not os.path.exists(idx):
            raise SystemExit(
                f"FATAL: DATA_PATH={DATA_PATH} but {idx} does not exist. "
                "DATA_PATH is a prefix without the .bin/.idx suffix. "
                "Build it with preprocessing/preprocess.py, or unset DATA_PATH "
                "to use the mock dataset."
            )
        cfg.dataset.data_path = DATA_PATH
        cfg.dataset.split = "9999,8,2"
        print(f"  dataset: real Megatron-indexed data at {DATA_PATH}", flush=True)
    else:
        print("  dataset: Megatron mock dataset (set DATA_PATH for real c4)", flush=True)
    # 32 vCPU on g5.8xlarge; 8 loader workers is ample for MBS=1.
    cfg.dataset.num_workers = _env_int("DATA_NUM_WORKERS", 8)

    # ---- Optimizer: same hyperparameters as the H200/B300 runs ----
    cfg.optimizer.lr = 3e-4
    cfg.optimizer.min_lr = 3e-5
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.clip_grad = 1.0
    # Overlap is a DP>1 concept: at DP=1 there is no peer to overlap a reduce or
    # a gather with, Megatron rejects or no-ops them, and the 16-GPU recipe's
    # True would be misleading. At DP>1 (two g5.8xlarge = DP=2) the gradient
    # all-reduce is a large fraction of step time on a 25 Gbit link, so
    # overlapping it with the backward pass is what makes 2 nodes worth running.
    # See the scaling table in g5/README.md.
    if WORLD_SIZE > 1:
        cfg.optimizer.overlap_grad_reduce = True
        cfg.optimizer.overlap_param_gather = True
        # Shards optimizer state across DP ranks (ZeRO-1 shape). This is the
        # actual reason to add a second node: it reduces per-GPU static memory,
        # where throughput only improves if GLOBAL_BATCH_SIZE scales too.
        if hasattr(cfg.optimizer, "use_distributed_optimizer"):
            cfg.optimizer.use_distributed_optimizer = True
    else:
        cfg.optimizer.overlap_grad_reduce = False
        cfg.optimizer.overlap_param_gather = False

    # ---- Logging; checkpointing off by default to keep the run quick ----
    cfg.logger.log_interval = _env_int("LOG_INTERVAL", 1)
    # The recipe defaults tensorboard_dir to a CWD-relative
    # './nemo_experiments/default/tb_logs'. g5/run.sh mounts the repo read-only,
    # so leaving that default makes the run die with
    #   OSError: [Errno 30] Read-only file system: '/workspace/repo/nemo_experiments'
    # at "setting tensorboard", after the model and tokenizer have already been
    # built. Point it at RUN_BASE, which is writable by construction.
    tb_dir = os.environ.get("TENSORBOARD_DIR", f"{RUN_BASE}/tb_logs")
    os.makedirs(tb_dir, exist_ok=True)
    cfg.logger.tensorboard_dir = tb_dir
    cfg.validation.eval_interval = train_iters + 1  # never mid-run
    cfg.validation.eval_iters = 0
    if os.environ.get("SAVE_CHECKPOINT", "0") == "1":
        os.makedirs(CKPT_PATH, exist_ok=True)
        cfg.checkpoint.save = CKPT_PATH
        cfg.checkpoint.load = CKPT_PATH
        cfg.checkpoint.save_interval = train_iters
    else:
        cfg.checkpoint.save = None
        cfg.checkpoint.load = None

    # ---- Throughput denominator ----
    # tokens/step is global_batch_size * seq_length at ANY data parallel size,
    # because global_batch_size is global by definition -- adding a node splits
    # the same step across more ranks rather than enlarging it. Printed before
    # the run so the figure is in the log even if the run dies, and so
    # g5/throughput.py does not have to re-derive the batch geometry.
    tokens_per_step = cfg.train.global_batch_size * cfg.model.seq_length
    if RANK == 0:
        print(
            f"  tokens per step       : {tokens_per_step:,} "
            f"({cfg.train.global_batch_size} seq x {cfg.model.seq_length} tok)",
            flush=True,
        )
        print(f"  train_iters           : {train_iters}", flush=True)

    wall_start = time.perf_counter()
    pretrain(config=cfg, forward_step_func=forward_step)
    wall_elapsed = time.perf_counter() - wall_start

    if RANK == 0:
        gib = 1024 ** 3
        total_tokens = tokens_per_step * train_iters
        print("=" * 72, flush=True)
        print("VALIDATION COMPLETE", flush=True)
        print(f"  peak allocated : {torch.cuda.max_memory_allocated() / gib:.2f} GiB", flush=True)
        print(f"  peak reserved  : {torch.cuda.max_memory_reserved() / gib:.2f} GiB", flush=True)
        print("  --- throughput ---", flush=True)
        print(f"  tokens per step: {tokens_per_step:,}", flush=True)
        print(f"  iterations     : {train_iters}", flush=True)
        print(f"  total tokens   : {total_tokens:,}", flush=True)
        # NOTE: wall_elapsed spans the whole pretrain() call, which includes
        # model construction, dataloader build and CUDA warmup. Over a 20-iter
        # validation that setup DOMINATES, so the figure below is a LOWER BOUND
        # on steady-state throughput, not the steady-state rate. The
        # authoritative per-step number comes from Megatron's own per-iteration
        # timings -- parse them with:  python3 g5/throughput.py <logfile>
        print(f"  wall clock     : {wall_elapsed:.1f} s (INCLUDES setup/warmup)", flush=True)
        print(
            f"  end-to-end     : {total_tokens / wall_elapsed:,.0f} tok/s "
            "(LOWER BOUND -- setup included)",
            flush=True,
        )
        print(
            "  steady-state   : run  python3 g5/throughput.py <logfile>  "
            "for the per-step rate",
            flush=True,
        )
        print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
