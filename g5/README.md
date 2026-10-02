# Single-GPU stack validation on g5.8xlarge (1x A10G 24 GB)

A fast, cheap way to check that this repo's training stack — NeMo 26.04,
Megatron-Bridge, the Qwen3 model definition, the tokenizer and the data path —
is healthy, without booking a 2-node H200 or B300 cluster.

> **This is not a Qwen3-8B pre-training run, and it produces no performance
> numbers.** Qwen3-8B needs 137.3 GiB of static state at DP=1 against 22.5 GiB
> of A10G, 6.11x over. See [`../docs/single-gpu-memory-budget.md`](../docs/single-gpu-memory-budget.md)
> for the arithmetic and for why `hidden_size`, not just `num_layers`, has to
> come down. Throughput, TFLOP/s and MFU comparisons belong to
> [`../h200/`](../h200/) and [`../b300/`](../b300/) only.

## What it does validate

Running on the **same entrypoint** as `h200/train.py` and `b300/train.py`
(`qwen3_8b_pretrain_config()` -> `pretrain(config=..., forward_step_func=forward_step)`):

- the NeMo 26.04 / Megatron-Bridge / Megatron-Core import graph and CUDA init on
  Ampere `sm_86`
- Qwen3 model construction: GQA, RoPE, RMSNorm, SwiGLU
- the real Qwen3 tokenizer and the full 151,936-entry vocab
- the dataset path — Megatron's mock dataset by default, or real
  Megatron-indexed `.bin`/`.idx` from `g5/prepare_c4.py`
- a real forward -> backward -> Adam step loop in BF16, with 0 skipped and 0
  NaN iterations
- measured peak memory against the A10G envelope

> **The loss curve on mock data is not a learning signal.** Both measured runs
> fell from ~12.2 to ~0.2 in 50 steps, which is degenerate fitting of 400
> synthetic samples, not convergence. First-iteration loss sitting just above
> `ln(151936) = 11.9312` is the correct signature of a fresh random init, and
> is as much as the mock path can tell you. For an interpretable curve, use
> real data — see "Using real data" below.

## Differences from the cluster path, and why

| | `h200/`, `b300/` | `g5/` |
|---|---|---|
| Scheduler | Slurm (`sbatch`) | none, one process |
| Container runtime | PyXis + Enroot `.sqsh` | Docker |
| Image | repo `Dockerfile` (NeMo + EFA + gdrcopy) | `nvcr.io/nvidia/nemo:26.04` directly |
| Interconnect | EFA GDRDMA | none needed |
| Filesystem | FSx for Lustre `/fsx` | instance-local disk |
| Parallelism | TP=1, PP=1, DP=16 | TP=1, PP=1, DP=1 |

The repo's root `Dockerfile` exists to add the EFA stack and gdrcopy for
*inter-node* NCCL. On one GPU in one node there is no inter-node traffic, so
that layer is dead weight and `g5/run.sh` runs the upstream NeMo image as-is.
This skips a ~50 GB `docker build` plus `enroot import`.

`overlap_grad_reduce` and `overlap_param_gather` are set to `False`: at DP=1
there is no peer to overlap a reduce or gather against, so inheriting the
16-GPU recipe's `True` would be meaningless at best.

## Running it

### On an existing GPU box

```bash
./g5/run.sh
```

Needs the NVIDIA driver, Docker and the NVIDIA container toolkit — any Deep
Learning AMI has all three. The script preflights all of them and the GPU count
before pulling anything.

### From scratch on EC2

```bash
# Launch one g5.8xlarge. Walks AZs on InsufficientInstanceCapacity, which is
# common for g5, and will fall back across AZs automatically.
./g5/launch-instance.sh                      # us-west-2 by default
REGION=us-east-1 ./g5/launch-instance.sh     # more g5 capacity pools

aws ssm start-session --target <instance-id> --region <region>

# on the instance
git clone https://github.com/paragao/qwen3-8b-optimised-pretraining.git
cd qwen3-8b-optimised-pretraining && ./g5/run.sh

# ALWAYS tear down: g5.8xlarge is ~$2.45/hr on-demand
./g5/terminate-instance.sh
```

`launch-instance.sh` is private by default: the security group it creates has
**zero ingress rules** and the script asserts that invariant rather than
assuming it, there is no SSH key pair and no port 22, access is via SSM Session
Manager, IMDSv2 is required and the root volume is encrypted.

## Configuration

Every knob is an environment variable; nothing needs editing. Defaults are the
`smoke` profile, which is the configuration actually validated on g5.8xlarge.

| Variable | Default | Notes |
|---|---|---|
| `NUM_LAYERS` | `4` | Qwen3-8B: 36 |
| `HIDDEN_SIZE` | `1024` | Qwen3-8B: 4096 |
| `FFN_HIDDEN_SIZE` | `3072` | keep at 3x `HIDDEN_SIZE` |
| `NUM_ATTENTION_HEADS` | `8` | must divide `HIDDEN_SIZE` |
| `NUM_QUERY_GROUPS` | `2` | must divide heads; 4:1 matches Qwen3-8B |
| `SEQ_LENGTH` | `1024` | Qwen3-8B: 4096 |
| `MICRO_BATCH_SIZE` | `1` | |
| `GLOBAL_BATCH_SIZE` | `8` | grad accumulation = GBS/MBS at DP=1 |
| `TRAIN_ITERS` | `20` | |
| `LOG_INTERVAL` | `1` | |
| `DATA_PATH` | unset | set to a `.bin`/`.idx` prefix for real c4; unset = mock |
| `DATA_NUM_WORKERS` | `8` | 32 vCPU available |
| `SAVE_CHECKPOINT` | `0` | `1` writes one checkpoint at the end |
| `CONTAINER_IMAGE` | `nvcr.io/nvidia/nemo:26.04` | |
| `RUN_BASE` | `~/qwen3-g5/run` | logs, HF cache, checkpoints |

`train.py` prints an analytic parameter count and an 18 B/param static-memory
estimate *before* allocating anything, and warns if that exceeds 60% of VRAM.
Its `param_count()` reproduces real Qwen3-8B at 8.190 B parameters, which is the
check that the estimate can be trusted.

### Profiles

| Profile | Layers | Hidden | FFN | Heads/KV | Seq | Params | Static (18 B/param) | % of 22.5 GiB |
|---|---|---|---|---|---|---|---|---|
| `smoke` (default) | 4 | 1024 | 3072 | 8/2 | 1024 | 359.4 M | 6.02 GiB | 26.8% |
| `wider` | 6 | 1536 | 4608 | 12/3 | 1024 | 629.5 M | 10.55 GiB | 46.9% |
| `deeper` | 12 | 1024 | 3072 | 8/2 | 2048 | 455.9 M | 7.64 GiB | 34.0% |

The `Static` column is analytic and excludes activations. All three profiles
are now measured:

| Profile | Static | Measured peak alloc | Measured reserved | % of 22.49 GiB total | Median tok/s | MODEL_TFLOP/s | MFU |
|---|---|---|---|---|---|---|---|
| `smoke` | 6.02 GiB | **6.84 GiB** | 7.26 GiB | 30.4% | **24,757** | 30.9 | 24.7% |
| `wider` | 10.55 GiB | **11.76 GiB** | 12.25 GiB | 52.3% | **14,357** | 34.9 | 27.9% |
| `deeper` | 7.64 GiB | **9.66 GiB** | 10.03 GiB | 43.0% | **18,793** | **36.7** | **29.4%** |

Percentages are against the A10G's **22.49 GiB total** (`nvidia-smi` reports
23028 MiB). All three ran 0 skipped and 0 NaN iterations, so the BF16 path is
stable on `sm_86` across 4-12 layers, hidden 1024-1536 and seq 1024-2048.

Two results worth reading off that table:

**Sequence length buys more efficiency than width.** `deeper` is the *most*
efficient of the three (36.7 MODEL_TFLOP/s, 29.4% MFU) at the *narrowest*
width, having doubled seq instead. Doubling seq beat a 50% hidden increase.

**Parameter count is not a proxy for step cost.** `deeper` has fewer
parameters than `wider` (455.9 M vs 629.5 M) but does 1.605x the FLOPs per
step, because seq 2048 doubles tokens/step and quadruples the attention term.

### Sizing an untested profile: measure it

`g5/predict.py --profile <name>` predicts memory, but **its memory model is
refuted** and its prediction should be treated as a rough bound, not a number:

```bash
python3 g5/predict.py --form-test   # shows why; exits 1 by design
```

The flat "+14% over static" heuristic is definitely wrong — `wider` refuted it.
But the decomposed replacement is *also* wrong: it predicted `deeper` at
10.21-10.52 GiB against a measured 9.66, outside both bands, and no model of
that form fits all three points (one leave-one-out fit needs a *negative*
bytes-per-activation-unit). The root cause is that none of the three profiles
is a single-variable change from another, so a model fitted to two of them has
no basis for extrapolation.

Empirically, across the three measured shapes, peak allocated ran **11-26%
above** the 18 B/param static figure. Treat 18 B/param as a floor, budget
generously, and measure a new shape rather than predicting it. A `seq` sweep at
fixed layers and hidden, plus a `layers` sweep at fixed seq and hidden, would
settle the form — about 12 runs of under a minute each.

**The throughput model, by contrast, is validated.** Calibrated on `smoke`
alone and never refitted, it predicts FLOPs per step to within **0.04%** on all
three profiles, across changes in layers, hidden size and sequence length:

| | predicted | implied by measurement | error |
|---|---|---|---|
| `smoke` | 10.22 TFLOP | 10.23 | −0.04% |
| `wider` | 19.94 TFLOP | 19.93 | +0.04% |
| `deeper` | 31.99 TFLOP | 32.00 | −0.03% |

All three keep `head_dim` at 128 and the GQA ratio at 4:1, as in Qwen3-8B.

```bash
# wider -- on the instance
NUM_LAYERS=6 HIDDEN_SIZE=1536 FFN_HIDDEN_SIZE=4608 \
  NUM_ATTENTION_HEADS=12 NUM_QUERY_GROUPS=3 ./g5/run.sh

# wider -- driven remotely from a laptop over SSH-via-SSM
NUM_LAYERS=6 HIDDEN_SIZE=1536 FFN_HIDDEN_SIZE=4608 \
  NUM_ATTENTION_HEADS=12 NUM_QUERY_GROUPS=3 \
  TRAIN_ITERS=50 ./g5/finish-run.sh

# deeper, longer sequence
NUM_LAYERS=12 SEQ_LENGTH=2048 ./g5/run.sh
```

`finish-run.sh` must name every variable explicitly when it builds the remote
command, because **ssh does not inherit the caller's environment**. A knob that
is not in that list is silently dropped and `train.py` falls back to its
`smoke` default, which looks exactly like a successful run of the profile you
asked for. Always confirm the `TOTAL` parameter line in the log matches the
profile you intended (`wider` is 629.5 M, `smoke` is 359.4 M) before believing
any number from the run.

Pushing much past ~50% static leaves too little for activations and the FP32
vocab logits, which are `SEQ_LENGTH x 151936 x 4 B` per micro-batch copy and are
the largest single activation in the model: 622 MB at `seq=1024`, 1.24 GB at
`seq=2048`. If a run OOMs, cut `NUM_LAYERS` or `SEQ_LENGTH` first.

## Using real data instead of the mock dataset

The mock dataset keeps this validation self-contained, but its loss curve is
**not interpretable** — see the warning under Profiles. For a real loss curve:

```bash
# 1. Build a bounded c4 dataset on the instance (CPU only, no GPU needed)
./g5/prepare-c4.sh                        # 50M tokens, the default budget
NUM_TOKENS=200000000 ./g5/prepare-c4.sh   # bigger budget

# 2. Train against it. DATA_PATH is the IN-CONTAINER path.
DATA_PATH=/workspace/run/datasets/c4_qwen3 TRAIN_ITERS=1000 ./g5/run.sh

# Or both in one step, driven from a laptop:
PREPARE_C4=1 TRAIN_ITERS=1000 ./g5/finish-run.sh
```

**`DATA_PATH` is a path inside the container, not on the host.** `g5/run.sh`
mounts `RUN_BASE` at `/workspace/run` and passes `DATA_PATH` straight through,
so a host path like `~/qwen3-g5/run/datasets/...` does not exist in the
container and `train.py`'s `.idx` check will reject it. Use the
`/workspace/run/...` form. `g5/finish-run.sh` translates between the two for
its pre-flight check and prints both.

`train.py` fails fast with an explicit message if `DATA_PATH` is set but the
`.idx` file is missing, rather than silently falling back to mock data.

### Why not `preprocessing/preprocess.py`

That script targets a p5en node with FSx, 192 CPUs and a 1B-token budget, and
is unusable on a single g5.8xlarge:

| | `preprocessing/preprocess.py` | `g5/prepare_c4.py` |
|---|---|---|
| download | `split="train[:N]"` (line 127) — a slice still resolves and downloads **every** `en` shard before slicing | `streaming=True`, stops at the token budget |
| `HF_TOKEN` | hard `sys.exit(1)` without it (line 55) | optional; `allenai/c4` is public |
| doc length | hardcoded 4096 (line 144) | `--doc-length`, default 4096 |
| verification | none | `--self-test` and `--verify-only` |

The streaming approach is what this repo's own
`docs/data-loading-explained.md` (line 41) already prescribes;
`preprocess.py` is what diverged from it.

Verify the format writer offline, with numpy alone and no network:

```bash
python3 g5/prepare_c4.py --self-test          # round-trip + corruption rejection
python3 g5/prepare_c4.py --verify-only PREFIX # check an existing build
```

`prepare_c4.py` reuses `write_idx_file` lifted out of
`preprocessing/preprocess.py` rather than copying it, so the on-disk Megatron
format cannot drift between the two paths.
