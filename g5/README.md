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
  Megatron-indexed `.bin`/`.idx` from `preprocessing/preprocess.py`
- a real forward -> backward -> Adam step loop in BF16, with the loss moving
- measured peak memory against the A10G envelope

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

All three keep `head_dim` at 128 and the GQA ratio at 4:1, as in Qwen3-8B.

```bash
# wider
NUM_LAYERS=6 HIDDEN_SIZE=1536 FFN_HIDDEN_SIZE=4608 \
  NUM_ATTENTION_HEADS=12 NUM_QUERY_GROUPS=3 ./g5/run.sh

# deeper, longer sequence
NUM_LAYERS=12 SEQ_LENGTH=2048 ./g5/run.sh
```

Pushing much past ~50% static leaves too little for activations and the FP32
vocab logits, which are `SEQ_LENGTH x 151936 x 4 B` per micro-batch copy and are
the largest single activation in the model: 622 MB at `seq=1024`, 1.24 GB at
`seq=2048`. If a run OOMs, cut `NUM_LAYERS` or `SEQ_LENGTH` first.

## Using real data instead of the mock dataset

The mock dataset keeps this validation self-contained. To exercise the real
tokenized path:

```bash
export HF_TOKEN=<your huggingface token>
python preprocessing/preprocess.py \
  --output-prefix ~/qwen3-g5/run/datasets/c4_qwen3_8b \
  --num-tokens 100000000 --workers 32

DATA_PATH=~/qwen3-g5/run/datasets/c4_qwen3_8b ./g5/run.sh
```

`train.py` fails fast with an explicit message if `DATA_PATH` is set but the
`.idx` file is missing, rather than silently falling back to mock data.
