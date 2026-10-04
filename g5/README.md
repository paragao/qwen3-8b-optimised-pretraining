# Single-GPU stack validation on g5.8xlarge (1x A10G 24 GB)

# Stack validation on g5.8xlarge: 1 or 2 nodes, Docker or EKS

A fast, cheap way to check that this repo's training stack — NeMo 26.04,
Megatron-Bridge, the Qwen3 model definition, the tokenizer and the data path —
is healthy, without booking a 2-node H200 or B300 cluster.

Four layouts, same `g5/train.py` in every one:

| | nodes | GPUs | launcher | how |
|---|---|---|---|---|
| single instance | 1 | 1 | `g5/run.sh` | [On an existing GPU box](#on-an-existing-gpu-box) |
| two instances | 2 | 2 | `g5/run.sh` per node | [Two nodes](#two-nodes-2x-g58xlarge) |
| EKS, one pod | 1 | 1 | `g5/eks/pretrain.yaml` | [Running on EKS](#running-on-eks) |
| EKS, two pods | 2 | 2 | `g5/eks/pretrain.yaml` | [Running on EKS](#running-on-eks) |

Each g5.8xlarge has exactly **one** A10G (`ec2 describe-instance-types` reports
`GpuInfo.Gpus[0].Count = 1`, 22888 MiB, 25 Gigabit, `EfaSupported: true`), so
the node count **is** the data-parallel degree and `nproc_per_node` is always 1.

Both mock and real **c4** data are supported — see
[Training on the c4 dataset](#training-on-the-c4-dataset). Use real data if you
care about the loss curve at all; the mock dataset's curve is meaningless.

> **This is not a Qwen3-8B pre-training run, and it produces no comparable
> performance numbers.** Qwen3-8B needs 137.3 GiB of static state at DP=1
> against 22.5 GiB of A10G, 6.11x over. See
> [`../docs/single-gpu-memory-budget.md`](../docs/single-gpu-memory-budget.md)
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
| **`1b`** | **20** | **1536** | **4608** | **12/3** | **1024** | **1,009.4 M** | **16.92 GiB** | **75.2%** |
| `deeper4k` | 12 | 1024 | 3072 | 8/2 | **4096** | 455.9 M | 7.64 GiB | 34.0% |
| `1b2k` | 20 | 1536 | 4608 | 12/3 | **2048** | 1,009.4 M | 16.92 GiB | 75.2% |

The `Static` column is analytic and excludes activations. Four profiles are
measured; `deeper4k` is predicted but not yet run:

| Profile | Static | Measured peak alloc | Measured reserved | % of 22.49 GiB total | Median tok/s | MODEL_TFLOP/s | MFU |
|---|---|---|---|---|---|---|---|
| `smoke` | 6.02 GiB | **6.84 GiB** | 7.26 GiB | 30.4% | **24,757** | 30.9 | 24.7% |
| `wider` | 10.55 GiB | **11.76 GiB** | 12.25 GiB | 52.3% | **14,357** | 34.9 | 27.9% |
| `deeper` | 7.64 GiB | **9.66 GiB** | 10.03 GiB | 43.0% | **18,793** | **36.7** | **29.4%** |
| **`1b`** | 16.92 GiB | **19.08 GiB** | 19.43 GiB | **84.8%** | **7,085** | 34.3 | 27.4% |
| `deeper4k` | 7.64 GiB | not run (predicted 11.14) | — | ~50% | predicted ~17,400 | — | — |
| `1b2k` | 16.92 GiB | not run (predicted 20.73) | — | ~92% | predicted ~6,800 | — | — |

Percentages are against the A10G's **22.49 GiB total** (`nvidia-smi` reports
23028 MiB). All four ran 0 skipped and 0 NaN iterations, so the BF16 path is stable on
`sm_86` across 4-20 layers, hidden 1024-1536 and seq 1024-2048.

`deeper4k` is `deeper` at seq 4096 and `1b2k` is `1b` at seq 2048 — both
single-variable changes, and `deeper4k` is the first profile matching
Qwen3-8B's own `seq_length`. **2048 is the ceiling for the 1B shape** (the
absolute limit is ~2,724 tokens, so seq 4096 at 1B would need 24.01 GiB and
OOM); seq 4096 is only affordable on the 455.9 M geometry. `seq_length` does not affect the
parameter count, so its static state is identical to `deeper`'s and every
change is in activations. See
[`results/deeper4k-prediction.md`](results/deeper4k-prediction.md).

Two results worth reading off that table:

**Efficiency comes from GEMM shape, not from how many GEMMs there are.** Three
cleanly separated effects across the four runs:

| change | effect on MODEL_TFLOP/s |
|---|---|
| hidden 1024 -> 1536 (`smoke` -> `wider`) | **+12.9%** (30.9 -> 34.9) |
| seq 1024 -> 2048 (`smoke` -> `deeper`) | **+18.8%** (30.9 -> 36.7) |
| layers 6 -> 20 at fixed width and seq (`wider` -> `1b`) | **−1.7%** (34.9 -> 34.3) |

That last row is a clean attribution, because `wider` -> `1b` changes
`num_layers` and nothing else. **Depth does not buy efficiency.** It was
predicted to help (more independent work for the scheduler to overlap) and it
did not — see [`results/1b-prediction.md`](results/1b-prediction.md), R4.

**Parameter count is not a proxy for step cost.** `deeper` has fewer
parameters than `wider` (455.9 M vs 629.5 M) but does 1.605x the FLOPs per
step, because seq 2048 doubles tokens/step and quadruples the attention term.

### The `1b` profile: 1 billion parameters on one A10G

```bash
# on the instance
NUM_LAYERS=20 HIDDEN_SIZE=1536 FFN_HIDDEN_SIZE=4608 \
  NUM_ATTENTION_HEADS=12 NUM_QUERY_GROUPS=3 \
  DATA_PATH=/workspace/run/datasets/c4_qwen3 TRAIN_ITERS=50 ./g5/run.sh

# driven from a laptop
NUM_LAYERS=20 HIDDEN_SIZE=1536 FFN_HIDDEN_SIZE=4608 \
  NUM_ATTENTION_HEADS=12 NUM_QUERY_GROUPS=3 \
  TRAIN_ITERS=50 ./g5/finish-run.sh
```

**1,009,385,472 parameters. Measured: 19.08 GiB peak allocated (84.8% of the
card), 7,085 tok/s, 0 skipped and 0 NaN iterations over 50 steps on real c4.**
It fits with **3.41 GiB spare**. Predicted 18.64-19.77 GiB beforehand, so the
measurement landed inside the bracket though outside both individual bands.

It is by far the tightest profile — `wider`, the previous largest, peaked at
52%. If you push past 1B, the fallback shape trades aspect ratio for activation
memory:

```bash
NUM_LAYERS=8 HIDDEN_SIZE=2048 FFN_HIDDEN_SIZE=6144 \
  NUM_ATTENTION_HEADS=16 NUM_QUERY_GROUPS=4 ./g5/run.sh   # 1.0082 B, ~18.7 GiB
```

Two deliberate choices in the shape, both worth knowing:

**Hidden 1536 / 20 layers, not 2048 / 8.** Aspect ratio 77 hidden per layer
against Qwen3-8B's 114; the shallower alternative is 256, more than 2x off.

**It is a single-variable change from `wider`** — same hidden, ffn, heads,
query groups and seq_length, with only `num_layers` moving 6 -> 20. That is the
experiment the refuted memory model needed and that none of the first three
profiles could provide: because both have `seq_length = 1024`, the FP32 logits
term is identical and **cancels exactly** in the difference of their residuals,
so

```
k = (residual_1b - 1.2065 GiB) / 22,020,096 units
```

measures the per-layer activation constant with **no assumption about the
logits term at all**. The two pair-fits that disagreed after `deeper` predict
18.64 GiB (k=24.81 B/unit) and 19.77 GiB (k=80.17), 1.14 GiB apart, so the run
picks one. `predict.py --self-check` asserts the single-variable property so a
future edit cannot silently break it.

Note the vocab share: embedding + LM head is **466.7 M of 1,009.4 M (46.2%)**,
against 15.2% for Qwen3-8B. A 1B proxy carrying the full 151,936 vocab is
structurally more vocab-dominated than the model it proxies. That is the price
of keeping the real tokenizer and the real logits/loss path, which is what the
validation exists to exercise.

Full prediction and thresholds: [`results/1b-prediction.md`](results/1b-prediction.md).



### Sizing an untested profile: measure it

`g5/predict.py --profile <name>` predicts memory. The **two-term** model it
started with is refuted, but the `1b` run measured the term that was missing,
and a three-term form now fits every measurement closely:

```bash
python3 g5/predict.py --form-test   # the whole story; exits 1 by design
```

The flat "+14% over static" heuristic is wrong — `wider` refuted it. The
decomposed two-term replacement is *also* wrong: it predicted `deeper` at
10.21-10.52 GiB against a measured 9.66, and across the four points its worst
pairwise extrapolation is 2.34 GiB, with one pair even requiring a *negative*
bytes-per-activation-unit. The root cause was that no pair among the first
three profiles was a single-variable change, so nothing could isolate a term.

**`1b` fixed that.** It differs from the measured `wider` in `num_layers`
alone, and both share `seq_length`, so the logits term cancels in the
difference and the per-layer constant is **measured, not fitted**:

```
k = (2.1589 − 1.2065) GiB / 22,020,096 units = 46.44 B/unit
```

Both earlier fits were wrong in opposite directions (24.81 and 80.17 bracket
it). With `k` known, the leftover is *not* a single constant — `smoke` and
`wider`/`1b` differ by 0.165 GiB at the same `seq_length` — so a third term is
needed:

| term | fitted over all four runs |
|---|---|
| fixed | 0.5366 GiB |
| per token | 149,765 B/token (**24.6%** of a full FP32 vocab logits row) |
| per `layers x seq x hidden x batch` | 51.02 B/unit |

Worst error **0.0787 GiB**, under 1% of peak on every profile. The per-token
coefficient landing at a quarter of a full FP32 logits row suggests Megatron
does not hold the whole logits tensor resident at peak — a hypothesis the
number is consistent with, not a demonstration.

**Still not validated**, and the weak direction is specific: three parameters
against four points leaves one degree of freedom, and the per-token coefficient
is pinned by a **single** point (`deeper` is the only profile at seq 2048;
dropping it makes the fit singular). A `seq` sweep at fixed layers and hidden —
`smoke` geometry at 512, 2048, 4096 — is three sub-minute runs and would settle
it.

The three-term model is what `predict.py --profile <name>` now reports, with a
±5% band. It reproduces all four measured runs to within 0.93% of peak.

**A caution on what ±5% of peak actually tests.** Static state (18 B/param) is
exact analytic arithmetic and is 75-89% of peak, so a ±5% band on *peak* allows
a ±42-44% error in the *residual* — the only part the model estimates.
`--self-check` reports the sensitivity directly: the per-token coefficient could
be wrong by **±53%** and still pass every assertion, because the largest
`seq_length` measured so far is 2048, where that term is at most 0.29 GiB. A
+20% error in it was mutation-tested and survives. The per-unit constant is
tighter at ±10%, and is independently *measured* at 46.44 by the `wider`/`1b`
pair.

That is what `deeper4k` is for: at seq 4096 the per-token term doubles to 0.57
GiB, and two candidate mechanisms for it separate by 1.75 GiB — disjoint even
at ±5%.

For sizing without a model: peak allocated ran **11-26% above** the 18 B/param
static figure across the four measured shapes. Treat 18 B/param as a floor and
budget to the top of that range.

**The throughput model, by contrast, is validated.** Calibrated on `smoke`
alone and never refitted, it predicts FLOPs per step to within **0.04%** on all
three profiles, across changes in layers, hidden size and sequence length:

| | predicted | implied by measurement | error |
|---|---|---|---|
| `smoke` | 10.22 TFLOP | 10.23 | −0.04% |
| `wider` | 19.94 TFLOP | 19.93 | +0.04% |
| `deeper` | 31.99 TFLOP | 32.00 | −0.03% |
| `1b` | 39.69 TFLOP | 39.65 | +0.10% |

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

## Training on the c4 dataset

The mock dataset keeps the validation self-contained and offline, but **its
loss curve is not a learning signal** — see the warning under
[What it does validate](#what-it-does-validate). Real c4 is two commands.

### Step 1: build the dataset

```bash
./g5/prepare-c4.sh                        # 50M tokens (default), ~200 MB .bin
NUM_TOKENS=200000000 ./g5/prepare-c4.sh   # bigger budget
```

CPU only — it requests no GPU, so it can run while the A10G is idle or busy.
It streams `allenai/c4` and stops at the token budget, so the download is
bounded by what you actually consume. **No `HF_TOKEN` is needed**; `allenai/c4`
is public, and a measured run confirmed it streams unauthenticated.

It is idempotent: an existing verified build is reused rather than
re-downloaded, so re-running after a failure is cheap.

Sizing the budget: a run consumes `TRAIN_ITERS x GLOBAL_BATCH_SIZE x
SEQ_LENGTH` tokens. The 50M default is ~6x a 1000-step `smoke` run, so no token
is seen twice — which is what makes a falling loss curve meaningful rather than
memorisation.

### Step 2: train on it

```bash
DATA_PATH=/workspace/run/datasets/c4_qwen3 TRAIN_ITERS=1000 ./g5/run.sh
```

> **`DATA_PATH` is a path INSIDE the container, not on the host.** `g5/run.sh`
> mounts `RUN_BASE` at `/workspace/run` and passes `DATA_PATH` straight
> through, so a host path like `~/qwen3-g5/run/datasets/...` does not exist in
> the container and `train.py` will reject it. Use the `/workspace/run/...`
> form. `g5/finish-run.sh` translates between the two and prints both.

`train.py` fails fast with an explicit message if `DATA_PATH` is set but the
`.idx` is missing, rather than silently falling back to mock data.

### Or both in one command, from a laptop

```bash
PREPARE_C4=1 TRAIN_ITERS=1000 ./g5/finish-run.sh
```

This tunnels SSH over SSM (**no security-group rule needed** — the SSM agent
dials out), builds the dataset, verifies it, runs, parses throughput and
retrieves the log. It defaults `DATA_PATH` after a `PREPARE_C4=1` build, so you
cannot pay for a download and then silently train on mock data.

### Confirming you actually got real data

Three lines in the log, and they are worth checking because the failure mode is
a *successful-looking* mock run:

```
dataset: real Megatron-indexed data at /workspace/run/datasets/c4_qwen3
mock=False
> total number of epochs: 1
```

`epochs: 1` is the one that matters: it means no token was seen twice. A
measured 1000-step `smoke` run on real c4 moved the loss **12.1809 -> 5.8633**,
against the mock run's **12.1481 -> 0.2801** at identical geometry. The mock
collapse is degenerate fitting of 400 synthetic samples; 5.86 nats is a
plausible early-training cross-entropy for a 359M model that has seen 8.2M
tokens once (0.114% of Chinchilla-optimal for its size).

If a real-data run drops below ~2.0, treat it as a **defect**, not fast
convergence: the dataset is probably repeating or far smaller than reported.

### Real data adds a step-time tail

Measured on the 1000-step run: the median step is **unchanged** (0.331 s, same
as mock) but **9 of 999 steps (0.90%)** exceed 2x the median, p99 is still
1.01x, and 1.2% of wall clock is lost to stalls. Quote the **median** — the
mean understates throughput by 0.6%, and a short run that catches a stall
understates it much more.

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
[`../docs/data-loading-explained.md`](../docs/data-loading-explained.md) (line
41) already prescribes; `preprocess.py` is what diverged from it.

Document length need not equal the training `seq_length`: Megatron's
`GPTDataset` concatenates documents and re-splits them into `seq_length + 1`
samples.

Verify the format writer offline, with numpy alone and no network:

```bash
python3 g5/prepare_c4.py --self-test           # round-trip + corruption rejection
python3 g5/prepare_c4.py --verify-only PREFIX  # check an existing build
```

`prepare_c4.py` reuses `write_idx_file` lifted out of
`preprocessing/preprocess.py` rather than copying it, so the on-disk Megatron
format cannot drift between the two paths.

## Two nodes (2x g5.8xlarge)

Two instances give **DP=2**: two nodes, one A10G each. Launch both at once:

```bash
NODES=2 ./g5/launch-instance.sh
```

That puts both in **one subnet** (one AZ — cross-AZ would add latency to every
all-reduce) and adds exactly **one** security-group ingress rule: TCP
**1-65535** whose source is *the security group itself*. Only instances in that
group can reach any of it — not the VPC, not the internet. The script asserts
that shape and refuses to launch if the rule has a CIDR source, or if the port
range is too narrow to carry NCCL.

**Why the whole TCP range and not just 29500.** Opening only the rendezvous
port looks tighter and does not work. torchrun's TCPStore rendezvous does use
29500, but NCCL then builds its communicator over its **own** sockets on
**ephemeral** ports: each rank listens on a kernel-assigned port and the peer
opens a *new inbound* connection to it. Security groups are stateful only for
return traffic on an already-established flow, so those fresh inbound
connections are dropped. The rendezvous succeeds, rank 0 prints
`NCCL version ...`, and the run then **hangs silently** — measured here at 55
minutes with ~140 bytes/sec of inter-node traffic while both nodes billed.
This is AWS's documented requirement, not a workaround: see
[EFA and NCCL](https://docs.aws.amazon.com/us_en/AWSEC2/latest/UserGuide/efa-start-nccl.html)
and [AWS PCS security groups](https://docs.aws.amazon.com/pcs/latest/userguide/working-with_networking_sg.html).
Enabling EFA additionally needs `IpProtocol=-1`, since EFA is not TCP.

If you have a cluster whose group still carries the old single-port rule,
converge it without launching anything:

```bash
./g5/fix-cluster-sg.sh        # REGION=/NAME= to target another cluster
```

Two guards now exist for this failure mode: `g5/launch-instance.sh` asserts the
range spans the ephemeral ports before it launches, and
`g5/finish-run-2node.sh` aborts the run (and kills the remote containers) after
`STALL_TIMEOUT` seconds — default 420 — of *both* rank logs producing no output
at all, so a network-level hang costs seven minutes of billing rather than
however long it takes someone to notice. 420 is **below** PyTorch's own 600 s
collective timeout on purpose: when both were 600 the two timers were a dead
heat, NCCL's watchdog won, and the run died with a `SIGABRT` backtrace instead
of this driver's explanation. That timeout comes from the process group's
options and is **not** settable by environment variable, so the margin is
enforced on the shell side — the script refuses to start if `STALL_TIMEOUT` is
raised to 600 or beyond. Multi-node runs also default `NCCL_DEBUG=INFO`,
because at `WARN` a transport hang prints nothing.

**If a previous 2-node run was interrupted, the next one fails with
`EADDRINUSE`.** Killing the local ssh client does not stop the *remote*
container, so a Ctrl-C leaves rank 0's container still holding `MASTER_PORT`
under host networking, and both containers still holding their GPU. The next
attempt then dies at rendezvous creation with:

```
torch.distributed.DistNetworkError: The server socket has failed to listen on
any local network address. port: 29500, ... EADDRINUSE: address already in use
```

and rank 1 fails with it, because its rendezvous has no server. This needs no
manual recovery now: `g5/finish-run-2node.sh` kills stale containers on both
nodes in a preflight, then **asserts** `MASTER_PORT` is actually free (retrying
five times) and refuses to launch if a wedged container survived `docker kill`.
Its cleanup trap also stops the remote containers on *any* exit, Ctrl-C
included, so the state stops accumulating in the first place.

**A third failure looks like a network hang and is not one.** Once the security
group is correct, NCCL comes up cleanly and a run can still die like this:

```
WorkNCCL(SeqNum=15, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
ran for 600070 milliseconds before timing out.
Last enqueued NCCL work: 15, last completed NCCL work: 14
```

Read the NCCL log before blaming the network. If it contains
`ncclCommInitRankConfig ... Init COMPLETE` and `Connected all rings`, the
transport is **working** — those lines require the peer to participate. What
failed is a **rank desync**: one rank entered a collective the other never
reached. A 1-element `ALLREDUCE` is a `torch.distributed.barrier()`, and the
timestamp arithmetic locates it precisely: the collective above was enqueued
600,070 ms before it was caught, which lands on the `[after model, optimizer,
and learning rate scheduler are built]` line — Megatron's barrier at the
model-setup → data-setup boundary.

Rank 0's log is misleading on its own here, in two ways. First, it keeps
logging *after* the collective that times out, because a NCCL barrier's
`wait()` synchronises the CUDA stream rather than the CPU thread, so rank 0
runs ahead while the GPU-side barrier sits unmatched. Second, rank 0 only ever
records that it *waited*; the rank that failed to arrive is the one holding the
cause. `g5/finish-run-2node.sh` therefore retrieves **every** rank's log as
`results/run-2node-<stamp>-rank<N>.log`, and on failure also preserves the
stdout it streamed from each rank, since a rank that dies before writing its
own logfile leaves no other trace.

**The measured cause of that desync was the Megatron index cache, not the
datasets.** The next failure mode below has the detail; it is worth reading
first, because it is the only cause so far observed on this path and it is
unconditional without a shared filesystem.

A second possible cause is **non-identical datasets**. There is no shared
filesystem, so each node builds c4 by streaming from the Hub, and two
independent builds can diverge — a truncated shard, or the rate limit behind
that `You are sending unauthenticated requests to the HF Hub` warning. Each
data-parallel rank derives its own sample and shuffle index from its own
`.bin`/`.idx`, so differing files give differing sample counts, differing
microbatch counts, and therefore different *sequences of collectives*. Checking
that both files merely **exist** does not catch this; the driver now compares
size **and** md5 of both files across nodes and refuses to launch on a
mismatch, which costs two checksums instead of ten minutes of billed silence
followed by an error that never mentions the dataset. This has not yet been
observed here — on 2026-10-03 and 2026-10-04 the checksums matched and the
cache was the cause — but the check is cheap and the symptom is identical.

**The fourth failure mode is the one that actually bit: Megatron's index cache
assumes a shared filesystem, and there is none here.** Rank 1 dies about a
minute into the run with:

```
FileNotFoundError: [Errno 2] No such file or directory:
'/workspace/run/datasets/c4_qwen3/cache/GPTDataset_indices/
 f245ba7ae1d1d6c2d1d1b825d10bf511-GPTDataset-train-document_index.npy'
```

while rank 0's log shows it *loading* the very same three files. That asymmetry
is not a race and not a corrupted copy. It is by design, in
`megatron/core/datasets/gpt_dataset.py`, where the build branch is guarded by:

```python
if not path_to_cache or (
        not cache_hit
        and (not torch.distributed.is_initialized()
             or torch.distributed.get_rank() == 0)):
```

On any rank other than 0 that branch is **unreachable** whenever a cache path
is set — and one always is, because leaving `path_to_cache` unset makes
Megatron derive `<data prefix>/cache/GPTDataset_indices`. So rank 1 never
builds; it always falls through to `numpy.load()`. The builder states the
assumption in its own comment, in `blended_megatron_dataset_builder.py`:

```python
# Then, build on other ranks; guaranteed to be data_cache hit
```

That guarantee holds on a cluster with a shared filesystem. On two independent
EC2 instances it is false, which makes the failure **unconditional**: it does
not matter whether the cache is cold on both nodes or warm on one, because rank
0 is the only rank that can ever create it. Worse, rank 0 gets *past* its own
build and then blocks on the post-build `torch.distributed.barrier()` waiting
for a rank that is already dead — which is why this surfaces ten minutes later
as the `SeqNum=15 ALLREDUCE` timeout above, an error that mentions neither
datasets nor files.

`g5/finish-run-2node.sh` handles it in two steps:

1. **Pre-sync.** Before launching, it copies node 0's index cache to node 1 and
   verifies it *on node 1* with `md5sum -c` against node 0's own checksum list.
   Copying is deliberate: data-parallel ranks must agree on the shuffle index,
   and a copy makes them identical by construction rather than trusting two
   hosts' numpy RNG to produce the same permutation. The check uses node 0's
   file list rather than a whole-directory digest, so entries left from an
   earlier `TRAIN_ITERS` — which have different hashes in their names and are
   never read — are not a reason to refuse to start.
2. **Self-heal.** A cold cache cannot be pre-synced, because nothing but a real
   run will create it. So if an attempt fails *and* either rank's log carries
   that `FileNotFoundError`, the driver syncs the cache rank 0 just built and
   retries **once**. No other failure is retried: this one is self-correcting
   only because the failed attempt did the useful half.

**That premise was false, and this driver's own watchdog falsified it.**
Measured 2026-10-04, the self-heal fired and the retry still failed:

| | rank 0 | rank 1 died on |
|---|---|---|
| attempt 1 | built **train**, built **valid**, killed at 420s before **test** | the `train` index |
| the sync | copied 8 files, "verified byte-identical" — **no test index** | |
| attempt 2 | loaded train+valid, built **test**, finished | the `test` index |

Building the sample and shuffle indices for ~49k samples is a long numpy
operation that logs **nothing** while it writes. The stall watchdog read that
silence as a hang and killed rank 0 mid-build — the same mistake as the output
buffering above, in a different place: **log silence is not the absence of
progress.** It then synced a faithful copy of an incomplete cache and called it
verified, because "byte-identical" was being checked against the wrong thing.

Two fixes:

- The watchdog now probes node 0's index-cache size when the logs go quiet, and
  resets the stall timer while the cache is **growing**. It is only probed
  while quiet, so a normal training run costs nothing extra. It fails safe: an
  unchanged cache, a cache that does not exist, and a failed probe all still
  abort, so a genuine hang is still caught.
- The sync now asserts all three splits (`train`, `valid`, `test`) have a
  document index, and reports an incomplete cache as incomplete instead of
  reporting success. Byte-identical is necessary and not sufficient — a
  faithful copy of two-thirds of a cache fails on the remaining third.

The indices are keyed by a hash of the dataset path, sequence length, random
seed and `TRAIN_ITERS × GLOBAL_BATCH_SIZE`, so **changing `TRAIN_ITERS` makes a
previously warm cache a miss** and costs one self-healed attempt.

`g5/throughput.py` now names this directly rather than reporting a generic
abort, and the driver parses **every** retrieved rank log instead of rank 0's
alone — on 2026-10-04 rank 0's log yielded "no explicit failure marker was
found" while rank 1's held the whole explanation.

**EFA is supported on `g5.8xlarge` but was never ATTACHED.** An earlier revision
of this section said "there is no EFA on `g5.8xlarge`" and that was **wrong** --
contradicted by this repo's own line 19 and by the API:

```
$ aws ec2 describe-instance-types --instance-types g5.8xlarge \
    --query 'InstanceTypes[0].NetworkInfo.{Efa:EfaSupported,Max:EfaInfo.MaximumEfaInterfaces}'
Efa: True,  Max: 1
```

What was actually true is that `g5/launch-instance.sh` attached a plain ENA, so
the nodes reported `InterfaceType: "interface"` rather than `"efa"`, and
libfabric therefore found no provider:

```
NET/OFI No eligible providers were found
NET/OFI Selected provider is tcp, fabric is 172.31.32.0/20 (found 1 nics)
NET/OFI Need to force simple protocol: GDR not supported
```

So inter-node traffic fell back to plain TCP over `ens5` with no GPUDirect and
`NCCL_PROTO=simple` forced -- a **launch-configuration gap, not a property of
the instance type.** Measured consequence: 4.08 Gbit/s of a nominal 25 Gbit/s,
and a 3x slowdown versus one node (see the measured table below).

#### FIXED: the launcher now attaches an EFA at 2 nodes

`USE_EFA` defaults to `1` when `NODES=2` and to `0` at `NODES=1`, where there is
no inter-node traffic for it to carry. **Four** things had to change together,
because any one alone leaves NCCL on TCP or unable to connect at all:

| layer | change | why alone it is not enough |
|---|---|---|
| interface | `run-instances --network-interfaces "...InterfaceType=efa..."` | without a device, libfabric has no provider to open |
| SG inbound | `IpProtocol=-1` self-referencing, replacing `tcp/1-65535` | **EFA is not TCP**, so a TCP rule silently blocks it |
| SG outbound | `IpProtocol=-1` self-referencing, **added** to the default `0.0.0.0/0` | a **CIDR** rule cannot match non-IP EFA traffic, so the default egress does not carry it |
| container | `--device /dev/infiniband` plus `FI_PROVIDER=efa` | the host having an EFA does not make it visible in the container |

The security-group layers are the ones that look done and are not. AWS's guide
says it in one sentence: "the self-referencing inbound **and outbound** rules
(allowing all traffic to and from the security group itself) are mandatory for
EFA to function. Without these rules, EFA traffic between instances will be
blocked and NCCL communication will fail." I read that sentence, implemented the
inbound half, and missed the outbound half -- which cost a run (see "FOURTH
LAYER" below). The launcher now asserts both, and **refuses to launch** against
a `tcp/1-65535` inbound rule or a missing self-referencing outbound rule.

The outbound rule is **added alongside** the existing `0.0.0.0/0` egress, never
instead of it: the nodes need outbound internet for the ~77 GB image pull and
the c4 download. Exposure is unchanged in kind -- every rule added here is
self-referencing, with no CIDR. `IpProtocol=-1` widens the protocol set, not the
source set.

No custom image is needed. The stock `nvcr.io/nvidia/nemo:26.04` already logged
`Initializing aws-ofi-nccl 1.17.3` and `Using Libfabric version 2.3` on the
failing run -- the software was always there, with no device to use. The
container-side detection is on `/dev/infiniband` rather than a flag, so the same
`run.sh` stays correct on a `USE_EFA=0` node.

After launch the script asserts every node reports `InterfaceType: "efa"` and
refuses otherwise, because requesting an interface is not evidence of getting
one -- the previous run launched cleanly and the only symptom was being 3x slow
35 minutes later. An EFA **cannot be attached to a running instance**, so
enabling this requires relaunching.

#### CONFIRMED: EFA negotiates. Then a fourth layer bit.

The relaunch on 2026-10-04 proved the three changes above work. Both ranks:

```
NET/OFI Selected provider is efa, fabric is efa (found 1 nics)
NET/OFI NIC group 0 device #0 0000:00:1d.0
```

`No eligible providers were found` is **gone**. Interface, security group and
container device are all correct, and libfabric opens the fabric.

The run then aborted instantly, both ranks, same second, `SIGABRT`:

```
FI_EFA_USE_DEVICE_RDMA=1 was set by user, but EFA device has no
rdma-read capability.  Application will abort().
```

That variable was **mine**, copied from `eks/pretrain.yaml` and described in the
commit as matching "the working EKS path". It was not working -- that manifest
has `USE_EFA: "0"`, so its `FI_EFA_USE_DEVICE_RDMA: "1"` had never once been
exercised. I propagated an untested value and called its source a precedent.

Worse, the evidence was already in hand: the earlier TCP run's log says
`Need to force simple protocol: GDR not supported`, and GDR *is* GPUDirect
RDMA -- the identical capability. g5's A10G-based EFA does not have it.

So the variable is now **unset** on both paths, and libfabric uses the device's
real capability. EFA still works; only zero-copy GPU reads are unavailable, and
they never existed on this hardware. Two subtleties worth keeping:

- Emptying the ConfigMap value is **not sufficient** on the EKS path, because
  the pod script read it as `${FI_EFA_USE_DEVICE_RDMA:-1}` and `:-` substitutes
  on EMPTY as well as unset -- the default would have silently restored `1`.
  The test is now on a non-empty value, and an explicit `1` gets a warning
  naming the abort.
- Set it only on hardware with rdma-read (p4d/p5 and similar), and confirm with
  `fi_info -p efa` rather than assuming.

Whether EFA recovers the predicted ~1.8x is **still untested**: no 2-node run
has yet completed a step over it. The arithmetic says it is the one change with
the leverage, since ~92% of the step is transfer.
`eks/set-efa.sh` does the equivalent on the EKS path.

#### FOURTH LAYER: the handshake needs a self-referencing OUTBOUND rule

With RDMA unset, the next run got further and failed differently -- no
`SIGABRT`, a clean Python exception on both ranks:

```
torch.distributed.DistBackendError: NCCL error ... ncclRemoteError
NET/OFI Request ... completed with error. RC: 103. Error: 4126
(Unresponsive receiver (reachable by EFA device but handshake failed)
 My EFA addr: fi_addr_efa://[fe80::8ff:e2ff:fe9a:b815]:0:102020407
 My host id: i-0f9e509bdf5c054bc
 Peer EFA addr: fi_addr_efa://[fe80::8ff:f7ff:fefc:bc43]:0:1879318055
 Peer host id: N/A)
```

It died in `_initialize_distributed` at `torch.distributed.barrier()` -- the
first collective, before any model work. "Reachable by EFA device but handshake
failed" is precise: addressing resolved, and `Peer host id: N/A` says the reply
never came.

The cause was the security group, again, and specifically the half of AWS's
sentence I had not implemented. Measured state at the time of failure:

| | rule | carries EFA? |
|---|---|---|
| inbound | `-1` from the group itself | yes |
| outbound | `-1` to `0.0.0.0/0` | **no** |

`0.0.0.0/0` is a **CIDR**, and EFA traffic is not IP, so a CIDR rule cannot
match it -- the default egress permits every IP packet and no EFA packet. The
same JMESPath query counting self-referencing `-1` rules returned **1 for
inbound and 0 for outbound** on the live group, which is both the diagnosis and
proof the check is not vacuous.

Everything else was already right and was verified: same subnet
(`subnet-b3815cee`), same AZ (`us-west-2c`), `InterfaceType: efa` on both
nodes, one security group, `Selected provider is efa` on both ranks.

**This one needs no relaunch.** Unlike an EFA interface, a security group rule
applies to running instances immediately:

```bash
./g5/fix-cluster-sg.sh        # adds the outbound rule, launches nothing
```

`fix-cluster-sg.sh` now converges both halves and asserts both, and the
launcher does the same for a fresh cluster. Three successive too-narrow shapes
have each cost a run here -- `tcp/29500` only, then `tcp/1-65535` inbound only,
then inbound-only all-protocol -- so each one now has an assertion that refuses
rather than a comment that warns.

Then on **each** node — identical except `NODE_RANK`, and `MASTER_ADDR` is
rank 0's **private** address on both (the launcher prints the exact commands):

```bash
# rank 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.1.42 GLOBAL_BATCH_SIZE=16 \
  DATA_PATH=/workspace/run/datasets/c4_qwen3 TRAIN_ITERS=1000 ./g5/run.sh

# rank 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.1.42 GLOBAL_BATCH_SIZE=16 \
  DATA_PATH=/workspace/run/datasets/c4_qwen3 TRAIN_ITERS=1000 ./g5/run.sh
```

`run.sh` refuses a `MASTER_ADDR` that is not RFC1918 or cluster-internal DNS,
and refuses a non-loopback address when `NNODES=1`, so a rendezvous cannot be
bound to a public interface by accident.

Run `./g5/prepare-c4.sh` on **both** nodes: there is no shared filesystem on
this path, so each needs its own copy of the dataset. (The EKS path uses one
ReadWriteMany volume instead.)

### Or drive the whole 2-node run from a laptop

```bash
NODES=2 ./g5/launch-instance.sh                   # launch both
TRAIN_ITERS=1000 ./g5/finish-run-2node.sh         # build c4 + run on both
```

`finish-run-2node.sh` resolves both instance ids and rank 0's private address,
pushes a 60-second ephemeral SSH key to each over SSM (no security-group rule,
nothing persisted to `authorized_keys`), copies the changed `g5/` files, builds
the dataset on both nodes concurrently, starts **rank 1 first** so the
rendezvous has both ends present, waits for both, then parses rank 0's
throughput and retrieves its log.

It defaults `GLOBAL_BATCH_SIZE` to **16**, not 8, for the reason below. It also
refuses to start if `g5/train.py` is not DP-aware, if either node's SSM agent
is not Online, or if rank 0's address is not RFC1918 — and warns loudly if the
two nodes landed in different AZs, since every all-reduce would then cross an
AZ boundary.

Tear both down in one call:

```bash
./g5/terminate-instance.sh                    # reads g5/.last-instance-id
./g5/terminate-instance.sh i-0aaa i-0bbb      # or explicitly
```

### GLOBAL_BATCH_SIZE=16 is not optional

`tokens/step` is `GLOBAL_BATCH_SIZE x seq_length` **regardless of node count**.
So at a *fixed* batch size, adding a node halves the compute per rank but does
not shrink the gradient all-reduce, whose volume is set by parameter count.
Using the measured per-GPU TFLOP/s against the 25 Gbit link:

| profile | all-reduce | 1-node step | 2-node best | speedup |
|---|---|---|---|---|
| `smoke` | 719 MB | 331 ms | 311 ms | **1.07x** |
| `wider` | 1259 MB | 571 ms | 543 ms | **1.05x** |
| `deeper` | 912 MB | 872 ms | 469 ms | 1.86x |

`smoke` and `wider` are **comm-bound** (all-reduce / compute = 1.54 and 1.57),
so a second instance doubles the bill for ~5%. Doubling `GLOBAL_BATCH_SIZE`
instead keeps per-rank compute at its 1-node value and doubles tokens/step,
amortising the same all-reduce over twice the work:

| profile | tok/step | 1-node tok/s | 2-node tok/s | speedup |
|---|---|---|---|---|
| `smoke` | 16,384 | 24,757 | 44,781 | **1.81x** |
| `wider` | 16,384 | 14,357 | 25,677 | **1.79x** |
| `deeper` | 32,768 | 18,793 | 37,589 | **2.00x** |

Both tables are **derivations, now REFUTED by measurement on `g5.8xlarge`.**

#### MEASURED: a second `g5.8xlarge` makes this model 3x SLOWER

First completed 2-node run, 2026-10-04, 1000 iterations, real c4, identical
geometry to the `smoke` single-node baseline (4 layers, hidden 1024, ffn 3072,
359.4 M params, seq 1024):

| | tok/step | median step | median tok/s | vs 1 node |
|---|---|---|---|---|
| 1 node, GBS 8 | 8,192 | 0.331 s | 24,727 | — |
| 2 nodes, GBS 16 | 16,384 | **2.026 s** | **8,086** | **0.33x** |

The table above predicted **1.81x**; the measurement is **0.33x**, so the
derivation is wrong by a factor of 5.5. Doubling tokens/step should have left
step time at 0.331 s if the all-reduce overlapped; it rose **6.12x** instead.

**The step time IS the network transfer time.** CloudWatch `NetworkOut` on both
nodes held a steady ~153,000 MB per 300 s during the run, i.e. **510 MB/s per
node**, symmetric. Over one 2.026 s step that is **1,033 MB per node**, which is
what this model's gradient exchange costs: 359.4 M params x 2 B (bf16) = 719 MB
of gradients, and the distributed optimizer adds a parameter all-gather on top.
Dividing that volume by the achieved bandwidth gives 1,033 / 510 = **2.03 s**,
which is the entire step. Per-rank compute (~0.166 s, half the 1-node 0.331 s)
is wholly hidden inside it, so `overlap_grad_reduce` has nothing left to win:
communication is ~12x compute, not comparable to it.

The achieved **510 MB/s is 4.08 Gbit/s on a nominal 25 Gbit/s link -- 16%.**
That is the EFA assumption failing. Crucially it fails for a **fixable** reason:
`g5.8xlarge` reports `EfaSupported: true` with one EFA interface available, but
`g5/launch-instance.sh` attaches a plain ENA, so both nodes report
`InterfaceType: "interface"` and libfabric reports `NET/OFI No eligible
providers were found`, falling back to `Selected provider is tcp` with
`GDR not supported`. The 311 ms figure predicted above needs 719 MB in 0.311 s
= 18.5 Gbit/s, which TCP at 4.08 Gbit/s cannot approach but EFA on a 25 Gbit
link plausibly can.

**Practical conclusion: do not add a second `g5.8xlarge` without EFA.** The
0.33x above is what a plain-ENA pair measures, and there one node is 3x faster
and half the price. The derivation was not wrong about the hardware -- it was
wrong to assume the launcher configured it.

`g5/launch-instance.sh` now attaches an EFA by default at `NODES=2` (see
"FIXED" and "CONFIRMED" above), so a cluster launched today does not reproduce
that figure. EFA is confirmed to **negotiate** -- both ranks log
`Selected provider is efa, fabric is efa` -- but whether it recovers the
predicted ~1.8x is **still untested**: no run has completed a training step
over it, so the only 2-node throughput on record remains the TCP one. The
arithmetic says it is the single change with the leverage, since ~92% of the
step is transfer, but that is a prediction, not a result. `USE_EFA=0 NODES=2`
reproduces the slow baseline deliberately if the two need comparing.

An EFA interface cannot be added to a running instance, so moving an existing
cluster onto it means terminating and relaunching.

The original derivation is reproduced from the single-node measured figures in
[`results/validation-run.md`](results/validation-run.md). Note that scaling the
batch changes the optimisation (larger effective batch), so a 2-node run is not
a like-for-like comparison with a 1-node one.

`g5/train.py` enables `overlap_grad_reduce`, `overlap_param_gather` and the
distributed optimizer automatically when `WORLD_SIZE > 1`, and leaves all three
off at `WORLD_SIZE == 1` — so the three recorded single-node measurements remain
valid. It also rejects a `GLOBAL_BATCH_SIZE` that is not divisible by
`nodes x MICRO_BATCH_SIZE` with an actionable message.

**The better reason to add a node is memory, not speed.** The distributed
optimizer shards optimizer state across DP ranks, so a larger proxy fits per
GPU. Measure it rather than predicting it — this repo's analytic memory model
was refuted by the `deeper` run.

## Running on EKS

[`g5/eks/pretrain.yaml`](eks/pretrain.yaml) runs the same `g5/train.py` on 1 or
2 g5.8xlarge pods. One file, six resources, all namespaced.

### Prerequisites

1. An EKS cluster with a g5.8xlarge node group (1 or 2 nodes).
2. The NVIDIA device plugin, so nodes advertise `nvidia.com/gpu`:
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
   kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
   ```
3. A StorageClass for the dataset volume. **2 nodes needs ReadWriteMany
   (EFS)** — both pods mmap the same `.bin`/`.idx`. An EBS volume cannot be
   mounted by two nodes and the second pod hangs `Pending`. For 1 node, switch
   the PVC to `ReadWriteOnce` on your EBS class.
4. The repo reachable by `git clone` at the ref in the ConfigMap. An init
   container clones it rather than baking an image, so `g5/train.py` in the
   cluster is the same file as in the repo and cannot drift.

### Run it

```bash
./g5/eks/set-nodes.sh 1          # or 2
python3 g5/eks/validate.py       # catches mismatches kubectl would accept
kubectl apply -f g5/eks/pretrain.yaml

kubectl -n qwen3-pretrain wait --for=condition=complete job/c4-prep --timeout=30m
kubectl -n qwen3-pretrain logs -f job/qwen3-pretrain -c train

kubectl delete namespace qwen3-pretrain   # teardown
```

Use `set-nodes.sh` rather than editing by hand. Four values must move together
(`parallelism`, `completions`, `NNODES`, `GLOBAL_BATCH_SIZE`), and the failure
modes are quiet: if `parallelism` is 2 but `NNODES` is 1 you get two
independent `WORLD_SIZE=1` runs that look fine, and the reverse waits forever
for a rank that never joins. `validate.py` checks all four agree, plus the
Service cross-references and the exposure properties.

### How the pods find each other

`completionMode: Indexed` gives each pod a stable `JOB_COMPLETION_INDEX` (used
directly as `--node_rank`) and a stable hostname `<job-name>-<index>`. With the
headless Service as `subdomain`, rank 0 is always at:

```
qwen3-pretrain-0.qwen3-rdzv.qwen3-pretrain.svc.cluster.local
```

`publishNotReadyAddresses: true` is required — rank 1 must resolve rank 0
*before* rank 0 is serving, or the rendezvous deadlocks waiting for a DNS record
that only appears once the pod is ready. An init container waits for that name
to resolve and fails with a pointed message after 60s.

### Network exposure

The rendezvous Service is **headless** (`clusterIP: None`): DNS records for
pod-to-pod traffic, no cluster IP, no NodePort, no LoadBalancer, no
`hostNetwork`, no `hostPort`. The training port is reachable only by pods in
the cluster, which is the job's own network.

**Do not change it to `type: NodePort` or `type: LoadBalancer` to make
debugging easier.** That publishes a port carrying an unauthenticated PyTorch
rendezvous store onto the node or the internet. To observe a run use
`kubectl logs` or `kubectl port-forward`, which tunnel through the API server
and need no exposure. `g5/eks/validate.py` asserts all of this.

### EFA

g5.8xlarge **does** support EFA (`ec2 describe-instance-types` reports
`NetworkInfo.EfaSupported: true`), with one EFA interface. EFA bypasses the
kernel network stack for NCCL, which cuts all-reduce latency — and the gradient
all-reduce is the limiting factor for 2-node training here, so this is the one
knob that moves the scaling tables above toward their ceilings.

It ships **off**, because it needs cluster-side setup this manifest cannot do
for you. Turn it on with:

```bash
./g5/eks/set-efa.sh on          # or: off, or: on --dry-run
python3 g5/eks/validate.py      # asserts USE_EFA and the resource agree
```

Two things must change together and only one is an env var, which is why there
is a script:

| | what it does |
|---|---|
| `USE_EFA` in the ConfigMap | selects the libfabric provider at run time |
| `vpc.amazonaws.com/efa` in `resources` | makes the device visible to the pod |

A Kubernetes **resource request cannot be driven by an env var**, so enabling
EFA is a real edit. Setting `USE_EFA=1` without the resource gets you a pod with
no EFA device: NCCL falls back to TCP and the run is simply slower. The training
container detects exactly that and says so rather than hiding it, and
`validate.py` fails the mismatch.

Cluster prerequisites, and the second one is what usually bites:

```bash
# 1. the EFA device plugin
helm repo add eks https://aws.github.io/eks-charts
helm install aws-efa-k8s-device-plugin --namespace kube-system \
  eks/aws-efa-k8s-device-plugin

# 2. confirm nodes actually advertise the resource
kubectl get nodes -o custom-columns=NAME:.metadata.name,EFA:.status.allocatable.'vpc\.amazonaws\.com/efa'
```

If that column is empty the pod stays `Pending` with
`Insufficient vpc.amazonaws.com/efa`. **The node group must have been created
with EFA on its launch template** — neither the script nor the plugin can add
an interface to a running node.

EFA only matters at 2 nodes. At 1 node there is no inter-node traffic, so it
changes nothing.

### Notes

- The prep Job requests **no GPU**, so dataset building never occupies the
  A10G.
- `/dev/shm` is a 16Gi Memory-medium `emptyDir` (the Kubernetes equivalent of
  `--shm-size=16g`). It counts against the pod memory limit, which is why that
  limit is 96Gi.
- `backoffLimit: 0`: a distributed job cannot usefully restart one rank out of
  two, so it fails rather than retrying into a half-dead rendezvous.
- EFA is supported on g5.8xlarge but **not** wired up here; the manifest uses
  the TCP path, which needs no device plugin. The scaling tables above assume
  EFA, so measured 2-node throughput will be somewhat below them.
