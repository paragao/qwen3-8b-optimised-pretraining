# Scenario 2 — two `g5.8xlarge` over EFA, DP=2, maximum memory

Two A10Gs, data-parallel, with the gradient all-reduce on EFA rather than TCP.

## Measured: the EFA path works

`g5/results/run-2node-20261004-221909-rank1.log`, 2026-10-04, 1000 iterations
on real c4:

| | |
|---|---|
| transport | `Selected provider is efa` on **both** ranks |
| throughput | **25,056 tok/s** median, stdev 460 (1.8%) |
| step time | 0.654 s at 16,384 tok/step |
| steady-state steps | 99 |
| stability | loss 11.9155 → 5.5249, 0 skipped, 0 NaN |

Against the alternatives at the same geometry:

| | tok/s | vs 1 node |
|---|---|---|
| 1 node, GBS 8 | 24,727 | — |
| 2 nodes, **TCP** | 8,086 | 0.33x |
| 2 nodes, **EFA** | 25,056 | **1.01x** |

**EFA is worth 3.10x against TCP, and the second node is still worth +1.3%.**
Both numbers matter: EFA is not optional at two nodes, and two nodes is not a
speed-up for this geometry. The step is bandwidth-bound either way — 1,033 MB
moves per step, and per-rank compute is ~0.331 s against 0.654 s of
communication, so overlap cannot hide it. Full derivation in the main
[`../README.md`](../README.md).

## That measurement used a different geometry from this scenario

It ran the 4-layer `smoke` shape and peaked at **4.83 GiB — 21% of the card**.
So it validates the fabric, the throughput and the stability, and says
**nothing** about memory occupancy. The scenario's own geometry was measured
separately the next day — see below — and the two together are what cover both
halves: `smoke` establishes that EFA works and what it costs against TCP, and
18 × 2048 establishes what fits.

To reproduce that run exactly rather than this scenario:

```bash
NUM_LAYERS=4 HIDDEN_SIZE=1024 FFN_HIDDEN_SIZE=3072 \
  NUM_ATTENTION_HEADS=8 NUM_QUERY_GROUPS=2 \
  GLOBAL_BATCH_SIZE=16 TRAIN_ITERS=1000 ./g5/finish-run-2node.sh
```

## This scenario: 18 layers x hidden 2048 at DP=2 — MEASURED

2026-10-05, 1000 iterations on real c4 over EFA, this exact geometry
(`g5/results/run-2node-20261005-101859-rank*.log`):

| | |
|---|---|
| parameters | **1,490,550,784** |
| peak allocated | **19.27 GiB** — 85.7% of 22.49 GiB |
| peak reserved | 19.92 GiB (fragmentation 1.034) |
| throughput | **5,753 tok/s** median, stdev 26 (0.45%) |
| step time | 2.848 s at 16,384 tok/step |
| stability | loss 11.3351 → 5.4236, 0 skipped, 0 NaN |

`2048/18 = 113.8` and Qwen3-8B is `4096/36 = 113.8` — identical to three
significant figures, so this is the best-proportioned proxy at this memory
budget. Larger shapes exist (6 × 3072 reaches 1.58 B) and sit 4.5x off that
ratio.

### The prediction was recorded first, and it held

`max-model.py` predicted this geometry at **19.13 GiB** before it was run:

| | |
|---|---|
| predicted | 19.13 GiB |
| measured | 19.270 GiB |
| error | **+0.140 GiB, +0.73%** |
| headroom vs the 19.76 GiB ceiling | +0.49 GiB (predicted +0.62) |

Worth stating plainly where that sits: +0.140 GiB is **1.8x the model's own
worst error** over its four DP=1 calibration runs (0.0787 GiB). A confirmation,
at the loose end — which is what an extrapolation to a new width (2048, never
measured), a new depth and a new DP degree all at once should look like.

Megatron reports memory in decimal **GB**, not GiB. The conversion was checked
against the `1b` run rather than assumed: its raw `20.488 GB` gives
`19.081 GiB`, matching the 19.08 on record.

### And it settled the sharding question

`max-model.py` shipped saying the 18 B/param split was "a reading, not a
measurement" — specifically that whether the 4 B fp32 gradient buffer shards
"depends on the reduce-scatter schedule", a ~1.9 GiB band of uncertainty.
Megatron prints its own `weight and optimizer` accounting, which answers it:

| reading | static at DP=2 | vs Megatron's 16.6599 GiB |
|---|---|---|
| **conservative, 12 B/param** | **16.6582 GiB** | **−0.01%** |
| optimistic, 10 B/param | 13.8818 GiB | −16.68% |
| unsharded, 18 B/param | 24.9873 GiB | +49.98% |

It implies **12.0012 B/param**. The conservative reading is right to four
significant figures and **the optimistic one is refuted**: the fp32 gradient
buffer does not shard in this accounting. `--self-check` now asserts both
halves, and mutation-testing confirms flipping the reading fails it by 2.9 GiB
on peak.

### The step is still entirely the transfer, at 4.1x the model size

| | |
|---|---|
| parameters vs the 4-layer run | 4.147x |
| bytes/step, scaled from the 1,033 MB measured there | 4,284 MB |
| bytes/step, measured (1,510 MB/s × 2.848 s) | **4,299 MB** |
| agreement | **+0.4%** |
| implied transfer time | 2.837 s vs measured step **2.848 s** |

Gradient volume scales *exactly* with parameter count, and compute is again
wholly hidden inside the transfer. So the earlier finding was not an artefact of
a tiny model.

It also confirms the algebra behind the sizing search: communication scales with
parameters while compute scales with parameters **×** tokens-per-rank, so at
fixed tokens/rank the ratio is parameter-independent. **A bigger model does not
improve the two-node economics** — it stays ~2x communication-bound. The lever
is tokens per step per rank.

### Still open: SEQ_LENGTH=2048

`SEQ_LENGTH=2048` at the 20 × 1536 shape **OOMed on one node** (20.05 GiB
allocated, 38 MiB free). At DP=2 the sharding leaves room for it. With the
sharding term now measured rather than assumed, that prediction is worth more
than it was:

```bash
SEQ_LENGTH=2048 ./g5/finish-run-2node.sh      # after sourcing scenario.env
```

At this 18 × 2048 geometry, seq 2048 is predicted to **OOM** — run
`max-model.py --dp 2 --seq 2048` for the shapes that do fit there (16 × 2048 is
the deepest, and at +0.16 GiB it is inside the unresolved band).

## How deep and how wide can two nodes go?

```bash
python3 g5/multi-node/max-model.py              # DP=2, seq 1024
python3 g5/multi-node/max-model.py --self-check # prove DP=1 reproduces reality first
python3 g5/multi-node/max-model.py --dp 1       # the single-node ceiling, for contrast
```

It imports `g5/predict.py` rather than restating any of it, so the parameter
formula and the three measured activation constants cannot drift from the
validated copy. It adds only the DP sharding term and searches.

**Maxima at DP=2, seq 1024, micro-batch 1**, against the 19.76 GiB ceiling on
peak allocated (derived from the `1b2k` OOM, not `nvidia-smi`'s 22.49):

| hidden | max layers | params | h/L vs Qwen3-8B | predicted peak | headroom |
|---|---|---|---|---|---|
| 1024 | 84 | 1.32 B | 0.11x | 19.66 GiB | +0.09 — **too tight to trust** |
| 1536 | 36 | 1.44 B | 0.38x | 19.50 GiB | +0.25 — tight |
| **2048** | **18** | **1.49 B** | **1.00x** | **19.13 GiB** | **+0.62** |
| 2560 | 10 | 1.53 B | 2.25x | 19.04 GiB | +0.71 |
| 3072 | 6 | 1.58 B | 4.50x | 19.29 GiB | +0.47 |

**18 layers × hidden 2048 is the pick.** Not because it is the largest — 6 × 3072
reaches 1.58 B — but because its aspect ratio is `2048/18 = 113.8`, and
Qwen3-8B's is `4096/36 = 113.8`. Identical to three significant figures, so it
is the best-proportioned proxy available at this memory budget, and it carries
the most headroom of the deep options.

The single-node ceiling for contrast, same model, same script at `--dp 1`:

| hidden | max layers (DP=1) | max layers (DP=2) | params DP=1 → DP=2 |
|---|---|---|---|
| 1024 | 55 | 84 | 0.97 B → 1.32 B |
| 1536 | 21 | 36 | 1.04 B → 1.44 B |
| 2048 | 9 | 18 | 1.06 B → 1.49 B |

So the second node roughly **doubles the depth** at any width, and lifts the
parameter ceiling from ~1.06 B to ~1.5 B — about **+50%**. It buys capacity, not
speed: throughput at this geometry is parity with one node (see the main
README).

### What these numbers are and are not

Every DP>1 figure here other than the 18 × 2048 row is a **prediction**. One
2-node memory reading now exists — this scenario's own, which came in at
+0.73% of the predicted value and pinned the sharding term to 12.0012 B/param —
so the extrapolation has one anchor rather than none. That is one point, at one
width and one DP degree: it does not validate the whole surface.

`--self-check` establishes what is actually load-bearing. It asserts the DP=1
path reproduces `predict.py` exactly on all five profiles and agrees with every
observed outcome **including the `1b2k` OOM**, and it now also checks the DP=2
prediction against the measurement and the sharding reading against Megatron's
own accounting. Mutation-tested: flipping the sharding reading to optimistic
fails it by 2.9 GiB on peak, and corrupting the recorded measurement by 1 GiB
fails it too.

One caveat that still changes what you should do:

- **Treat anything inside ~0.5 GiB of the ceiling as unresolved.** The model's
  own worst error is 0.08 GiB, its one DP=2 error is 0.14 GiB, and the ceiling
  derives from a single OOM — so `84 × 1024` at +0.09 GiB is not a fitting
  configuration, it is a coin flip. This scenario sits at +0.49 GiB measured,
  just outside that band.

The sharding caveat that used to sit here is **resolved**: it read "whether the
4 B fp32 gradient buffer also shards depends on the reduce-scatter schedule",
a ~1.9 GiB band. Megatron's own accounting settled it at 12.0012 B/param, so
the conservative reading is correct and the optimistic one is refuted.

## Reproduce on EC2 directly

**First: [`../PREREQUISITES.md`](../PREREQUISITES.md).** It covers the AWS
credentials, the `session-manager-plugin` install, the IAM permission the
launcher needs and the GPU vCPU quota — and ends with a paste-in block that
checks all of them before you spend anything. **This scenario needs 64 vCPUs of
G-instance quota** (32 per node), twice the single-node requirement, and raising
a quota can take a day.

Then, from a clone of this repo:

```bash
git clone https://github.com/paragao/qwen3-8b-optimised-pretraining.git
cd qwen3-8b-optimised-pretraining
git checkout feat/g5-single-gpu-validation

export REGION=us-west-2                   # or wherever you have G quota

# 1. Load the scenario FIRST -- see the ordering note below.
set -a; . g5/multi-node/scenario.env; set +a

# 2. Launch two g5.8xlarge with EFA attached (~5 min).
NODES=2 ./g5/launch-instance.sh

# 3. Only if your security group predates EFA support in this repo. Safe to
#    run always: it converges the group and is a no-op when already correct.
./g5/fix-cluster-sg.sh

# 4. Bootstrap both nodes, build and verify the dataset on each, sync
#    Megatron's index cache, then train 1000 iterations. Allow ~30 min on cold
#    nodes for the 77 GB container pull, then ~48 min of training.
./g5/finish-run-2node.sh

# 5. TERMINATE BOTH. The pair bills ~$4.90/hr whether or not it is busy.
#    Every argument is an instance id; the launcher prints both.
./g5/terminate-instance.sh i-0aaa i-0bbb
```

Total wall clock is roughly **90 minutes** and about **$7**.

Order matters at step 1. `USE_EFA` is a **provisioning** knob —
`launch-instance.sh` attaches the interface and chooses the security-group
shape, and `fix-cluster-sg.sh` converges an existing group. The run itself
**autodetects** the fabric from `/dev/infiniband`, because the device is either
there or it is not and a flag could only disagree with the hardware. Sourcing
`scenario.env` after the launch would leave `USE_EFA` with nothing to act on.
(`USE_EFA=1` is already the default at `NODES=2`, so a launch without it is
still correct — the ordering is about the file meaning what it says.)

The driver builds the c4 dataset on both nodes, verifies they are
byte-identical, syncs Megatron's index cache (there is no shared filesystem),
and retrieves both ranks' logs into `g5/results/`.

If the run dies partway, the nodes keep their container image and dataset, so
re-running step 4 resumes from there rather than repeating the 30-minute cold
start. Nothing is cleaned up implicitly — step 5 is the only thing that stops
the billing.

## Reproduce on Kubernetes

```bash
kubectl apply -k g5/multi-node/
kubectl -n qwen3-pretrain logs -f job/qwen3-pretrain
```

**Three settings must agree, and the overlay sets all three** because any one
alone is a silent misconfiguration:

| | value | alone it gives you |
|---|---|---|
| `NNODES` in the ConfigMap | 2 | with `parallelism: 1`, a hang at the rendezvous waiting for a pod that was never created |
| Job `parallelism` / `completions` | 2 | with `NNODES: 1`, two pods each training alone |
| `vpc.amazonaws.com/efa` on the container | 1 | without it, `USE_EFA=1` selects the provider, finds no device, and falls back to TCP — a measured 3x loss |

### Cluster prerequisites beyond the base manifest's

1. **The EFA device plugin**, or an EFA-requesting pod stays `Pending` with
   `Insufficient vpc.amazonaws.com/efa`:
   ```bash
   helm repo add eks https://aws.github.io/eks-charts
   helm install aws-efa-k8s-device-plugin --namespace kube-system \
     eks/aws-efa-k8s-device-plugin
   ```
2. **A node group created with EFA on its launch template.** An EFA interface
   cannot be added to a running node, so a node group without one must be
   replaced. Confirm the nodes advertise it:
   ```bash
   kubectl get nodes -o custom-columns=\
   NAME:.metadata.name,EFA:.status.allocatable.'vpc\.amazonaws\.com/efa'
   ```
   An empty column means the pods will never schedule.
3. **A security group allowing all traffic both ways to and from itself.** Not
   just inbound, and not a CIDR rule — EFA is not IP, so the default
   `0.0.0.0/0` egress cannot carry it. Missing the outbound half cost a run:
   libfabric selected the efa provider and the handshake then failed with
   `Unresponsive receiver (reachable by EFA device but handshake failed)`.
4. **Both nodes in one subnet and one AZ.** Cross-AZ puts every gradient
   all-reduce over an AZ boundary, and the all-reduce is already the limit.

## Check the two paths still agree

```bash
./g5/scenario-check.sh multi-node
```

Asserts every key in `scenario.env` matches the **built** manifest, that
`NNODES` equals the Job's `parallelism` and `completions`, and that the EFA
settings are mutually consistent — including that `FI_EFA_USE_DEVICE_RDMA`
stays empty, since g5's EFA has no rdma-read and setting it aborts both ranks
on startup.

Mutation-tested: breaking the `parallelism` coupling, setting
`FI_EFA_USE_DEVICE_RDMA=1`, and requesting `USE_EFA=1` without the device
resource all fail it.

## Expected output

```
==> Syncing the Megatron index cache from node 0 to node 1
    all three splits present (train, valid, test) -- the cache is complete

NET/OFI Selected provider is efa, fabric is efa (found 1 nics)
 iteration       10/    1000 | ... lm loss: 1.19E+01
```

`NET/OFI No eligible providers were found` means the EFA device did not reach
the container — the run will still complete, about 3x slower, over TCP.
