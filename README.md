# Qwen3-8B Pre-Training: H200 vs B300 (NeMo/Megatron)

Pre-training **Qwen3-8B** (8.2B dense parameters) on 1T tokens comparing two GPU generations — p5en.48xlarge (H200) and p6-b300.48xlarge (B300) — using NeMo/Megatron on 2-node / 16-GPU topologies with EFA GDRDMA interconnect.

## Results

| Metric | H200 (p5en) | B300 (p6-b300) | B300/H200 | g5 1-node | g5 2-node |
|--------|-------------|----------------|-----------|-----------|-----------|
| Model trained | Qwen3-8B, 8.2B | Qwen3-8B, 8.2B | same | *proxy*, 1.01B | *proxy*, 1.49B |
| GPUs | 16× H200 141 GB | 16× B300 288 GB | — | 1× A10G 24 GB | 2× A10G 24 GB |
| Sequence length | 4096 | 4096 | same | 1024 | 1024 |
| **TFLOP/s per GPU** | 497 | **976** | 1.96× | 34.3 | 21.0 |
| **Throughput** | 162K tok/s | **318K tok/s** | 1.96× | 7,085 tok/s | 5,753 tok/s |
| **Time to 1T tokens** | ~71 days | **~36 days** | 1.97× | ~1,630 days | ~2,010 days |
| Step time (100 iters) | 3.23s | 1.65s | 1.96× | 1.16s | 2.85s |
| Peak memory/GPU | ~114 GB / 141 GB | ~173 GB / 288 GB | — | 19.1 / 22.5 GiB (85%) | 19.3 / 22.5 GiB (86%) |
| MFU | 0.50 | 0.50 | — | 0.27 | 0.17 |

Both clusters are compute-saturated with perfect communication overlap. AllReduce and AllGather are fully hidden behind compute.

**The two g5 columns are not a fourth and fifth data point in the same
comparison.** They train a proportionally scaled *proxy* model — 1.01B and
1.49B parameters at sequence length 1024, not Qwen3-8B at 4096 — because a
24 GB A10G holds 22.5 GiB against Qwen3-8B's 137.3 GiB of static state. They are
in this table to show what the cheap path does and does not reproduce, not to
rank A10G against H200:

- The `Time to 1T tokens` row is arithmetic on a *different and much smaller*
  model. It is in the thousands of days, which is the point: this is a stack and
  memory validation, not a training run.
- The **g5 MFU falls from 0.27 to 0.17 when the second node is added**, while
  the clusters hold 0.50 at 16 GPUs. That is the one genuinely transferable
  finding — at this geometry the g5 step time *is* the gradient all-reduce, with
  compute entirely hidden inside it, confirmed at two model sizes 4.1× apart.
  The clusters are the opposite case: compute-bound with communication hidden.
- g5 `TFLOP/s per GPU` and `MFU` are derived from the repo's own FLOP model
  ([`g5/predict.py`](g5/predict.py), calibrated once and never refitted) times
  the measured step time. The model reproduces measured throughput to 0.03%
  (5,753 implied vs 5,753 measured; 7,087 vs 7,085). Memory, throughput and step
  time are read directly from the runs.

## No cluster? Validate the whole stack on one cheap GPU

The results above need a Slurm cluster with H200s or B300s. If you want to
exercise **the same NeMo 26.04 / Megatron-Bridge stack** — same training script,
same container, real `c4` data, real EFA — on hardware you can start in five
minutes for a couple of dollars, use **[`g5/`](g5/README.md)**.

It runs a proportionally scaled Qwen3 on `g5.8xlarge` (one 24 GB A10G, ~$2.45/hr)
instead of 16 H200s, as two reproducible scenarios with both an EC2 and a
Kubernetes path:

| | hardware | model | measured | cost |
|---|---|---|---|---|
| **[`g5/single-node/`](g5/single-node/README.md)** | 1x g5.8xlarge | 1.01 B params | 19.08 GiB, 7,085 tok/s | ~$2 |
| **[`g5/multi-node/`](g5/multi-node/README.md)** | 2x g5.8xlarge + EFA | 1.49 B params | 19.27 GiB, 5,753 tok/s | ~$7 |

Start with **[`g5/PREREQUISITES.md`](g5/PREREQUISITES.md)** — AWS credentials,
the GPU vCPU quota (the usual hard stop), and a paste-in block that checks
everything before you spend anything.

What it is good for, and what it is not: it validates the stack, the fabric and
the memory model, and it carries a calibrated memory predictor
([`g5/predict.py`](g5/predict.py)) that reproduces every run measured so far.
It is **not** a performance comparison with the clusters above — a 24 GB A10G
holds a model ~5x smaller than Qwen3-8B. Each scenario README is explicit about
which of its numbers are measured and which are still predicted.

## Prerequisites

- **Slurm** workload manager with **PyXis + Enroot** container runtime
- **EFA networking** with GDRDMA support (for multi-node communication)
- **FSx for Lustre** shared filesystem mounted at `/fsx/`
- **Docker** (for building container images)

> **Don't have a cluster?** Deploy a fully functional HPC cluster in under 1 hour using [Amazon SageMaker HyperPod](https://awslabs.github.io/ai-on-sagemaker-hyperpod/). The guide walks you through deploying a ready-to-use cluster with Slurm, EFA, PyXis/Enroot, and FSx for Lustre pre-configured.

## Quick Start

> **Disk space:** The container build requires ~50 GB of disk space in TMPDIR.
> `enroot import` needs `sudo` and TMPDIR pointing to FSx or another file system (not `/tmp`, which is too small).

### Clone this repo and change it its directory
```bash
git clone https://github.com/paragao/qwen3-8b-optimised-pretraining.git
cd qwen3-8b-optimised-pretraining
```

### Prepare datasets (allenai/c4/en)
```bash
# export your Hugging Face token, if you have one
export HF_TOKEN=<your token>

# Prepare the dataset
sbatch preprocessing/preprocess.sh
```
Without your Hugging Face token, the download will be throttled. The script requires a token. 
Datasets are tokenized and transformed into binary mmap accessible files to avoid streaming data (`.idx` and `.bin` files).

### Build the container
```bash
# Build container
docker build -t qwen3-8b-pretraining:latest .

# Setup directories to run
mkdir -p /fsx/tmp && mkdir -p /fsx/ubuntu/qwen3-8b-pretraining/containers/

# Create the squash file with Enroot
sudo TMPDIR=/fsx/tmp ENROOT_TEMP_PATH=/fsx/tmp enroot import --output /fsx/ubuntu/qwen3-8b-pretraining/containers/nemo-efa-26.04.sqsh dockerd://qwen3-8b-pretraining:latest
```

### H200 Cluster (2x p5en.48xlarge)

```bash
# 1. Change to directory
cd h200

# 2. Submit training job
sbatch slurm/run.sh
```
Logs will be written to `/fsx/ubuntu/qwen3-8b-pretraining/logs`.
Checkpoints are saved to `/fsx/ubuntu/qwen3-8b-pretraining/checkpoints`.

### B300 Cluster (2x p6-b300.48xlarge)

```bash
# 1. Change to directory
cd b300

# 2. Submit training job
sbatch slurm/run.sh
```
Logs will be written to `/fsx/ubuntu/qwen3-8b-pretraining/logs`.
Checkpoints are saved to `/fsx/ubuntu/qwen3-8b-pretraining/checkpoints`.

## Running the g5 scenarios on EKS

The two [`g5/`](g5/README.md) scenarios also run on Kubernetes, as the `g5 1-node`
and `g5 2-node` columns in [Results](#results). This path is **self-contained**:
it does *not* need Slurm, FSx, the `sbatch preprocessing/preprocess.sh` step, the
`docker build`, or the `enroot import` above. An init container clones this repo
into the pod and a CPU-only Job builds the `c4` dataset in-cluster, so
`g5/train.py` running on the node is the same file as in the repo and cannot
drift from it.

Both scenarios are kustomize **overlays** on
[`g5/eks/pretrain.yaml`](g5/eks/pretrain.yaml), which stays the single definition
of the Namespace, ConfigMap, PVC, rendezvous Service, dataset Job and training
Job. `kubectl apply -k` needs no extra tooling — kustomize is built into
`kubectl`.

> **Local edits do not reach the cluster.** The init container clones
> `REPO_URL` at `REPO_REF` from the ConfigMap — currently pinned to the
> `feat/g5-single-gpu-validation` branch, which is where the g5 work lives until
> it merges. This is the opposite of the EC2 path, which `scp`s your local files
> to the instance. If you change `g5/train.py` and want the cluster to run it,
> push the change and point `REPO_REF` at your branch or commit.

### Cluster prerequisites (both scenarios)

```bash
# 1. An EKS cluster with a g5.8xlarge node group — 1 node for scenario 1,
#    2 nodes IN ONE SUBNET AND AZ for scenario 2.

# 2. The NVIDIA device plugin, so the nodes advertise nvidia.com/gpu
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
```

An empty `GPU` column means the training pod will never schedule.

**StorageClass.** The base PVC requests `ReadWriteMany` on a class named
`efs-sc`, because at 2 nodes both pods mmap the same `.bin`/`.idx` and an EBS
volume cannot be mounted by two nodes — the second pod hangs `Pending`. If your
cluster has no `efs-sc`, the PVC stays `Pending` and nothing starts. For
**scenario 1 only**, any `ReadWriteOnce` class will do; add this to
`g5/single-node/kustomization.yaml` under `patches:` and substitute your own
class name:

```yaml
  - target:
      kind: PersistentVolumeClaim
      name: qwen3-data
    patch: |-
      apiVersion: v1
      kind: PersistentVolumeClaim
      metadata:
        name: qwen3-data
        namespace: qwen3-pretrain
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: gp3
```

### Scenario 1 — single node, ~1B parameters, 85% of one A10G

```bash
# 1. Check the Kubernetes and EC2 definitions of the scenario still agree
./g5/scenario-check.sh single-node

# 2. Create the namespace, config, PVC, Service and both Jobs
kubectl apply -k g5/single-node/

# 3. Wait for the CPU-only dataset build (~10 min; 30 min cold, 77 GB image pull)
kubectl -n qwen3-pretrain wait --for=condition=complete job/c4-prep --timeout=30m

# 4. Watch the training run
kubectl -n qwen3-pretrain logs -f job/qwen3-pretrain -c train

# 5. Teardown — everything is namespaced
kubectl delete namespace qwen3-pretrain
```

Measured: **1,009,385,472 parameters, 19.08 GiB peak allocated (84.8% of
22.49 GiB), 7,085 tok/s**, 0 skipped and 0 NaN iterations. `SEQ_LENGTH=1024` is a
measured ceiling, not a cautious default — 2048 at this geometry OOMed after one
iteration with 38 MiB free. EFA is deliberately off: at one node there is no
inter-node traffic for it to carry.

### Scenario 2 — two nodes over EFA, DP=2, ~1.5B parameters

Two further cluster prerequisites, and the second is the one that usually bites:

```bash
# 1. The EFA device plugin
helm repo add eks https://aws.github.io/eks-charts
helm install aws-efa-k8s-device-plugin --namespace kube-system \
  eks/aws-efa-k8s-device-plugin

# 2. Confirm the nodes actually advertise the resource
kubectl get nodes -o custom-columns=NAME:.metadata.name,EFA:.status.allocatable.'vpc\.amazonaws\.com/efa'
```

If that column is empty the pods stay `Pending` with
`Insufficient vpc.amazonaws.com/efa`. **The node group must have been created
with EFA on its launch template** — an EFA interface cannot be added to a running
node, so a node group without one has to be replaced. Neither the plugin nor any
script here can do it for you.

The node security group must also allow **all traffic both ways to and from
itself**. Not inbound only, and not a CIDR rule: EFA is not IP, so the default
`0.0.0.0/0` egress rule cannot match it. Missing that outbound rule cost a run on
2026-10-04 — libfabric selected the `efa` provider and the handshake then failed
with `Unresponsive receiver (reachable by EFA device but handshake failed)`.

```bash
./g5/scenario-check.sh multi-node
kubectl apply -k g5/multi-node/

kubectl -n qwen3-pretrain wait --for=condition=complete job/c4-prep --timeout=30m

# Both ranks at once. `kubectl logs -f job/...` follows only ONE pod, so a
# 2-pod job needs the label selector and a raised request cap. Rank 0 carries
# the geometry and the memory summary; rank 1 carries the per-iteration
# timings, because Megatron's print_rank_last writes there.
kubectl -n qwen3-pretrain logs -f -l app=qwen3-pretrain -c train \
  --prefix --max-log-requests 2

kubectl delete namespace qwen3-pretrain
```

Measured: **1,490,550,784 parameters, 19.27 GiB peak allocated (85.7%), 5,753
tok/s** over 1000 iterations, loss 11.3351 → 5.4236, 0 skipped and 0 NaN.
`2048/18 = 113.8` matches Qwen3-8B's `4096/36 = 113.8` exactly, so this is the
best-proportioned proxy available at this memory budget.

The second node buys **+48% model capacity at lower throughput**, not speed. It
is the only honest summary of the 2-node result: `use_distributed_optimizer`
shards 12 of the 18 bytes/parameter of static state, so per-rank static memory
drops and a larger model fits — while the step time stays pinned to the gradient
all-reduce.

### What the overlays change, and why all three values move together

The multi-node overlay sets **three** things, because any one of them alone is a
*silent* misconfiguration rather than an error:

| | set alone, you get |
|---|---|
| `NNODES: "2"` in the ConfigMap | a hang at the rendezvous, waiting for a pod that was never created |
| `parallelism`/`completions: 2` on the Job | two pods each training alone at `WORLD_SIZE=1`, both looking healthy |
| `vpc.amazonaws.com/efa: 1` on the container | silent fallback to TCP — a **3.1× measured** throughput loss |

`./g5/scenario-check.sh` asserts these against the **built** manifest rather than
the patch text, so a patch that silently fails to apply is caught, and it
cross-checks every key against the `scenario.env` the direct-EC2 path reads. Two
copies of a geometry drift invisibly: both files stay valid and both paths keep
running while measuring different models.

`python3 g5/eks/validate.py` additionally checks the base manifest's own
invariants — the four node-count values, the Service cross-references, and that
the rendezvous stays headless with no NodePort, LoadBalancer or `hostNetwork`.

> **Do not change the rendezvous Service to `type: NodePort` or
> `type: LoadBalancer` to make debugging easier.** It is headless
> (`clusterIP: None`) on purpose; publishing it would expose an unauthenticated
> PyTorch rendezvous store on the node or the internet. Use `kubectl logs` or
> `kubectl port-forward`, which tunnel through the API server.

Each scenario's own README has the full detail, including which figures are
measured and which are still predicted:
[`g5/single-node/README.md`](g5/single-node/README.md),
[`g5/multi-node/README.md`](g5/multi-node/README.md). For the EC2 path instead of
Kubernetes, start at [`g5/PREREQUISITES.md`](g5/PREREQUISITES.md).

## Model Architecture: Qwen3-8B



| Parameter | Value |
|-----------|-------|
| Layers | 36 |
| Hidden dim (d_model) | 4096 |
| Q-heads | 32 |
| KV-heads | 8 (GQA) |
| FFN dim | 12288 (SwiGLU) |
| Vocab size | 151,936 |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Sequence length | 4096 |
| Precision | BF16 |
| Total params | 8.2B |

## Parallelism Strategy

**Pure Data Parallelism (DP=16)** — the model fits entirely on a single GPU.

| Component | Setting |
|-----------|---------|
| Tensor Parallel | 1 |
| Pipeline Parallel | 1 |
| Data Parallel | 16 |
| Distributed Optimizer | Yes (shards Adam states across DP ranks) |
| Overlap Grad Reduce | Yes |
| Overlap Param Gather | Yes |

**Why TP=1 is optimal:** At 8.2B params, the model + optimizer states fit on one GPU with distributed optimizer. Adding tensor parallelism introduces all-reduce communication for every transformer layer — validated experimentally: TP=2 was 11% slower (868 vs 976 TFLOP/s on B300).

## Best Configuration Per Cluster

| Parameter | H200 (p5en.48xlarge) | B300 (p6-b300.48xlarge) |
|-----------|---------------------|------------------------|
| GPUs | 16× H200 (141 GB HBM3) | 16× B300 (288 GB HBM3e) |
| Parallelism | TP=1, PP=1, DP=16 | TP=1, PP=1, DP=16 |
| **Micro-batch size** | **2** | **4** |
| Global batch size | 128 (grad_accum=4) | 128 (grad_accum=2) |
| Sequence length | 4096 | 4096 |
| Precision | BF16 | BF16 |
| **Gradient checkpointing** | Selective (core_attn only) | Selective (core_attn only) | 
| Distributed optimizer | Yes (sharded Adam) | Yes (sharded Adam) |
| Overlap grad reduce | Yes | Yes |
| Overlap param gather| Yes | Yes |
| Framework | Megatron-Bridge (NeMo 26.04) | Megatron-Bridge (NeMo 26.04) |

## Key Findings

1. **Both clusters are compute-saturated with perfect communication overlap.** AllReduce and AllGather are fully hidden behind compute — verified by single-GPU benchmarks showing lower TFLOP/s due to reduced batch arithmetic intensity.

2. **Both clusters use the Megatron-Bridge recipe API.** NeMo 26.04 for both H200 and B300.

3. **Pure data parallelism is optimal** when the model fits in single-GPU memory. Distributed optimizer + overlapped grad reduce eliminate the memory penalty.

4. **Selective gradient checkpointing used on both clusters:** lightweight core_attn recompute is Megatron-Core's standard behavior, keeping H200 peak at ~114 GB (MBS=2) and B300 at ~173 GB (MBS=4).

## Hardware

| | H200 Cluster | B300 Cluster |
|---|---|---|
| Instance | p5en.48xlarge | p6-b300.48xlarge |
| Nodes | 2 | 2 |
| GPUs per node | 8× H200 | 8× B300 |
| GPU Memory | 141 GB HBM3 | 288 GB HBM3e |
| Interconnect | EFA GDRDMA (3200 Gbps) | EFA GDRDMA (6400 Gbps) |
| Intra-node | NVLink (900 GB/s) | NVLink (1800 GB/s) |

## Project Structure

```
├── README.md              ← You are here
│   Dockerfile             ← NeMo 26.04 + EFA container
├── h200/
│   ├── train.py           ← Megatron-Bridge training script
│   └── slurm/
│       └── run.sh         ← Slurm submission script
├── b300/
│   ├── train.py           ← Megatron-Bridge training script
│   └── slurm/
│       └── run.sh         ← Slurm submission script
└── g5/                    ← single-GPU validation on cheap hardware, no Slurm
    ├── README.md          ← measured results + the memory model
    ├── PREREQUISITES.md   ← START HERE: tools, credentials, quota, cost
    ├── predict.py         ← calibrated memory predictor (--self-check)
    ├── launch-instance.sh ← provision, terminate-instance.sh tears down
    ├── prepare-c4.sh      ← tokenize real c4 on the instance
    ├── finish-run.sh      ← 1-node driver; finish-run-2node.sh for 2
    ├── single-node/       ← scenario 1: scenario.env + kustomization.yaml
    ├── multi-node/        ← scenario 2: + max-model.py sizing search
    ├── eks/               ← shared kustomize base for both scenarios
    └── results/           ← logs from every run on record
```

## License

MIT-0
