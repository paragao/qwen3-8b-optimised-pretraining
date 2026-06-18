# B300 Cluster — Qwen3-8B Pre-Training

## Cluster Specification

| | |
|---|---|
| Instance | p6-b300.48xlarge |
| Nodes | 2 |
| GPUs | 16× NVIDIA B300 (288 GB HBM3e) |
| Interconnect | EFA GDRDMA (3200 Gbps) |
| Framework | NeMo 26.02, Megatron-Bridge v2.9.0 |
| Container | `nemo-efa-26.02.sqsh` |

---

## Prerequisites

- **Slurm** workload manager (in PATH on B300 clusters)
- **PyXis + Enroot** container runtime
- **EFA networking** with GDRDMA on p6-b300.48xlarge instances
- **FSx for Lustre** shared filesystem mounted at `/fsx/`
- **Docker** (for building container images)

> **Don't have a cluster?** Deploy a fully functional HPC cluster in under 1 hour using [Amazon SageMaker HyperPod](https://awslabs.github.io/ai-on-sagemaker-hyperpod/). The guide walks you through deploying a ready-to-use cluster with Slurm, EFA, PyXis/Enroot, and FSx for Lustre pre-configured.

---

## Best Configuration

| Parameter | Value |
|-----------|-------|
| Tensor Parallel | 1 |
| Pipeline Parallel | 1 |
| Data Parallel | 16 |
| Micro-batch size | 4 |
| Global batch size | 128 (grad_accum=2) |
| Sequence length | 4096 |
| Precision | BF16 |
| Gradient checkpointing | None (288 GB allows MBS=4 without recompute) |
| Distributed optimizer | Yes (shards Adam across DP ranks) |
| Overlap grad reduce | Yes |
| Overlap param gather | Yes |

---

## Result

| Metric | Value |
|--------|-------|
| TFLOP/s per GPU | 976 |
| Throughput | 318K tok/s |
| Time to 1T tokens | ~36 days |
| Step time | 1.65s |
| Peak memory/GPU | ~173 GB / 288 GB |
| MFU | 0.50 |

---

## Optimization Sweep Results

| MBS | Seq Len | Grad Ckpt | TFLOP/s/GPU | tok/s (16 GPU) |
|-----|---------|-----------|-------------|----------------|
| 2 | 4096 | Yes | 612 | 199K |
| 2 | 4096 | No | 784 | 255K |
| 4 | 4096 | Yes | 768 | 250K |
| **4** | **4096** | **No** | **976** | **318K** |
| 8 | 4096 | No | 941 | 306K |
| 4 | 8192 | No | 892 | 145K |
| 8 | 8192 | No | 914 | 149K |

### TP=2 Experiment (Negative Result)

TP=2, DP=8, MBS=4: 868 TFLOP/s/GPU, 283K tok/s — **11% worse than TP=1**. Unnecessary all-reduce communication overhead for a model that fits on one GPU. Pure DP wins.

---

## Reproduction Steps

### Prerequisites

- Slurm cluster with PyXis and Enroot (Slurm is in PATH on B300 clusters)
- EFA-enabled p6-b300 instances
- FSx for Lustre at `/fsx/`
- Docker installed

### 1. Build the Container

> **Disk space:** The container build requires ~50 GB of disk space in TMPDIR.
> `enroot import` needs `sudo` and TMPDIR pointing to FSx (not `/tmp`, which is too small).

```bash
cd /fsx/ubuntu/qwen3-8b/containers/
sudo docker build -t qwen3-8b-b300:latest -f Dockerfile .

sudo TMPDIR=/fsx/ubuntu/qwen3-8b/tmp ENROOT_TEMP_PATH=/fsx/ubuntu/qwen3-8b/tmp \
  enroot import --output /fsx/ubuntu/qwen3-8b/containers/nemo-efa-26.02.sqsh \
  dockerd://qwen3-8b-b300:latest

sudo chown ubuntu:ubuntu /fsx/ubuntu/qwen3-8b/containers/nemo-efa-26.02.sqsh
```

### 2. Deploy Training Script

```bash
mkdir -p /fsx/ubuntu/qwen3-8b/{logs,checkpoints,code}
cp b300/scripts/train.py /fsx/ubuntu/qwen3-8b/code/train_bridge.py
```

### 3. Submit Job

```bash
sbatch b300/scripts/run.sh
squeue -u ubuntu
tail -f /fsx/ubuntu/qwen3-8b/logs/<JOB_ID>.out
```

---

## Key Differences from H200

| | H200 | B300 |
|---|---|---|
| Slurm path | `/opt/slurm/bin/` (not in PATH) | In PATH |
| Framework | Megatron-Core (NeMo 25.07) | Megatron-Bridge (NeMo 26.02) |
| Gradient checkpointing | Required (141 GB limit) | Not needed (288 GB) |
| Micro-batch size | 2 (memory constrained) | 4 (memory available) |
| Training API | `pretrain()` with argparse | `pretrain(config=cfg)` with config object |
