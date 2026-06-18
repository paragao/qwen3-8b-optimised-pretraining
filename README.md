# Qwen3-8B Optimised Pre-Training

## TL;DR

Pre-training **Qwen3-8B** (8.2B params) on 1T tokens. B300 achieves **976 TFLOP/s** (318K tok/s, ~36 days). H200 achieves **497 TFLOP/s** (162K tok/s, ~71 days). **1.96× speedup.**

Both clusters run pure data parallelism (DP=16) with distributed optimizer and overlapped communication on 2-node / 16-GPU topologies connected via EFA GDRDMA.

---

## Hardware

| | H200 Cluster | B300 Cluster |
|---|---|---|
| Instance | p5en.48xlarge | p6-b300.48xlarge |
| Nodes | 2 | 2 |
| GPUs per node | 8× H200 | 8× B300 |
| GPU Memory | 141 GB HBM3 | 288 GB HBM3e |
| Interconnect | EFA GDRDMA (3200 Gbps) | EFA GDRDMA (3200 Gbps) |
| Intra-node | NVLink | NVLink |

---

## Prerequisites

- **Slurm** workload manager
- **PyXis + Enroot** container runtime (for running GPU containers via Slurm)
- **EFA networking** with GDRDMA support (for multi-node communication)
- **FSx for Lustre** shared filesystem mounted at `/fsx/`
- **Docker** (for building container images)

> **Don't have a cluster?** Deploy a fully functional HPC cluster in under 1 hour using [Amazon SageMaker HyperPod](https://awslabs.github.io/ai-on-sagemaker-hyperpod/). The guide walks you through deploying a ready-to-use cluster with Slurm, EFA, PyXis/Enroot, and FSx for Lustre pre-configured.

---

## Quick Start

### H200 Cluster (p5en)

```bash
# 1. Build container
cd h200/ && docker build -t qwen3-8b-h200:latest .
sudo enroot import --output /fsx/ubuntu/qwen3-8b/containers/nemo-efa-25.07.sqsh dockerd://qwen3-8b-h200:latest

# 2. Submit job
/opt/slurm/bin/sbatch h200/scripts/run.sh
```

### B300 Cluster (p6-b300)

```bash
# 1. Build container
cd b300/ && docker build -t qwen3-8b-b300:latest .
sudo enroot import --output /fsx/ubuntu/qwen3-8b/containers/nemo-efa-26.02.sqsh dockerd://qwen3-8b-b300:latest

# 2. Submit job
sbatch b300/scripts/run.sh
```

---

## Results

| Metric | H200 | B300 | Ratio |
|--------|------|------|-------|
| **TFLOP/s per GPU** | 497 | **976** | 1.96× |
| **Throughput** | 162K tok/s | **318K tok/s** | 1.96× |
| **Time to 1T tokens** | ~71 days | **~36 days** | 1.97× |
| Step time (100 iters) | 3.23s | 1.65s | 1.96× |
| Peak memory/GPU | ~138 GB / 141 GB | ~173 GB / 288 GB | — |
| MFU | 0.50 | 0.50 | — |

Both clusters are compute-saturated with perfect communication overlap. AllReduce and AllGather are fully hidden behind compute.

---

## Architecture: Qwen3-8B

| Parameter | Value |
|-----------|-------|
| Layers | 36 |
| Hidden dim (d_model) | 4096 |
| Q-heads | 32 |
| KV-heads | 8 (GQA) |
| FFN dim | 14336 (SwiGLU) |
| Vocab size | 151,936 |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Sequence length | 4096 |
| Precision | BF16 |
| Total params | 8.2B |

---

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

---

## Reproducing

See detailed per-cluster guides:
- [H200 Reproduction Guide](h200/README.md)
- [B300 Reproduction Guide](b300/README.md)

### Prerequisites (both clusters)
- Slurm cluster with PyXis and Enroot
- EFA-enabled instances with GDRDMA
- FSx for Lustre mounted at `/fsx/`
- Docker (for container builds)

---

## Key Findings

1. **Both clusters are compute-saturated with perfect communication overlap.** AllReduce and AllGather are fully hidden behind compute — verified by single-GPU benchmarks showing lower TFLOP/s due to reduced batch arithmetic intensity.

2. **Best software stack per GPU generation.** NeMo 25.07 (Megatron-Core v0.13.1) for H200; NeMo 26.02 (Megatron-Bridge v2.9.0) for B300. Each maximizes hardware utilization for its target architecture.

3. **Pure data parallelism is optimal** when the model fits in single-GPU memory. Distributed optimizer + overlapped grad reduce eliminate the memory penalty.

4. **Gradient checkpointing is the key differentiator:** required on H200 (141 GB) for MBS≥2, unnecessary on B300 (288 GB) for MBS≤4 — saving ~20% recompute overhead.

---

## Project Structure

```
├── README.md              ← You are here
├── executive-summary.md   ← One-page comparison
├── h200/                  ← H200 cluster (p5en.48xlarge)
│   ├── README.md          ← Reproduction guide
│   ├── Dockerfile         ← Container build
│   ├── scripts/
│   │   ├── train.py       ← Megatron-Core training script
│   │   └── run.sh         ← Slurm submission script
│   └── results/
│       └── benchmark.md   ← Detailed results
├── b300/                  ← B300 cluster (p6-b300.48xlarge)
│   ├── README.md
│   ├── Dockerfile
│   ├── scripts/
│   │   ├── train.py       ← Megatron-Bridge training script
│   │   └── run.sh         ← Slurm submission script
│   └── results/
│       └── benchmark.md
└── docs/
    ├── data-loading-explained.md
    └── lessons-learned.md
```

---

## License

Apache 2.0
