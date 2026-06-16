# Qwen3-8B Pre-Training Plan: H200 vs B300 Clusters

## Executive Summary

This document outlines the plan to run a **Qwen3-8B pre-training** workload on **1 trillion tokens** (C4/similar dataset) across two clusters:

| Cluster | Instances | GPUs | GPU Type | Memory/GPU |
| --- | --- | --- | --- | --- |
| **Cluster A** | 8× p5en.48xlarge | 64 | H200 SXM | 141 GB |
| **Cluster B** | 8× p6-b300.48xlarge | 64 | B300 Ultra | 268 GB |

Both clusters use **Slurm + PyXis/Enroot** for container orchestration and **FSx for Lustre** as the shared filesystem.

---

## 1. Model Architecture

| Parameter | Value |
| --- | --- |
| Parameters | 8.2B (dense) |
| Layers | 36 |
| Hidden Dim | 4096 |
| Attention Heads (Q) | 32 |
| KV Heads (GQA) | 8 |
| Head Dim | 128 |
| FFN Dim | 14336 (SwiGLU) |
| Vocabulary | 151,936 |
| Context Length | 4096 |
| Precision | BF16 |

---

## 2. Infrastructure Calculator Results

> ⚠️ **Assumptions**: MFU = 0.40 (40% of peak GPU FLOPS). This is a conservative planning estimate. Actual MFU varies by workload — well-optimized training can achieve 0.45–0.55. Results scale linearly with MFU.

### 2.1 Memory Analysis (Phase 1)

The model fits comfortably on a single GPU with **no tensor parallelism (TP=1)** and **no pipeline parallelism (PP=1)** on both clusters. Pure **Data Parallelism (DP=64)** is optimal.

| Metric | H200 (141 GB) | B300 (268 GB) |
| --- | --- | --- |
| Model Weights | 18.2 GB | 18.2 GB |
| Gradients | 18.2 GB | 18.2 GB |
| Optimizer (ZeRO-1, DP=64) | 1.7 GB | 1.7 GB |
| Activations (MBS=4) | 13.1 GB | 13.1 GB |
| Buffers (NCCL) | 6.0 GB | 6.0 GB |
| **Total (MBS=4)** | **70.9 GB** | **70.9 GB** |
| **Headroom** | 58.9 GB (50%) | 175.7 GB (74%) |
| Max micro-batch before OOM | 16 | 16+ |

**Recommendation**: Use **micro-batch size = 4** on H200 and **micro-batch size = 8** on B300 to maximize GPU utilization while leaving headroom for spikes.

### 2.2 Optimal Parallelism Strategy

| Parameter | Both Clusters |
| --- | --- |
| Tensor Parallelism (TP) | 1 |
| Pipeline Parallelism (PP) | 1 |
| Data Parallelism (DP) | 64 |
| Context Parallelism (CP) | 1 |
| ZeRO Stage | **1** (best throughput) |

**Rationale**: At 8.2B parameters, the model fits entirely on a single GPU with TP=1. This eliminates all intra-node communication overhead from tensor parallelism and maximizes throughput. ZeRO-1 (optimizer state sharding only) gives the best efficiency (no gradient communication penalty).

### 2.3 Batch Configuration (Phase 2)

| Config | Micro-Batch | Grad Accum | Global Batch (tokens) | Steps |
| --- | --- | --- | --- | --- |
| **Optimal** | 3 | 4 | 3.1M | 317,891 |
| Good | 2 | 4 | 2.1M | 476,837 |
| Good | 4 | 4 | 4.2M | 238,418 |
| Good (larger) | 2 | 8 | 4.2M | 238,418 |

**Recommended**: `micro_batch=4, grad_accum=4` → **4.2M tokens/batch, ~238K steps**. This is a standard batch size for 8B model pre-training.

### 2.4 Training Time Estimates (Phase 3)

| Cluster | ZeRO Stage | Time (days) | Range | Tokens/sec | Tokens/sec/GPU |
| --- | --- | --- | --- | --- | --- |
| **H200 (p5en)** | 1 | **22.5** | 16.9 – 28.1 | 514,602 | 8,041 |
| **B300 (p6-b300)** | 1 | **6.6** | 4.9 – 8.2 | 1,756,098 | 27,439 |

**B300 is ~3.4× faster** due to its 3.4× higher peak BF16 TFLOPS (3375 vs 989).

### 2.5 Communication Overhead (Phase 4)

| Metric | Both Clusters |
| --- | --- |
| ZeRO-1 AllGather Volume | 14.4 GB |
| ZeRO-1 AllGather Time | ~36 ms/step |
| Inter-node Bandwidth (EFA) | 400 Gbps |

With 8 nodes and DP=64, the all-reduce communication is well-amortized across the gradient accumulation steps.

---

## 3. Recommended Training Framework

**NeMo + Megatron-LM** is the recommended framework for this workload:

- Native Qwen3 support
- Optimized FSDP/ZeRO-1 implementation
- Built-in data pipeline with C4 support
- Checkpoint management and resumption
- Integrated logging (W&B, TensorBoard)

Alternative: **torchtitan** or **Hugging Face Accelerate + DeepSpeed ZeRO-1**

---

## 4. Container Build Plan

Based on the [nccl-tests.Dockerfile](https://github.com/awslabs/awsome-distributed-ai/blob/main/micro-benchmarks/nccl-tests/nccl-tests.Dockerfile) reference, here's the training container:

### 4.1 Base Image & Key Versions

| Component | Version | Notes |
| --- | --- | --- |
| Base Image | `nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu22.04` | Latest CUDA 13 |
| EFA Installer | 1.48.0 | Ships OFI NCCL plugin bundled |
| NCCL | v2.30.4-1 | Latest stable |
| GDRCopy | v2.5.2 | GPU Direct RDMA |
| PyTorch | 2.6+ (nightly) | CUDA 13 support |
| NeMo | latest main | Qwen3 support |
| Transformers | >= 4.46 | Qwen3 tokenizer |

### 4.2 Dockerfile Skeleton

```dockerfile
ARG CUDA_VERSION=13.0.2
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# ==== AWS Infrastructure Layer ====
ARG EFA_INSTALLER_VERSION=1.48.0
ARG NCCL_VERSION=v2.30.4-1
ARG GDRCOPY_VERSION=v2.5.2

# Remove stale packages
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get remove -y --allow-change-held-packages \
    ibverbs-utils libibverbs-dev libibverbs1 \
    libmlx5-1 libnccl2 libnccl-dev
RUN rm -rf /opt/hpcx /usr/local/mpi /etc/ld.so.conf.d/hpcx.conf && ldconfig

# System dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake curl git gcc gdb kmod \
    libtool openssh-client openssh-server pkg-config \
    python3-distutils python3-dev vim wget
RUN apt-get purge -y cuda-compat-*

# SSH for multi-node MPI
RUN mkdir -p /var/run/sshd
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking\).*/\1 no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes\).*/\1 no/g' /etc/ssh/sshd_config

# GDRCopy
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy && make prefix=/opt/gdrcopy install

# EFA Installer (includes OFI NCCL plugin)
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer

# NCCL from source (sm_80, sm_90, sm_100, sm_103)
RUN git clone -b ${NCCL_VERSION} https://github.com/NVIDIA/nccl.git /opt/nccl \
    && cd /opt/nccl \
    && make -j $(nproc) src.build CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 \
                  -gencode=arch=compute_100,code=sm_100 \
                  -gencode=arch=compute_103,code=sm_103"

# ==== Environment Variables ====
ENV LD_LIBRARY_PATH=/opt/gdrcopy/lib:/opt/nccl/build/lib:\
/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:\
/opt/amazon/openmpi/lib:/usr/local/cuda/extras/CUPTI/lib64:\
/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:\
/opt/gdrcopy/bin:$PATH
ENV OMPI_MCA_pml=^ucx \
    OMPI_MCA_btl=tcp,self \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent \
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo,veth \
    PMIX_MCA_gds=hash \
    LD_PRELOAD=/opt/nccl/build/lib/libnccl.so

# ==== ML Framework Layer ====
RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3 /tmp/get-pip.py

# PyTorch with CUDA 13 support
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Training frameworks
RUN pip3 install --no-cache-dir \
    transformers>=4.46 \
    datasets \
    tokenizers \
    accelerate \
    deepspeed \
    wandb \
    tensorboard \
    pynvml \
    awscli \
    flash-attn --no-build-isolation

# NeMo (for Megatron-LM based training)
RUN pip3 install --no-cache-dir \
    nemo_toolkit[all] \
    megatron-core

# ==== Health Check ====
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

```

### 4.3 NCCL Gencode Notes

- **H200**: `sm_90` (Hopper)
- **B300**: `sm_100` / `sm_103` (Blackwell)
- Build NCCL with all targets for a universal container, or build separate images per cluster.

---

## 5. Slurm Job Submission (PyXis/Enroot)

### 5.1 Enroot Container Import

```bash
# Pull and convert container to Enroot squashfs
enroot import docker://your-ecr-registry/qwen3-8b-training:latest
# Or from a local .sqsh file on FSx

```

### 5.2 Slurm SBATCH Script

```bash
#!/bin/bash
#SBATCH --job-name=qwen3-8b-pretrain
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --output=/fsx/paragao/qwen3-8b/logs/qwen3-8b-%j.out
#SBATCH --error=/fsx/paragao/qwen3-8b/logs/qwen3-8b-%j.err
#SBATCH --time=30-00:00:00
#SBATCH --partition=gpu

# ============ Container & Mount Config ============
CONTAINER_IMAGE="/fsx/paragao/qwen3-8b/containers/qwen3-8b-training.sqsh"
MOUNT="/fsx:/fsx"

# ============ NCCL / EFA Tuning ============
# The AWS OFI NCCL tuner plugin auto-configures optimal NCCL parameters.
# The latest EFA installer ensures NCCL uses EFA by default.
export NCCL_DEBUG=INFO
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo,veth

# ============ Training Config ============
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))

# ============ Launch ============
srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${MOUNT}" \
     --no-container-mount-home \
     torchrun \
     --nnodes=${SLURM_NNODES} \
     --nproc-per-node=8 \
     --rdzv-id=${SLURM_JOB_ID} \
     --rdzv-backend=c10d \
     --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
     /fsx/paragao/qwen3-8b/code/train_qwen3_8b.py \
     --model-name Qwen/Qwen3-8B \
     --dataset allenai/c4 \
     --dataset-subset en \
     --seq-length 4096 \
     --micro-batch-size 4 \
     --global-batch-size 1024 \
     --grad-accum-steps 4 \
     --lr 3e-4 \
     --min-lr 3e-5 \
     --lr-scheduler cosine \
     --warmup-steps 2000 \
     --max-steps 238418 \
     --weight-decay 0.1 \
     --grad-clip 1.0 \
     --bf16 \
     --zero-stage 1 \
     --flash-attn \
     --checkpoint-dir /fsx/paragao/qwen3-8b/checkpoints \
     --checkpoint-interval 1000 \
     --log-interval 10 \
     --wandb-project qwen3-8b-pretrain

```

---

## 6. FSx for Lustre Configuration

### 6.1 Sizing Recommendations

| Aspect | Recommendation |
| --- | --- |
| **Capacity** | ≥ 20 TB (dataset + checkpoints + logs) |
| **Throughput** | PERSISTENT_2, 1000 MB/s/TiB minimum |
| **Deployment Type** | PERSISTENT_2 (for long-running training) |
| **Stripe Count** | Maximize (OST count × stripe size) for parallel I/O |
| **Data Compression** | LZ4 (reduce storage, minimal CPU overhead) |

### 6.2 Filesystem Layout

```
/fsx/paragao/qwen3-8b/
├── datasets/
│   └── c4/                    # Pre-tokenized C4 dataset (parquet/arrow)
├── checkpoints/               # Model checkpoints (every 1000 steps)
├── code/
│   └── train_qwen3_8b.py     # Training script
├── containers/
│   └── qwen3-8b-training.sqsh  # Enroot container image
├── logs/
│   ├── slurm/                 # Slurm stdout/stderr
│   └── tensorboard/           # TensorBoard logs
└── tokenizer/
    └── qwen3/                 # Tokenizer files

```

### 6.3 Data Loading Optimization

- **Pre-tokenize** the C4 dataset offline and store as memory-mapped files (`.bin` + `.idx`)
- Use **multiple dataloader workers** (8–12 per GPU) to saturate Lustre read bandwidth
- Set Lustre **stripe count = number of OSTs** for dataset files to parallelize reads
- Use `lfs setstripe -c -1` on the dataset directory for maximum parallelism

```bash
# Set striping for dataset directory (maximum parallelism)
lfs setstripe -c -1 -S 1M /fsx/paragao/qwen3-8b/datasets/c4/

# Set striping for checkpoints (moderate — large sequential writes)
lfs setstripe -c 4 -S 4M /fsx/paragao/qwen3-8b/checkpoints/

```

---

## 7. Key Considerations & Recommendations

### 7.1 EFA Networking

| Feature | p5en (H200) | p6-b300 (B300) |
| --- | --- | --- |
| EFA NICs | 32 | 16 |
| Per-NIC Bandwidth | 100 Gbps | 400 Gbps |
| Aggregate Node BW | 3200 Gbps | 6400 Gbps |
| Bisection BW (EFA) | 400 GB/s | 800 GB/s |
| NVLink (intra-node) | 900 GB/s | 1800 GB/s |

### 7.2 Cluster Health & Pre-Flight Checks

Before starting a multi-day training run:

1. **NCCL All-Reduce Test**: Run the `nccl-tests` benchmark (`all_reduce_perf`) across all 8 nodes to verify EFA connectivity and expected bandwidth
2. **GPU Health**: Run `dcgmi diag -r 3` on each node to detect failing GPUs
3. **Lustre Throughput**: Run `ior` or `fio` to validate expected storage bandwidth
4. **Container Verification**: Run a single-node 100-step smoke test before launching full training

### 7.3 Fault Tolerance & Checkpointing

| Strategy | Recommendation |
| --- | --- |
| Checkpoint frequency | Every 1000 steps (~30 min on B300, ~90 min on H200) |
| Async checkpointing | Use `torch.distributed.checkpoint` async save |
| Slurm requeue | `#SBATCH --requeue` + training script auto-resume from last checkpoint |
| Health monitoring | DCGM + custom Slurm prolog/epilog scripts |
| Expected GPU failures | ~0.03/day for 64 GPUs (MTBF ~50,000 hrs) |

### 7.4 Hyperparameters

| Hyperparameter | Value | Notes |
| --- | --- | --- |
| Learning Rate | 3e-4 | Standard for 8B models |
| Min LR | 3e-5 | 10× decay |
| Warmup Steps | 2000 | ~0.8% of total |
| Weight Decay | 0.1 | Standard |
| Gradient Clipping | 1.0 | Prevent divergence |
| Adam β1, β2 | 0.9, 0.95 | Qwen3 defaults |
| Adam ε | 1e-8 | Standard |
| Scheduler | Cosine | Smooth decay |

### 7.5 Monitoring & Observability

- **Weights & Biases** (or TensorBoard): Loss curves, gradient norms, learning rate
- **DCGM Exporter + Prometheus + Grafana**: GPU utilization, memory, temperature, ECC errors
- **Slurm sacct**: Job accounting, node utilization
- **Custom alerts**: Loss spike detection, GPU utilization drops below threshold

### 7.6 Dataset Preparation (C4)

```python
# Pre-tokenize C4 for efficient training
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
dataset = load_dataset("allenai/c4", "en", streaming=True)

# Tokenize and pack sequences to 4096 tokens
# Save as binary files for memory-mapped loading

```

**Storage estimate**: ~1T tokens × ~1.5 bytes/token (with packing overhead) ≈ **1.5 TB** tokenized dataset.

---

## 8. Performance Comparison Summary

| Metric | H200 Cluster | B300 Cluster | B300 Advantage |
| --- | --- | --- | --- |
| Peak BF16 TFLOPS/GPU | 989 | 3,375 | 3.4× |
| GPU Memory | 141 GB | 268 GB | 1.9× |
| Training Time (est.) | 22.5 days | 6.6 days | 3.4× faster |
| Tokens/sec (total) | 514K | 1.76M | 3.4× |
| Tokens/sec/GPU | 8,041 | 27,439 | 3.4× |
| Max Micro-Batch | 16 | 16+ (memory-bound later) | More headroom |
| NVLink BW | 900 GB/s | 1,800 GB/s | 2× intra-node |

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| GPU failure mid-training | Training interruption | Checkpointing every 1000 steps + Slurm auto-requeue |
| EFA link flap | Collective hangs | NCCL timeout tuning (`NCCL_TIMEOUT=1800`), Slurm health checks |
| Lustre I/O bottleneck | Dataloader stalls | Pre-tokenized binary data, aggressive striping, 12 workers/GPU |
| Loss divergence | Wasted compute | Gradient clipping, LR warmup, monitor loss every 10 steps |
| B300 driver/CUDA 13 immaturity | Runtime crashes | Pin CUDA/driver versions, test thoroughly before long runs |
| Container image too large | Slow startup | Use squashfs compression, pre-stage on Lustre, shared cache |

---

## 10. Next Steps

1. **Build & push container image** to ECR (one per GPU arch, or universal with multi-gencode NCCL)
2. **Pre-tokenize C4 dataset** and stage on FSx for Lustre
3. **Run NCCL all-reduce benchmark** on both clusters to validate networking
4. **Run single-node smoke test** (100 steps) to validate memory usage and throughput
5. **Run 8-node short run** (1000 steps) to validate multi-node scaling efficiency
6. **Launch full training** with monitoring and alerting in place
7. **Compare actual MFU** between H200 and B300 after 1000 steps to calibrate estimates

