# H200 Cluster — Qwen3-8B Pre-Training

## Cluster Specification

| | |
|---|---|
| Instance | p5en.48xlarge |
| Nodes | 2 |
| GPUs | 16× NVIDIA H200 (141 GB HBM3) |
| Interconnect | EFA GDRDMA (3200 Gbps) |
| Framework | NeMo 25.07, Megatron-Core v0.13.1, Transformer Engine 2.5 |
| Container | `nemo-efa-25.07.sqsh` (31 GB) |

---

## Best Configuration

| Parameter | Value |
|-----------|-------|
| Tensor Parallel | 1 |
| Pipeline Parallel | 1 |
| Data Parallel | 16 |
| Micro-batch size | 2 |
| Global batch size | 128 (grad_accum=4) |
| Sequence length | 4096 |
| Precision | BF16 |
| Gradient checkpointing | Full recompute (mandatory — 141 GB limit) |
| Distributed optimizer | Yes (shards Adam across DP ranks) |
| Overlap grad reduce | Yes |
| Overlap param gather | Yes |

---

## Result

| Metric | Value |
|--------|-------|
| TFLOP/s per GPU | 497 |
| Throughput | 162K tok/s |
| Time to 1T tokens | ~71 days |
| Step time | 3.23s |
| Peak memory/GPU | ~138 GB / 141 GB |
| MFU | 0.50 |

---

## Reproduction Steps

### Prerequisites

- Slurm cluster with PyXis (v3.4.1+) and Enroot (v3.4.1+)
- EFA-enabled p5en instances
- FSx for Lustre at `/fsx/`
- Docker installed
- **Important:** Slurm binaries are at `/opt/slurm/bin/` (NOT in PATH on these clusters)

### 1. Build the Container

```bash
cd /fsx/paragao/qwen3-8b/containers/
# Copy h200/Dockerfile to the build context
sudo docker build -t qwen3-8b-h200:latest -f Dockerfile .

# Convert to Enroot squashfs (MUST use enroot import, never mksquashfs)
sudo TMPDIR=/fsx/paragao/qwen3-8b/tmp ENROOT_TEMP_PATH=/fsx/paragao/qwen3-8b/tmp \
  enroot import --output /fsx/paragao/qwen3-8b/containers/nemo-efa-25.07.sqsh \
  dockerd://qwen3-8b-h200:latest

sudo chown paragao:paragao /fsx/paragao/qwen3-8b/containers/nemo-efa-25.07.sqsh
```

### 2. Setup FSx Paths

```bash
mkdir -p /fsx/paragao/qwen3-8b/{logs,checkpoints,code,containers,tmp}
# Copy training script
cp h200/scripts/train.py /fsx/paragao/qwen3-8b/code/train_megatron.py
```

### 3. Submit Job

```bash
/opt/slurm/bin/sbatch h200/scripts/run.sh
/opt/slurm/bin/squeue -u paragao
tail -f /fsx/paragao/qwen3-8b/logs/<JOB_ID>.out
```

---

## Gotchas

| Issue | Fix |
|-------|-----|
| Slurm NOT in PATH | Always use full path: `/opt/slurm/bin/sbatch`, `/opt/slurm/bin/scontrol` |
| Env vars don't pass into PyXis containers | Use `--container-env=VAR1,VAR2` in srun |
| `scontrol` unavailable inside container | Resolve `MASTER_ADDR` before `srun` |
| `torch.compile` crashes | Set `TORCH_COMPILE_DISABLE=1` (always) |
| EFA silent fallback to TCP | Ensure `LD_LIBRARY_PATH`, `NCCL_TUNER_PLUGIN`, `FI_PROVIDER=efa` are all set |
| enroot import needs TMPDIR | `/tmp` is too small; point at `/fsx/` |
| Distributed optimizer at DP=1 | Crashes — only use with DP≥2 |
