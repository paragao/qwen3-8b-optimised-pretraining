# Lessons Learned

Hard-won knowledge from optimizing Qwen3-8B pre-training on H200 and B300 clusters. Each section represents a problem that cost hours to debug.

---

## EFA Silent Fallback to TCP

**Symptom:** Multi-node training runs but at single-node throughput. NCCL reports no errors.

**Root cause:** NCCL silently falls back to TCP sockets when the OFI plugin isn't loaded.

**Fix — all three are required:**

```bash
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-tuner-aws-ofi.so
export FI_PROVIDER=efa
```

**Verification:** Look for `NCCL INFO NET/OFI` in logs (not `NET/Socket`).

The OFI NCCL plugin directory may use either `/opt/amazon/ofi-nccl/lib/` or `/opt/amazon/aws-ofi-nccl/lib/` depending on the EFA installer version. Check which exists.

Symlinks required in some versions:
- `libnccl-net-aws-ofi.so` → `libnccl-net-ofi.so`
- `libnccl-tuner-aws-ofi.so` → `libnccl-tuner-ofi.so`

---

## Enroot Import Workflow

**Never use `mksquashfs` directly.** It produces images that PyXis can't launch.

**Correct workflow:**

```bash
# 1. Build with Docker
sudo docker build -t my-image:latest .

# 2. Import with enroot (requires sudo for Docker socket)
sudo TMPDIR=/fsx/paragao/qwen3-8b/tmp \
     ENROOT_TEMP_PATH=/fsx/paragao/qwen3-8b/tmp \
     enroot import --output /fsx/path/to/image.sqsh dockerd://my-image:latest

# 3. Fix permissions
sudo chown $USER:$USER /fsx/path/to/image.sqsh
```

**Three requirements:**
1. `sudo` — Docker socket is `root:docker`, user not in docker group
2. `TMPDIR` on FSx — NeMo containers are 30+ GB, `/tmp` won't fit
3. `docker buildx use default` — image must be in main containerd store

---

## Slurm/PyXis Gotchas

### Slurm NOT in PATH
On p5en clusters, Slurm binaries live at `/opt/slurm/bin/`. Always use full paths:
```bash
/opt/slurm/bin/sbatch script.sh
/opt/slurm/bin/squeue -u paragao
/opt/slurm/bin/scontrol show hostname $SLURM_NODELIST
```

### Shell vars don't pass into containers
PyXis containers don't inherit the calling shell's environment. Pass explicitly:
```bash
srun --container-env=VAR1,VAR2,VAR3 ...
```

### Resolve MASTER_ADDR before srun
`scontrol` is not available inside PyXis containers. Compute the head node in the batch script, before the `srun` call:
```bash
export MASTER_ADDR=$(/opt/slurm/bin/scontrol show hostname $SLURM_NODELIST | head -n1)
```

### ntasks-per-node for torchrun
When using `torchrun` (which spawns GPU workers itself), set `--ntasks-per-node=1`. If using raw `python` with NCCL init, use `--ntasks-per-node=8`.

### Single-node: disable EFA
For intra-node-only jobs, `FI_PROVIDER=efa` causes NCCL failures. Remove it or set `FI_PROVIDER=shm` for single-node debugging.

---

## torch.compile — NEVER USE

Set `TORCH_COMPILE_DISABLE=1` in all environments. It fails in every configuration tested:

| Stack | Failure Mode |
|-------|-------------|
| DeepSpeed + HF | Compilation error |
| HF Qwen3 model (multi-node) | Silent hang/crash |
| NeMo 25.07 container | `ldconfig` error during compilation |
| NeMo 26.02 container | Kernel compatibility issues |

The performance gain (if it worked) would be minimal since Transformer Engine already provides fused kernels.

---

## Distributed Optimizer Trap at DP=1

`--use-distributed-optimizer` with `DP=1` (single GPU or TP-only parallelism) causes a crash. The sharding logic divides by DP world size and expects DP≥2.

**Rule:** Only enable distributed optimizer when DP≥2. For single-GPU debugging, remove the flag.

---

## Megatron-Bridge API Gotchas (NeMo 26.02)

### cfg.logger.log_interval
The logging interval is on the `logger` sub-config, not `train`:
```python
cfg.logger.log_interval = 5  # correct
 cfg.train.log_interval = 5   # wrong — silently ignored
```

### Recompute settings
To disable gradient checkpointing, set `cfg.train.recompute_granularity = None`. Setting it to `""` or `False` may error.

### Checkpoint directory
```python
cfg.train.dir = "/path/to/checkpoints"  # not cfg.train.save or cfg.train.checkpoint_dir
```

### Qwen3 bridge recipe
`qwen3_8b_pretrain_config()` provides correct model dimensions. Don't manually override hidden_size/num_layers unless you're experimenting with a different size.

---

## Memory Budget: H200 vs B300

Understanding why configs differ between clusters:

```
H200 (141 GB available):
  Model (BF16):           16 GB
  Gradients (BF16):       16 GB
  Optimizer (sharded/16):  3 GB
  Activations (recompute): ~100 GB  ← with full recompute, MBS=2
  Overhead:                 3 GB
  Total:                 ~138 GB ✓

B300 (288 GB available):
  Model (BF16):           16 GB
  Gradients (BF16):       16 GB
  Optimizer (sharded/16):  3 GB
  Activations (no recomp): ~135 GB  ← NO recompute needed, MBS=4
  Overhead:                 3 GB
  Total:                 ~173 GB ✓ (115 GB headroom)
```

B300's extra memory means no recompute overhead → ~20% fewer FLOPs per step → directly translates to 1.96× throughput combined with higher peak FLOPS.
