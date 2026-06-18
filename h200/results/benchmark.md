# H200 Benchmark Results

## Configuration

- **Cluster:** 2× p5en.48xlarge (16× H200, 141 GB HBM3)
- **Framework:** NeMo 25.07, Megatron-Core v0.13.1, Transformer Engine 2.5
- **Parallelism:** TP=1, PP=1, DP=16
- **Micro-batch size:** 2
- **Global batch size:** 128 (grad_accum=4)
- **Sequence length:** 4096
- **Precision:** BF16
- **Gradient checkpointing:** Full recompute (uniform, 1 layer)
- **Distributed optimizer:** Yes (overlapped grad reduce + param gather)

## Results (100 iterations)

| Metric | Value |
|--------|-------|
| TFLOP/s per GPU | 497 |
| Throughput | 162,000 tok/s |
| Step time (avg) | 3.23s |
| Peak memory per GPU | ~138 GB / 141 GB |
| MFU | 0.50 |
| Projected time to 1T tokens | ~71 days |

## Memory Breakdown (estimated)

| Component | Size |
|-----------|------|
| Model parameters (BF16) | ~16 GB |
| Gradients (BF16) | ~16 GB |
| Optimizer states (FP32, sharded /16) | ~3 GB |
| Activations (with full recompute) | ~100 GB |
| NCCL buffers + framework overhead | ~3 GB |
| **Total** | **~138 GB** |

## Why Gradient Checkpointing is Required

Without full recompute, activations for 36 transformer layers at MBS=2, seq=4096 would consume ~200+ GB — exceeding H200's 141 GB limit. Full recompute trades ~20% extra compute for fitting in memory.

## Optimization Path

| Config | Throughput | Notes |
|--------|-----------|-------|
| HF Accelerate streaming | 17.5K tok/s (8 GPU) | Baseline |
| + SDPA + ZeRO-1 + MBS=4 | 37K tok/s (8 GPU) | 2.1× |
| + Pre-tokenized mmap | 50.1K tok/s (8 GPU) | 2.9× |
| 2-node HF (mmap) | 105K tok/s (16 GPU) | 6× |
| Megatron-Core single-node | 70.3K tok/s (8 GPU) | 4× |
| **Megatron-Core 2-node** | **162K tok/s (16 GPU)** | **9.3×** |
