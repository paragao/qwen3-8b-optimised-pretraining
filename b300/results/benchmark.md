# B300 Benchmark Results

## Configuration

- **Cluster:** 2× p6-b300.48xlarge (16× B300, 288 GB HBM3e)
- **Framework:** NeMo 26.02, Megatron-Bridge v2.9.0
- **Parallelism:** TP=1, PP=1, DP=16
- **Micro-batch size:** 4
- **Global batch size:** 128 (grad_accum=2)
- **Sequence length:** 4096
- **Precision:** BF16
- **Gradient checkpointing:** None (not needed with 288 GB)
- **Distributed optimizer:** Yes (overlapped grad reduce + param gather)

## Results (100 iterations)

| Metric | Value |
|--------|-------|
| TFLOP/s per GPU | 976 |
| Throughput | 318,000 tok/s |
| Step time (avg) | 1.65s |
| Peak memory per GPU | ~173 GB / 288 GB |
| MFU | 0.50 |
| Projected time to 1T tokens | ~36 days |

## Memory Breakdown (estimated)

| Component | Size |
|-----------|------|
| Model parameters (BF16) | ~16 GB |
| Gradients (BF16) | ~16 GB |
| Optimizer states (FP32, sharded /16) | ~3 GB |
| Activations (no recompute, MBS=4) | ~135 GB |
| NCCL buffers + framework overhead | ~3 GB |
| **Total** | **~173 GB** |

## Full Sweep Results

| MBS | Seq Len | Grad Ckpt | TFLOP/s/GPU | tok/s (16 GPU) | Notes |
|-----|---------|-----------|-------------|----------------|-------|
| 2 | 4096 | Yes | 612 | 199K | Conservative |
| 2 | 4096 | No | 784 | 255K | |
| 4 | 4096 | Yes | 768 | 250K | |
| **4** | **4096** | **No** | **976** | **318K** | **Best** |
| 8 | 4096 | No | 941 | 306K | Slight regression at MBS=8 |
| 4 | 8192 | No | 892 | 145K | Longer seq = fewer tok/s |
| 8 | 8192 | No | 914 | 149K | |

## TP=2 Experiment (Negative Result)

| Config | TFLOP/s/GPU | tok/s | vs Best |
|--------|-------------|-------|---------|
| TP=1, DP=16, MBS=4 | 976 | 318K | baseline |
| TP=2, DP=8, MBS=4 | 868 | 283K | -11% |

TP=2 introduces unnecessary all-reduce communication for every transformer layer. For an 8B model that fits on one GPU, pure DP is strictly better.

## Communication Analysis

Single-GPU (DP=1) benchmark: 803 TFLOP/s/GPU vs DP=16: 976 TFLOP/s/GPU.

The DP=16 result is **higher** than DP=1 because the larger effective batch (128 vs 4) increases arithmetic intensity, while AllReduce is fully overlapped with backward compute. Communication cost is effectively zero — both clusters are compute-saturated.
