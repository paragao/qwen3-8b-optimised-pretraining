# Qwen3-8B Pre-Training: H200 vs B300 Performance Comparison

## Overview

Pre-training **Qwen3-8B** (8.2B dense parameters, 36 layers, GQA 32/8) on 1T tokens (C4), comparing two GPU generations across identical 2-node / 16-GPU topologies with EFA GDRDMA interconnect.

---

## Best Configuration Per Cluster

| Parameter | H200 (p5en.48xlarge) | B300 (p6-b300.48xlarge) |
|-----------|---------------------|------------------------|
| GPUs | 16× H200 (141 GB HBM3) | 16× B300 (288 GB HBM3e) |
| Parallelism | TP=1, PP=1, DP=16 | TP=1, PP=1, DP=16 |
| Micro-batch size | 2 | 4 |
| Global batch size | 128 (grad_accum=4) | 128 (grad_accum=2) |
| Sequence length | 4096 | 4096 |
| Precision | BF16 | BF16 |
| Gradient checkpointing | Full recompute | None |
| Distributed optimizer | Yes (sharded Adam) | Yes (sharded Adam) |
| Overlap grad reduce | Yes | Yes |
| Framework | Megatron-Core (NeMo 25.07) | Megatron-Bridge (NeMo 26.02) |
| Container | nemo-efa-25.07.sqsh | nemo-efa-26.02.sqsh |

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

---

## Key Findings

1. **Both clusters are compute-saturated with perfect communication overlap.** AllReduce and AllGather are fully hidden behind compute — verified by DP=1 single-GPU benchmarks showing lower TFLOP/s (803) than DP=16 (976) due to reduced batch arithmetic intensity. Communication cost is effectively zero.

2. **Best software stack per GPU generation.** NeMo 25.07 (Megatron-Core v0.13.1) delivers optimal kernel performance on H200; NeMo 26.02 (Megatron-Bridge v2.9.0) delivers optimal performance on B300. Each stack maximizes hardware utilization for its target architecture.

3. **Pure data parallelism is optimal** when the model fits in single-GPU memory. Distributed optimizer + overlapped grad reduce eliminate the memory penalty while enabling larger effective batch sizes.

4. **Gradient checkpointing** is the key differentiator: required on H200 (141 GB) for MBS≥2, unnecessary on B300 (288 GB) for MBS≤4 — saving ~20% recompute overhead and enabling higher per-step throughput.
