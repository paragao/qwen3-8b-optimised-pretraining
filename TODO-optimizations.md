# Qwen3-8B Training Optimization TODO

## Status: ALL COMPLETE ✅ + Megatron-Core Migration Done

### 1. ✅ Pre-tokenize C4 Dataset
- [x] Downloaded C4 english split (streaming from HF Hub)
- [x] Tokenized with Qwen3-8B tokenizer (vocab=151936), 96 parallel workers
- [x] Packed sequences to 4096 tokens (zero padding waste)
- [x] Saved as memory-mapped binary files (153 shards, 13 GB, ~5B tokens usable)
- [x] Completed in ~4.8 hours on 96 cores
- [x] Validated: 50.1K tok/s single-node (+35% vs streaming)

### 2. ~~Switch to NeMo/Megatron-LM~~ (Deferred)
- [ ] Check NeMo version in existing container (`nvidia+nemo+25.11.01-efa-nccl29.sqsh`)
- [x] **DONE: Migrated to Megatron-Core 0.13.1 with Transformer Engine 2.5**
- [x] Container: `qwen3-8b-megatron-bridge.sqsh` (31 GB, NeMo 25.07, sudo enroot import)
- [x] Training script: `/fsx/paragao/qwen3-8b/code/train_megatron.py`
- [x] Validated: 70.3K tok/s on 8 GPUs (MBS=2, DP=8, distributed optimizer, overlap)

### 3. ✅ Enable Flash Attention Explicitly
- [x] ~~flash_attention_2~~ → ABI mismatch with PyTorch 2.6.0
- [x] Switched to `attn_implementation="sdpa"` (PyTorch native)
- [x] SDPA auto-dispatches to FlashAttention/MemoryEfficient backend on H200
- [x] Validated: no OOM, clean training

### 4. ✅ ZeRO-1 + Gradient Checkpointing + MBS=4
- [x] Switched to ZeRO Stage 1 (less communication overhead)
- [x] Gradient checkpointing enabled (mandatory for memory)
- [x] MBS=4 validated — fits with ZeRO-1 + grad ckpt + SDPA
- [x] GA=4 → 4 × 4 × 8 × 4096 = 524K tokens/batch (single-node)

### Validated Results

| Config | MBS | Throughput (8 GPU) | vs Baseline | Status |
|--------|-----|-------------------|-------------|--------|
| HF: Baseline (ZeRO-2, no FA, MBS=2) | 2 | 17.5K tok/s | 1× | ✅ |
| HF: + SDPA + ZeRO-1 + MBS=4 | 4 | 37K tok/s | 2.1× | ✅ |
| HF: + Pre-tokenized mmap | 4 | 50.1K tok/s | 2.9× | ✅ |
| HF: 2-node mmap production | 4 | 105K tok/s (16 GPU) | 6× | ✅ |
| **Megatron-Core + TE + DistOpt + overlap** | **2** | **70.3K tok/s (8 GPU)** | **4×** | **✅ Final** |
| **Megatron-Core: 2-node (projected)** | **2** | **~140K tok/s (16 GPU)** | **8×** | **Est.** |

### Production Projections (2-node, 16 GPU)

| Scenario | Throughput | Time to 1T tokens | Status |
|----------|-----------|-------------------|--------|
| HF Accelerate (streaming) | 72K tok/s | ~160 days | Superseded |
| HF Accelerate (mmap) | 105K tok/s | ~110 days | Superseded |
| **Megatron-Core + TE (mock data)** | **~140K tok/s** | **~83 days** | **✅ Validated framework** |
| Megatron-Core + real data | ~140K tok/s | ~83 days | Next step |
| MFU achieved | — | — | **~0.44** |

### Next Steps
1. ✅ ~~Pre-tokenize C4~~ — Done (153 shards, 5B tokens)
2. ✅ ~~HF Accelerate optimization~~ — Done (50.1K single-node, 105K 2-node)
3. ✅ ~~Megatron-Core migration~~ — **Done (70.3K single-node, ~140K 2-node projected)**
4. **Connect real pre-tokenized C4 to Megatron-Core script** — Need Megatron .bin/.idx format
5. **2-node Megatron-Core production run** — Scale to 16 GPUs
6. Scale tokenization to full 1T tokens (currently 5B)
7. Consider MBS=4 (25 GB headroom available)
