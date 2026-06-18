# Data Loading for Qwen3-8B Pre-Training: Streaming vs Memory-Mapped Shards

## Problem Statement

Training an 8.2B parameter model on 1 trillion tokens requires moving data to GPUs as fast as the model can consume it. The data pipeline must not become the bottleneck — any idle GPU time waiting for data is wasted compute ($$$).

We tested two approaches and measured a **+35% throughput improvement** switching from streaming to mmap (validated June 12, 2026 on 8× H200 GPUs).

---

## Approach 1: HuggingFace Streaming (Original)

### How It Works

```
Internet → HF Hub → Python Iterator → Tokenizer (CPU) → Pad/Truncate → Tensor → GPU
```

Each training step:
1. Pull raw text documents from HuggingFace servers over the network
2. Tokenize each document using the Qwen3-8B tokenizer (CPU-bound)
3. Truncate to `seq_length=4096` or pad shorter docs
4. Collate into a batch tensor
5. Transfer to GPU

### Code (Original DataLoader)

```python
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

class StreamDS(IterableDataset):
    """Wrap HF streaming dataset as PyTorch IterableDataset."""
    def __init__(self, ds):
        self.ds = ds
    def __iter__(self):
        return iter(self.ds)

# Load streaming dataset (no download, iterates on-demand)
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def collate_fn(examples):
    """Tokenize + pad/truncate each document per step."""
    texts = [ex["text"] for ex in examples]
    enc = tokenizer(
        texts,
        truncation=True,       # Cut docs longer than 4096
        max_length=4096,
        padding="max_length",  # Pad shorter docs to 4096
        return_tensors="pt",
    )
    enc["labels"] = enc["input_ids"].clone()
    return enc

dataloader = DataLoader(
    StreamDS(dataset),
    batch_size=4,              # micro-batch size
    collate_fn=collate_fn,     # tokenize on-the-fly
    num_workers=4,
    pin_memory=True,
)
```

### Problems

| Issue | Impact |
| --- | --- |
| **Network I/O** | Each step requires HTTP requests to HF servers |
| **CPU tokenization** | Tokenizer runs per-document on every step (single-threaded in collate) |
| **Padding waste** | Short documents padded to 4096 → ~30-50% of tokens are padding (wasted FLOPs) |
| **No shuffling** | Streaming datasets can't shuffle — sequential order only |
| **No resumption** | If training restarts, you re-iterate from the beginning |
| **Throughput** | ~37K tok/s on 8× H200 GPUs (GPU often idle waiting for data) |

---

## The Bridge: Pre-Tokenization Pipeline

Before we can use the fast dataloader, we need to transform the raw dataset once into an optimized binary format.

### Pipeline Stages

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────────┐
│  1. DATA SOURCE │     │  2. TOKENIZE & PACK  │     │  3. BINARY SHARDS       │
│                 │     │                      │     │     (on Lustre)          │
│  HuggingFace    │────▶│  Qwen3-8B Tokenizer  │────▶│  c4_w00_s0000.bin       │
│  allenai/c4     │     │  vocab=151,936       │     │  c4_w00_s0001.bin       │
│  (streaming)    │     │                      │     │  c4_w01_s0000.bin       │
│                 │     │  Concatenate all docs │     │  ...                    │
│                 │     │  Pack → 4096 chunks   │     │  (200 MB each, uint32)  │
└─────────────────┘     └──────────────────────┘     └─────────────────────────┘
```

**Key insight**: All documents are concatenated into one continuous stream of tokens, then sliced into exact 4096-token chunks. No padding, no truncation, no document boundaries — every single token is used for training.

### Pre-Tokenization Code (Parallel, 96 cores)

```python
#!/usr/bin/env python3
"""Pre-tokenize C4 for Qwen3-8B — Parallel (96 cores)"""
import os, struct, time, numpy as np
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer
from datasets import load_dataset

OUTPUT_DIR = "/fsx/paragao/qwen3-8b/datasets/c4"
SEQ_LENGTH = 4096
MODEL = "Qwen/Qwen3-8B"
NUM_WORKERS = 96  # Use all physical cores on p5en
MAX_TOKENS = 10_000_000_000  # 10B tokens (scale to 1T for full run)
TOKENS_PER_SHARD = 50_000_000  # 50M tokens per shard file (~200 MB)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def tokenize_worker(worker_id):
    """Each worker processes every Nth document (round-robin)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    buffer = []
    total_tokens = 0
    shard_idx = 0
    max_tokens_per_worker = MAX_TOKENS // NUM_WORKERS
    
    bin_path = os.path.join(OUTPUT_DIR, f"c4_w{worker_id:02d}_s{shard_idx:04d}.bin")
    bin_file = open(bin_path, "wb")
    shard_tokens = 0
    
    for i, example in enumerate(dataset):
        # Round-robin: each worker takes every Nth document
        if i % NUM_WORKERS != worker_id:
            continue
        if total_tokens >= max_tokens_per_worker:
            break
        
        tokens = tokenizer.encode(example["text"], add_special_tokens=False)
        buffer.extend(tokens)
        
        # Pack complete sequences
        while len(buffer) >= SEQ_LENGTH:
            seq = buffer[:SEQ_LENGTH]
            buffer = buffer[SEQ_LENGTH:]
            
            # Write as uint32 (4 bytes/token, supports vocab > 65535)
            arr = np.array(seq, dtype=np.uint32)
            bin_file.write(arr.tobytes())
            
            total_tokens += SEQ_LENGTH
            shard_tokens += SEQ_LENGTH
            
            # Roll to next shard at 50M tokens
            if shard_tokens >= TOKENS_PER_SHARD:
                bin_file.close()
                shard_idx += 1
                bin_path = os.path.join(OUTPUT_DIR, f"c4_w{worker_id:02d}_s{shard_idx:04d}.bin")
                bin_file = open(bin_path, "wb")
                shard_tokens = 0
    
    bin_file.close()
    return total_tokens

if __name__ == "__main__":
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(tokenize_worker, range(NUM_WORKERS))
    print(f"Total: {sum(results)/1e9:.2f}B tokens across {NUM_WORKERS} workers")
```

### File Format

Each `.bin` shard is a flat array of `uint32` values:

```
┌──────────────────────────────────────────────────────────────────┐
│ Sequence 0 (4096 × uint32 = 16 KB)                              │
├──────────────────────────────────────────────────────────────────┤
│ Sequence 1 (4096 × uint32 = 16 KB)                              │
├──────────────────────────────────────────────────────────────────┤
│ Sequence 2 (4096 × uint32 = 16 KB)                              │
├──────────────────────────────────────────────────────────────────┤
│ ...                                                              │
├──────────────────────────────────────────────────────────────────┤
│ Sequence 12,206 (last in shard, 50M tokens / 4096 = 12,207 seqs)│
└──────────────────────────────────────────────────────────────────┘
```

**One shard**: 50M tokens × 4 bytes = **200 MB**  
**Full 1T tokens**: ~20,000 shards, ~4 TB on disk

### Lustre Optimization

```bash
# Maximum striping for parallel reads across all OSTs
lfs setstripe -c -1 -S 1M /fsx/paragao/qwen3-8b/datasets/c4/
```

With `-c -1` (stripe across ALL OSTs), a single 200 MB shard is distributed across every storage server. When 8+ DataLoader workers read different sequences simultaneously, they hit different OSTs — achieving aggregate I/O bandwidth close to the filesystem maximum.

---

## Approach 2: Memory-Mapped Binary Shards (Optimized)

### How It Works

```
Lustre (mmap) → OS Page Cache → numpy view → torch.Tensor → GPU
```

Each training step:
1. Index into the memory-mapped shard file (just a pointer arithmetic operation)
2. OS kernel serves the page from RAM cache (or triggers a Lustre read on first access)
3. Cast numpy array to PyTorch tensor
4. Transfer to GPU

**No tokenization. No padding. No network I/O. No Python file handling.**

### Code (Optimized DataLoader)

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler, DataLoader
import glob

class MmapShardDataset(Dataset):
    """Memory-mapped dataset from pre-tokenized binary shards.
    
    Each shard is a flat array of uint32 token IDs.
    Every seq_length consecutive tokens form one training sequence.
    Random access via numpy memmap — zero-copy from OS page cache.
    """
    
    def __init__(self, data_dir, seq_length=4096):
        self.seq_length = seq_length
        
        # Discover all shard files
        self.shard_paths = sorted(glob.glob(f"{data_dir}/c4_w*_s*.bin"))
        assert len(self.shard_paths) > 0, f"No shards found in {data_dir}"
        
        # Memory-map each shard (lazy — no data loaded until accessed)
        self.shards = []
        self.cumulative_seqs = [0]
        
        for path in self.shard_paths:
            mmap = np.memmap(path, dtype=np.uint32, mode='r')
            num_seqs = len(mmap) // seq_length
            self.shards.append(mmap)
            self.cumulative_seqs.append(self.cumulative_seqs[-1] + num_seqs)
        
        self.total_seqs = self.cumulative_seqs[-1]
        print(f"Loaded {len(self.shards)} shards, {self.total_seqs:,} sequences "
              f"({self.total_seqs * seq_length / 1e9:.1f}B tokens)")
    
    def __len__(self):
        return self.total_seqs
    
    def __getitem__(self, idx):
        # Binary search: which shard contains this sequence?
        shard_idx = np.searchsorted(self.cumulative_seqs[1:], idx, side='right')
        local_idx = idx - self.cumulative_seqs[shard_idx]
        
        # Direct memory access — zero copy from page cache
        start = local_idx * self.seq_length
        tokens = self.shards[shard_idx][start : start + self.seq_length]
        
        # Convert to tensor (int64 for PyTorch embedding layer)
        input_ids = torch.from_numpy(tokens.astype(np.int64))
        return {"input_ids": input_ids, "labels": input_ids.clone()}


# Integration in training script:
DATA_DIR = "/fsx/paragao/qwen3-8b/datasets/c4"

dataset = MmapShardDataset(DATA_DIR, seq_length=4096)

# DistributedSampler enables proper shuffling across GPUs
sampler = DistributedSampler(dataset, shuffle=True)

dataloader = DataLoader(
    dataset,
    batch_size=4,           # micro-batch size
    sampler=sampler,        # shuffled, distributed
    num_workers=8,          # parallel workers hitting page cache
    pin_memory=True,        # fast GPU transfer
    drop_last=True,
)
# No collate_fn needed — data is already packed as tensors!
```

### Why This Is Fast

| Operation | Streaming | mmap Shards |
| --- | --- | --- |
| `__getitem__` cost | ~5-50 ms (network + tokenize) | ~1-5 µs (pointer math) |
| I/O mechanism | HTTP download per document | OS page fault → Lustre DMA |
| CPU work per step | Tokenizer + pad + collate | numpy slice + dtype cast |
| Token utilization | ~70% (padding waste) | **100%** (packed sequences) |
| Shuffling | No (streaming can't shuffle) | Yes (DistributedSampler) |
| Multi-epoch | No (re-download everything) | Yes (instant re-access) |
| Resume training | No (restart from beginning) | Yes (sampler tracks position) |
| Workers parallelism | Blocked on single network stream | 8 workers → 8 OSTs simultaneously |

### Memory Access Pattern

```
Training Step N:
  Worker 0 → mmap[seq_42891]  → page fault → Lustre OST-3 → RAM cache → GPU 0
  Worker 1 → mmap[seq_108234] → page fault → Lustre OST-7 → RAM cache → GPU 1
  Worker 2 → mmap[seq_73019]  → page cache hit (already in RAM) → GPU 2
  ...
  
Training Step N+1:
  Worker 0 → mmap[seq_42892]  → page cache hit → GPU 0  (sequential = prefetched!)
  ...
```

The OS kernel's readahead prefetcher detects sequential access patterns within each shard and pre-fetches pages before they're needed. Combined with Lustre's client-side caching, most reads after the first epoch come directly from RAM.

---

## Performance Comparison (Validated)

| Metric | Streaming (Original) | mmap Shards (Optimized) | Improvement |
| --- | --- | --- | --- |
| Throughput (8 GPU) | 37K tok/s | **50.1K tok/s** | +35% |
| Throughput (16 GPU) | 72K tok/s | **105K tok/s** | **+46%** |
| GPU idle time (data stalls) | 30-40% | <5% | — |
| Token waste (padding) | ~30% | 0% | — |
| Effective MFU | ~0.21 | **~0.33** | +57% |
| Time to 1T tokens (16 GPU) | ~160 days | **~110 days** | **-50 days** |

---

## Summary

| | Streaming | Pre-tokenized mmap |
| --- | --- | --- |
| **Setup effort** | None (just run) | One-time preprocessing (20-30 min for 10B tokens) |
| **Throughput** | Limited by network + CPU | Limited by GPU compute (desired) |
| **Scalability** | Degrades with more GPUs (shared network) | Scales linearly (parallel OST reads) |
| **Production readiness** | Debugging/prototyping only | Standard for large-scale training |
| **Used by** | Quick experiments | Megatron-LM, NeMo, GPT-NeoX, OLMo, Llama |

**Bottom line**: Streaming is fine for initial debugging. For any serious training run, pre-tokenize your data. The one-time cost (~30 minutes on 96 cores for 10B tokens) pays for itself within the first hour of training.
