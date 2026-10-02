# Why Qwen3-8B does not pre-train on one GPU

This repo's H200 and B300 runs use **DP=16 with Megatron's distributed
optimizer**. That detail is not a tuning preference, it is what makes the run
possible at all. This note works out the single-GPU budget, because the answer
is counter-intuitive: full-Adam Qwen3-8B pre-training does not fit on a single
**H200** either, let alone a 24 GB A10G.

All figures below are analytic. `g5/train.py:param_count()` implements the same
arithmetic and reproduces Qwen3-8B at **8.190 B** parameters against the
README's 8.2 B, which is the check that the formula is right.

## Parameter accounting

Qwen3-8B: 36 layers, `hidden=4096`, `ffn=12288`, 32 Q heads / 8 KV heads
(GQA 4:1), `head_dim=128`, `vocab=151936`, untied LM head.

| Block | Formula | Params |
|---|---|---|
| Input embedding | `vocab x hidden` | 622.3 M |
| Output LM head (untied) | `vocab x hidden` | 622.3 M |
| Attention / layer | `h*h + 2*(h*kv_dim) + h*h`, `kv_dim = 8*128 = 1024` | 2.62 M |
| MLP / layer (SwiGLU) | `3 * h * ffn` | 150.99 M |
| Per layer | | 192.94 M |
| 36 layers | | 6.946 B |
| **Total** | | **8.190 B** |

## Static state per parameter

Megatron BF16 mixed-precision training with Adam keeps five tensors per weight:

| Tensor | dtype | Bytes/param |
|---|---|---|
| Working weights | BF16 | 2 |
| Main gradients | FP32 | 4 |
| Master weights | FP32 | 4 |
| Adam `exp_avg` (*m*) | FP32 | 4 |
| Adam `exp_avg_sq` (*v*) | FP32 | 4 |
| **Total** | | **18** |

**8.190e9 x 18 B = 147.4e9 bytes = 137.3 GiB**, before a single activation.

## The result, against real hardware

At **DP=1** nothing is shardable, so 137.3 GiB is the floor. Capacities below
are what `nvidia-smi` reports, and everything is in GiB to avoid the GB/GiB trap
(a "141 GB" H200 reports 140.4 GiB):

| GPU | Capacity | 137.3 GiB static at DP=1 |
|---|---|---|
| A10G (g5) | 22.5 GiB | No — **6.11x over** |
| A100 | 40.0 GiB | No — 3.43x over |
| A100 | 80.0 GiB | No — 1.72x over |
| H100 | 80.0 GiB | No — 1.72x over |
| H200 (p5en) | 140.4 GiB | Static fits, **3.1 GiB left** |
| B200 | 179.1 GiB | Static fits, 41.8 GiB left |
| B300 (p6-b300) | 288.0 GiB | Static fits, 150.7 GiB left |

The H200 row is the interesting one and it is a near miss, not a pass. The
static state alone consumes 97.8% of the card, leaving 3.1 GiB for activations,
gradient buffers, NCCL buffers and fragmentation. For scale, this repo's own
H200 run peaks at ~114 GB with only ~21 GiB of that resident state at DP=16, so
activations and buffers account for the large majority of its footprint at
MBS=2 / seq=4096. Three GiB does not cover that at any useful batch size.
Single-GPU full-Adam Qwen3-8B pre-training is therefore out of reach on an H200
in practice, and only comfortable from B200 upward.

This is also why the repo's H200 run reports **~114 GB peak rather than 147 GB**
of state: at DP=16 the distributed optimizer shards the FP32 master + *m* + *v*
(12 B/param) across 16 ranks, so resident state is ~15.3 GiB of replicated BF16
weights plus ~5.7 GiB of sharded optimizer, about 21 GiB total. Sharding is
load-bearing, not an optimisation.

## The non-obvious part: width is the binding constraint, not depth

The instinct on a small GPU is to cut `num_layers`. On this model that barely
helps, because the vocab projections are not in the layer stack:

```
embedding + untied LM head = 2 x 151936 x 4096 = 1.245 B params
                           x 18 B/param        = 20.87 GiB
```

That is **92.8% of a 22.5 GiB A10G consumed before layer zero exists**, leaving
about 1.6 GiB for every activation, the gradient buffer, the CUDA context and
the FP32 vocab logits — which at `seq=4096` are `4096 x 151936 x 4 B = 2.5 GB`
per micro-batch copy on their own. A hypothetical Qwen3-8B truncated to
`num_layers=1` is already out of memory. So `hidden_size` has to come down too:
it is the only lever that shrinks the vocab projections while keeping the full
151,936-entry vocabulary and the real Qwen3 tokenizer, both of which are worth
exercising.

Nor does scaling out rescue the A10G: the BF16 working weights (15.3 GiB) are
*replicated* on every rank under Megatron's distributed optimizer, so they alone
consume 68% of the card no matter how large DP grows. Fitting the real model on
24 GB cards would need the weights themselves sharded (FSDP / ZeRO-3) plus
aggressive recompute, which is a different parallelism strategy from the one
this repo benchmarks.

## What `g5/train.py` therefore does

It scales **width and depth** and preserves everything else, so the code path,
the tokenizer, the vocab, and the architectural ratios are the real ones:

| Knob | Qwen3-8B | g5 proxy (`smoke`) | Preserved? |
|---|---|---|---|
| `num_layers` | 36 | 4 | scaled |
| `hidden_size` | 4096 | 1024 | scaled |
| `ffn_hidden_size` | 12288 | 3072 | ratio 3x hidden |
| `num_attention_heads` | 32 | 8 | — |
| `num_query_groups` | 8 | 2 | GQA ratio 4:1 |
| `head_dim` | 128 | 128 | identical |
| `vocab_size` | 151936 | 151936 | identical |
| Tokenizer | Qwen3 | Qwen3 | identical |
| Precision | BF16 | BF16 | identical |
| Attention | GQA + RoPE | GQA + RoPE | identical |
| Norm / MLP | RMSNorm / SwiGLU | RMSNorm / SwiGLU | identical |
| Entrypoint | `qwen3_8b_pretrain_config` -> `pretrain` | same | identical |

Proxy totals: **359.4 M params -> 6.02 GiB static (18 B/param), 26.8% of a
22.49 GiB A10G.** The remaining headroom absorbs activations and the FP32 vocab
logits, which at `seq=1024` are `1024 x 151936 x 4 B = 622 MB` per micro-batch
copy and are the largest single activation in the model.

**Measured 2026-10-02** (50 iterations on an A10G, log
`g5/results/run-20261002-162239.log`): peak allocated **6.84 GiB**, peak
reserved **7.26 GiB**. So the analytic static figure was low by 0.82 GiB
(13.6%), and that gap is the activation term this table does not model. It is
consistent with the 622 MB logits estimate above being the dominant activation,
leaving ~218 MB for everything else plus allocator workspace — which is a point
in favour of the logits analysis, though it does not isolate the term. Treat
18 B/param as a **floor** and add an activation margin when sizing.

## What this validates, and what it does not

Validated on g5.8xlarge:

- the NeMo 26.04 / Megatron-Bridge / Megatron-Core import graph and CUDA init on
  Ampere `sm_86`
- Qwen3 model construction: GQA, RoPE, RMSNorm, SwiGLU
- the real Qwen3 tokenizer and the 151,936-entry vocab
- the dataset path, mock or real Megatron-indexed `.bin`/`.idx`
- a genuine forward -> backward -> Adam step loop in BF16, with loss moving
- peak memory measured against the A10G envelope

**Not** validated, and not validatable on this hardware:

- any throughput, TFLOP/s or MFU number — a 359 M proxy on one A10G says nothing
  about 8.2 B on 16 H200s, and single-GPU figures are not comparable to the
  DP=16 results in the top-level README
- EFA / NCCL / GDRDMA inter-node behaviour — there is no second node
- the distributed optimizer's sharding, overlap-grad-reduce and
  overlap-param-gather paths, all of which are no-ops at DP=1 and are explicitly
  disabled in `g5/train.py`
- convergence of the real model
