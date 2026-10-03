# `deeper4k` — `deeper` geometry at seq 4096 — prediction recorded BEFORE the run

**Status: PREDICTION ONLY. Nothing here is measured.**
Reproduce with `python3 g5/predict.py --profile deeper4k`.

Memory and FLOP predictions are held to the operator's accepted **±5%**.

## What changes, and what does not

| | `deeper` (measured) | `deeper4k` |
|---|---|---|
| num_layers | 12 | 12 |
| hidden_size | 1024 | 1024 |
| ffn_hidden_size | 3072 | 3072 |
| heads / kv groups | 8 / 2 | 8 / 2 |
| **seq_length** | **2048** | **4096** |
| params | 455,868,416 | **455,868,416 — unchanged** |
| static @18 B/param | 7.642 GiB | **7.642 GiB — unchanged** |
| activation units | 25,165,824 | **50,331,648** |
| tokens/step | 16,384 | **32,768** |

`seq_length` does not enter the parameter count, so this is still a 455.9 M
model and the static state is identical. **Every change is in activations**,
which is what makes it a clean probe.

Two further properties:

- **Single-variable change from a measured profile.** Only `seq_length` moves,
  so the per-layer term scales by exactly 2x and nothing else is confounded.
- **It is the first profile matching Qwen3-8B's own `seq_length` of 4096.** The
  other four are at 1024 or 2048.

## Why this run is the one that matters now

The three-term model fitted over the four completed runs is:

```
residual = 0.5366 GiB + 149,765 B/token x seq + 51.02 B/unit x units
```

It reproduces all four measurements to within 0.93% of peak. But its per-token
coefficient is the **weakly determined** constant, and the sensitivity report
in `--self-check` quantifies how weakly:

| constant | could be wrong by | why |
|---|---|---|
| `k` (per unit) | **±10%** | and independently *measured* at 46.44 by the `wider`/`1b` pair |
| `c1` (per token) | **±53%** | largest measured `seq_length` is 2048, so the term is at most 0.29 GiB |

Mutation-tested: a **+20% error in `c1` passes every assertion** in the
self-check. The existing data simply does not constrain it. At seq 4096 the
term doubles to 0.57 GiB, and the two candidate mechanisms separate by 1.75
GiB — far outside ±5%.

## The two hypotheses

| | residual | predicted peak | ±5% band |
|---|---|---|---|
| **T1** three-term model (logits ~24.6% resident) | 3.50 GiB | **11.14 GiB** (49.5%) | **[10.58, 11.70]** |
| **T2** full FP32 vocab logits resident | 5.25 GiB | **12.89 GiB** (57.3%) | **[12.24, 13.53]** |

A full FP32 logits tensor at seq 4096 is `4096 x 151,936 x 4 B = 2.32 GiB` on
its own. The fitted coefficient says only about a quarter of that is resident
at peak, which would be consistent with a chunked vocab projection or
cross-entropy. **T1 vs T2 is a factor-4 difference in that term and the bands
are disjoint**, so this run decides it.

## Does it fit?

**Yes, comfortably, under either hypothesis.**

| | peak | headroom |
|---|---|---|
| T1 | 11.14 GiB (49.5% of card) | **+11.35 GiB** |
| T2 | 12.89 GiB (57.3%) | +9.60 GiB |

This is a far easier fit than the `1b` profile's 19.08 GiB, because the
parameter count is less than half and static state dominates peak. Doubling
`seq_length` is cheap *here* precisely because this is a small model.

### For contrast: `1b` at seq 4096 would NOT fit

```
static 16.92 + residual 7.09 = 24.01 GiB = 107% of the card -> OOM
```

So seq 4096 is affordable on the 455.9 M geometry and not on the 1.0 B one.
Worth knowing before anyone combines the two changes.

## Throughput prediction

**68.93 TFLOP/step**, 2.155x `deeper` and 6.742x `smoke`, at 32,768 tokens/step.

The attention term grows as seq², so its share of total FLOPs roughly doubles:

| | `deeper` (seq 2048) | `deeper4k` (seq 4096) |
|---|---|---|
| dense | 92.3% | **85.6%** |
| attention | 7.7% | **14.4%** |

| achieved efficiency | s/step | tok/s |
|---|---|---|
| 34.3 TFLOP/s (`1b`'s) | 2.010 | 16,305 |
| 36.7 TFLOP/s (`deeper`'s) | 1.878 | **17,445** |
| 38.5 TFLOP/s | 1.790 | 18,301 |

Central estimate **17,445 tok/s** at `deeper`'s measured 36.7 TFLOP/s, ±5%
band **[16,573, 18,318]**.

On efficiency I am genuinely uncertain of the direction, and say so in advance.
Sequence length has been the strongest driver so far (+18.8% from 1024 to
2048), which argues for ≥36.7. But attention is now 14.4% of the work rather
than 7.7%, and attention is more memory-bandwidth-bound than the dense GEMMs,
which argues for a flattening or a slight fall. **I expect 36-39 TFLOP/s and
would not be surprised by 35.**

## Thresholds

| # | Condition | What it means |
|---|---|---|
| **T1** | peak in **[10.58, 11.70] GiB** | three-term model confirmed; logits not fully resident |
| **T2** | peak in **[12.24, 13.53] GiB** | full FP32 logits ARE resident; `c1` is wrong by ~4x |
| **T3** | peak outside both, no OOM | neither; the measured per-token value is the result |
| **R3** | median tok/s outside **[13,954, 18,967]** | the FLOP model or the efficiency assumption |
| **R4** | MODEL_TFLOP/s **< 30.9** | would refute "bigger GEMMs are at least as efficient" |
| **R5** | **OOM** at a predicted 50% of card | a large unmodelled allocation scales with seq |
| **R6** | log reports TOTAL **!= 455.9 M** | **run is void** — env override dropped |
| **R7** | any skipped or NaN iteration | BF16 stability at seq 4096 |

The superseded predictions, recorded so they cannot be quietly reused: the
smoke-only two-term calibration says 12.79 GiB, the two-point says 13.41 GiB,
and the flat "+14%" rule says 8.68 GiB. All three are refuted and none is under
test.

## The command

```bash
NUM_LAYERS=12 HIDDEN_SIZE=1024 FFN_HIDDEN_SIZE=3072 \
  NUM_ATTENTION_HEADS=8 NUM_QUERY_GROUPS=2 SEQ_LENGTH=4096 \
  DATA_PATH=/workspace/run/datasets/c4_qwen3 TRAIN_ITERS=50 \
  ./g5/finish-run.sh
```

`DATA_PATH` is optional — memory and throughput do not depend on the data
source, and the mock dataset gives the same figures. Including it keeps the
loss curve interpretable at no cost, and the existing instance already has the
50M-token c4 build.

One note on the dataset: it was built with `--doc-length 4096`, so at
`seq_length 4096` each indexed document maps to exactly one training sample.
That is fine — `GPTDataset` concatenates and re-splits regardless — but it does
mean the 12,247 documents yield about 12,247 samples rather than 48,939, which
is still 245x the 50 steps this run consumes.
