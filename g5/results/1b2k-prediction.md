# `1b2k` — the 1B model at seq 2048 — prediction recorded BEFORE the run

**Status: PREDICTION ONLY. Nothing here is measured.**
Reproduce with `python3 g5/predict.py --profile 1b2k`.

## What changes

| | `1b` (measured) | `1b2k` |
|---|---|---|
| num_layers | 20 | 20 |
| hidden_size | 1536 | 1536 |
| ffn_hidden_size | 4608 | 4608 |
| heads / kv groups | 12 / 3 | 12 / 3 |
| **seq_length** | **1024** | **2048** |
| params | 1,009,385,472 | **unchanged** |
| static @18 B/param | 16.921 GiB | **unchanged** |
| activation units | 31,457,280 | **62,914,560** |
| tokens/step | 8,192 | **16,384** |

Single-variable change from the measured `1b`. Parameter count and static state
are untouched; the entire cost is activations.

## Does it fit? Predicted yes, with the thinnest margin yet

| | |
|---|---|
| predicted peak | **20.73 GiB** (92.2% of the card) |
| ±5% band | **[19.70, 21.77] GiB** |
| headroom | **1.76 GiB** |
| anchored on measured `1b` | 20.58-20.72 GiB (91.5-92.1%) |

**2048 is the ceiling for this shape.** The absolute limit is ~2,724 tokens, so
3072 (22.37 GiB, 99.5%) and 4096 (24.01 GiB, 106.8%) both OOM.

**The OOM risk here is real and larger than any previous run.** The top of the
±5% band is 21.77 GiB, which is 96.8% of the card. Every earlier profile had
its whole band comfortably inside. If it OOMs, that is informative rather than
surprising — fall back to seq 1024, or drop to ~12-13 layers at hidden 1536 if
you need seq 4096 (which costs about 0.2 B parameters).

### Why I am fairly confident anyway

The seq-dependent slope is **1,717,099 B/token**, and it is dominated by the
*well-determined* constant:

| component | B/token | share of slope |
|---|---|---|
| `c1` (per token, ±53% uncertain) | 149,765 | **8.7%** |
| `k × layers × hidden` (`k` ±10%, measured 46.44) | 1,567,334 | **91.3%** |

So the ±53% uncertainty in `c1` moves the slope by only ±4.6%. Pushing both
constants to their pessimistic extremes gives a ceiling of 2,395 tokens — still
above 2048. Pushing both optimistic gives 3,158 — still below 4096. **seq 2048
fits under every variant of the model, and seq 4096 fails under all of them.**

## What the run measures

Because `1b` and `1b2k` share layers and hidden, the difference of their
residuals is exactly `Δseq × (c1 + k × layers × hidden)` — it measures the
**seq slope** directly:

```
slope = (residual_1b2k − 2.1589 GiB) / 1,024 tokens
c1    = slope − 46.44 × 30,720  =  slope − 1,426,637 B/token
```

With `k` already measured at 46.44 from the `wider`/`1b` pair, this **inverts
to give `c1` on its own** — the constant the existing data barely constrains
(±53%). That is the scientific value of the run.

## What it does NOT do

**It does not discriminate the logits hypothesis.** At seq 2048 a full FP32
logits tensor would be 1.16 GiB against the model's 0.29 GiB — a 0.87 GiB
difference, which sits *inside* the ±5% band of a 20.73 GiB peak. The tool
flags this: `bands disjoint at ±5%? False`.

`deeper4k` is the run for that question — at seq 4096 the same two hypotheses
separate by 1.75 GiB and the bands are disjoint.

Likewise the two anchored candidates here are only 0.13 GiB apart, which is
inside the model's own worst error (0.0787 GiB). They are not a meaningful
pass/fail, and the tool says so rather than presenting them as one.

## Throughput prediction

**82.47 TFLOP/step**, 2.078x the `1b` run at seq 1024, at 16,384 tokens/step.

| achieved efficiency | s/step | tok/s |
|---|---|---|
| 30.9 TFLOP/s | 2.669 | 6,139 |
| 34.3 TFLOP/s (`1b`'s measured) | 2.404 | **6,814** |
| 38.0 TFLOP/s | 2.170 | 7,549 |

**Expect throughput to fall slightly**, to roughly 6,800 tok/s from 7,085.
Tokens per step double but FLOPs per step more than double, because the
attention term grows as seq² (3.9% → 7.5% of total FLOPs). That is arithmetic,
not a regression.

On efficiency: `deeper` gained +18.8% from doubling seq at hidden 1024, so
there is a case for a rise here too. Against that, `1b` already showed depth
costs a little (−1.7% vs `wider`), and attention is more bandwidth-bound than
the dense GEMMs. I expect **34-37 TFLOP/s** and would treat anything above 37
as a genuine surprise.

## Thresholds

| # | Condition | What it means |
|---|---|---|
| **M1** | peak in **[19.70, 21.77] GiB** | three-term model holds; the run is a clean slope measurement |
| **M2** | **OOM** | the model under-predicts at >90% occupancy; 2048 is not usable at 1B |
| **M3** | peak outside the band without an OOM | report the measured slope and the `c1` it implies |
| **R3** | median tok/s outside **[5,832, 7,927]** | the FLOP model or the efficiency assumption |
| **R4** | MODEL_TFLOP/s **< 30.9** | would refute "bigger GEMMs are at least as efficient" |
| **R6** | log reports TOTAL **!= 1009.4 M** | **run is void** — env override dropped |
| **R7** | any skipped or NaN iteration | BF16 stability at 20 layers / seq 2048 |

Superseded and not under test: the smoke-only two-term calibration says 21.61
GiB, the two-point says 22.62 GiB, the flat "+14%" rule says 19.21 GiB.

## The command

```bash
NUM_LAYERS=20 HIDDEN_SIZE=1536 FFN_HIDDEN_SIZE=4608 \
  NUM_ATTENTION_HEADS=12 NUM_QUERY_GROUPS=3 SEQ_LENGTH=2048 \
  DATA_PATH=/workspace/run/datasets/c4_qwen3 TRAIN_ITERS=50 \
  ./g5/finish-run.sh
```

50 steps at 16,384 tokens/step is 819,200 tokens, against ~12,247 samples
available in the c4 build — no repetition. At ~2.4 s/step expect roughly 2
minutes of training plus setup.
