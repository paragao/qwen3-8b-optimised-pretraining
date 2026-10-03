# `1b` profile (1.0094 B params) — prediction recorded BEFORE the run

**Status: PREDICTION ONLY. Nothing here is measured.**
Reproduce with `python3 g5/predict.py --profile 1b`.

## Geometry, and why this shape

| | `1b` | `wider` (measured) | Qwen3-8B |
|---|---|---|---|
| num_layers | **20** | 6 | 36 |
| hidden_size | 1536 | 1536 | 4096 |
| ffn_hidden_size | 4608 | 4608 | 12288 |
| heads / kv groups | 12 / 3 | 12 / 3 | 32 / 8 |
| head_dim | 128 | 128 | 128 |
| GQA ratio | 4:1 | 4:1 | 4:1 |
| seq_length | 1024 | 1024 | 4096 |
| **params** | **1,009,385,472** | 629.5 M | 8.190 B |
| aspect (hidden/layers) | 77 | 256 | 114 |

Two reasons for hidden 1536 / 20 layers rather than a wider, shallower shape at
the same parameter count:

1. **Aspect ratio.** 77 hidden per layer against Qwen3-8B's 114. The
   alternative at ~1B, hidden 2048 / 8 layers, is 256 — more than 2x off.
2. **It is a single-variable change from `wider`.** Same hidden, ffn, heads,
   query groups and seq_length; only `num_layers` moves, 6 -> 20. Asserted by
   `predict.py --self-check` so a future edit cannot silently break it.

That second property is the valuable one, and it is why this run is worth more
than a model-size check.

## This run settles the memory model question

The `deeper` run refuted the residual model's *form*, and the diagnosis was
that none of the first three profiles was a single-variable change from any
other, so no pair could isolate a term. `1b` and `wider` are such a pair.

Because both have `seq_length = 1024`, the FP32 logits term is **identical in
both** and **cancels exactly** in the difference of their residuals. So:

```
k = (residual_1b - residual_wider) / (units_1b - units_wider)
  = (residual_1b - 1.2065 GiB) / 22,020,096
```

carries **no assumption about the logits term at all**. This is the first clean
measurement of the per-layer activation constant this repo can make.

The two pair-fits that disagreed after `deeper` predict different answers:

| per-layer constant | source | predicted `1b` peak |
|---|---|---|
| 24.81 B/unit | smoke+deeper fit | **18.64 GiB** (82.9% of card) |
| 80.17 B/unit | smoke+wider fit | **19.77 GiB** (87.9% of card) |

**1.14 GiB apart**, so the measurement picks one. Note these are *anchored*
predictions: static is analytic and exact, `wider`'s residual is measured, and
only `k` is extrapolated.

## Does it fit?

| | |
|---|---|
| static @18 B/param | **16.921 GiB** (75.2% of card) |
| worst anchored prediction | **19.77 GiB** (87.9%) |
| headroom at worst | **2.72 GiB** |
| A10G total | 22.488 GiB (`nvidia-smi`: 23028 MiB) |

**It should fit, but this is the tightest profile yet** — `wider`, the previous
largest, peaked at 11.76 GiB (52%). Treat an OOM as a real possibility rather
than a surprise.

The vocab is now a smaller share than at smaller sizes but still large:
embedding + LM head is **466.7 M of 1,009.4 M (46.2%)**. For Qwen3-8B the same
term is 15.2%, so a 1B proxy with the full 151,936 vocab is *structurally*
more vocab-dominated than the model it proxies. Keeping the real vocab is
deliberate — it preserves the tokenizer and the logits/loss path, which is what
the validation exists to exercise.

### Fallback if it OOMs

```bash
NUM_LAYERS=8 HIDDEN_SIZE=2048 FFN_HIDDEN_SIZE=6144 \
  NUM_ATTENTION_HEADS=16 NUM_QUERY_GROUPS=4 ./g5/run.sh
```

1.0082 B params, 16.90 GiB static, but only **16.8 M activation units against
1b's 31.5 M** — worst case ~18.7 GiB. It loses both properties above (aspect
256, and it is no longer a single-variable change from `wider`), so prefer the
20-layer shape and fall back only if needed.

## Throughput prediction

39.69 TFLOP/step, **3.882x `smoke`** and 1.991x `wider`, at 8,192 tokens/step.

| achieved efficiency | s/step | tok/s |
|---|---|---|
| 30.9 TFLOP/s (= `smoke`) | 1.284 | 6,378 |
| 35.0 TFLOP/s | 1.134 | **7,224** |
| 38.0 TFLOP/s | 1.044 | 7,843 |

Throughput falls versus every earlier profile because tokens/step is fixed at
8,192 while the work per token nearly quadruples. That is arithmetic, not a
regression.

Efficiency should land at or above `wider`'s 34.9 TFLOP/s: identical width, and
20 layers give the scheduler more independent work to overlap. `deeper` reached
36.7 at a *narrower* width, so 35-37 is the reasonable expectation.

## Refutation thresholds

| # | Condition | What it means |
|---|---|---|
| **A1** | peak in **[18.49, 18.79] GiB** | per-layer constant k ~ 24.81 B/unit |
| **A2** | peak in **[19.62, 19.92] GiB** | per-layer constant k ~ 80.17 B/unit |
| **A3** | peak outside both, but **no OOM** | k is neither; report the measured k, which the pair gives directly |
| **R3** | median tok/s outside **[6,059, 8,236]** | the FLOP model or the efficiency assumption |
| **R4** | MODEL_TFLOP/s **< 34.9** (`wider`'s) | unexpected: same width, more layers to overlap |
| **R5** | **OOM** | 18 B/param is not a floor at this scale; fall back to hidden 2048 / 8 layers |
| **R6** | log reports TOTAL **!= 1009.4 M** | **run is void** — env override dropped |
| **R7** | any skipped or NaN iteration | BF16 stability at 20 layers on `sm_86` |

A1 and A2 are disjoint by construction (±0.15 GiB against a 1.14 GiB gap).

**A3 is not a failure.** Unlike the earlier runs, this pair *measures* k rather
than testing a prediction about it, so a value outside both bands is the result,
not a refutation. The only genuinely bad outcomes are R5 (OOM) and R6 (void).

## What this will and will not establish

**Will:** that a 1B-parameter model trains on a single 24 GB A10G at DP=1; the
per-layer activation constant, measured with the logits term cancelled; and
whether efficiency keeps rising with depth at fixed width.

**Will not:** anything about Qwen3-8B. 1.0094 B is still 8.1x smaller, at 20
layers against 36 and hidden 1536 against 4096, and Qwen3-8B needs 137.3 GiB of
static state against this card's 22.5 GiB.

---

# OUTCOME — measured 2026-10-03 09:12

Log: `g5/results/run-20261003-101233.log`.
Reproduce the analysis with `python3 g5/predict.py --form-test`.

**Headline: a 1B-parameter model trains on one 24 GB A10G at 19.08 GiB peak
(84.8% of the card). A3 fired — the measurement landed outside both bands,
which is the anticipated result, because this pair measures the per-layer
constant rather than testing a prediction about it. It is 46.44 B/unit.**

## Validity gates

| Gate | Measured | Verdict |
|---|---|---|
| **R6** TOTAL params | **1009.4 M** (expected 1009.4 M) | **VALID** |
| **R7** skipped / NaN | **0 / 0** over 50 steps | PASS — BF16 stable at 20 layers |
| **R5** OOM | none; 19.08 GiB, **3.41 GiB headroom** | PASS — it fits |

The log also confirms real data (`dataset: real Megatron-indexed data at
/workspace/run/datasets/c4_qwen3`, `mock: False`) and the new DP reporting
(`data parallel size: 1`, `grad accum per rank: 8`).

## Memory: 19.08 GiB, between the two bands

| | Predicted | Band | Measured |
|---|---|---|---|
| A1 (k = 24.81 B/unit) | 18.64 GiB | [18.49, 18.79] | **19.08 GiB** — miss |
| A2 (k = 80.17 B/unit) | 19.77 GiB | [19.62, 19.92] | **19.08 GiB** — miss |
| **A3** outside both, no OOM | — | — | **FIRED** |

Fragmentation was 0.35 GiB (reserved 19.43), the lowest of the four runs.

## The measurement: k = 46.44 B/unit

This is the point of the run. `wider` and `1b` differ in `num_layers` alone and
share `seq_length`, so the logits term is identical in both and cancels:

```
k = (2.1589 − 1.2065) GiB / 22,020,096 units = 46.44 B/unit
```

**No assumption about the logits term enters this number.** Both earlier
pair-fits were wrong, and in opposite directions — 24.81 and 80.17 bracket the
measured 46.44. That is why neither band hit, and why A3 was written in advance
as a result rather than a failure.

## A three-term model now fits all four points

With k known, the residual's remaining part is not constant across profiles at
the same `seq_length`: `smoke` gives 0.6337 GiB and `wider`/`1b` give 0.7984
GiB, both at seq 1024. So a plain "logits + per-layer" form cannot be right.
Splitting the remainder into a fixed part and a per-token part:

```
residual = c0 + c1 x seq_length + k x (layers x seq x hidden x batch)
```

Least squares over all four measurements:

| | |
|---|---|
| c0 (fixed) | **0.5366 GiB** |
| c1 (per token) | **149,765 B/token** = **24.6%** of a full FP32 vocab logits row (607,744 B) |
| k (per unit) | **51.02 B/unit** |

| profile | predicted | measured | error |
|---|---|---|---|
| `smoke` | 0.8787 | 0.8151 | +0.0636 GiB (+0.93% of peak) |
| `wider` | 1.1278 | 1.2065 | −0.0787 GiB (−0.67%) |
| `deeper` | 2.0179 | 2.0179 | −0.0000 GiB |
| `1b` | 2.1740 | 2.1589 | +0.0151 GiB (+0.08%) |

**Worst error 0.0787 GiB**, against the two-term model's worst pairwise
extrapolation of 2.3354 GiB. Under 1% of peak everywhere.

The per-token coefficient coming out at **24.6% of a full FP32 logits row** is
the interesting part: it suggests Megatron does **not** hold the entire FP32
vocab-logits tensor resident at peak, which would be consistent with a chunked
vocab projection or cross-entropy. That is a hypothesis the number is
consistent with, not something this run demonstrates.

### Why this is still not a validated model

Three parameters against four points leaves **one degree of freedom**, and the
free direction is the weak one: **`c1` is pinned by a single point.** `deeper`
is the only profile at seq 2048; the other three are all at 1024, so dropping
`deeper` makes the fit singular. The model is suggestive, not validated.

What would settle it: a **seq sweep at fixed layers and hidden** — e.g. the
`smoke` geometry at seq 512, 2048 and 4096. Three more runs of under a minute
each, and `c1` stops resting on one measurement.

## Throughput: the FLOP model holds on a fourth architecture

| | Predicted | Measured | |
|---|---|---|---|
| FLOPs/step | 39.69 TFLOP | 39.65 TFLOP | **+0.10%** |
| median tok/s | 6,378–7,843 (R3 [6,059, 8,236]) | **7,085** | **HIT** |
| s/step | 1.044–1.284 | 1.156 (stdev 0.003) | in range |

| | predicted | implied | error |
|---|---|---|---|
| smoke | 10.22 TFLOP | 10.23 | −0.04% |
| wider | 19.94 TFLOP | 19.93 | +0.04% |
| deeper | 31.99 TFLOP | 32.00 | −0.03% |
| **1b** | **39.69 TFLOP** | **39.65** | **+0.10%** |

Calibrated on `smoke` alone, never refitted, across a **3.9x span in FLOPs per
step** and changes in all three of layers, hidden size and sequence length.

## R4 fired: depth does not buy efficiency

I predicted efficiency would land at or above `wider`'s 34.9 TFLOP/s, reasoning
that identical width plus 20 layers would give the scheduler more independent
work to overlap. **That was wrong.** Measured **34.3 TFLOP/s (27.4% MFU)**,
1.7% *below* `wider`.

And this is a clean attribution, because `wider` -> `1b` is a single-variable
pair for throughput too:

| | hidden | seq | layers | TFLOP/s | MFU |
|---|---|---|---|---|---|
| `smoke` | 1024 | 1024 | 4 | 30.9 | 24.7% |
| `wider` | **1536** | 1024 | 6 | 34.9 | 27.9% |
| `deeper` | 1024 | **2048** | 12 | **36.7** | **29.4%** |
| `1b` | 1536 | 1024 | **20** | 34.3 | 27.4% |

Width raised efficiency (+12.9%), sequence length raised it more (+18.8%), and
**depth did not** — 6 -> 20 layers at fixed width and seq moved it −1.7%.
Efficiency on this card is set by the shape of the GEMMs, not by how many of
them there are.

## The loss curve

12.2330 -> **7.8027** over 50 steps on real c4 (409,600 tokens, 400 samples of
48,939 available, 1 epoch — no token seen twice). First-iteration loss sits
0.30 nats above `ln(151936) = 11.9312`, the correct signature of a fresh init.

For scale, the `smoke` real-data run reached 5.86 after 8.2M tokens. This is
7.80 after 0.41M — 20x fewer tokens, and a consistent trajectory. Nothing here
is converged, and nothing should be read as a quality claim.

## What this establishes

A 1B-parameter model, with the full 151,936 vocab and the real Qwen3 tokenizer,
pre-trains on a single g5.8xlarge: 19.08 GiB of 22.49 GiB, 7,085 tok/s, 0
skipped and 0 NaN iterations, on real c4 data. The per-layer activation
constant is now measured rather than fitted.

It still says nothing about Qwen3-8B, which is 8.1x larger and needs 137.3 GiB
of static state against this card's 22.5 GiB.
