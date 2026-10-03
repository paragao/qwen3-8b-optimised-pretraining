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
