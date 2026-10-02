# `deeper` profile — prediction recorded BEFORE the run

**Status: PREDICTION ONLY. Nothing here is measured.**
Reproduce with `python3 g5/predict.py --profile deeper`.

## What this run is for

`smoke` and `wider` are both measured, and together they solve the memory
residual exactly — which is the problem. Two equations in two unknowns have
**zero degrees of freedom**, so that solve cannot fail and proves nothing on
its own. `deeper` is the third point that over-determines the system and makes
it testable.

It discriminates because it moves `seq_length` to 2048 while dropping
`hidden_size` back to 1024. The two calibrations disagree there by **0.31 GiB**:

| Calibration | per-unit | logits term | `deeper` predicted peak |
|---|---|---|---|
| smoke only | 60.29 B/unit | full FP32 copy (1.0000) | **10.21 GiB** |
| smoke + wider (two-point) | 80.17 B/unit | 0.8660 of a copy | **10.52 GiB** |

The `wider` run already refuted the flat "+14% of static" rule, which predicts
8.68 GiB here. That is **not** the hypothesis under test — re-testing a dead
rule would waste the run.

## Geometry

| | `deeper` | `smoke` (measured) | `wider` (measured) |
|---|---|---|---|
| num_layers | **12** | 4 | 6 |
| hidden_size | **1024** | 1024 | 1536 |
| ffn_hidden_size | 3072 | 3072 | 4608 |
| heads / kv groups | 8 / 2 | 8 / 2 | 12 / 3 |
| head_dim | 128 | 128 | 128 |
| seq_length | **2048** | 1024 | 1024 |
| tokens/step | **16,384** | 8,192 | 8,192 |
| TOTAL params | 455.9 M | 359.4 M | 629.5 M |

Note `deeper` is *fewer* parameters than `wider` (455.9 M vs 629.5 M) but does
**more** FLOPs per step (31.99 vs 19.94 TFLOP), because doubling `seq_length`
doubles tokens per step and quadruples the attention term. Parameter count is
not a proxy for step cost.

## Memory prediction, decomposed

Under the smoke-only calibration (**the R1 hypothesis**):

| Term | Scaling | `deeper` |
|---|---|---|
| static state | 18 B/param | 7.64 GiB |
| FP32 vocab logits | `mbs x seq x 151936 x 4`, linear in seq | 1.16 GiB |
| other activations | `layers x seq x hidden x mbs` @ 60.29 B/unit | 1.41 GiB |
| **predicted peak** | | **10.21 GiB** (45.4% of card) |

Under the two-point calibration (**the R2 hypothesis**): 7.64 + 1.00 + 1.88 =
**10.52 GiB** (46.8%).

The logits term more than doubles versus `wider` (0.58 -> 1.16 GiB) purely from
`seq_length`, which is the term the flat rule cannot represent.

## Throughput prediction

31.99 TFLOP/step, **3.129x** `smoke` and 1.605x `wider`, at 16,384 tokens/step.

| Achieved efficiency | s/step | tok/s |
|---|---|---|
| 30.9 TFLOP/s (= `smoke`) | 1.035 | 15,824 |
| 35.0 TFLOP/s | 0.914 | **17,924** |
| 38.0 TFLOP/s | 0.842 | 19,460 |

**`deeper` should be FASTER in tok/s than `wider`** (14,357) despite doing more
work per step, because seq 2048 doubles the tokens each step amortises over.
If it comes out slower than `wider`, the FLOP model is wrong.

On efficiency: `deeper` returns to hidden 1024, the same GEMM width as `smoke`,
so the +12.9% gain `wider` showed should **not** fully carry over — width is
what drove it. But seq 2048 doubles the GEMM M dimension, which helps
independently. Best guess is 32-35 TFLOP/s, between the two measured points.

## Refutation thresholds

| # | Condition | What it means |
|---|---|---|
| **R1** | peak in **[10.07, 10.35] GiB** | smoke-only calibration confirmed (full FP32 logits, 60.29 B/unit) |
| **R2** | peak in **[10.39, 10.66] GiB** | two-point calibration confirmed (0.866x logits, 80.17 B/unit) |
| **R2b** | peak outside **both** | both wrong; an unmodelled term dominates |
| **R3** | median tok/s outside **[15,033, 20,433]** | the FLOP model or the efficiency assumption |
| **R4** | MODEL_TFLOP/s **< 30.9** | plausible here, unlike for `wider` — hidden is back to 1024, so a *drop* toward smoke's efficiency would be informative, not a failure |
| **R5** | OOM at a predicted 45% of card | 18 B/param is not a floor |
| **R6** | log reports TOTAL **!= 455.9 M** | **run is void** — env override dropped |
| **R7** | any skipped or NaN iteration | BF16 stability at 12 layers / seq 2048 |
| **R8** | tok/s **< 14,357** (slower than `wider`) | refutes the seq-amortisation argument |

R1 and R2 are disjoint by construction (±0.14 GiB tolerance against a 0.31 GiB
gap), so this run picks one.

**R4 is weaker here than it was for `wider`** and I want that on the record
before the fact: `wider` raised efficiency by widening GEMMs, and `deeper` does
not widen them. Reading a drop as a refutation would be wrong; the question
`deeper` actually answers about efficiency is whether sequence length
substitutes for width, and I do not have a confident prediction either way.

## Self-check

`python3 g5/predict.py --self-check` now also asserts the two-point solve
round-trips **both** measured points (6.84 and 11.76 GiB) and reproduces the
0.8660 logits fraction. That round-trip is the only check the exact solve can
fail, and it is what proves the solve itself is arithmetically right even
though it is not a validated model.
