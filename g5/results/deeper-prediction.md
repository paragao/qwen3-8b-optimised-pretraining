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

---

# OUTCOME — measured 2026-10-02 18:25

Log: `g5/results/run-20261002-192536.log`.
Reproduce the analysis with `python3 g5/predict.py --form-test`.

**Headline: R2b fired. The measurement fell outside BOTH bands, and that
refutes the memory model's FORM, not just its constants.**

## Validity gates

| Gate | Measured | Verdict |
|---|---|---|
| **R6** TOTAL params | **455.9 M** (expected 455.9 M) | **VALID** |
| **R7** skipped / NaN | **0 / 0** over 50 steps | PASS — BF16 stable at 12 layers / seq 2048 |

## Memory: both predictions were wrong, in the same direction

| Hypothesis | Predicted | Band | Measured | Verdict |
|---|---|---|---|---|
| R1 smoke-only calibration | 10.21 GiB | [10.07, 10.35] | **9.66 GiB** | **MISS** (over by 0.55) |
| R2 two-point calibration | 10.52 GiB | [10.39, 10.66] | **9.66 GiB** | **MISS** (over by 0.86) |
| R2b outside both | — | — | **9.66 GiB** | **FIRED** |

Peak reserved 10.03 GiB, so fragmentation is 0.37 GiB. At 9.66 GiB the profile
used 43.0% of the card, against a predicted 45.4-46.8%.

## Why: the model form cannot fit three points

The residual model has two free parameters:

```
residual = alpha * logits_bytes(seq) + beta * (layers * seq * hidden)
```

Two points fix it exactly. If the form were right, any two points would predict
the third. All three leave-one-out fits fail:

| Fit on | alpha | beta (B/unit) | Predicts | Measured | Error |
|---|---|---|---|---|---|
| wider + deeper | 3.1044 | **−67.44** | smoke 1.5358 | 0.8151 | **+0.72 GiB (+88.4%)** |
| smoke + deeper | 1.2391 | 24.81 | wider 0.9363 | 1.2065 | −0.27 GiB (−22.4%) |
| smoke + wider | 0.8660 | 80.17 | deeper 2.8828 | 2.0179 | **+0.86 GiB (+42.9%)** |

The first fit requires a **negative** bytes-per-activation-unit. No physical
allocation can have that, so the form is refuted on its own terms before the
error magnitudes are even considered.

## The experiment design was weak, and that is the real lesson

None of the three profiles is a single-variable change from any other:

| | co-varies |
|---|---|
| `smoke` -> `wider` | num_layers 4->6, **and** hidden_size 1024->1536 |
| `smoke` -> `deeper` | num_layers 4->12, **and** seq_length 1024->2048 |
| `wider` -> `deeper` | num_layers, hidden_size **and** seq_length |

A two-parameter model fitted to one such pair has no basis for extrapolating to
a third point. **`wider` appearing to confirm the decomposed model was
interpolation luck, not validation** — it happened to sit near the
smoke-calibrated line. `deeper` is the first point far enough away to expose
that, and it did.

### Correcting the previous conclusion

`g5/results/wider-prediction.md` reports "the decomposed memory model was
CONFIRMED". That needs splitting in two:

- **Still true:** at `wider`, the decomposed prediction (11.66) beat the flat
  "+14%" rule (11.98) against a measured 11.76. That was a head-to-head
  comparison of two predictions against one measurement, and the flat rule's
  refutation stands.
- **No longer true:** that the decomposed form is *correct*. It is not. One
  point of agreement is not validation, and the third point refutes it.

## What would actually settle it

A **single-variable sweep**, which none of these profiles provides:

1. **seq sweep** at fixed layers and hidden (`smoke` geometry at seq 512,
   1024, 2048, 4096). At fixed layers and hidden both candidate terms are
   linear in seq, so the residual must be linear in seq. Curvature would show
   directly whether the logits term grows sub-linearly — which is the leading
   suspicion, since both failing fits over-predict the high-seq point.
2. **layers sweep** at fixed seq and hidden (`smoke` at 4, 8, 12, 16 layers).
   Separates the per-layer term cleanly.

Each is ~6 runs of under a minute. Until then the honest sizing advice is the
empirical one: 18 B/param is a floor, measured residuals ran 11-26% above it
across these three shapes, and an untested shape should be measured rather than
predicted.

## Throughput: every prediction held

| | Predicted | Measured | |
|---|---|---|---|
| FLOPs/step | 31.99 TFLOP | 32.00 TFLOP | **−0.03%** |
| median tok/s | 15,824-19,460 (R3 band [15,033, 20,433]) | **18,793** | **HIT** |
| s/step | 0.842-1.035 | 0.872 | in range |
| R8 faster than `wider`? | yes | 18,793 > 14,357 | **PASS** |
| R4 MODEL_TFLOP/s >= 30.9 | yes | **36.7** | PASS |

**The FLOP model is now validated on three architectures.** Calibrated on
`smoke` alone and never refitted:

| | predicted | implied by measurement | error |
|---|---|---|---|
| smoke | 10.22 TFLOP | 10.23 | −0.04% |
| wider | 19.94 TFLOP | 19.93 | +0.04% |
| deeper | 31.99 TFLOP | 32.00 | **−0.03%** |

That spans changes in `num_layers`, `hidden_size` **and** `seq_length`. The
contrast with the memory model is the useful result of this run: one model
generalised across all three axes, the other did not survive its first
genuinely out-of-sample point.

## Sequence length substitutes for width, and then some

I recorded before the run that R4 was weak here and that I had no confident
prediction, because `deeper` returns to hidden 1024 and does not widen GEMMs.
The answer is unambiguous:

| | hidden | seq | MODEL_TFLOP/s | MFU |
|---|---|---|---|---|
| `smoke` | 1024 | 1024 | 30.9 | 24.7% |
| `wider` | **1536** | 1024 | 34.9 (+12.9%) | 27.9% |
| `deeper` | 1024 | **2048** | **36.7 (+18.8%)** | **29.4%** |

`deeper` is the **most** efficient of the three at the **narrowest** width.
Doubling sequence length bought more than increasing hidden size by 50% did,
presumably by enlarging the GEMM M dimension while leaving weights cache-

friendly. Also worth noting: `deeper` has *fewer* parameters than `wider`
(455.9 M vs 629.5 M) yet does 1.605x the FLOPs per step, so parameter count is
not a proxy for step cost.

Loss moved 12.1410 -> 0.2943 on mock data, which means nothing — see the
real-data run for an interpretable curve.
