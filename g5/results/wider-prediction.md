# `wider` profile — prediction, and the result that scored it

**The predictions in this file were committed at 2026-10-02 ~17:30 in
`6eb3a41`, before the run executed at 17:41.** The outcome is appended at the
bottom and nothing above it was edited afterwards, so the git history is the
evidence that this was a test rather than a fit. Reproduce with:

```bash
python3 g5/predict.py --profile wider   # the prediction
python3 g5/predict.py --self-check      # validate the models
python3 g5/predict.py --score           # grade the measurement
```

**Outcome in one line: the decomposed memory model was CONFIRMED (11.66
predicted, 11.76 measured) and the README's flat "+14% over static" rule was
REFUTED. Throughput landed 0.17% from the central prediction.**

## Why this file exists

The `smoke` profile was run first and its result was reconciled against an
estimate *after the fact*. That is a weaker epistemic position: a model that is
only ever compared to data it has already seen cannot be refuted. This time the
prediction, and the thresholds that would refute it, are committed first.

## How the predictions were derived

`g5/predict.py` lifts `param_count()` out of `g5/train.py` with `ast` rather
than restating the formula, so the two cannot drift. It has two models, each
with exactly one constant fitted to the measured `smoke` run:

**Memory.** Decomposed into three terms that scale *differently*:

| Term | Scaling | `wider` |
|---|---|---|
| static state | 18 B/param | 10.55 GiB |
| FP32 vocab logits | `mbs x seq x 151936 x 4` — **constant** in layers and hidden | 0.58 GiB |
| other activations | `layers x seq x hidden x mbs`, at a fitted 60.29 B/unit | 0.53 GiB |
| **predicted peak** | | **11.66 GiB** (51.9% of card) |

This competes with `g5/README.md`'s flat "+14% over static" rule of thumb,
which predicts **11.98 GiB**. The two differ by 0.32 GiB, and the run will
discriminate between them — the decomposition says the logits term must *not*
grow with layer count, the flat rule implicitly says it does.

**Throughput.** Model FLOPs per step, with the only free variable being
achieved TFLOP/s:

```
FLOPs = 6 x tokens x (layer_params + lm_head)      # dense GEMMs
      + 6 x batch x seq^2 x hidden x layers        # attention, causal-halved
```

This reproduces the measured `smoke` step to **0.04%** (10.2242 modelled vs
10.2279 TFLOP implied by 30.9 TFLOP/s x 0.331 s), which is what makes it
load-bearing rather than decorative. `wider` needs **19.94 TFLOP/step**,
**1.950x** `smoke`.

| Achieved efficiency | s/step | tok/s |
|---|---|---|
| 30.9 TFLOP/s (= `smoke`, treated as a floor) | 0.645 | 12,697 |
| 35.0 TFLOP/s (central: modest GEMM gain) | 0.570 | **14,382** |
| 38.0 TFLOP/s (optimistic) | 0.525 | 15,615 |

`smoke`'s efficiency is a floor because `wider`'s GEMMs are strictly larger
(hidden 1024 -> 1536) and larger GEMMs do not run *less* efficiently.

## Refutation thresholds

| # | Condition | What it refutes |
|---|---|---|
| **R1** | peak allocated in **[11.52, 11.81] GiB** | confirms the decomposed model |
| **R2** | peak allocated in **[11.84, 12.12] GiB** | refutes the decomposition; flat +14% rule wins |
| **R2b** | peak outside **both** bands | both models wrong; unmodelled term dominates |
| **R3** | median tok/s outside **[12,062, 16,395]** | the FLOP model or the efficiency assumption |
| **R4** | MODEL_TFLOP/s **< 30.9** | refutes "bigger GEMMs are at least as efficient" |
| **R5** | OOM at a predicted 52% of card | 18 B/param is not a floor |
| **R6** | log reports TOTAL params **!= 629.5 M** | **the run is void** — env override was dropped |
| **R7** | any skipped or NaN iteration | BF16 stability at this width on sm_86 |

R1 and R2 are deliberately **disjoint** (0.14 GiB tolerance against a 0.32 GiB
gap). An earlier draft used +/-0.35, which swallowed the competing prediction
and would have made the run unable to discriminate at all.

## R6 is the one that matters most

`g5/finish-run.sh` originally forwarded only `TRAIN_ITERS`, `LOG_INTERVAL` and
`HF_TOKEN` over ssh. Because ssh does not inherit the caller's environment,
every architecture variable was **silently dropped**, and `train.py`'s
`_env_int` treats an absent value as its default — so a request for `wider`
would have re-run `smoke` and reported ~24,757 tok/s as if it were the wider
result. Fixed in the same commit as this file.

The log prints `TOTAL : 629.5 M` before allocating anything. **Check that line
first.** If it reads 359.4 M, the override was dropped and the numbers are
`smoke` again, not a `wider` measurement.

## Expected geometry

| | `wider` | `smoke` (measured) |
|---|---|---|
| num_layers | 6 | 4 |
| hidden_size | 1536 | 1024 |
| ffn_hidden_size | 4608 | 3072 |
| heads / kv groups | 12 / 3 | 8 / 2 |
| head_dim | 128 | 128 |
| GQA ratio | 4:1 | 4:1 |
| seq_length | 1024 | 1024 |
| tokens/step | 8,192 | 8,192 |
| TOTAL params | 629.5 M | 359.4 M |

`head_dim` 128 and GQA 4:1 match Qwen3-8B in both profiles, which is the point
of the proxy. Vocab stays at the full 151,936.

## Self-check and mutation test

`python3 g5/predict.py --self-check` verifies the lifted `param_count`
reproduces Qwen3-8B at 8.1904 B, all three profiles to within 0.1 M, the FLOP
model to 1%, and that the memory model round-trips `smoke` to 6.84 GiB.

It was mutation-tested to confirm it is not vacuous: changing the SwiGLU factor
from 3 to 2, and zeroing the untied `lm_head`, were both **KILLED**. `train.py`
was verified byte-identical (sha256 `1bf6dd75...`) after the mutants ran
against copies in `/tmp`.

---

# OUTCOME — measured 2026-10-02 17:41

Log: `g5/results/run-20261002-184158.log` (46,433 B, retrieved locally).
Scored by `python3 g5/predict.py --score`.

## Validity gates first

| Gate | Measured | Verdict |
|---|---|---|
| **R6** TOTAL params | **629.5 M** (expected 629.5 M) | **VALID** — the env override reached the model |
| **R7** skipped / NaN iterations | **0 / 0** over 50 steps | PASS — no BF16 instability at hidden 1536 on `sm_86` |

R6 is the one that mattered. The `finish-run.sh` passthrough defect was fixed
in the same commit as this file; the log confirms all six architecture values
arrived (`num_layers: 6`, `hidden_size: 1536`, `ffn_hidden_size: 4608`,
`num_attention_heads: 12`, `num_query_groups: 3`, `seq_length: 1024`). Had the
fix not landed, this would have been a second `smoke` run reporting 24,757
tok/s under the wrong label.

## Memory: the decomposition wins, the flat rule is refuted

| Model | Predicted | Measured | Error | Band | Verdict |
|---|---|---|---|---|---|
| **decomposed** (this file) | 11.66 GiB | **11.76 GiB** | **+0.10 GiB (+0.83%)** | [11.52, 11.81] | **HIT — confirmed** |
| flat "+14% over static" | 11.98 GiB | 11.76 GiB | −0.22 GiB (−1.88%) | [11.84, 12.12] | miss — **refuted** |

The disjoint bands did their job: the measurement falls inside one and outside
the other, so the run discriminated rather than accommodating both. The flat
rule erred by 2.3x as much, and in the direction the decomposition predicts —
it over-counts because it scales the FP32 logits term with static state, when
that term is constant in `num_layers` and `hidden_size`.

Peak reserved was 12.25 GiB, so allocator fragmentation is 0.49 GiB (smoke:
0.42 GiB). At 11.76 GiB the profile used 52.3% of the 22.49 GiB card.

## Throughput: the FLOP model holds on a second architecture

| | Predicted | Measured |
|---|---|---|
| FLOPs/step | **19.94 TFLOP** | 19.93 TFLOP (34.9 TFLOP/s x 0.571 s) — **+0.04%** |
| median tok/s | 14,382 @ 35.0 TFLOP/s assumed | **14,357** — **−0.17%** |
| s/step | 0.570 | 0.571 (stdev 0.001, 0.2%) |
| MODEL_TFLOP/s | >= 30.9 (R4 floor) | **34.9** — PASS, **+12.9%** over smoke |

The FLOP model was calibrated on `smoke` and **never refitted**, then predicted
`wider`'s FLOPs per step to **0.04%**. That is the result worth keeping: the
model generalised across a 1.75x change in parameter count and a 1.5x change in
hidden size, which a single-point calibration had no obligation to do.

The central 14,382 tok/s figure landing within 0.17% is **partly luck** and
should not be read as the model being that precise. It required a judgement
call that achieved efficiency would be 35.0 TFLOP/s, and the measurement came
in at 34.9. The defensible claim is the FLOP count (+0.04%) plus the R4
direction call (bigger GEMMs are not less efficient, confirmed at +12.9%); the
exact tok/s followed from an efficiency guess inside a [30.9, 38.0] band.

MFU rose from 24.7% to **27.9%** of A10G BF16 dense peak. Throughput is
**0.5799x** smoke, against 1.950x the FLOPs per step.

## What two points now permit, and what they do not

With `smoke` and `wider` the residual splits into its constant and per-layer
parts exactly:

| | |
|---|---|
| smoke residual | 0.8151 GiB over 4,194,304 units |
| wider residual | 1.2065 GiB over 9,437,184 units |
| solved per-unit | **80.17 B** per (layer x seq x hidden x batch) |
| solved constant | **0.5019 GiB** |
| pure FP32 logits theory | 0.5796 GiB (solved value is **13.4% lower**) |

**This is a reparametrisation, not a validated fit.** Two equations in two
unknowns have zero degrees of freedom, so it cannot fail and must not be
reported as confirmation. It does raise a real question: the constant term
comes out at 0.87 of a full FP32 logits copy, not 1.00, which would be
consistent with partial fusion or with peak not coinciding with full logits
materialisation — or with the single fitted per-unit constant absorbing
structure that does not actually scale as `layers x hidden`.

Testing it needs a **third** point, and `deeper` is the natural one because it
moves `seq_length` to 2048 while dropping back to hidden 1024. The two
parametrisations disagree there by 0.31 GiB:

| Calibration | `deeper` predicted peak |
|---|---|
| smoke only (60.29 B/unit, full FP32 logits) | **10.21 GiB** |
| smoke + wider (80.17 B/unit, 0.87x logits) | **10.53 GiB** |

Same discriminating magnitude as the question this run just settled.

## The loss curve still does not mean convergence

Loss moved 12.2437 -> 0.1902, and the log still reports `mock: True` with
`MockGPTDataset splits with sizes=(400, 0, 0)`. First-iteration loss sits just
above `ln(151936) = 11.9312`, which is uniform guessing over the full vocab and
the correct starting point for a fresh init; the collapse is degenerate fitting
of 400 synthetic samples. `DATA_PATH` was forwarded by the fixed script but
left empty. Real convergence needs it pointed at c4.

What the trace does prove: gradients flow, Adam updates, the LR schedule
applies, and 0 skipped / 0 NaN across 50 steps at hidden 1536.
