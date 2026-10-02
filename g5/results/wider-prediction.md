# `wider` profile — prediction recorded BEFORE the run

**Status: PREDICTION ONLY. Nothing here is measured.**
Written 2026-10-02, before the `wider` profile was ever executed, so the run is
a test of these numbers rather than a fit to them. Regenerate with:

```bash
python3 g5/predict.py --profile wider
python3 g5/predict.py --self-check
```

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
