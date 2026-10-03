#!/usr/bin/env python3
"""Derive memory and throughput predictions for the g5 proxy profiles.

Why this file exists
--------------------
A measurement whose predicted values exist only in a conversation is a fit, not
a test. This script writes the predictions down, with explicit refutation
thresholds, BEFORE a profile is run, so the result can refute them.

It deliberately reuses `param_count()` out of `g5/train.py` rather than
restating the formula, so the two can never drift apart. `train.py` imports
`megatron.bridge` and `torch` at module scope and neither is installed on a
laptop, so the function is lifted out with `ast` instead of imported. The
`--self-check` assertions below prove the lift grabbed the real function.

Usage:
    python3 g5/predict.py              # print predictions for every profile
    python3 g5/predict.py --self-check # verify against known-good values
"""

import argparse
import ast
import os
import sys

GIB = 1024 ** 3

# Measured on the g5.8xlarge A10G: nvidia-smi reports 23028 MiB total.
A10G_TOTAL_BYTES = 23028 * 1024 * 1024

# A10G BF16 dense tensor-core peak, used only for the MFU figure.
A10G_BF16_PEAK_TFLOPS = 125.0

VOCAB_SIZE = 151936

TRAIN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")


def load_param_count(path: str = TRAIN_PY):
    """Lift `param_count` out of train.py without importing its dependencies."""
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "param_count":
            module = ast.Module(body=[node], type_ignores=[])
            namespace: dict = {}
            exec(compile(module, path, "exec"), namespace)  # noqa: S102
            return namespace["param_count"]
    raise SystemExit(f"FATAL: param_count() not found in {path}; did it get renamed?")


# The three documented proxy profiles, matching g5/README.md.
PROFILES = {
    "smoke": dict(num_layers=4, hidden_size=1024, ffn_hidden_size=3072,
                  num_attention_heads=8, num_query_groups=2, seq_length=1024),
    "wider": dict(num_layers=6, hidden_size=1536, ffn_hidden_size=4608,
                  num_attention_heads=12, num_query_groups=3, seq_length=1024),
    "deeper": dict(num_layers=12, hidden_size=1024, ffn_hidden_size=3072,
                   num_attention_heads=8, num_query_groups=2, seq_length=2048),
    # ~1B parameters. Deliberately hidden 1536 rather than a wider/shallower
    # shape at the same parameter count, for two reasons:
    #
    #  1. Aspect ratio. 1536/20 = 77 hidden per layer, against Qwen3-8B's
    #     4096/36 = 114. The alternative (hidden 2048 / 8 layers) is 256, more
    #     than 2x off, so this is the better proxy for the real architecture.
    #  2. It is a SINGLE-VARIABLE change from `wider` -- same hidden, ffn,
    #     heads, query groups and seq_length, only num_layers differs (6 -> 20).
    #     Because seq_length is identical, the logits term cancels exactly in
    #     the difference of the two residuals, so the pair MEASURES the
    #     per-layer activation constant directly. That is the experiment the
    #     refuted memory model needed and that none of the first three
    #     profiles could provide.
    #
    # The cost of that choice is activation memory: 31.5M activation units
    # against 16.8M for hidden 2048 / 8 layers. If this profile OOMs, the
    # fallback is NUM_LAYERS=8 HIDDEN_SIZE=2048 FFN_HIDDEN_SIZE=6144
    # NUM_ATTENTION_HEADS=16 NUM_QUERY_GROUPS=4 (1.0082B, ~18.7 GiB worst case).
    "1b": dict(num_layers=20, hidden_size=1536, ffn_hidden_size=4608,
               num_attention_heads=12, num_query_groups=3, seq_length=1024),
}

# Qwen3-8B itself, as a sanity anchor for the lifted formula.
QWEN3_8B = dict(num_layers=36, hidden_size=4096, ffn_hidden_size=12288,
                num_attention_heads=32, num_query_groups=8, seq_length=4096)

# ---------------------------------------------------------------------------
# Ground truth from the completed `smoke` run (2026-10-02), which is what makes
# the predictions below calibrated rather than guessed.
#   g5/results/validation-run.md and g5/results/run-20261002-162239.log
# ---------------------------------------------------------------------------
SMOKE_MEASURED = dict(
    peak_allocated_gib=6.84,
    peak_reserved_gib=7.26,
    step_time_s=0.331,
    tokens_per_step=8192,
    tokens_per_s=24757,
    model_tflop_s=30.9,
)

# ---------------------------------------------------------------------------
# Measured `wider` run (2026-10-02 17:41), AFTER the predictions above were
# committed in g5/results/wider-prediction.md. Kept separate from
# SMOKE_MEASURED, which is the only data either model was calibrated on.
#   g5/results/run-20261002-184158.log
# ---------------------------------------------------------------------------
WIDER_MEASURED = dict(
    peak_allocated_gib=11.76,
    peak_reserved_gib=12.25,
    step_time_s=0.571,
    tokens_per_step=8192,
    tokens_per_s=14357,
    model_tflop_s=34.9,
    params_m=629.5,
    skipped_iterations=0,
    nan_iterations=0,
    loss_first=12.2437,
    loss_last=0.1902,
)

# ---------------------------------------------------------------------------
# Measured `deeper` run (2026-10-02 18:25). This is the THIRD point, and it
# fell outside BOTH predicted bands (R2b) -- see g5/results/deeper-prediction.md.
#   g5/results/run-20261002-192536.log
# ---------------------------------------------------------------------------
DEEPER_MEASURED = dict(
    peak_allocated_gib=9.66,
    peak_reserved_gib=10.03,
    step_time_s=0.872,
    tokens_per_step=16384,
    tokens_per_s=18793,
    model_tflop_s=36.7,
    params_m=455.9,
    skipped_iterations=0,
    nan_iterations=0,
    loss_first=12.1410,
    loss_last=0.2943,
)

# ---------------------------------------------------------------------------
# Measured REAL-DATA run (2026-10-02 18:55): `smoke` geometry, real c4.
# Controlled single-variable change against SMOKE_MEASURED -- only the data
# source differs. 1000 iterations, 1 epoch, no token seen twice.
#   g5/results/run-20261002-195550.log
# ---------------------------------------------------------------------------
C4_MEASURED = dict(
    peak_allocated_gib=6.84,
    peak_reserved_gib=7.26,
    step_time_s=0.331,
    tokens_per_s=24727,
    iterations=1000,
    tokens_consumed=8_192_000,
    dataset_sequences=12247,
    dataset_tokens=12247 * 4096,
    samples_available=48939,
    samples_consumed=8000,
    epochs=1,
    loss_first=12.1809,
    loss_last=5.8633,
    skipped_iterations=0,
    nan_iterations=0,
)

# train.py defaults: micro_batch_size=1, global_batch_size=8 (lines 204-205).
# So a global step is 8 gradient-accumulation micro-steps, and peak activation
# memory is that of a SINGLE micro-batch of size 1.
MICRO_BATCH_SIZE = 1
GLOBAL_BATCH_SIZE = 8


def logits_bytes(arch: dict, micro_batch: int = MICRO_BATCH_SIZE) -> int:
    """FP32 vocab logits for one micro-batch: the largest single activation.

    Constant in num_layers and hidden_size, linear in seq_length -- which is
    exactly why scaling peak memory by a flat percentage of the static figure
    (the README's '+14%' rule of thumb) is the wrong shape.
    """
    return micro_batch * arch["seq_length"] * VOCAB_SIZE * 4


def activation_units(arch: dict, micro_batch: int = MICRO_BATCH_SIZE) -> int:
    """Scaling units for everything stashed per layer, excluding the logits.

    Stored activations go as (num_layers x seq_length x hidden_size x batch).
    The bytes-per-unit constant is fitted from the measured `smoke` run rather
    than assumed, in `memory_model()` below.
    """
    return arch["num_layers"] * arch["seq_length"] * arch["hidden_size"] * micro_batch


def memory_model(param_count, arch: dict, calibration: dict) -> dict:
    """Predict peak allocated memory, decomposed into terms that scale differently."""
    counts = param_count(arch, VOCAB_SIZE)
    static = counts["total"] * 18          # 18 B/param: BF16 w + FP32 grad/master/m/v
    logits = logits_bytes(arch)
    other = calibration["bytes_per_activation_unit"] * activation_units(arch)
    return dict(
        params=counts["total"],
        static=static,
        logits=logits,
        other=other,
        predicted_peak=static + logits + other,
        naive_peak=static * calibration["naive_ratio"],
    )


def calibrate_memory(param_count) -> dict:
    """Fit the one free memory constant against the measured `smoke` run."""
    smoke = PROFILES["smoke"]
    counts = param_count(smoke, VOCAB_SIZE)
    static = counts["total"] * 18
    measured = SMOKE_MEASURED["peak_allocated_gib"] * GIB
    residual = measured - static - logits_bytes(smoke)
    return dict(
        bytes_per_activation_unit=residual / activation_units(smoke),
        naive_ratio=measured / static,
        smoke_static=static,
        smoke_residual=residual,
    )


def calibrate_two_point(param_count) -> dict:
    """Solve the residual's constant and per-unit terms from smoke AND wider.

    Two equations, two unknowns -- an EXACT solve with zero degrees of freedom.
    This is a reparametrisation, not a validated fit: it cannot fail, so it
    must not be reported as confirmation. Its value is that it disagrees with
    the smoke-only calibration about `deeper`, which makes `deeper` a test.

    The two terms are separable because they scale differently in the
    architecture: the logits term is constant in num_layers and hidden_size
    while the activation term is linear in their product.
    """
    s, w = PROFILES["smoke"], PROFILES["wider"]
    us, uw = activation_units(s), activation_units(w)
    rs = (SMOKE_MEASURED["peak_allocated_gib"] * GIB
          - param_count(s, VOCAB_SIZE)["total"] * 18)
    rw = (WIDER_MEASURED["peak_allocated_gib"] * GIB
          - param_count(w, VOCAB_SIZE)["total"] * 18)
    if uw == us:
        raise SystemExit("FATAL: smoke and wider have equal activation units; "
                         "the two terms are not separable from this pair.")
    k = (rw - rs) / (uw - us)
    const = rs - k * us
    # Both reference runs use seq_length 1024, so `const` is the logits term at
    # that sequence length. Express it as a fraction of a full FP32 copy so it
    # can be re-scaled for a profile with a different seq_length.
    return dict(
        bytes_per_activation_unit=k,
        logits_fraction=const / logits_bytes(s),
        constant_at_seq1024=const,
    )


def memory_model_two_point(param_count, arch: dict, cal2: dict) -> dict:
    """Predict peak memory using the two-point calibration."""
    counts = param_count(arch, VOCAB_SIZE)
    static = counts["total"] * 18
    logits = cal2["logits_fraction"] * logits_bytes(arch)
    other = cal2["bytes_per_activation_unit"] * activation_units(arch)
    return dict(
        params=counts["total"], static=static, logits=logits, other=other,
        predicted_peak=static + logits + other,
    )


def anchored_prediction(param_count, target: str, anchor: str = "wider") -> dict:
    """Predict `target`'s peak by extrapolating only the num_layers delta.

    Stronger than either calibration for a profile that differs from a MEASURED
    profile in num_layers alone. Starting from the anchor's measured residual
    and adding only `k * delta_units` means:

      * the static term is exact (analytic, from param_count)
      * the logits term CANCELS, because both have the same seq_length -- so
        the prediction carries no assumption about it at all
      * the only extrapolated quantity is the per-layer constant k

    Conversely, the measurement inverts to give k directly:
        k = (residual_target - residual_anchor) / (units_target - units_anchor)
    which is the first clean measurement of k this repo can make.
    """
    measured = {"smoke": SMOKE_MEASURED, "wider": WIDER_MEASURED,
                "deeper": DEEPER_MEASURED}
    if anchor not in measured:
        raise SystemExit(f"FATAL: anchor {anchor!r} has no measurement")
    a, t = PROFILES[anchor], PROFILES[target]
    diffs = sorted(k for k in t if a[k] != t[k])
    if diffs != ["num_layers"]:
        raise SystemExit(
            f"FATAL: {target} differs from {anchor} in {diffs}, not num_layers "
            "alone, so the logits term does not cancel and this anchoring is "
            "invalid. Use the ordinary report instead."
        )
    resid_anchor = (measured[anchor]["peak_allocated_gib"] * GIB
                    - param_count(a, VOCAB_SIZE)["total"] * 18)
    d_units = activation_units(t) - activation_units(a)
    static = param_count(t, VOCAB_SIZE)["total"] * 18
    # The two per-layer constants the earlier pair-fits disagree about.
    candidates = {
        "k=24.81 B/unit (smoke+deeper fit)": 24.81,
        "k=80.17 B/unit (smoke+wider fit)": 80.17,
    }
    out = {}
    for label, k in candidates.items():
        out[label] = static + resid_anchor + k * d_units
    return dict(static=static, resid_anchor=resid_anchor, d_units=d_units,
                predictions=out, anchor=anchor)


def flops_per_step(param_count, arch: dict) -> dict:
    """Model FLOPs for one optimizer step.

    Dense term: 6 x tokens x (layer params + LM head), the standard
    forward+backward multiplier. The embedding lookup is a gather, not a
    matmul, so only the LM head half of `embedding_and_head` counts.

    Attention term: 6 x batch x seq^2 x hidden x layers. That is the usual
    12 x b x s^2 x h halved for causal masking, which is what makes this
    reproduce the measured `smoke` figure to 0.04% (see --self-check).
    """
    counts = param_count(arch, VOCAB_SIZE)
    lm_head = counts["embedding_and_head"] // 2
    tokens = GLOBAL_BATCH_SIZE * arch["seq_length"]
    dense = 6 * tokens * (counts["layers_total"] + lm_head)
    attn = (6 * GLOBAL_BATCH_SIZE * arch["seq_length"] ** 2
            * arch["hidden_size"] * arch["num_layers"])
    return dict(dense=dense, attn=attn, total=dense + attn, tokens=tokens)


def self_check(param_count) -> int:
    """Prove the lifted formula and both calibrated models are sound."""
    failures = []

    def check(name, got, want, tol, unit=""):
        ok = abs(got - want) <= tol
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} "
              f"got {got:>10,.4f}{unit}  want {want:>10,.4f}{unit}  (+/-{tol:g})")
        if not ok:
            failures.append(name)

    print("--- lifted param_count() reproduces known architectures ---")
    check("Qwen3-8B total params",
          param_count(QWEN3_8B, VOCAB_SIZE)["total"] / 1e9, 8.190, 0.005, " B")
    check("smoke total params",
          param_count(PROFILES["smoke"], VOCAB_SIZE)["total"] / 1e6, 359.4, 0.1, " M")
    check("wider total params",
          param_count(PROFILES["wider"], VOCAB_SIZE)["total"] / 1e6, 629.5, 0.1, " M")
    check("deeper total params",
          param_count(PROFILES["deeper"], VOCAB_SIZE)["total"] / 1e6, 455.9, 0.1, " M")
    check("1b total params",
          param_count(PROFILES["1b"], VOCAB_SIZE)["total"] / 1e9, 1.0094, 0.0005, " B")

    print("\n--- 1b must be a single-variable change from wider ---")
    # This is the property that makes the pair able to measure the per-layer
    # activation constant with the logits term cancelled. If a future edit
    # changes any other field, that capability is silently lost.
    _diffs = sorted(k for k in PROFILES["1b"]
                    if PROFILES["wider"][k] != PROFILES["1b"][k])
    _ok = _diffs == ["num_layers"]
    print(f"  [{'PASS' if _ok else 'FAIL'}] only num_layers differs from wider"
          f"{'':<10} got {_diffs}")
    if not _ok:
        failures.append("1b is not a single-variable change from wider")

    print("\n--- FLOP model reproduces the MEASURED smoke run ---")
    measured = SMOKE_MEASURED["model_tflop_s"] * 1e12 * SMOKE_MEASURED["step_time_s"]
    modelled = flops_per_step(param_count, PROFILES["smoke"])["total"]
    check("smoke FLOPs/step", modelled / 1e12, measured / 1e12,
          0.01 * measured / 1e12, " TF")

    print("\n--- memory model round-trips the smoke measurement ---")
    cal = calibrate_memory(param_count)
    rt = memory_model(param_count, PROFILES["smoke"], cal)
    check("smoke predicted peak", rt["predicted_peak"] / GIB,
          SMOKE_MEASURED["peak_allocated_gib"], 0.01, " GiB")

    print("\n--- two-point solve round-trips BOTH measured points ---")
    # By construction it must hit both exactly. If it does not, the solve is
    # wrong. This is the only check the exact solve can actually fail.
    cal2 = calibrate_two_point(param_count)
    for name, measured in (("smoke", SMOKE_MEASURED), ("wider", WIDER_MEASURED)):
        got = memory_model_two_point(param_count, PROFILES[name], cal2)
        check(f"two-point {name} peak", got["predicted_peak"] / GIB,
              measured["peak_allocated_gib"], 0.01, " GiB")
    check("two-point logits fraction", cal2["logits_fraction"], 0.8660, 0.0005)

    print("\n--- geometry constraints Megatron enforces ---")
    for name, arch in PROFILES.items():
        hd = arch["hidden_size"] // arch["num_attention_heads"]
        ok = (arch["hidden_size"] % arch["num_attention_heads"] == 0
              and arch["num_attention_heads"] % arch["num_query_groups"] == 0
              and hd == 128)
        gqa = arch["num_attention_heads"] // arch["num_query_groups"]
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<8} head_dim={hd:<5} GQA={gqa}:1"
              f"   h%heads={arch['hidden_size'] % arch['num_attention_heads']}"
              f"   heads%qg={arch['num_attention_heads'] % arch['num_query_groups']}")
        if not ok:
            failures.append(f"{name} geometry")

    print()
    if failures:
        print(f"SELF-CHECK FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("SELF-CHECK PASSED")
    return 0


def report(param_count, target: str) -> None:
    """Print the pre-run prediction for one profile, with refutation thresholds."""
    cal = calibrate_memory(param_count)
    arch = PROFILES[target]
    mem = memory_model(param_count, arch, cal)
    flops = flops_per_step(param_count, arch)
    smoke_flops = flops_per_step(param_count, PROFILES["smoke"])["total"]

    # Throughput: the only free variable is achieved TFLOP/s. The measured
    # smoke value is the floor, because wider's GEMMs are strictly larger
    # (hidden 1024 -> 1536) and larger GEMMs do not run less efficiently.
    floor_tflops = SMOKE_MEASURED["model_tflop_s"]
    band = [("floor (= smoke efficiency)", floor_tflops),
            ("central (modest GEMM gain)", 35.0),
            ("optimistic", 38.0)]

    print("=" * 74)
    print(f"PRE-RUN PREDICTION: '{target}' profile on g5.8xlarge / A10G")
    print("=" * 74)
    print(f"  calibrated against : the measured 'smoke' run "
          f"({SMOKE_MEASURED['peak_allocated_gib']} GiB, "
          f"{SMOKE_MEASURED['tokens_per_s']:,} tok/s)")
    print(f"  fitted constant    : {cal['bytes_per_activation_unit']:.2f} B per "
          f"(layer x seq x hidden x batch)")
    print()
    print("  --- geometry (must appear in the run log, else override was dropped) ---")
    for key in ("num_layers", "hidden_size", "ffn_hidden_size",
                "num_attention_heads", "num_query_groups", "seq_length"):
        print(f"  {key:<22}: {arch[key]}")
    print(f"  {'TOTAL params':<22}: {mem['params'] / 1e6:.1f} M"
          f"   <-- the observable that proves the profile took effect")
    print()
    print("  --- predicted peak allocated memory ---")
    print(f"  static @18 B/param    : {mem['static'] / GIB:>7.2f} GiB")
    print(f"  FP32 vocab logits     : {mem['logits'] / GIB:>7.2f} GiB  "
          f"(constant in layers/hidden)")
    print(f"  other activations     : {mem['other'] / GIB:>7.2f} GiB  "
          f"(scales layers x seq x hidden)")
    print(f"  PREDICTED PEAK        : {mem['predicted_peak'] / GIB:>7.2f} GiB  "
          f"({100 * mem['predicted_peak'] / A10G_TOTAL_BYTES:.1f}% of 22.49 GiB card)")
    print(f"  (README +14% rule     : {mem['naive_peak'] / GIB:>7.2f} GiB  "
          f"<-- competing prediction, see below)")

    # Once smoke AND wider are measured, a second calibration exists and
    # disagrees. For any profile other than those two, that disagreement is a
    # live test -- so show it.
    cal2 = calibrate_two_point(param_count)
    m2 = memory_model_two_point(param_count, arch, cal2)
    if target not in ("smoke", "wider"):
        print()
        print("  --- competing calibration: two-point (smoke + wider) ---")
        print(f"  per-unit constant     : {cal2['bytes_per_activation_unit']:.2f} B "
              f"(vs {cal['bytes_per_activation_unit']:.2f} B from smoke alone)")
        print(f"  logits fraction       : {cal2['logits_fraction']:.4f} of a full FP32 copy "
              f"(vs 1.0000 assumed)")
        print(f"  static @18 B/param    : {m2['static'] / GIB:>7.2f} GiB")
        print(f"  logits term           : {m2['logits'] / GIB:>7.2f} GiB")
        print(f"  other activations     : {m2['other'] / GIB:>7.2f} GiB")
        print(f"  PREDICTED PEAK (2pt)  : {m2['predicted_peak'] / GIB:>7.2f} GiB  "
              f"({100 * m2['predicted_peak'] / A10G_TOTAL_BYTES:.1f}% of card)")
        print(f"  DISCRIMINATING GAP    : "
              f"{abs(m2['predicted_peak'] - mem['predicted_peak']) / GIB:>7.2f} GiB")

    # A profile differing from a MEASURED profile in num_layers alone gets a
    # stronger prediction: anchor on that profile's measured residual and
    # extrapolate only the layer delta, which cancels the logits term entirely.
    anchored = None
    for _anchor in ("wider", "smoke", "deeper"):
        if target == _anchor:
            continue
        _d = sorted(k for k in arch if PROFILES[_anchor][k] != arch[k])
        if _d == ["num_layers"]:
            anchored = anchored_prediction(param_count, target, _anchor)
            break
    if anchored:
        print()
        print(f"  --- ANCHORED on measured `{anchored['anchor']}` "
              f"(single-variable: num_layers only) ---")
        print(f"  This is the STRONGEST prediction available for this profile.")
        print(f"  Both share seq_length, so the FP32 logits term cancels exactly")
        print(f"  in the difference and is not assumed at all.")
        print(f"  {anchored['anchor']} measured residual : "
              f"{anchored['resid_anchor'] / GIB:>7.4f} GiB")
        print(f"  delta activation units : {anchored['d_units']:>12,}")
        for label, peak in sorted(anchored["predictions"].items(),
                                  key=lambda kv: kv[1]):
            print(f"  {label:<36} -> {peak / GIB:>6.2f} GiB "
                  f"({100 * peak / A10G_TOTAL_BYTES:.1f}% of card)")
        _vals = sorted(anchored["predictions"].values())
        print(f"  DISCRIMINATING GAP     : {(_vals[-1] - _vals[0]) / GIB:>7.2f} GiB")
        print(f"  The measurement INVERTS to give the per-layer constant:")
        print(f"    k = (residual - {anchored['resid_anchor'] / GIB:.4f} GiB) "
              f"/ {anchored['d_units']:,} units")

    print()
    print("  --- predicted throughput ---")
    print(f"  tokens/step           : {flops['tokens']:,} "
          f"(unchanged: GBS {GLOBAL_BATCH_SIZE} x seq {arch['seq_length']})")
    print(f"  model FLOPs/step      : {flops['total'] / 1e12:.2f} TFLOP "
          f"({flops['total'] / smoke_flops:.3f}x smoke)")
    for label, tf in band:
        step = flops["total"] / (tf * 1e12)
        print(f"  @ {tf:>4.1f} TFLOP/s {label:<26}: "
              f"{step:.3f} s/step  {flops['tokens'] / step:>7,.0f} tok/s")
    print()
    print("  --- REFUTATION THRESHOLDS (recorded before the run) ---")
    # Pick the LIVE competing hypothesis. For smoke/wider that was the flat
    # "+14% of static" rule. Once wider refuted it, the open question became
    # which CALIBRATION of the decomposed model holds -- smoke-only or the
    # two-point solve. Testing an already-refuted rule would waste the run.
    if target not in ("smoke", "wider"):
        competitor_name = "two-point calibration (smoke + wider)"
        competitor = m2["predicted_peak"]
        print(f"  NOTE: the flat +14% rule predicts {mem['naive_peak'] / GIB:.2f} GiB here, but it was")
        print(f"        already REFUTED by the wider run; it is not the hypothesis under test.")
    else:
        competitor_name = "flat +14% of static rule"
        competitor = mem["naive_peak"]
    # The tolerance must be strictly under half the gap or the bands overlap
    # and the run cannot discriminate between them. That would defeat the point.
    gap = abs(competitor - mem["predicted_peak"]) / GIB
    tol = min(0.15, 0.45 * gap)
    lo, hi = mem["predicted_peak"] / GIB - tol, mem["predicted_peak"] / GIB + tol
    nlo, nhi = competitor / GIB - tol, competitor / GIB + tol
    print(f"  competing predictions are {gap:.2f} GiB apart; "
          f"tolerance +/-{tol:.2f} GiB keeps the bands disjoint")
    print(f"  R1 peak allocated inside [{lo:.2f}, {hi:.2f}] GiB")
    print(f"     -> CONFIRMS the smoke-only calibration "
          f"({mem['predicted_peak'] / GIB:.2f} GiB: full FP32 logits, 60.29 B/unit).")
    print(f"  R2 peak allocated inside [{nlo:.2f}, {nhi:.2f}] GiB")
    print(f"     -> CONFIRMS the {competitor_name} ({competitor / GIB:.2f} GiB).")
    print(f"  R2b peak outside BOTH bands")
    print(f"     -> both wrong; an unmodelled term dominates.")


    t_floor = flops["tokens"] / (flops["total"] / (floor_tflops * 1e12))
    t_opt = flops["tokens"] / (flops["total"] / (38.0 * 1e12))
    print(f"  R3 median tok/s outside [{t_floor * 0.95:,.0f}, {t_opt * 1.05:,.0f}]")
    print(f"     -> the FLOP model or the efficiency assumption is wrong.")
    print(f"  R4 MODEL_TFLOP/s < {floor_tflops} (below smoke)")
    print(f"     -> refutes 'bigger GEMMs are at least as efficient'.")
    print(f"  R5 OOM at a predicted {100 * mem['predicted_peak'] / A10G_TOTAL_BYTES:.0f}% of card")
    print(f"     -> an unmodelled allocation dominates; 18 B/param is not a floor.")
    print(f"  R6 log reports TOTAL params != {mem['params'] / 1e6:.1f} M")
    print(f"     -> the env override was DROPPED; the run is void, not a result.")
    print(f"  R7 any skipped or NaN iteration")
    print(f"     -> BF16 instability at this width on sm_86.")
    print("=" * 74)


def score(param_count) -> int:
    """Grade the measured `wider` run against the thresholds committed before it.

    Returns 0 if the run is valid (R6/R7 clean), regardless of which memory
    model won -- a refuted prediction is a result, not an error.
    """
    cal = calibrate_memory(param_count)
    arch = PROFILES["wider"]
    mem = memory_model(param_count, arch, cal)
    flops = flops_per_step(param_count, arch)
    m = WIDER_MEASURED

    gap = abs(mem["naive_peak"] - mem["predicted_peak"]) / GIB
    tol = min(0.15, 0.45 * gap)
    r1 = (mem["predicted_peak"] / GIB - tol, mem["predicted_peak"] / GIB + tol)
    r2 = (mem["naive_peak"] / GIB - tol, mem["naive_peak"] / GIB + tol)
    peak = m["peak_allocated_gib"]

    print("=" * 74)
    print("SCORED: 'wider' measurement vs thresholds committed BEFORE the run")
    print("=" * 74)

    print("\n  --- validity gates (these decide whether the run counts at all) ---")
    r6_ok = abs(m["params_m"] - mem["params"] / 1e6) < 0.1
    print(f"  R6 TOTAL params      : {m['params_m']} M vs expected "
          f"{mem['params'] / 1e6:.1f} M -> {'VALID' if r6_ok else 'VOID (override dropped)'}")
    r7_ok = m["skipped_iterations"] == 0 and m["nan_iterations"] == 0
    print(f"  R7 skipped / NaN     : {m['skipped_iterations']} / {m['nan_iterations']}"
          f" -> {'PASS, no BF16 instability' if r7_ok else 'FAIL'}")
    if not r6_ok:
        print("\n  Run is VOID. Nothing below is a wider measurement.")
        return 1

    print("\n  --- memory: two competing models, disjoint bands ---")
    in_r1 = r1[0] <= peak <= r1[1]
    in_r2 = r2[0] <= peak <= r2[1]
    print(f"  measured peak        : {peak:.2f} GiB")
    print(f"  R1 decomposed model  : predicted {mem['predicted_peak'] / GIB:.2f}, "
          f"band [{r1[0]:.2f}, {r1[1]:.2f}] -> {'HIT' if in_r1 else 'miss'}")
    print(f"  R2 flat +14% rule    : predicted {mem['naive_peak'] / GIB:.2f}, "
          f"band [{r2[0]:.2f}, {r2[1]:.2f}] -> {'HIT' if in_r2 else 'miss'}")
    for label, pred in (("decomposed", mem["predicted_peak"] / GIB),
                        ("flat +14%", mem["naive_peak"] / GIB)):
        err = peak - pred
        print(f"    {label:<12} error     : {err:+.2f} GiB ({100 * err / peak:+.2f}%)")
    if in_r1 and not in_r2:
        print("  VERDICT: at wider, the decomposed prediction beat the flat +14% rule,")
        print("           which is REFUTED. But see --form-test: the decomposed FORM")
        print("           is itself refuted by the deeper run (9.66 measured vs 10.21")
        print("           and 10.52 predicted). Agreeing here was interpolation luck,")
        print("           not validation.")
    elif in_r2 and not in_r1:
        print("  VERDICT: flat +14% rule CONFIRMED; decomposition REFUTED.")
    else:
        print("  VERDICT: R2b -- outside both bands; an unmodelled term dominates.")

    print("\n  --- FLOP model: calibrated on smoke, never refitted ---")
    implied = m["model_tflop_s"] * 1e12 * m["step_time_s"]
    err = 100 * (flops["total"] - implied) / implied
    print(f"  predicted FLOPs/step : {flops['total'] / 1e12:.2f} TFLOP")
    print(f"  implied by measurement: {implied / 1e12:.2f} TFLOP "
          f"({m['model_tflop_s']} TFLOP/s x {m['step_time_s']} s)")
    print(f"  error                : {err:+.2f}%  -> "
          f"{'CONFIRMED on a second architecture' if abs(err) < 1 else 'REFUTED'}")

    print("\n  --- throughput ---")
    t_floor = flops["tokens"] / (flops["total"] / (SMOKE_MEASURED["model_tflop_s"] * 1e12))
    t_opt = flops["tokens"] / (flops["total"] / (38.0 * 1e12))
    lo, hi = t_floor * 0.95, t_opt * 1.05
    in_r3 = lo <= m["tokens_per_s"] <= hi
    central = flops["tokens"] / (flops["total"] / (35.0 * 1e12))
    print(f"  measured median      : {m['tokens_per_s']:,} tok/s @ {m['step_time_s']} s/step")
    print(f"  R3 band              : [{lo:,.0f}, {hi:,.0f}] -> {'HIT' if in_r3 else 'MISS'}")
    print(f"  central prediction   : {central:,.0f} tok/s @ 35.0 TFLOP/s assumed")
    print(f"    error vs central   : {m['tokens_per_s'] - central:+,.0f} tok/s "
          f"({100 * (m['tokens_per_s'] - central) / m['tokens_per_s']:+.2f}%)")
    r4_ok = m["model_tflop_s"] >= SMOKE_MEASURED["model_tflop_s"]
    print(f"  R4 efficiency floor  : {m['model_tflop_s']} vs smoke "
          f"{SMOKE_MEASURED['model_tflop_s']} TFLOP/s -> "
          f"{'PASS, bigger GEMMs no less efficient' if r4_ok else 'REFUTED'}")
    print(f"    efficiency gain    : {100 * (m['model_tflop_s'] / SMOKE_MEASURED['model_tflop_s'] - 1):+.1f}%")
    print(f"    MFU               : {100 * m['model_tflop_s'] / A10G_BF16_PEAK_TFLOPS:.1f}% "
          f"(smoke: {100 * SMOKE_MEASURED['model_tflop_s'] / A10G_BF16_PEAK_TFLOPS:.1f}%)")
    print(f"  vs smoke throughput  : {m['tokens_per_s'] / SMOKE_MEASURED['tokens_per_s']:.4f}x")

    print("\n  --- two-point separation of the residual terms ---")
    print("  With smoke AND wider, the constant term and the per-layer term can be")
    print("  solved exactly. NOTE: 2 equations, 2 unknowns -- zero degrees of")
    print("  freedom, so this is a REPARAMETRISATION, not a validated fit.")
    s, w = PROFILES["smoke"], PROFILES["wider"]
    us, uw = activation_units(s), activation_units(w)
    rs = SMOKE_MEASURED["peak_allocated_gib"] * GIB - param_count(s, VOCAB_SIZE)["total"] * 18
    rw = m["peak_allocated_gib"] * GIB - param_count(w, VOCAB_SIZE)["total"] * 18
    k = (rw - rs) / (uw - us)
    const = rs - k * us
    print(f"  smoke residual       : {rs / GIB:.4f} GiB over {us:,} units")
    print(f"  wider residual       : {rw / GIB:.4f} GiB over {uw:,} units")
    print(f"  solved per-unit      : {k:.2f} B per (layer x seq x hidden x batch)")
    print(f"  solved constant      : {const / GIB:.4f} GiB")
    print(f"  FP32 logits theory   : {logits_bytes(w) / GIB:.4f} GiB "
          f"({100 * (const - logits_bytes(w)) / logits_bytes(w):+.1f}% vs solved)")
    print("=" * 74)
    return 0 if (r6_ok and r7_ok) else 1


def form_test(param_count) -> int:
    """Test whether a 2-parameter linear residual model fits all THREE points.

    `wider` appeared to confirm the decomposed model. `deeper` landed outside
    both predicted bands, which moves the question from "which constants" to
    "is the FORM right". A model of the shape

        residual = alpha * logits_bytes(seq) + beta * (layers * seq * hidden)

    has two free parameters, so any two points fix it exactly. If the form were
    right, the fit from any two points would predict the third. This checks all
    three leave-one-out fits.
    """
    measured = {
        "smoke": SMOKE_MEASURED["peak_allocated_gib"],
        "wider": WIDER_MEASURED["peak_allocated_gib"],
        "deeper": DEEPER_MEASURED["peak_allocated_gib"],
    }
    pts = {}
    print("=" * 74)
    print("FORM TEST: does one 2-parameter model fit all three measured points?")
    print("=" * 74)
    print("\n  --- measured residuals over static @18 B/param ---")
    for name, peak in measured.items():
        arch = PROFILES[name]
        static = param_count(arch, VOCAB_SIZE)["total"] * 18
        pts[name] = dict(arch=arch, peak=peak, static=static,
                         resid=peak * GIB - static,
                         L=logits_bytes(arch), u=activation_units(arch))
        p = pts[name]
        print(f"  {name:<7} seq={arch['seq_length']:<5} layers={arch['num_layers']:<3} "
              f"hidden={arch['hidden_size']:<5} | static {static / GIB:6.3f} "
              f"resid {p['resid'] / GIB:6.4f} GiB")

    def solve(a, b):
        A, B = pts[a], pts[b]
        det = A["L"] * B["u"] - B["L"] * A["u"]
        if det == 0:
            return None
        return ((A["resid"] * B["u"] - B["resid"] * A["u"]) / det,
                (A["L"] * B["resid"] - B["L"] * A["resid"]) / det)

    print("\n  --- leave-one-out: fit on two points, predict the third ---")
    names = list(pts)
    worst, negative_beta = 0.0, False
    for held in names:
        pair = [n for n in names if n != held]
        sol = solve(*pair)
        if sol is None:
            print(f"  fit on {pair} is singular")
            continue
        alpha, beta = sol
        H = pts[held]
        pred = alpha * H["L"] + beta * H["u"]
        err = (pred - H["resid"]) / GIB
        worst = max(worst, abs(err))
        if beta < 0:
            negative_beta = True
        flag = "  <-- NEGATIVE, physically impossible" if beta < 0 else ""
        print(f"  fit {pair[0]:<6}+{pair[1]:<7} alpha={alpha:7.4f} "
              f"beta={beta:7.2f} B/unit{flag}")
        print(f"      predicts {held:<7} {pred / GIB:6.4f} GiB vs measured "
              f"{H['resid'] / GIB:6.4f}  ERROR {err:+.4f} GiB "
              f"({100 * err / (H['resid'] / GIB):+.1f}%)")

    print(f"\n  worst leave-one-out error : {worst:.4f} GiB")
    if negative_beta:
        print("  one fit needs a NEGATIVE bytes-per-activation-unit, which no")
        print("  physical allocation can have. That alone refutes the form.")

    print("\n  --- why the form was never testable from these profiles ---")
    for a, b in (("smoke", "wider"), ("smoke", "deeper"), ("wider", "deeper")):
        A, B = pts[a]["arch"], pts[b]["arch"]
        diffs = [k for k in ("num_layers", "hidden_size", "seq_length")
                 if A[k] != B[k]]
        print(f"  {a:<7} -> {b:<7} co-varies {len(diffs)}: "
              + ", ".join(f"{k} {A[k]}->{B[k]}" for k in diffs))
    print("  No pair is a single-variable change, so a model fitted to one pair")
    print("  has no basis for extrapolating. The apparent success on `wider`")
    print("  was interpolation luck, not a validated form.")

    print("\n  --- what IS validated: the FLOP model ---")
    for name, m in (("smoke", SMOKE_MEASURED), ("wider", WIDER_MEASURED),
                    ("deeper", DEEPER_MEASURED)):
        pred = flops_per_step(param_count, PROFILES[name])["total"]
        implied = m["model_tflop_s"] * 1e12 * m["step_time_s"]
        print(f"  {name:<7} predicted {pred / 1e12:6.2f} TFLOP  "
              f"implied {implied / 1e12:6.2f}  err {100 * (pred - implied) / implied:+6.2f}%")
    print("  Calibrated on smoke alone and never refitted, it holds to within")
    print("  0.04% across changes in layers, hidden_size AND seq_length.")
    print("=" * 74)
    # Non-zero: the memory form IS refuted, and this should be visible in CI.
    return 1 if worst > 0.35 else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true",
                    help="verify the lifted formula and calibrated models")
    ap.add_argument("--score", action="store_true",
                    help="grade the measured wider run against committed thresholds")
    ap.add_argument("--form-test", action="store_true",
                    help="test whether one 2-parameter memory model fits all "
                         "three measured points (it does not)")
    ap.add_argument("--profile", default="wider", choices=sorted(PROFILES),
                    help="profile to predict (default: wider)")
    args = ap.parse_args()

    param_count = load_param_count()
    if args.self_check:
        return self_check(param_count)
    if args.score:
        return score(param_count)
    if args.form_test:
        return form_test(param_count)
    report(param_count, args.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
