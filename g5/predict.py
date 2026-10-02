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
    # The two memory models differ by only (naive - predicted) GiB, so the
    # tolerance must be strictly under half that gap or the bands overlap and
    # the run cannot discriminate between them. That would defeat the point.
    gap = abs(mem["naive_peak"] - mem["predicted_peak"]) / GIB
    tol = min(0.15, 0.45 * gap)
    lo, hi = mem["predicted_peak"] / GIB - tol, mem["predicted_peak"] / GIB + tol
    nlo, nhi = mem["naive_peak"] / GIB - tol, mem["naive_peak"] / GIB + tol
    print(f"  competing predictions are {gap:.2f} GiB apart; "
          f"tolerance +/-{tol:.2f} GiB keeps the bands disjoint")
    print(f"  R1 peak allocated inside [{lo:.2f}, {hi:.2f}] GiB")
    print(f"     -> CONFIRMS the decomposed model (logits constant in layers).")
    print(f"  R2 peak allocated inside [{nlo:.2f}, {nhi:.2f}] GiB")
    print(f"     -> REFUTES the decomposition; the flat +14% rule wins.")
    print(f"  R2b peak outside BOTH bands")
    print(f"     -> both models wrong; an unmodelled term dominates.")

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true",
                    help="verify the lifted formula and calibrated models")
    ap.add_argument("--profile", default="wider", choices=sorted(PROFILES),
                    help="profile to predict (default: wider)")
    args = ap.parse_args()

    param_count = load_param_count()
    if args.self_check:
        return self_check(param_count)
    report(param_count, args.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
