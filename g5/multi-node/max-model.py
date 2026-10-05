#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""How deep and how wide can the 2-node scenario go before it OOMs?

WHY THIS EXISTS
`g5/predict.py` carries the calibrated memory model, but every profile in it is
single-node: it has no notion of data parallelism, so it cannot answer "what
fits on two nodes". Two nodes is not just twice the memory -- per-rank memory
actually DROPS, because `g5/train.py` enables `use_distributed_optimizer` when
WORLD_SIZE > 1 and that shards optimizer state across the data-parallel group.

This imports predict.py rather than restating any of it, so the parameter
formula and the three measured activation constants cannot drift from the
validated copy. Nothing here re-derives the model; it only adds the DP term and
searches.

WHAT IS MEASURED AND WHAT IS NOT
  MEASURED  the three-term activation model, fitted over four single-node runs
            (worst error 0.0787 GiB), and the 19.76 GiB ceiling on peak
            allocated, derived from the `1b2k` OOM
  MEASURED  that 2 nodes over EFA runs at all: 25,056 tok/s on the 4-layer
            `smoke` shape, which peaked at 4.83 GiB
  NOT       any 2-node MEMORY reading. The DP sharding term below is arithmetic
            on top of a single-node-calibrated model. Every number this script
            prints for DP=2 is a PREDICTION.

HOW THE 18 B/param SPLITS, which is the whole DP question
  BF16 weights          2 B/param   replicated -- every rank needs them to run
  FP32 gradients        4 B/param   reduce-scattered by overlap_grad_reduce
  FP32 master weights   4 B/param   sharded by the distributed optimizer
  FP32 Adam m           4 B/param   sharded
  FP32 Adam v           4 B/param   sharded
                       18 B/param

So at DP=N the sharded part is divided by N. Two readings of "sharded":

  conservative  only master+m+v shard (12 B/param -> 12/N)
                DP=2 gives 2 + 4 + 6 = 12 B/param
  optimistic    the FP32 gradient buffer shards too (16 B/param -> 16/N)
                DP=2 gives 2 + 2 + 6 = 10 B/param

The conservative figure is the headline here. Megatron's distributed optimizer
is ZeRO-1-shaped, where the gradient buffer's residency depends on the
reduce-scatter schedule, and this script is not the place to guess it. Both are
printed so the gap is visible: it is ~1.9 GiB at 1B parameters, which is the
width of the uncertainty, not a rounding error.

GEOMETRY CONSTRAINTS Megatron actually enforces, taken from the shapes that
ran: head_dim is 128, so heads = hidden/128; GQA is 4:1, so query_groups =
heads/4; and ffn = 3 x hidden. query_groups must be a whole number, which makes
hidden a multiple of 512. The search only visits shapes satisfying all of that,
because a shape that violates them does not OOM -- it refuses to start.

Usage:
  python3 g5/multi-node/max-model.py              # DP=2, seq 1024
  python3 g5/multi-node/max-model.py --seq 2048
  python3 g5/multi-node/max-model.py --dp 1       # reproduce the single-node ceiling
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import predict  # noqa: E402  -- path set above

GIB = predict.GIB
CEILING = predict.A10G_ALLOC_CEILING_BYTES

# The 18 B/param decomposition. Sums to 18 by assertion below, so a future edit
# cannot silently change the total.
B_WEIGHTS = 2      # BF16, replicated
B_GRAD = 4         # FP32
B_MASTER = 4       # FP32
B_ADAM_M = 4       # FP32
B_ADAM_V = 4       # FP32
assert B_WEIGHTS + B_GRAD + B_MASTER + B_ADAM_M + B_ADAM_V == 18, \
    "the decomposition must sum to predict.py's 18 B/param"

SHARDED_CONSERVATIVE = B_MASTER + B_ADAM_M + B_ADAM_V      # 12
SHARDED_OPTIMISTIC = SHARDED_CONSERVATIVE + B_GRAD          # 16

# Taken from predict.py's own QWEN3_8B rather than written down, so it cannot
# drift from the architecture the proxies are proxies FOR.
QWEN3_RATIO = (predict.QWEN3_8B["hidden_size"]
               / predict.QWEN3_8B["num_layers"])


def bytes_per_param(dp: int, optimistic: bool = False) -> float:
    """Static bytes per parameter per rank at data-parallel degree `dp`."""
    sharded = SHARDED_OPTIMISTIC if optimistic else SHARDED_CONSERVATIVE
    replicated = 18 - sharded
    return replicated + sharded / dp


def arch_for(layers: int, hidden: int, seq: int) -> dict:
    """A geometry obeying the constraints the measured profiles all satisfy."""
    return dict(num_layers=layers, hidden_size=hidden,
                ffn_hidden_size=3 * hidden,
                num_attention_heads=hidden // 128,
                num_query_groups=hidden // 512,
                seq_length=seq)


def valid(arch: dict) -> bool:
    """Reject shapes Megatron refuses to start on, so OOM is the only failure."""
    h, heads, qg = arch["hidden_size"], arch["num_attention_heads"], arch["num_query_groups"]
    return (heads > 0 and qg > 0
            and h % heads == 0
            and heads % qg == 0
            and h // heads == 128)


def peak_bytes(param_count, arch: dict, dp: int, micro_batch: int,
               optimistic: bool = False) -> dict:
    """Per-rank peak allocated, measured activation model + DP-sharded static."""
    counts = param_count(arch, predict.VOCAB_SIZE)
    static = counts["total"] * bytes_per_param(dp, optimistic)
    # Activations are per-rank and depend on micro_batch and seq, NOT on the
    # global batch size: gradient accumulation does not raise peak activations.
    # So they are unchanged by DP at fixed micro_batch -- which is why sharding
    # the optimizer is pure gain.
    fixed = predict.THREE_TERM["c0_bytes"]
    per_token = predict.THREE_TERM["c1_bytes_per_token"] * arch["seq_length"]
    per_layer = (predict.THREE_TERM["k_bytes_per_unit"]
                 * predict.activation_units(arch, micro_batch))
    peak = static + fixed + per_token + per_layer
    return dict(params=counts["total"], static=static, peak=peak,
                acts=fixed + per_token + per_layer,
                fits=peak <= CEILING, headroom=CEILING - peak)


def deepest(param_count, hidden: int, seq: int, dp: int, micro_batch: int):
    """Most layers that fit at this width. Memory is monotonic in layers."""
    best = None
    for layers in range(1, 513):
        a = arch_for(layers, hidden, seq)
        if not valid(a):
            return None
        m = peak_bytes(param_count, a, dp, micro_batch)
        if not m["fits"]:
            break
        best = (layers, a, m)
    return best


def widest(param_count, layers: int, seq: int, dp: int, micro_batch: int):
    """Largest hidden that fits at this depth, over multiples of 512."""
    best = None
    for hidden in range(512, 16385, 512):
        a = arch_for(hidden=hidden, layers=layers, seq=seq)
        if not valid(a):
            continue
        m = peak_bytes(param_count, a, dp, micro_batch)
        if not m["fits"]:
            break
        best = (hidden, a, m)
    return best


def row(label, arch, m, dp):
    # hidden/layers is the aspect ratio. Qwen3-8B is 4096/36 = 113.8, and the
    # PROFILES comment in predict.py picks shapes on it deliberately: a geometry
    # that fits the card but sits 2-4x off that ratio is a worse proxy for the
    # real architecture even at the same parameter count.
    ratio = arch["hidden_size"] / arch["num_layers"]
    off = ratio / QWEN3_RATIO
    print(f"  {label:<26} layers {arch['num_layers']:>3}  hidden {arch['hidden_size']:>5}  "
          f"ffn {arch['ffn_hidden_size']:>5}  heads {arch['num_attention_heads']:>3}/"
          f"{arch['num_query_groups']:<3} "
          f"| {m['params'] / 1e9:5.2f} B | h/L {ratio:6.1f} ({off:4.2f}x Qwen3) "
          f"| static {m['static'] / GIB:5.2f} + acts {m['acts'] / GIB:4.2f} "
          f"= {m['peak'] / GIB:5.2f} GiB ({m['headroom'] / GIB:+.2f})")


def self_check(param_count) -> int:
    """Prove the DP=1 path reproduces predict.py and the observed outcomes.

    Without this the DP=2 numbers rest on nothing: if the model cannot
    reproduce the single-node runs that calibrated it, its extrapolation to two
    nodes is not worth reading.
    """
    print("--- DP=1 must reproduce predict.py's own model exactly ---")
    bad = 0
    for name in ("smoke", "wider", "deeper", "1b", "1b2k"):
        arch = predict.PROFILES[name]
        mine = peak_bytes(param_count, arch, dp=1, micro_batch=1)["peak"]
        theirs = predict.memory_model_three_term(param_count, arch)["predicted_peak"]
        ok = abs(mine - theirs) < 1024            # a KiB, i.e. float noise only
        bad += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<7} mine {mine / GIB:6.3f} "
              f"predict.py {theirs / GIB:6.3f} GiB")

    print("\n--- and must agree with what actually happened ---")
    observed = {"smoke": True, "wider": True, "deeper": True, "1b": True,
                "1b2k": False}
    for name, should_fit in observed.items():
        m = peak_bytes(param_count, predict.PROFILES[name], dp=1, micro_batch=1)
        ok = m["fits"] == should_fit
        bad += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<7} predicted "
              f"{'fits' if m['fits'] else 'OOM':<4} observed "
              f"{'fits' if should_fit else 'OOM'}")

    print("\n--- the DP term itself ---")
    for dp, want in ((1, 18.0), (2, 12.0), (4, 9.0)):
        got = bytes_per_param(dp)
        ok = abs(got - want) < 1e-9
        bad += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] DP={dp} conservative "
              f"{got:.1f} B/param (want {want:.1f})")

    # The scenario's own geometry must be reproduced by arch_for(), or the
    # search is exploring shapes unrelated to the ones that were measured.
    print("\n--- arch_for() reproduces the measured 1b geometry ---")
    got = arch_for(20, 1536, 1024)
    want = predict.PROFILES["1b"]
    diffs = sorted(k for k in want if got.get(k) != want[k])
    ok = not diffs
    bad += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] arch_for(20, 1536, 1024) == PROFILES['1b']"
          f"{'' if ok else f'  differs: {diffs}'}")

    print()
    print("SELF-CHECK " + ("PASSED" if not bad else f"FAILED ({bad})"))
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dp", type=int, default=2, help="data-parallel degree (default 2)")
    ap.add_argument("--seq", type=int, default=1024, help="sequence length (default 1024)")
    ap.add_argument("--micro-batch", type=int, default=1, help="micro batch (default 1)")
    ap.add_argument("--self-check", action="store_true",
                    help="prove DP=1 reproduces predict.py and the observed outcomes")
    args = ap.parse_args()

    param_count = predict.load_param_count()
    if args.self_check:
        return self_check(param_count)

    dp, seq, mb = args.dp, args.seq, args.micro_batch
    print("=" * 108)
    print(f"MAXIMUM GEOMETRY AT DP={dp}, seq={seq}, micro_batch={mb}")
    print("=" * 108)
    print(f"  ceiling on peak allocated : {CEILING / GIB:.2f} GiB per rank "
          f"(derived from the 1b2k OOM, not nvidia-smi's 22.49)")
    print(f"  static                    : {bytes_per_param(dp):.1f} B/param "
          f"conservative, {bytes_per_param(dp, True):.1f} B/param optimistic")
    print(f"  activations               : measured three-term model, worst error "
          f"{predict.THREE_TERM['worst_error_gib']:.4f} GiB over 4 runs")
    print()

    print("  --- reference: the shapes that have actually been run ---")
    for name in ("smoke", "1b", "1b2k"):
        a = predict.PROFILES[name]
        m = peak_bytes(param_count, a, dp=1, micro_batch=1)
        tag = {"smoke": "ran, 2-node measured", "1b": "ran, single-node ceiling",
               "1b2k": "RAN AND OOMED"}[name]
        row(f"{name} @ DP=1 ({tag})", a, m, 1)

    print()
    print(f"  --- DEEPEST that fits at DP={dp}, by width ---")
    for hidden in (1024, 1536, 2048, 2560, 3072):
        got = deepest(param_count, hidden, seq, dp, mb)
        if got:
            layers, a, m = got
            row(f"hidden {hidden}", a, m, dp)
        else:
            print(f"  hidden {hidden:<20} nothing fits, even at 1 layer")

    print()
    print(f"  --- WIDEST that fits at DP={dp}, by depth ---")
    for layers in (4, 8, 12, 20, 32):
        got = widest(param_count, layers, seq, dp, mb)
        if got:
            hidden, a, m = got
            row(f"{layers} layers", a, m, dp)
        else:
            print(f"  {layers:<3} layers{'':<17} nothing fits, even at hidden 512")

    print()
    print("  Every DP>1 row above is a PREDICTION. No 2-node memory reading")
    print("  exists; the only 2-node measurement is throughput on the 4-layer")
    print("  smoke shape, which used 4.83 GiB and so tested none of this.")
    print("  The headroom column is against a ceiling derived from ONE OOM, and")
    print("  the model's own worst error is 0.08 GiB -- so treat anything inside")
    print("  ~0.5 GiB of the ceiling as unresolved rather than safe.")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    sys.exit(main())
