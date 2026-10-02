#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Derive per-step throughput from a g5 validation log.

WHY THIS EXISTS
---------------
`g5/train.py` can only report an end-to-end wall-clock rate across the whole
`pretrain()` call, which includes model construction, tokenizer download,
dataloader build and CUDA warmup. Over a 20-iteration validation that setup
dominates, so the end-to-end figure is a lower bound and not the steady-state
step rate.

Megatron logs its own per-iteration elapsed time. Parsing that gives the real
per-step rate, lets the warmup iteration be excluded, and yields a median and
spread rather than a single conflated average.

tokens/step at DP=1 is exactly global_batch_size * seq_length. There is no
data-parallel multiplier to forget here, which is why this is trustworthy.

USAGE
-----
    python3 g5/throughput.py <logfile>
    python3 g5/throughput.py <logfile> --warmup 2
    python3 g5/throughput.py <logfile> --tokens-per-step 8192

If --tokens-per-step is omitted it is read from the "tokens per step" line that
train.py prints, falling back to "global batch size" from the Megatron log line
multiplied by a --seq-len argument.

HONESTY NOTE
------------
The iteration-line regexes below have NOT been validated against a real
Megatron-Bridge 26.04 log, because the validation run never reached iteration 1.
If no iteration lines match, this script does NOT report zero or guess -- it
exits non-zero and prints the candidate lines it found, so the pattern can be
corrected against real output.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys

# Megatron-Core's training loop logs one line per `log_interval` iterations.
# Several spellings are accepted because the exact key has not been confirmed
# against a real 26.04 log; whichever matches first is used.
ELAPSED_PATTERNS = (
    re.compile(r"elapsed time per iteration \(ms\):\s*([0-9]*\.?[0-9]+)"),
    re.compile(r"elapsed time per iteration\s*\(s\):\s*([0-9]*\.?[0-9]+)"),
    re.compile(r"train_step_timing in s:\s*([0-9]*\.?[0-9]+)"),
    re.compile(r"iter_time\s*[:=]\s*([0-9]*\.?[0-9]+)"),
)
# Index of the pattern above -> multiplier to convert the captured value to
# seconds. ms for the first, already-seconds for the rest.
ELAPSED_TO_SECONDS = (1e-3, 1.0, 1.0, 1.0)

ITERATION_RE = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)")
LOSS_RE = re.compile(r"lm loss:\s*([0-9]*\.?[0-9]+(?:[eE][+\-]?\d+)?)")
GBS_RE = re.compile(r"global batch size:\s*(\d+)")
TOKENS_PER_STEP_RE = re.compile(r"tokens per step\s*:\s*([0-9,]+)")


def parse_log(path: str) -> dict:
    """Extract per-iteration timings and the token geometry from a log file."""
    try:
        with open(path, "r", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        raise SystemExit(f"FATAL: cannot read log {path}: {exc}") from exc

    if not lines:
        raise SystemExit(f"FATAL: log {path} is empty -- nothing to parse.")

    steps: list[dict] = []
    tokens_per_step: int | None = None
    gbs: int | None = None
    candidates: list[str] = []

    for line in lines:
        tps_match = TOKENS_PER_STEP_RE.search(line)
        if tps_match and tokens_per_step is None:
            tokens_per_step = int(tps_match.group(1).replace(",", ""))

        gbs_match = GBS_RE.search(line)
        if gbs_match and gbs is None:
            gbs = int(gbs_match.group(1))

        elapsed_s = None
        for idx, pattern in enumerate(ELAPSED_PATTERNS):
            match = pattern.search(line)
            if match:
                elapsed_s = float(match.group(1)) * ELAPSED_TO_SECONDS[idx]
                break

        if elapsed_s is None:
            # Keep a few lines that mention "iteration" so a failed parse can
            # be diagnosed against the real format instead of guessed at.
            if "iteration" in line.lower() and len(candidates) < 8:
                candidates.append(line.rstrip())
            continue

        if elapsed_s <= 0:
            raise SystemExit(
                f"FATAL: parsed a non-positive iteration time ({elapsed_s} s) from:\n  {line.rstrip()}"
            )

        iter_match = ITERATION_RE.search(line)
        loss_match = LOSS_RE.search(line)
        steps.append(
            {
                "iteration": int(iter_match.group(1)) if iter_match else len(steps) + 1,
                "seconds": elapsed_s,
                "loss": float(loss_match.group(1)) if loss_match else None,
            }
        )

    return {
        "steps": steps,
        "tokens_per_step": tokens_per_step,
        "global_batch_size": gbs,
        "candidates": candidates,
        "line_count": len(lines),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-step throughput from a g5 validation log.")
    ap.add_argument("logfile", help="path to the run log produced by g5/run.sh")
    ap.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="iterations to exclude as warmup (default 1; the first step pays "
        "cuDNN/cuBLAS autotune and allocator growth)",
    )
    ap.add_argument("--tokens-per-step", type=int, default=None, help="override tokens/step")
    ap.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="sequence length, used with the logged global batch size when "
        "train.py's 'tokens per step' line is absent",
    )
    args = ap.parse_args()

    parsed = parse_log(args.logfile)
    steps = parsed["steps"]

    if not steps:
        print(
            f"FATAL: no per-iteration timing lines matched in {args.logfile} "
            f"({parsed['line_count']} lines scanned).",
            file=sys.stderr,
        )
        print(
            "This means the Megatron log format differs from the patterns in "
            "ELAPSED_PATTERNS, not that throughput was zero.",
            file=sys.stderr,
        )
        if parsed["candidates"]:
            print("\nLines mentioning 'iteration' (fix the regex against these):", file=sys.stderr)
            for line in parsed["candidates"]:
                print(f"  {line}", file=sys.stderr)
        else:
            print(
                "\nNo lines mentioned 'iteration' at all -- the run probably died "
                "before the training loop started.",
                file=sys.stderr,
            )
        return 2

    tokens_per_step = args.tokens_per_step or parsed["tokens_per_step"]
    if tokens_per_step is None and parsed["global_batch_size"] and args.seq_len:
        tokens_per_step = parsed["global_batch_size"] * args.seq_len
    if tokens_per_step is None:
        print(
            "FATAL: could not determine tokens/step. Pass --tokens-per-step, "
            "or --seq-len so it can be combined with the logged global batch size.",
            file=sys.stderr,
        )
        return 2

    warmup = steps[: args.warmup]
    steady = steps[args.warmup :]
    if not steady:
        print(
            f"FATAL: only {len(steps)} iteration(s) logged, all consumed by "
            f"--warmup {args.warmup}. Nothing left to measure.",
            file=sys.stderr,
        )
        return 2

    rates = [tokens_per_step / step["seconds"] for step in steady]
    times = [step["seconds"] for step in steady]

    print("=" * 72)
    print("THROUGHPUT (parsed from Megatron per-iteration timings)")
    print("=" * 72)
    print(f"  log file           : {args.logfile}")
    print(f"  tokens per step    : {tokens_per_step:,}")
    print(f"  iterations logged  : {len(steps)}")
    warmup_times = ", ".join("{:.3f}s".format(s["seconds"]) for s in warmup)
    print(f"  warmup excluded    : {len(warmup)}  [{warmup_times}]")
    print(f"  steady-state steps : {len(steady)}")
    print("  --- step time ---")
    print(f"  median             : {statistics.median(times):.3f} s")
    print(f"  mean               : {statistics.fmean(times):.3f} s")
    if len(times) > 1:
        print(f"  stdev              : {statistics.stdev(times):.3f} s")
    print(f"  min / max          : {min(times):.3f} s / {max(times):.3f} s")
    print("  --- tokens/sec per training step ---")
    print(f"  MEDIAN             : {statistics.median(rates):,.0f} tok/s")
    print(f"  mean               : {statistics.fmean(rates):,.0f} tok/s")
    if len(rates) > 1:
        print(f"  stdev              : {statistics.stdev(rates):,.0f} tok/s")
    print(f"  min / max          : {min(rates):,.0f} / {max(rates):,.0f} tok/s")

    losses = [s["loss"] for s in steps if s["loss"] is not None]
    if len(losses) >= 2:
        print("  --- loss ---")
        print(f"  first / last       : {losses[0]:.4f} / {losses[-1]:.4f}")
        print(f"  moved              : {'DOWN' if losses[-1] < losses[0] else 'NOT DOWN'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
