#!/usr/bin/env python3
"""Build a small, bounded c4 dataset in Megatron indexed format for the g5 box.

Why this exists instead of `preprocessing/preprocess.py`
--------------------------------------------------------
That script targets a p5en node with FSx and 192 CPUs and a 1B-token budget.
On a single g5.8xlarge it is unusable for three separate reasons:

1. It calls `load_dataset("allenai/c4", "en", split=f"train[:{N}]")`. A split
   SLICE still resolves and downloads every shard of the `en` config -- about
   305 GB compressed across 1024 files -- and only then slices. The g5 root
   volume has well under that free. This script uses `streaming=True` and
   stops once the token budget is met, so the download is bounded by what is
   actually consumed.
2. It hard-exits without `HF_TOKEN`. `allenai/c4` is a public dataset and the
   Qwen3 tokenizer was already shown to download unauthenticated during the
   smoke run, so a token is optional here. It is still honoured if set.
3. Its tokenise workers are called with a hardcoded document length of 4096.
   That is kept as the DEFAULT here (so the on-disk convention matches) but is
   exposed as `--doc-length`.

Document length does not need to equal the training `seq_length`: Megatron's
GPTDataset concatenates documents and re-splits them into `seq_length + 1`
samples. It only needs enough total tokens.

Usage (inside the NeMo container, which already has datasets/transformers):
    python3 g5/prepare_c4.py --output-prefix /workspace/run/datasets/c4_qwen3
    python3 g5/prepare_c4.py --self-test     # no network, no HF, numpy only
"""

import argparse
import ast
import os
import struct
import sys
import time

import numpy as np

# Megatron MMapIndexedDataset constants, mirrored from preprocessing/preprocess.py.
_HDR_MAGIC = b"MMIDIDX\x00\x00"
_DTYPE = np.int32
_DTYPE_CODE = 4

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESS_PY = os.path.join(REPO_ROOT, "preprocessing", "preprocess.py")


def load_write_idx_file(path: str = PREPROCESS_PY):
    """Lift `write_idx_file` out of preprocess.py so the on-disk format cannot drift.

    preprocess.py imports `datasets` and `transformers` at module scope, so it
    cannot be imported on a machine without them (and --self-test must run
    with numpy alone). Extracting the single function with `ast` avoids that
    while still using the authoritative implementation rather than a copy.
    """
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: {path} not found; cannot lift write_idx_file")
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "write_idx_file":
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {
                "_HDR_MAGIC": _HDR_MAGIC, "_DTYPE": _DTYPE,
                "_DTYPE_CODE": _DTYPE_CODE, "np": np, "struct": struct,
            }
            exec(compile(module, path, "exec"), namespace)  # noqa: S102
            return namespace["write_idx_file"]
    raise SystemExit(f"FATAL: write_idx_file() not found in {path}")


def read_idx_file(idx_path: str) -> dict:
    """Parse a Megatron .idx file back, so what was written can be verified.

    Reading the index back is the only way to prove the writer produced
    something Megatron can consume. A .bin of the right byte count says
    nothing about whether the index that addresses it is coherent.
    """
    with open(idx_path, "rb") as f:
        magic = f.read(9)
        if magic != _HDR_MAGIC:
            raise SystemExit(f"FATAL: {idx_path} bad magic {magic!r}, "
                             f"expected {_HDR_MAGIC!r}")
        version = struct.unpack("<Q", f.read(8))[0]
        dtype_code = struct.unpack("<B", f.read(1))[0]
        n_sizes = struct.unpack("<Q", f.read(8))[0]
        n_docs = struct.unpack("<Q", f.read(8))[0]
        sizes = np.fromfile(f, dtype=np.int32, count=n_sizes)
        pointers = np.fromfile(f, dtype=np.int64, count=n_sizes)
        doc_idx = np.fromfile(f, dtype=np.int64, count=n_docs)
    return dict(version=version, dtype_code=dtype_code, sizes=sizes,
                pointers=pointers, doc_idx=doc_idx)


def verify(prefix: str, expect_tokens: int | None = None) -> dict:
    """Verify a built dataset is self-consistent. Raises SystemExit on any fault."""
    bin_path, idx_path = prefix + ".bin", prefix + ".idx"
    for p in (bin_path, idx_path):
        if not os.path.exists(p):
            raise SystemExit(f"FATAL: {p} does not exist")

    idx = read_idx_file(idx_path)
    sizes, pointers = idx["sizes"], idx["pointers"]
    bin_bytes = os.path.getsize(bin_path)
    itemsize = _DTYPE().itemsize
    total_tokens = int(sizes.sum())

    faults = []
    if idx["dtype_code"] != _DTYPE_CODE:
        faults.append(f"dtype code {idx['dtype_code']} != {_DTYPE_CODE}")
    if total_tokens * itemsize != bin_bytes:
        faults.append(f".idx accounts for {total_tokens * itemsize} bytes but "
                      f".bin is {bin_bytes}")
    # Pointers must be the exclusive cumulative byte offsets of sizes.
    expect_ptr = np.zeros(len(sizes), dtype=np.int64)
    if len(sizes):
        expect_ptr[1:] = np.cumsum(sizes[:-1].astype(np.int64)) * itemsize
    if not np.array_equal(pointers, expect_ptr):
        bad = int(np.argmax(pointers != expect_ptr))
        faults.append(f"pointer[{bad}]={pointers[bad]} != {expect_ptr[bad]}")
    if len(sizes) == 0:
        faults.append("dataset is empty (0 documents)")
    if expect_tokens is not None and total_tokens < expect_tokens:
        faults.append(f"only {total_tokens:,} tokens, wanted >= {expect_tokens:,}")

    # Spot-check that the last document is actually readable at its offset.
    if len(sizes) and not faults:
        with open(bin_path, "rb") as f:
            f.seek(int(pointers[-1]))
            tail = np.frombuffer(f.read(int(sizes[-1]) * itemsize), dtype=_DTYPE)
        if len(tail) != int(sizes[-1]):
            faults.append(f"last doc short read: {len(tail)} != {sizes[-1]}")
        elif tail.min() < 0:
            faults.append(f"negative token id {tail.min()} in last doc")

    if faults:
        raise SystemExit("FATAL: dataset verification failed:\n  - "
                         + "\n  - ".join(faults))

    return dict(documents=len(sizes), total_tokens=total_tokens,
                bin_bytes=bin_bytes, idx_bytes=os.path.getsize(idx_path),
                doc_length=int(sizes[0]) if len(sizes) else 0,
                max_token_id=None)


def build(prefix: str, tokenizer_name: str, num_tokens: int, doc_length: int,
          batch_size: int = 512) -> dict:
    """Stream c4/en, tokenise, and write a bounded Megatron indexed dataset."""
    # Imported here, not at module scope, so --self-test runs with numpy alone.
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            f"FATAL: {exc.name} is not installed. Run this inside the NeMo "
            "container (which has it), not on the host."
        ) from exc

    write_idx_file = load_write_idx_file()
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)

    # Optional: allenai/c4 is public. Honoured if present, never required.
    token = os.environ.get("HF_TOKEN") or None
    print(f"  HF_TOKEN           : {'set (will be used)' if token else 'unset (not needed)'}",
          flush=True)

    print(f"  tokenizer          : {tokenizer_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # streaming=True is the whole point: it fetches shards lazily and we stop
    # as soon as the budget is met, instead of materialising all 305 GB.
    print("  dataset            : allenai/c4 'en' train, STREAMING", flush=True)
    stream = load_dataset("allenai/c4", "en", split="train",
                          streaming=True, token=token)

    bin_path, idx_path = prefix + ".bin", prefix + ".idx"
    sizes: list[int] = []
    doc_idx: list[int] = [0]
    buffer: list[int] = []
    total = 0
    docs_read = 0
    t0 = time.time()

    with open(bin_path, "wb") as out:
        batch: list[str] = []
        for record in stream:
            batch.append(record["text"])
            docs_read += 1
            if len(batch) < batch_size:
                continue
            encoded = tok(batch, add_special_tokens=False)["input_ids"]
            batch = []
            for ids in encoded:
                buffer.extend(ids)
            while len(buffer) >= doc_length:
                np.asarray(buffer[:doc_length], dtype=_DTYPE).tofile(out)
                del buffer[:doc_length]
                sizes.append(doc_length)
                doc_idx.append(len(sizes))
                total += doc_length
            if total >= num_tokens:
                break
            if len(sizes) and len(sizes) % 2000 == 0:
                rate = total / max(time.time() - t0, 1e-9)
                print(f"    {total:>12,} / {num_tokens:,} tokens "
                      f"({100 * total / num_tokens:5.1f}%)  {rate:,.0f} tok/s",
                      flush=True)

    if total == 0:
        raise SystemExit("FATAL: produced 0 tokens. The stream yielded nothing "
                         "usable -- check network access to huggingface.co.")

    write_idx_file(idx_path, sizes, doc_idx)
    elapsed = time.time() - t0

    info = verify(prefix, expect_tokens=min(num_tokens, total))
    info.update(docs_read=docs_read, elapsed_s=elapsed)
    return info


def self_test() -> int:
    """Prove the writer/reader round-trip with no network and no HF deps."""
    import tempfile

    write_idx_file = load_write_idx_file()
    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
        if not cond:
            failures.append(name)

    print("--- write_idx_file lifted from preprocessing/preprocess.py ---")
    check("lift succeeded", callable(write_idx_file))

    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "synthetic")
        doc_length, n_docs = 4096, 7
        rng = np.random.default_rng(1234)
        truth = rng.integers(0, 151936, size=doc_length * n_docs, dtype=np.int32)

        print("\n--- build a synthetic dataset by hand ---")
        with open(prefix + ".bin", "wb") as out:
            truth.tofile(out)
        sizes = [doc_length] * n_docs
        write_idx_file(prefix + ".idx", sizes, list(range(n_docs + 1)))

        info = verify(prefix, expect_tokens=doc_length * n_docs)
        check("verify() accepts a well-formed dataset",
              info["total_tokens"] == doc_length * n_docs,
              f"{info['total_tokens']:,} tokens, {info['documents']} docs")

        print("\n--- round-trip: every token read back must match ---")
        idx = read_idx_file(prefix + ".idx")
        recovered = []
        with open(prefix + ".bin", "rb") as f:
            for ptr, size in zip(idx["pointers"], idx["sizes"]):
                f.seek(int(ptr))
                recovered.append(np.frombuffer(
                    f.read(int(size) * _DTYPE().itemsize), dtype=_DTYPE))
        recovered = np.concatenate(recovered)
        check("all tokens round-trip via .idx offsets",
              np.array_equal(recovered, truth),
              f"{len(recovered):,} tokens compared")

        print("\n--- verify() must REJECT corruption (non-vacuous) ---")
        # Truncate the .bin so the index over-accounts for it.
        with open(prefix + ".bin", "r+b") as f:
            f.truncate(os.path.getsize(prefix + ".bin") - 4)
        try:
            verify(prefix)
            check("truncated .bin rejected", False, "verify() accepted it")
        except SystemExit:
            check("truncated .bin rejected", True)

        # Corrupt the magic bytes.
        with open(prefix + ".idx", "r+b") as f:
            f.seek(0)
            f.write(b"XXXXXXXXX")
        try:
            verify(prefix)
            check("bad magic rejected", False, "verify() accepted it")
        except SystemExit:
            check("bad magic rejected", True)

        # An empty dataset must be rejected, not silently accepted.
        prefix2 = os.path.join(tmp, "empty")
        open(prefix2 + ".bin", "wb").close()
        write_idx_file(prefix2 + ".idx", [], [0])
        try:
            verify(prefix2)
            check("empty dataset rejected", False, "verify() accepted it")
        except SystemExit:
            check("empty dataset rejected", True)

    print()
    if failures:
        print(f"SELF-TEST FAILED: {', '.join(failures)}")
        return 1
    print("SELF-TEST PASSED")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-prefix", default="/workspace/run/datasets/c4_qwen3",
                   help="prefix without .bin/.idx (default: %(default)s)")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    p.add_argument("--num-tokens", type=int, default=50_000_000,
                   help="token budget; bounds the streaming download "
                        "(default: %(default)s)")
    p.add_argument("--doc-length", type=int, default=4096,
                   help="tokens per indexed document; need NOT equal the "
                        "training seq_length (default: %(default)s)")
    p.add_argument("--self-test", action="store_true",
                   help="verify the format writer/reader offline, numpy only")
    p.add_argument("--verify-only", metavar="PREFIX",
                   help="verify an already-built dataset and exit")
    args = p.parse_args()

    if args.self_test:
        return self_test()

    if args.verify_only:
        info = verify(args.verify_only)
        print(f"  VERIFIED {args.verify_only}")
        for k, v in info.items():
            if v is not None:
                print(f"    {k:<14}: {v:,}" if isinstance(v, int) else f"    {k:<14}: {v}")
        return 0

    if args.doc_length <= 0 or args.num_tokens <= 0:
        raise SystemExit("FATAL: --doc-length and --num-tokens must be positive")

    prefix = args.output_prefix
    # Idempotent: a verified dataset is reused rather than rebuilt.
    if os.path.exists(prefix + ".idx") and os.path.exists(prefix + ".bin"):
        try:
            info = verify(prefix, expect_tokens=args.num_tokens)
            print(f"=== dataset already present and verified, reusing ===")
            print(f"  {prefix}.bin  {info['bin_bytes'] / 1e9:.2f} GB")
            print(f"  {info['total_tokens']:,} tokens in {info['documents']:,} documents")
            return 0
        except SystemExit as exc:
            print(f"  existing dataset unusable, rebuilding: {exc}", flush=True)

    print("=== allenai/c4 (streaming, bounded) -> Megatron indexed format ===")
    print(f"  target tokens      : {args.num_tokens:,}")
    print(f"  doc length         : {args.doc_length:,} tokens")
    info = build(prefix, args.tokenizer, args.num_tokens, args.doc_length)
    print("\n=== Done ===")
    print(f"  documents          : {info['documents']:,}")
    print(f"  total tokens       : {info['total_tokens']:,}")
    print(f"  c4 docs consumed   : {info['docs_read']:,}")
    print(f"  {prefix}.bin : {info['bin_bytes'] / 1e9:.2f} GB")
    print(f"  {prefix}.idx : {info['idx_bytes'] / 1e6:.1f} MB")
    print(f"  elapsed            : {info['elapsed_s'] / 60:.1f} min")
    print(f"\n  Train with:  DATA_PATH={prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
