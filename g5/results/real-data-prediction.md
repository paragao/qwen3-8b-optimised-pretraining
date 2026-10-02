# Real c4 data — prediction recorded BEFORE the run

**Status: PREDICTION ONLY. Nothing here is measured.**

## What this run is for, and why the mock runs could not do it

Both completed runs used Megatron's mock dataset, and both produced a loss
trace that *looks* like learning and is not:

| | `smoke` (mock) | `wider` (mock) |
|---|---|---|
| first loss | 12.1481 | 12.2437 |
| last loss (50 steps) | **0.2801** | **0.1902** |
| data | `mock: True`, `MockGPTDataset splits with sizes=(400, 0, 0)` | same |

`ln(151936) = 11.9312` nats is the cross-entropy of uniform guessing over the
full vocab. Both runs started 0.22 and 0.31 nats above it, which is the correct
signature of a fresh random init. The collapse to ~0.2 is **degenerate fitting
of 400 synthetic samples**, not convergence — the model saw the same tiny
synthetic set repeatedly and memorised it.

This run replaces the data source and changes nothing else, so the loss curve
becomes interpretable.

## Configuration

Deliberately the `smoke` geometry, so this is a **controlled single-variable
change** against an already-measured baseline: identical architecture,
identical optimizer, only `DATA_PATH` differs.

| | value |
|---|---|
| profile | `smoke` (4 layers, hidden 1024, seq 1024) |
| iterations | 1000 |
| tokens consumed | 8,192,000 |
| dataset budget | 50,000,000 tokens (**6.1x headroom, so no repetition**) |
| baseline to compare | 6.84 GiB peak, 24,757 tok/s, loss 12.1481 -> 0.2801 |

The headroom matters: with 6.1x more tokens than the run consumes, the model
**cannot** memorise its way to a low loss. That is what makes a low final loss
falsifying rather than expected.

For scale, 8.2M tokens is **0.114%** of the ~7.2B tokens a 359.4M-parameter
model would need to be Chinchilla-optimal. This is a very undertrained model
and the prediction reflects that.

## Predictions

| # | Quantity | Predicted |
|---|---|---|
| **P1** | first-iteration loss | **11.5 - 12.5** (just above `ln(V)`, data-independent) |
| **P2** | final loss after 1000 steps | **6.0 - 9.5** |
| **P3** | peak allocated memory | **6.84 +/- 0.10 GiB** (unchanged — same shapes) |
| **P4** | median tok/s | **24,757 +/- 5%** (dataloading should not bottleneck) |
| **P5** | skipped / NaN iterations | 0 / 0 |

P2's reasoning: early training first learns the unigram token distribution,
which for English text under a 152k BPE vocab lands cross-entropy around 7-8
nats. A model this undertrained should reach roughly that and then improve only
slowly. Below 6.0 would be surprisingly good for 8.2M tokens; above 9.5 would
mean it barely learned.

## Refutation thresholds

| # | Condition | What it means |
|---|---|---|
| **D1** | log does **not** contain `dataset: real Megatron-indexed data at` | **RUN IS VOID** — fell back to mock; nothing else is interpretable |
| **D2** | log contains `mock: True` or `MockGPTDataset` | **RUN IS VOID** — same |
| **D3** | final loss **< 2.0** | the headline failure. With 6.1x token headroom this should be impossible; it would mean the dataset is repeating, far smaller than reported, or `DATA_PATH` silently ignored |
| **D4** | final loss **> 11.0** | no learning at all — suspect LR schedule, data corruption, or token IDs out of vocab range |
| **D5** | first loss outside [11.0, 13.0] | the init or the loss reduction is wrong, independent of data |
| **D6** | peak memory differs from 6.84 GiB by **> 0.3 GiB** | the data path changes GPU allocation, which it should not |
| **D7** | median tok/s **< 22,000** (-11%) | dataloading *is* a bottleneck at `DATA_NUM_WORKERS=8` |
| **D8** | any skipped or NaN iteration | real-data token distribution destabilises BF16 where synthetic did not |

**D3 is the whole point of this run.** The mock runs reached 0.19-0.28. If real
c4 also collapses below 2.0, the data path is not doing what it claims and the
"real loss curve" is an illusion — that outcome must be reported as a defect,
not as fast convergence.

## What a PASS would and would not establish

Establishes: the data path works end to end (c4 -> Qwen3 tokenizer -> Megatron
indexed format -> GPTDataset -> training step), and the loss curve is a genuine
if very early learning signal on real text.

Does **not** establish: anything about Qwen3-8B's convergence, any quality
claim, or that the proxy's hyperparameters are right. At 0.114% of
Chinchilla-optimal tokens, this measures that the pipeline is sound, nothing
more.

## Data path caveats, recorded up front

`g5/prepare_c4.py` is new and its **streaming download has never been run**.
Three things could surface on first execution, and all are the prep step's
problem rather than the training step's:

1. `allenai/c4` could require authentication for streaming despite being
   public, in which case `HF_TOKEN` becomes necessary after all.
2. The streaming shard layout could differ from what `load_dataset(...,
   streaming=True)` expects for the `en` config.
3. Tokenising 50M tokens single-process with batched encoding is untimed;
   I expect a few minutes on 32 vCPUs but have not measured it.

The format writer itself **is** verified: `python3 g5/prepare_c4.py
--self-test` round-trips 28,672 tokens through `.bin`/`.idx` offsets exactly,
and rejects a truncated `.bin`, corrupted magic bytes, and an empty dataset.
It reuses `write_idx_file` lifted from `preprocessing/preprocess.py` rather
than a copy, so the on-disk format cannot drift from the repo's own.

## Why `preprocessing/preprocess.py` was not used

That script is correct for its target (a p5en node, FSx, 192 CPUs, 1B tokens)
and wrong for this one, in three independent ways:

1. It calls `load_dataset("allenai/c4", "en", split=f"train[:{N}]")`
   (line 127). A split **slice** still resolves and downloads every shard of
   the `en` config before slicing. `allenai/c4`'s `en` train split is published
   as 1024 gzipped JSON shards totalling a few hundred GB — that figure is my
   own knowledge of the dataset, not something this repo states, so treat it as
   an estimate; the material point is that it is far more than the g5 root
   volume holds and the download would outlast the experiment.
2. It hard-exits without `HF_TOKEN` (line 55), which `allenai/c4` does not
   require.
3. Its tokenise workers are invoked with a hardcoded document length of 4096
   (line 144), not a parameter.

`g5/prepare_c4.py` uses `streaming=True` and stops at the token budget, makes
the token optional, and exposes `--doc-length`.

**The streaming choice is not an invention.** This repo's own
`docs/data-loading-explained.md` line 41 already prescribes exactly
`load_dataset("allenai/c4", "en", split="train", streaming=True)`, and line 100
describes the concatenate-then-slice-into-4096-token-chunks design that
`prepare_c4.py` implements. `preprocess.py` is what diverged from the
documented design, by switching to a split slice.

Document length need not equal the training `seq_length`: Megatron's GPTDataset
concatenates documents and re-splits them into `seq_length + 1` samples, which
is the same property that doc relies on.
