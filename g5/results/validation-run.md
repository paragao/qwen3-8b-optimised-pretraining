# g5.8xlarge validation run — 2026-10-02

Partial run. The stack came up and got as far as tokenizer construction, then hit
a real defect in the single-GPU path, which has been fixed in `g5/train.py` and
`g5/run.sh` but **not yet re-run to completion**. See "Status" at the bottom.

## Environment

| | |
|---|---|
| Account | 159553542841 (`compute-sa-team-Administrator`) |
| Region / AZ | us-east-1b — us-west-2 had **no g5.8xlarge capacity in any AZ** |
| Instance | `i-09ee99ff5540c60ec`, g5.8xlarge |
| AMI | `ami-0d2caa95fd63ebdbc` — Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20260929 |
| GPU | NVIDIA A10G, **23028 MiB (22.49 GiB)**, compute capability **8.6**, driver 595.91.07 |
| Host | 32 vCPU, 124 GB RAM, 290 GB gp3 root |
| Docker | 29.8.1 |
| Container | `nvcr.io/nvidia/nemo:26.04`, **77.4 GB** |
| Access | SSM Session Manager; security group had 0 ingress rules |

## Confirmed working

1. **A10G is `sm_86`** and the NeMo 26.04 container runs on it. No Ampere
   incompatibility in this stack — relevant because the repo otherwise only
   targets Hopper (`sm_90`) and Blackwell.
2. **`g5/run.sh` preflight** correctly detected the driver, Docker and a GPU
   count of 1, and reported the card's capacity and compute capability.
3. **The Megatron-Bridge import graph resolves** inside the stock NeMo image —
   `megatron.bridge.recipes.qwen.qwen3.qwen3_8b_pretrain_config`,
   `megatron.bridge.training.gpt_step.forward_step` and
   `megatron.bridge.training.pretrain.pretrain` all imported. The repo's
   EFA-augmented `Dockerfile` is **not** required for a single-GPU run.
4. **CUDA initialised** — the pre-flight memory report printed, which requires a
   live `torch.cuda.get_device_properties(0)`.
5. **`qwen3_8b_pretrain_config()` accepted every scaled override**, and the
   resolved config echoed back `vocab_size: 151936`, unchanged from Qwen3-8B.
6. **The real Qwen3 tokenizer downloaded and built**:
   ```
   tokenizer_type : 'HuggingFaceTokenizer'
   tokenizer_model: 'Qwen/Qwen3-8B'
   INFO:root:Using preset vocab_size: 151936 over the tokenizer
             vocab_size: 151669, dummy tokens: 267.
   [after tokenizer is built] datetime: 2026-10-02 12:21:31
   ```
   No `HF_TOKEN` was needed — `Qwen/Qwen3-8B` is a public repo. Note the 267
   dummy tokens: the tokenizer's real vocabulary is 151,669 and the recipe pads
   to 151,936 for divisibility.

## Defect found and fixed

The run died immediately after the tokenizer, at `> setting tensorboard ...`:

```
OSError: [Errno 30] Read-only file system: '/workspace/repo/nemo_experiments'
  File ".../megatron/bridge/training/state.py", line 179, in tensorboard_logger
```

The recipe leaves `logger.tensorboard_dir` at a **CWD-relative**
`./nemo_experiments/default/tb_logs`. The cluster path never trips over this
because it runs with the FSx working directory writable. `g5/run.sh` mounts the
repo read-only, which is the safer default and surfaced the latent assumption.

Fixed in two places:

- `g5/train.py` now sets `cfg.logger.tensorboard_dir` explicitly to a path under
  `RUN_BASE` (override with `TENSORBOARD_DIR`) and `makedirs` it, so the config
  never depends on the working directory.
- `g5/run.sh` runs with `-w /workspace/run` (writable) and an absolute script
  path `/workspace/repo/g5/train.py`, so any other stray CWD-relative write also
  lands somewhere writable while the repo stays read-only.

Both edits are verified locally (`python3 -m py_compile`, `bash -n`) and
committed. Neither has been exercised on the instance.

## Previously unverified — now closed

The first run failed *before* model construction, leaving these open. **All four
were closed by the completed run of 2026-10-02 15:22 UTC** (log:
`g5/results/run-20261002-162239.log`, 50/50 iterations):

| Open item | Result |
|---|---|
| Qwen3 model build on `sm_86` — GQA, RoPE, RMSNorm, SwiGLU | **confirmed**, 50 steps, 0 skipped, 0 NaN |
| mock dataset / dataloader | **confirmed**, `MockGPTDataset` sizes `(400, 0, 0)` |
| forward -> backward -> Adam step, loss movement | **confirmed**, loss responded, grad norm 1.0–3.8 |
| measured peak memory vs predicted 6.02 GiB static | **6.84 GiB allocated / 7.26 GiB reserved** |

### Memory: prediction vs measurement

```
predicted static (18 B/param)   6.02 GiB   <- analytic, weights+grads+Adam+master
measured peak allocated         6.84 GiB
                                --------
activations + workspace         0.82 GiB   (13.6% on top of static)
```

The analytic model was therefore **low by 13.5%** (0.8151 GiB against a precise
static figure of 6.0249 GiB; the rounded 6.02 gives 13.6%), and the gap is
exactly the
term it does not model: activations and allocator workspace. For sizing work,
treat the 18 B/param figure as a floor and add an activation margin. Peak
reserved (7.26 GiB) sits 0.42 GiB above allocated, which is allocator
fragmentation, not model state.

At 6.84 GiB the proxy uses **30.9% of the A10G's 22.1 GiB usable**, so the card
had ample headroom — the scaling was conservative.

### Superseded by the `wider` run

The "+14% activation margin" advice above was a one-point rule of thumb and is
**refuted**: `wider` measured 11.76 GiB against its 11.98 GiB flat-rule
prediction.

Its decomposed replacement is **also refuted**, by `deeper`. Do not use either
to size a new shape — measure it.

| | `smoke` | `wider` | `deeper` |
|---|---|---|---|
| static @18 B/param | 6.02 GiB | 10.55 GiB | 7.64 GiB |
| measured peak allocated | 6.84 GiB | **11.76 GiB** | **9.66 GiB** |
| measured peak reserved | 7.26 GiB | 12.25 GiB | 10.03 GiB |
| residual over static | 0.82 GiB (+13.5%) | 1.21 GiB (+11.4%) | **2.02 GiB (+26.4%)** |
| fragmentation (reserved − alloc) | 0.42 GiB | 0.49 GiB | 0.37 GiB |

The residual share does **not** move monotonically with model size (13.5% ->
11.4% -> 26.4%), which is the clearest single sign that no simple scaling of
the static figure works. `deeper` is the only profile at `seq_length` 2048 and
carries by far the largest residual share, consistent with sequence length
driving the activation terms — though these three profiles co-vary too many
parameters to attribute it properly.

**Empirical sizing rule:** peak allocated ran **11-26% above** the 18 B/param
static figure across these three shapes. Treat 18 B/param as a floor and budget
to the top of that range. `python3 g5/predict.py --form-test` shows why a
model-based prediction is not trustworthy here; full detail in
`g5/results/deeper-prediction.md`.

## Throughput: measured

Measured on the completed run of 2026-10-02 15:22 UTC, 50/50 iterations,
parsed by `g5/throughput.py` from Megatron's own per-iteration timings with
iteration 1 excluded as warmup (2.207 s against a 0.331 s steady state):

| Metric | Value |
|---|---|
| **tokens/sec per training step (median)** | **24,757 tok/s** |
| mean / stdev | 24,754 / 47 tok/s |
| min / max | 24,490 / 24,817 tok/s |
| median step time | 0.331 s |
| stdev of step time | 0.001 s (0.3% — very stable) |
| steady-state steps | 49 of 50 |
| tokens/step | 8,192 (GBS 8 x seq 1024) |
| reported compute | 30.9 MODEL_TFLOP/s/GPU ≈ **24.7% MFU** of A10G BF16 dense peak (125 TFLOP/s) |

Two independent cross-checks agree, so the figure is not a parser artefact:
`8192 / 0.331 = 24,749 tok/s` by hand against the parser's 24,757 tok/s, and
Megatron's own `Step Time : 0.33s` line matches its `elapsed time per iteration
(ms): 331.2`.

### Why the end-to-end rate is much lower

```
end-to-end     13,912 tok/s   (29.4 s wall, setup included)
steady-state   24,757 tok/s   (per-step, warmup excluded)
```

The end-to-end number is **1.78x pessimistic** because ~11.0 s of the 29.4 s
wall clock is one-off setup — model build, tokenizer download, dataset index
compilation (7.17 s on its own), CUDA warmup. At 50 iterations that overhead is
38% of the run. Quote the steady-state median; the end-to-end figure is only a
lower bound and is labelled as such in the output.

### Confirmed again on `wider`

The `wider` run of 2026-10-02 17:41 reproduces the same pattern and extends it
to a second architecture:

| | `smoke` | `wider` |
|---|---|---|
| median tok/s (steady state) | 24,757 | **14,357** |
| end-to-end tok/s (lower bound) | 13,912 | 10,004 |
| end-to-end pessimism | 1.78x | 1.44x |
| median step time | 0.331 s | 0.571 s |
| step-time stdev | 0.001 s (0.3%) | 0.001 s (0.2%) |
| MODEL_TFLOP/s | 30.9 | **34.9** |
| MFU of 125 TFLOP/s peak | 24.7% | **27.9%** |
| FLOPs/step | 10.23 TFLOP | 19.94 TFLOP (1.950x) |
| skipped / NaN | 0 / 0 | 0 / 0 |

Two findings from the pair. **Efficiency rises with width**: 30.9 -> 34.9
MODEL_TFLOP/s (+12.9%) going from hidden 1024 to 1536, which was predicted in
advance as a direction (R4) and confirmed — larger GEMMs are not less
efficient on this card. And **throughput is not a model-quality metric here**:
`wider` does 1.950x the FLOPs per step for the same 8,192 tokens, so its 0.58x
throughput is arithmetic, not a regression.

`g5/throughput.py`'s regexes, originally verified only against a synthetic log,
have now parsed two real Megatron-Bridge 26.04 logs and correctly identified
the warmup iteration in both (2.207 s and 2.423 s).

### What this number does and does not mean

It measures that **the step loop works and is stable on an A10G** at a given
proxy size. It is **not** a Qwen3-8B throughput figure and must not be
extrapolated to one: the proxy is 4 layers at hidden 1024 (359 M params) against
36 layers at hidden 4096 (8.19 B params), on one Ampere card rather than
Hopper, and Qwen3-8B cannot run here at all (see
`docs/single-gpu-memory-budget.md`).

For contrast, the 16x H200 run in `h200/results/benchmark.md` reaches 162,000
tok/s at 3.23 s/step with GBS=128 and seq=4096 (524,288 tokens/step) — a
different architecture, precision regime and parallelism, and not comparable.

### The loss curve is NOT evidence of learning

Loss moved 12.1481 -> 0.2801 over 50 iterations. **This does not show the model
learning anything**, and reading it that way would be a mistake. The data source
is Megatron's synthetic mock dataset, confirmed in the log:

```
dataset: Megatron mock dataset (set DATA_PATH for real c4)
Let mock = True, as both blend and blend_per_split are None
Building MockGPTDataset splits with sizes=(400, 0, 0)
```

The iteration-1 loss of 12.1481 sits just above `ln(151936) = 11.9312`, i.e.
uniform random guessing over the full vocab, which is the expected starting
point. The subsequent collapse to 0.28 on only 400 synthetic samples is
degenerate fitting of trivially predictable mock tokens, not convergence.

The valid conclusions from the loss trace are narrower and still useful:
gradients flow, the optimizer updates weights, the LR schedule is applied
(6.0e-05 warmup -> 3.0e-05), and **0 skipped / 0 NaN iterations** across all 50
steps — so no numerical instability in the BF16 path on `sm_86`. Judging real
convergence requires `DATA_PATH` pointed at real c4 tokens.

### Instrumentation gaps found while answering this, both fixed

1. **`g5/train.py` had no throughput instrumentation at all.** A completed run
   would have printed peak memory and no tok/s. It now prints `tokens per step`
   before the loop (so the figure survives a crash) and, on completion, total
   tokens plus an end-to-end rate.
2. **The end-to-end rate alone would have been misleading** — measured at 1.78x
   pessimistic above. It is labelled a lower bound in the output, with the
   authoritative per-step number coming from `g5/throughput.py`.

`g5/throughput.py` was tested pre-run against a synthetic log with a
hand-computed answer (median 2.000 s -> 4,096 tok/s). Its iteration-line
regexes are **now validated against a real Megatron-Bridge 26.04 log**: it
parsed all 50 iterations, identified the warmup step, and its median agreed
with the hand cross-check to within 8 tok/s (0.03%).

## Reproducing from here

The instance already has the repo cloned and the 77.4 GB image pulled, so
finishing takes one command from the repo root on this branch:

```bash
./g5/finish-run.sh                 # 20 iterations
TRAIN_ITERS=50 ./g5/finish-run.sh  # steadier median
```

That script pushes a 60-second ephemeral SSH key via EC2 Instance Connect,
tunnels SSH over SSM Session Manager (**no inbound security group rule needed** —
the SSM agent dials out and sshd is reached on the instance's own loopback),
copies the three fixed files, runs the validation, parses throughput with
`g5/throughput.py`, and retrieves the log into `g5/results/logs/`.

Expected on success: the pre-flight report, `tokens per step: 8,192`, 20 logged
iterations, a `VALIDATION COMPLETE` block with peak allocated and reserved
memory, then the throughput table with a median tok/s.

### Why the agent could not run this itself

**The operator ran `g5/finish-run.sh` from their own terminal on 2026-10-02 at
15:21 UTC and it completed cleanly** — that is the source of every measured
number above. The agent could not invoke it, for the reasons below. Note that
the script's own transport is `aws ssm start-session` (line 87), so the agent
running the script would have smuggled the blocked call past the policy inside
a file rather than satisfied the control; it declined on that basis.

Remote execution was blocked by host security policy, progressively:

| Channel | Status |
|---|---|
| `aws ssm send-command` (shell and `use_aws`) | blocked |
| `aws ssm start-session` (the SSH-over-SSM tunnel) | blocked |
| `aws ec2 authorize-security-group-ingress` | blocked |
| `aws ec2-instance-connect send-ssh-public-key` | **permitted** (`Success: true`) |

The key push alone is useless without a transport. Routes that remained —
an EC2 Instance Connect Endpoint, a replacement security group attached via
`modify-instance-attribute`, or CloudFormation — would each have been a
deliberate end-run around a control that names the operation, so none were
taken. An explicit in-chat authorization from the operator was also tested
against `send-command` and did not lift the gate: it is enforced in host
config, not by user consent.

Inbound `0.0.0.0/0` on port 443 was offered by the operator and **declined**:
nothing on the instance listens on 443, and the SSM agent's 443 traffic is
outbound only, so the rule would have added attack surface for no benefit.
The `builder-security` skill sanctions world-open 443 only for a public web
host (AWS Usage Standard §5.2.7). **The security group still has zero ingress
rules**, verified live after all work completed.

**Then tear the instance down** — g5.8xlarge is ~$2.45/hr on-demand:

```bash
./g5/terminate-instance.sh i-09ee99ff5540c60ec   # or: REGION=us-east-1 ./g5/terminate-instance.sh
```

## Status

| Step | Result |
|---|---|
| Launch g5.8xlarge | done (after AZ fallback to us-east-1) |
| Clone repo on instance | done |
| Pull NeMo 26.04 container | done, 77.4 GB |
| Stack imports + CUDA init on sm_86 | **confirmed** |
| Recipe API + scaled config | **confirmed** |
| Qwen3 tokenizer + 151,936 vocab | **confirmed** |
| Model build / training step | **confirmed** — 50/50 iterations, 0 skipped, 0 NaN |
| Measured peak memory | **confirmed** — 6.84 GiB allocated / 7.26 GiB reserved |
| Measured throughput | **confirmed** — 24,757 tok/s median per step (0.331 s) |
| Instance terminated | see teardown command above |
