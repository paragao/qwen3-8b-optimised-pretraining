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

## Not yet verified

The failure happened *before* model construction, so these remain open:

- Qwen3 model build on `sm_86` — GQA, RoPE, RMSNorm, SwiGLU kernel paths
- the mock dataset / dataloader
- a forward -> backward -> Adam optimizer step, and loss movement
- **measured** peak memory against the predicted 6.02 GiB static

The predicted figures are analytic only. `param_count()` reproducing Qwen3-8B at
8.190 B parameters shows the arithmetic is self-consistent, but no measurement
has confirmed it on hardware. Treat 6.02 GiB as an estimate, not a result.

## Reproducing from here

The instance already has the repo cloned, the 77.4 GB image pulled, and the
pre-fix scripts staged at
`/home/ubuntu/qwen3-g5/qwen3-8b-optimised-pretraining`. To finish:

```bash
aws ssm start-session --target i-09ee99ff5540c60ec --region us-east-1
sudo su - ubuntu
cd ~/qwen3-g5/qwen3-8b-optimised-pretraining
git fetch origin && git checkout feat/g5-single-gpu-validation   # picks up the fix
./g5/run.sh
```

Expected on success: the pre-flight report, ~20 training iterations at
`LOG_INTERVAL=1`, then a `VALIDATION COMPLETE` block with peak allocated and
peak reserved memory.

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
| Model build / training step / peak memory | **not run** — fix committed, re-run needed |
| Instance terminated | see teardown command above |
