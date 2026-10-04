# Scenario 1 — single `g5.8xlarge`, ~1B parameters, maximum memory

One A10G (22.49 GiB usable), one ~1B-parameter model, filling as much of the
card as it will actually hold.

## Measured

`g5/results/run-20261003-101233.log`, 2026-10-03, real c4 data:

| | |
|---|---|
| parameters | **1,009,385,472** |
| peak allocated | **19.08 GiB** — 84.8% of 22.49 GiB |
| peak reserved | 19.43 GiB |
| throughput | **7,085 tok/s** median |
| MODEL_TFLOP/s | 34.3 (MFU 27.4%) |
| stability | 0 skipped, 0 NaN over 50 steps |
| headroom left | 3.41 GiB |

## Why `SEQ_LENGTH=1024` is the ceiling

Because 2048 was tried and it failed. `g5/results/run-20261003-104814.log` is
the identical geometry at `SEQ_LENGTH=2048`: it completed **one** iteration,
reached 20.05 GiB allocated with **38.25 MiB free**, then died trying to
allocate 36 MiB.

So 84.8% is the measured ceiling for this shape on this card, and the remaining
3.41 GiB is allocator headroom, not spare capacity to claim. Measured
fragmentation overhead (reserved ÷ allocated) is 1.018 here.

The main README's profile table described `1b2k` as "not run (predicted
20.73)". It was run, and it OOMed — corrected there.

## Reproduce on EC2 directly

```bash
cd /path/to/qwen3-g5-validate
NODES=1 ./g5/launch-instance.sh          # one g5.8xlarge, zero-ingress SG
./g5/prepare-c4.sh                        # ~50M tokens of real c4
set -a; . g5/single-node/scenario.env; set +a
./g5/finish-run.sh
./g5/terminate-instance.sh                # g5.8xlarge bills ~$2.45/hr
```

`scenario.env` is the geometry and nothing else, so `set -a` exports it into
the environment `finish-run.sh` reads. It is also the file
`g5/scenario-check.sh` compares the Kubernetes manifest against.

## Reproduce on Kubernetes

```bash
kubectl apply -k g5/single-node/
kubectl -n qwen3-pretrain logs -f job/qwen3-pretrain
```

An overlay on `../eks`, which stays the single definition of the Namespace,
ConfigMap, PVC, rendezvous Service, dataset Job and training Job. The overlay
changes only the geometry — a copy would drift from the base, and the base has
been corrected several times.

Cluster prerequisites are the base manifest's: an EKS cluster with a
`g5.8xlarge` node group, the NVIDIA device plugin so nodes advertise
`nvidia.com/gpu`, and an RWX StorageClass for the dataset PVC. See the header of
[`../eks/pretrain.yaml`](../eks/pretrain.yaml).

**EFA is deliberately off.** At one node there is no inter-node traffic for it
to carry, so it would cost an interface and change nothing.

## Check the two paths still agree

```bash
./g5/scenario-check.sh single-node
```

Two definitions of one geometry drift silently — both files stay valid, both
paths keep running, and they measure different models. This compares them key
by key against the **built** manifest (not the patch text, so a patch that
fails to apply is caught) and asserts `NNODES` matches the Job's `parallelism`
and `completions`.

Mutation-tested: changing `HIDDEN_SIZE` in `scenario.env` alone, and setting
`USE_EFA=1` without an EFA device resource, both fail it.

## Expected output

```
  TOTAL                 :    1009.4 M
  tokens per step       : 8,192 (8 seq x 1024 tok)

VALIDATION COMPLETE
  peak allocated : 19.08 GiB
  peak reserved  : 19.43 GiB
```

Within a few hundred MiB of 19.08 GiB is a match. Materially lower means the
geometry did not take effect — check the resolved `num_layers` and
`hidden_size` echoed near the top of the log rather than trusting the knobs you
set.
