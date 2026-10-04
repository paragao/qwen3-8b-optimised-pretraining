# Scenario 2 — two `g5.8xlarge` over EFA, DP=2, maximum memory

Two A10Gs, data-parallel, with the gradient all-reduce on EFA rather than TCP.

## Measured: the EFA path works

`g5/results/run-2node-20261004-221909-rank1.log`, 2026-10-04, 1000 iterations
on real c4:

| | |
|---|---|
| transport | `Selected provider is efa` on **both** ranks |
| throughput | **25,056 tok/s** median, stdev 460 (1.8%) |
| step time | 0.654 s at 16,384 tok/step |
| steady-state steps | 99 |
| stability | loss 11.9155 → 5.5249, 0 skipped, 0 NaN |

Against the alternatives at the same geometry:

| | tok/s | vs 1 node |
|---|---|---|
| 1 node, GBS 8 | 24,727 | — |
| 2 nodes, **TCP** | 8,086 | 0.33x |
| 2 nodes, **EFA** | 25,056 | **1.01x** |

**EFA is worth 3.10x against TCP, and the second node is still worth +1.3%.**
Both numbers matter: EFA is not optional at two nodes, and two nodes is not a
speed-up for this geometry. The step is bandwidth-bound either way — 1,033 MB
moves per step, and per-rank compute is ~0.331 s against 0.654 s of
communication, so overlap cannot hide it. Full derivation in the main
[`../README.md`](../README.md).

## That measurement used a different geometry from this scenario

It ran the 4-layer `smoke` shape and peaked at **4.83 GiB — 21% of the card**.
So it validates the fabric, the throughput and the stability, and says
**nothing** about memory occupancy.

To reproduce that run exactly rather than this scenario:

```bash
NUM_LAYERS=4 HIDDEN_SIZE=1024 FFN_HIDDEN_SIZE=3072 \
  NUM_ATTENTION_HEADS=8 NUM_QUERY_GROUPS=2 \
  GLOBAL_BATCH_SIZE=16 TRAIN_ITERS=1000 ./g5/finish-run-2node.sh
```

## This scenario: the ~1B geometry at DP=2 — NOT YET MEASURED

`scenario.env` uses the same ~1B shape measured at 19.08 GiB on one node, with
`GLOBAL_BATCH_SIZE=16` so each rank still processes 8 sequences.

Per-rank memory should come out **lower** than 19.08 GiB, which is
counter-intuitive enough to be worth stating: `g5/train.py` enables
`use_distributed_optimizer` when `WORLD_SIZE > 1`, which **shards optimizer
state across the data-parallel group**. At 1.009 B parameters the Adam moments
plus fp32 master weights are ~12 B/param ≈ 11.3 GiB, so DP=2 removes ~5.6 GiB
per rank. Activations are unchanged, because per-rank sequences are unchanged.

Estimate: **~13.4 GiB per rank, ~9 GiB headroom.** That is arithmetic from two
measurements (the 1B static state and its activation residual), not a reading.
Treat it as a prediction this scenario is designed to test.

### The next test, and its refutation threshold

`SEQ_LENGTH=2048` at this geometry **OOMed on one node** (20.05 GiB allocated,
38 MiB free). The sharding above says it should fit at DP=2 with room to spare.
If it does not, the sharding estimate is wrong and should be re-measured rather
than re-fitted.

```bash
SEQ_LENGTH=2048 ./g5/finish-run-2node.sh      # after sourcing scenario.env
```

## Reproduce on EC2 directly

```bash
cd /path/to/qwen3-g5-validate
set -a; . g5/multi-node/scenario.env; set +a   # FIRST: USE_EFA is read by the
                                               # launcher, not by the run driver
NODES=2 ./g5/launch-instance.sh                # attaches EFA, sets the SG shape
./g5/fix-cluster-sg.sh                         # only if the SG predates EFA support
./g5/finish-run-2node.sh
./g5/terminate-instance.sh i-0aaa i-0bbb       # both ids; ~$4.90/hr for the pair
```

Order matters. `USE_EFA` is a **provisioning** knob — `launch-instance.sh`
attaches the interface and chooses the security-group shape, and
`fix-cluster-sg.sh` converges an existing group. The run itself **autodetects**
the fabric from `/dev/infiniband`, because the device is either there or it is
not and a flag could only disagree with the hardware. Sourcing `scenario.env`
after the launch would leave `USE_EFA` with nothing to act on. (`USE_EFA=1` is
already the default at `NODES=2`, so a launch without it is still correct — the
ordering is about the file meaning what it says.)

The driver builds the c4 dataset on both nodes, verifies they are
byte-identical, syncs Megatron's index cache (there is no shared filesystem),
and retrieves both ranks' logs.

## Reproduce on Kubernetes

```bash
kubectl apply -k g5/multi-node/
kubectl -n qwen3-pretrain logs -f job/qwen3-pretrain
```

**Three settings must agree, and the overlay sets all three** because any one
alone is a silent misconfiguration:

| | value | alone it gives you |
|---|---|---|
| `NNODES` in the ConfigMap | 2 | with `parallelism: 1`, a hang at the rendezvous waiting for a pod that was never created |
| Job `parallelism` / `completions` | 2 | with `NNODES: 1`, two pods each training alone |
| `vpc.amazonaws.com/efa` on the container | 1 | without it, `USE_EFA=1` selects the provider, finds no device, and falls back to TCP — a measured 3x loss |

### Cluster prerequisites beyond the base manifest's

1. **The EFA device plugin**, or an EFA-requesting pod stays `Pending` with
   `Insufficient vpc.amazonaws.com/efa`:
   ```bash
   helm repo add eks https://aws.github.io/eks-charts
   helm install aws-efa-k8s-device-plugin --namespace kube-system \
     eks/aws-efa-k8s-device-plugin
   ```
2. **A node group created with EFA on its launch template.** An EFA interface
   cannot be added to a running node, so a node group without one must be
   replaced. Confirm the nodes advertise it:
   ```bash
   kubectl get nodes -o custom-columns=\
   NAME:.metadata.name,EFA:.status.allocatable.'vpc\.amazonaws\.com/efa'
   ```
   An empty column means the pods will never schedule.
3. **A security group allowing all traffic both ways to and from itself.** Not
   just inbound, and not a CIDR rule — EFA is not IP, so the default
   `0.0.0.0/0` egress cannot carry it. Missing the outbound half cost a run:
   libfabric selected the efa provider and the handshake then failed with
   `Unresponsive receiver (reachable by EFA device but handshake failed)`.
4. **Both nodes in one subnet and one AZ.** Cross-AZ puts every gradient
   all-reduce over an AZ boundary, and the all-reduce is already the limit.

## Check the two paths still agree

```bash
./g5/scenario-check.sh multi-node
```

Asserts every key in `scenario.env` matches the **built** manifest, that
`NNODES` equals the Job's `parallelism` and `completions`, and that the EFA
settings are mutually consistent — including that `FI_EFA_USE_DEVICE_RDMA`
stays empty, since g5's EFA has no rdma-read and setting it aborts both ranks
on startup.

Mutation-tested: breaking the `parallelism` coupling, setting
`FI_EFA_USE_DEVICE_RDMA=1`, and requesting `USE_EFA=1` without the device
resource all fail it.

## Expected output

```
==> Syncing the Megatron index cache from node 0 to node 1
    all three splits present (train, valid, test) -- the cache is complete

NET/OFI Selected provider is efa, fabric is efa (found 1 nics)
 iteration       10/    1000 | ... lm loss: 1.19E+01
```

`NET/OFI No eligible providers were found` means the EFA device did not reach
the container — the run will still complete, about 3x slower, over TCP.
