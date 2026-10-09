# Qwen3-8B Pre-Training: H200 vs B300 (NeMo/Megatron)

Pre-training **Qwen3-8B** (8.2B dense parameters) on 1T tokens comparing two GPU generations — p5en.48xlarge (H200) and p6-b300.48xlarge (B300) — using NeMo/Megatron on 2-node / 16-GPU topologies with EFA GDRDMA interconnect.

## Results

| Metric | H200 (p5en) | B300 (p6-b300) | B300/H200 | g5 1-node | g5 2-node |
|--------|-------------|----------------|-----------|-----------|-----------|
| Model trained | Qwen3-8B, 8.2B | Qwen3-8B, 8.2B | same | *proxy*, 1.01B | *proxy*, 1.49B |
| GPUs | 16× H200 141 GB | 16× B300 288 GB | — | 1× A10G 24 GB | 2× A10G 24 GB |
| Sequence length | 4096 | 4096 | same | 1024 | 1024 |
| **TFLOP/s per GPU** | 497 | **976** | 1.96× | 34.3 | 21.0 |
| **Throughput** | 162K tok/s | **318K tok/s** | 1.96× | 7,085 tok/s | 5,753 tok/s |
| **Time to 1T tokens** | ~71 days | **~36 days** | 1.97× | ~1,630 days | ~2,010 days |
| Step time (100 iters) | 3.23s | 1.65s | 1.96× | 1.16s | 2.85s |
| Peak memory/GPU | ~114 GB / 141 GB | ~173 GB / 288 GB | — | 19.1 / 22.5 GiB (85%) | 19.3 / 22.5 GiB (86%) |
| MFU | 0.50 | 0.50 | — | 0.27 | 0.17 |

Both clusters are compute-saturated with perfect communication overlap. AllReduce and AllGather are fully hidden behind compute.

**The two g5 columns are not a fourth and fifth data point in the same
comparison.** They train a proportionally scaled *proxy* model — 1.01B and
1.49B parameters at sequence length 1024, not Qwen3-8B at 4096 — because a
24 GB A10G holds 22.5 GiB against Qwen3-8B's 137.3 GiB of static state. They are
in this table to show what the cheap path does and does not reproduce, not to
rank A10G against H200:

- The `Time to 1T tokens` row is arithmetic on a *different and much smaller*
  model. It is in the thousands of days, which is the point: this is a stack and
  memory validation, not a training run.
- The **g5 MFU falls from 0.27 to 0.17 when the second node is added**, while
  the clusters hold 0.50 at 16 GPUs. That is the one genuinely transferable
  finding — at this geometry the g5 step time *is* the gradient all-reduce, with
  compute entirely hidden inside it, confirmed at two model sizes 4.1× apart.
  The clusters are the opposite case: compute-bound with communication hidden.
- g5 `TFLOP/s per GPU` and `MFU` are derived from the repo's own FLOP model
  ([`g5/predict.py`](g5/predict.py), calibrated once and never refitted) times
  the measured step time. The model reproduces measured throughput to 0.03%
  (5,753 implied vs 5,753 measured; 7,087 vs 7,085). Memory, throughput and step
  time are read directly from the runs.

## No cluster? Validate the whole stack on one cheap GPU

The results above need a Slurm cluster with H200s or B300s. If you want to
exercise **the same NeMo 26.04 / Megatron-Bridge stack** — same training script,
same container, real `c4` data, real EFA — on hardware you can start in five
minutes for a couple of dollars, use **[`g5/`](g5/README.md)**.

It runs a proportionally scaled Qwen3 on `g5.8xlarge` (one 24 GB A10G, ~$2.45/hr)
instead of 16 H200s, as two reproducible scenarios with both an EC2 and a
Kubernetes path:

| | hardware | model | measured | cost |
|---|---|---|---|---|
| **[`g5/single-node/`](g5/single-node/README.md)** | 1x g5.8xlarge | 1.01 B params | 19.08 GiB, 7,085 tok/s | ~$2 |
| **[`g5/multi-node/`](g5/multi-node/README.md)** | 2x g5.8xlarge + EFA | 1.49 B params | 19.27 GiB, 5,753 tok/s | ~$7 |

Start with **[`g5/PREREQUISITES.md`](g5/PREREQUISITES.md)** — AWS credentials,
the GPU vCPU quota (the usual hard stop), and a paste-in block that checks
everything before you spend anything.

What it is good for, and what it is not: it validates the stack, the fabric and
the memory model, and it carries a calibrated memory predictor
([`g5/predict.py`](g5/predict.py)) that reproduces every run measured so far.
It is **not** a performance comparison with the clusters above — a 24 GB A10G
holds a model ~5x smaller than Qwen3-8B. Each scenario README is explicit about
which of its numbers are measured and which are still predicted.

## Prerequisites

- **Slurm** workload manager with **PyXis + Enroot** container runtime
- **EFA networking** with GDRDMA support (for multi-node communication)
- **FSx for Lustre** shared filesystem mounted at `/fsx/`
- **Docker** (for building container images)

> **Don't have a cluster?** Deploy a fully functional HPC cluster in under 1 hour using [Amazon SageMaker HyperPod](https://awslabs.github.io/ai-on-sagemaker-hyperpod/). The guide walks you through deploying a ready-to-use cluster with Slurm, EFA, PyXis/Enroot, and FSx for Lustre pre-configured.

## Quick Start

> **Disk space:** The container build requires ~50 GB of disk space in TMPDIR.
> `enroot import` needs `sudo` and TMPDIR pointing to FSx or another file system (not `/tmp`, which is too small).

### Clone this repo and change it its directory
```bash
git clone https://github.com/paragao/qwen3-8b-optimised-pretraining.git
cd qwen3-8b-optimised-pretraining
```

### Prepare datasets (allenai/c4/en)
```bash
# export your Hugging Face token, if you have one
export HF_TOKEN=<your token>

# Prepare the dataset
sbatch preprocessing/preprocess.sh
```
Without your Hugging Face token, the download will be throttled. The script requires a token. 
Datasets are tokenized and transformed into binary mmap accessible files to avoid streaming data (`.idx` and `.bin` files).

### Build the container
```bash
# Build container
docker build -t qwen3-8b-pretraining:latest .

# Setup directories to run
mkdir -p /fsx/tmp && mkdir -p /fsx/ubuntu/qwen3-8b-pretraining/containers/

# Create the squash file with Enroot
sudo TMPDIR=/fsx/tmp ENROOT_TEMP_PATH=/fsx/tmp enroot import --output /fsx/ubuntu/qwen3-8b-pretraining/containers/nemo-efa-26.04.sqsh dockerd://qwen3-8b-pretraining:latest
```

### H200 Cluster (2x p5en.48xlarge)

```bash
# 1. Change to directory
cd h200

# 2. Submit training job
sbatch slurm/run.sh
```
Logs will be written to `/fsx/ubuntu/qwen3-8b-pretraining/logs`.
Checkpoints are saved to `/fsx/ubuntu/qwen3-8b-pretraining/checkpoints`.

### B300 Cluster (2x p6-b300.48xlarge)

```bash
# 1. Change to directory
cd b300

# 2. Submit training job
sbatch slurm/run.sh
```
Logs will be written to `/fsx/ubuntu/qwen3-8b-pretraining/logs`.
Checkpoints are saved to `/fsx/ubuntu/qwen3-8b-pretraining/checkpoints`.

## Running the g5 scenarios on EKS

The two [`g5/`](g5/README.md) scenarios also run on Kubernetes, as the `g5 1-node`
and `g5 2-node` columns in [Results](#results). This path is **self-contained**:
it does *not* need Slurm, FSx, the `sbatch preprocessing/preprocess.sh` step, the
`docker build`, or the `enroot import` above. An init container clones this repo
into the pod and a CPU-only Job builds the `c4` dataset in-cluster, so
`g5/train.py` running on the node is the same file as in the repo and cannot
drift from it.

Both scenarios are kustomize **overlays** on
[`g5/eks/pretrain.yaml`](g5/eks/pretrain.yaml), which stays the single definition
of the Namespace, ConfigMap, PVC, rendezvous Service, dataset Job and training
Job. `kubectl apply -k` needs no extra tooling — kustomize is built into
`kubectl`.

> **Local edits do not reach the cluster.** The init container clones
> `REPO_URL` at `REPO_REF` from the ConfigMap — currently pinned to the
> `feat/g5-single-gpu-validation` branch, which is where the g5 work lives until
> it merges. This is the opposite of the EC2 path, which `scp`s your local files
> to the instance. If you change `g5/train.py` and want the cluster to run it,
> push the change and point `REPO_REF` at your branch or commit.

### Step 0 — credentials, kubeconfig, and proving you can reach the cluster

Do this first. `g5/PREREQUISITES.md` covers the AWS side and **applies to this
path too**, not only to EC2.

```bash
# 1. Prove you have credentials. If this fails, nothing below will work.
aws sts get-caller-identity

# 2. Write a kubeconfig entry for your cluster and select it.
aws eks update-kubeconfig --name <your-cluster> --region <your-region>

# 3. Prove you can actually reach the Kubernetes API.
kubectl get nodes
```

Step 3 is not redundant. **AWS credentials do not grant Kubernetes access** —
they are separate systems. `update-kubeconfig` only writes a local file and never
checks whether you can use it, so it succeeds even when you have no access at
all, and the failure surfaces one step later as:

```
error: You must be logged in to the server (the server has asked for the client to provide credentials)
```

That message reads like expired credentials and usually is not. If
`aws sts get-caller-identity` worked, the cause is RBAC — your principal has no
EKS access entry and no `aws-auth` mapping on that cluster. Check with
`aws eks list-access-entries --cluster-name <your-cluster> --region <your-region>`
and have the cluster owner add you; re-running `update-kubeconfig` will not help.

### Step 1 — confirm it is the RIGHT cluster

`kubectl apply -k` goes to whatever context is current, and **the GPU check below
cannot catch a wrong one.** If you have more than one cluster configured, this is
the step that stops you creating a namespace, a PVC and two training Jobs on
somebody else's cluster.

```bash
# Which cluster am I pointed at?
kubectl config current-context

# What are these nodes REALLY, and where?
kubectl get nodes -o custom-columns=NAME:.metadata.name,\
TYPE:.metadata.labels.'node\.kubernetes\.io/instance-type',\
ZONE:.metadata.labels.'topology\.kubernetes\.io/zone'
```

You want `g5.8xlarge` in the `TYPE` column. Any other GPU instance type will
advertise `nvidia.com/gpu` and sail through the next check while running a model
sized for a 24 GB A10G on the wrong hardware — the measured numbers in
[Results](#results) will not reproduce and nothing will tell you why.

**Both Jobs pin `nodeSelector: node.kubernetes.io/instance-type: g5.8xlarge`.**
So on anything else the pods do not run on the wrong hardware — they do not run
at all, with `0/N nodes are available: N node(s) didn't match Pod's node
affinity/selector`, which never names the label. That pin is deliberate (it keeps
the dataset build in the same AZ as the training pods). To run on a different
type on purpose, override it on both Jobs — note HyperPod reports types with an
`ml.` prefix:

```yaml
  - target:
      kind: Job
      name: c4-prep
    patch: |-
      - op: replace
        path: /spec/template/spec/nodeSelector/node.kubernetes.io~1instance-type
        value: ml.g6e.48xlarge
  - target:
      kind: Job
      name: qwen3-pretrain
    patch: |-
      - op: replace
        path: /spec/template/spec/nodeSelector/node.kubernetes.io~1instance-type
        value: ml.g6e.48xlarge
```

For **scenario 2**, the `ZONE` column must show two `g5.8xlarge` nodes in **the
same** zone. Cross-AZ puts every gradient all-reduce over an AZ boundary, and the
all-reduce is already the limiting factor. A node group spread across three AZs
is the normal default and is *not* what you want here.

### Cluster prerequisites (both scenarios)

```bash
# 1. An EKS cluster with a g5.8xlarge node group — 1 node for scenario 1,
#    2 nodes IN ONE SUBNET AND AZ for scenario 2 (verify with Step 1 above).

# 2. The NVIDIA device plugin, so the nodes advertise nvidia.com/gpu
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
```

An empty `GPU` column means the training pod will never schedule. A *populated*
one means only that the nodes have NVIDIA GPUs — it does not mean they are
A10Gs. Step 1 is what checks that.

**Node disk — the prerequisite that is easiest to miss and hardest to fix.** The
training image needs roughly **100 GB free on the node's root volume**, and a
node that fails this looks healthy by every other check here. Measured on
`nvcr.io/nvidia/nemo:26.04`: 25.9 GB compressed across 120 layers, ~77 GB
unpacked, and containerd needs **both at once** during the pull. A node with less
free space does not fail fast — the pull runs for twenty minutes, crosses
kubelet's eviction threshold, and the pod dies with `Init:ErrImagePull` plus
`The node was low on resource: ephemeral-storage`.

```bash
# Free disk per node, which `kubectl get nodes` does not show
for n in $(kubectl get nodes -o name | cut -d/ -f2); do
  kubectl get --raw "/api/v1/nodes/$n/proxy/stats/summary" 2>/dev/null \
  | python3 -c "import sys,json;f=json.load(sys.stdin)['node']['fs'];\
print('  $n  %.1f GB free of %.1f GB' % (f['availableBytes']/1e9,f['capacityBytes']/1e9))"
done
```

A 107 GB root volume — the SageMaker HyperPod default — is **not enough** unless
the node is nearly empty.

Two things to know before you try to work around it. First, **a large instance
store does not help here.** A `g6e.48xlarge` carries 6.5 TiB of free local NVMe
(see the next paragraph) and the pull still fails, because containerd's data root
is `/var/lib/containerd` on the *root* volume and is not symlinked onto the
instance store. Check the volume containerd actually uses, not the node's total
disk. Second, **the root volume itself cannot be resized, and the field that
looks like it does is not that field.** On HyperPod the instance group takes
`InstanceStorageConfigs` with `EbsVolumeConfig{VolumeSizeInGB, RootVolume}`, and
`RootVolume: true` **forbids** `VolumeSizeInGB` — "the size of the root volume is
determined for you". It exists only so you can supply your own KMS key for the
root volume. An `update-cluster` call pairing the two is rejected.

What works is the *secondary* volume, and the reason it works is a step most
readers would not look for:

```bash
# RootVolume: false is the working form. The size applies to a SECOND
# EBS volume, which HyperPod mounts at /opt/sagemaker.
InstanceStorageConfigs=[{EbsVolumeConfig={VolumeSizeInGB=500,RootVolume=false}}]
```

A second volume at `/opt/sagemaker` would be useless on its own, because
containerd reads from `/var/lib/containerd` on the root volume. The link is the
instance group's **lifecycle script**. AWS's sample `on_create.sh` — the one
HyperPod clusters are normally built from — carries exactly this conditional:

```bash
if [[ $(mount | grep /opt/sagemaker) ]]; then
  # Found secondary EBS volume. Set containerd data root to it.
  sed -i -e "/^[# ]*root\s*=/c\root = \"/opt/sagemaker/containerd/data-root\"" \
    /etc/eks/containerd/containerd-config.toml
fi
```

So the secondary volume **is** the image-storage fix — the volume is the
capacity, the script is what makes containerd use it, and you need both.

**Check that the script's target path exists before you attach anything.** That
conditional never runs on a cluster with no secondary volume, so it is routinely
untested, and on `p6-b200-eks-cluster` it is simply wrong: all six nodes keep
containerd's config at `/etc/containerd/config.toml` and have **no**
`/etc/eks/containerd/` directory at all. Since `on_create.sh` runs under
`set -e` and `sed -i` on a missing file exits non-zero, attaching the volume
there would terminate every node in the group and then fail provisioning on the
replacements.

`g5/eks/NODE-DISK-FIX.md` is the full runbook: how to check your own nodes, a
corrected script that finds the config instead of assuming its path, the
`update-cluster` payload with every field it demands, and how to verify
containerd actually moved rather than trusting the call. Either way it replaces
the nodes, so it is a cluster-admin change, not something a job can do. Failing
that, pre-pull the image onto the node or host it in a registry closer to the
cluster.

**Node-local scratch, and `/fsx`.** Worth knowing before you point `DATA_PATH` at
a network volume. On `g6e.48xlarge` the four 1.9 TB instance-store NVMe disks are
LVM-striped into one ext4 filesystem mounted at **`/opt/dlami/nvme`** — 6.86 TiB,
6.51 TiB free, mode `drwxrwxrwt`. It is **not** at `/scratch` or
`/local_scratch`; those do not exist. kubelet does not count it in
`ephemeral-storage`, so a `hostPath` volume into it is both large and immune to
the eviction threshold that kills the image pull. `/tmp` looks even bigger at
746 GiB but is `tmpfs`, so it spends RAM, not disk.

`/fsx` is a **pod** mount, not a host mount — nothing is mounted at `/fsx` on the
node. It arrives through the FSx CSI driver (`fsx.csi.aws.com`) as a
`ReadWriteMany` PVC, and on this cluster it is 1.2 TiB of Lustre with ~1.1 TiB
free. Check what you have, and where it lives, because a cross-AZ FSx mount is
read every training step:

```bash
# instance-store scratch and the real root-volume figure, per node
for n in $(kubectl get nodes -o name | cut -d/ -f2); do
  echo "== $n"
  kubectl get node "$n" -o jsonpath='   ephemeral-storage (root vol only): {.status.capacity.ephemeral-storage}{"\n"}'
done

# FSx: does a PVC exist, and is the filesystem in the same AZ as the nodes?
kubectl get pvc -A
kubectl get nodes -o custom-columns='NODE:.metadata.name,ZONE:.metadata.labels.topology\.kubernetes\.io/zone'

# Every FSx filesystem in the region, with the AZ of its subnet so you can
# compare against the node zones above. Cross-AZ mounts work but cost latency.
aws fsx describe-file-systems --region "$REGION" \
  --query 'FileSystems[].[FileSystemId,StorageCapacity,SubnetIds[0]]' --output text \
| while read -r fsid gib subnet; do
    az=$(aws ec2 describe-subnets --subnet-ids "$subnet" --region "$REGION" \
         --query 'Subnets[0].AvailabilityZone' --output text 2>/dev/null)
    printf '  %-24s %7s GiB  %-26s %s\n' "$fsid" "$gib" "$subnet" "$az"
  done
```

HyperPod can also mount FSx on the host: the same `InstanceStorageConfigs` takes
`FsxLustreConfig{DnsName, MountName, MountPath}`, where `MountPath` is "the local
path where the Amazon FSx for Lustre file system is mounted on instances" — so a
real host `/fsx` is a cluster-config change, not something these manifests do.

**StorageClass.** The base PVC requests `ReadWriteMany` on a class named
`efs-sc`. Check whether you have one *before* applying anything — a missing class
is the single most likely reason nothing starts:

```bash
kubectl get storageclass
kubectl get storageclass efs-sc    # NotFound here means read the next paragraph
```

RWX is required at 2 nodes, because both pods mmap the same `.bin`/`.idx` and an
EBS volume cannot be mounted by two nodes — the second pod hangs `Pending`. If
`efs-sc` does not exist, the PVC stays `Pending`, both pods stay `Pending`, and
the `wait` in step 4 burns its full timeout before telling you anything. For
**scenario 1 only**, any `ReadWriteOnce` class from `kubectl get storageclass`
will do — provided its `VOLUMEBINDINGMODE` is `WaitForFirstConsumer`. That
matters: **both** Jobs mount this one PVC, and an RWO volume can only be mounted
from one node, so they have to co-locate. `WaitForFirstConsumer` binds the volume
to the first scheduled pod's node and the second pod inherits that affinity,
which is what makes the patch safe. On an `Immediate` class the volume binds to
an arbitrary zone first and the pods can be stranded. Add this to
`g5/single-node/kustomization.yaml` at the end, under the existing `patches:`
key, substituting your own class name:

```yaml
  - target:
      kind: PersistentVolumeClaim
      name: qwen3-data
    patch: |-
      apiVersion: v1
      kind: PersistentVolumeClaim
      metadata:
        name: qwen3-data
        namespace: qwen3-pretrain
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: gp3
```

**Or skip the StorageClass entirely and use the node's own NVMe.** Two optional
kustomize components put the job's bytes on the instance store described above.
Add **one** of them — `local-nvme-data` already includes `local-nvme` — as a
`components:` key in the scenario overlay you are applying:

```yaml
# at the end of g5/single-node/kustomization.yaml (or g5/multi-node/)
components:
  - ../eks/local-nvme
```

| | `../eks/local-nvme` | `../eks/local-nvme-data` |
|---|---|---|
| moves logs/TensorBoard (`/workspace/run`) | yes | yes |
| moves the HuggingFace cache (`HF_HOME`) | yes | yes |
| moves the **c4 dataset** | no | yes |
| needs a StorageClass | yes, RWX at 2 nodes | **no — the PVC is removed** |
| node count | 1 or 2 | **1 only** |

Use `local-nvme` for the scaling benefit: both things it moves are per-node
scratch no other rank reads, so it is correct at either node count. The
`/workspace/run` emptyDir it replaces lives on the **root** volume and counts
against the same `ephemeral-storage` eviction budget the 100 GB image pull is
already straining, so a checkpoint written there can evict the pod that wrote
it. The HF cache is the largest transient write in the run — roughly 3x the
dataset — and on a cross-AZ EFS or FSx mount every byte of it crossed an AZ
boundary.

Use `local-nvme-data` when you have **no RWX StorageClass at all**. It deletes
the PVC, so there is nothing left to stay `Pending`. It is **single-node only**,
and not as a matter of taste: a hostPath is node-local, `c4-prep` runs once on
one node, and at DP=2 both ranks must mmap the same `.bin`/`.idx`. The failure is
at least loud — the base manifest's preflight initContainer checks for
`${DATA_PATH}.bin` before torchrun starts, so rank 1 stops with
`FATAL: /data/c4_qwen3.bin missing` rather than silently training on the mock
dataset. If you see that on exactly one of two pods, this is why.

Three things to check before applying either, in rough order of how likely they
are to bite:

```bash
# 1. The path must exist on every node the pod could land on. hostPath is NOT
#    schedulable-aware: the scheduler places the pod and the mount fails
#    afterwards, so a node without it gives you
#      hostPath type check failed: /opt/dlami/nvme is not a directory
#    This cluster has g6e nodes that HAVE it and c6i nodes that do not.
kubectl get nodes -o custom-columns='NODE:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type'

# 2. Pod Security Admission must allow hostPath. It is forbidden by BOTH the
#    `baseline` and `restricted` Pod Security Standards. Empty output means no
#    enforcement and hostPath is allowed; `restricted` means the Job is
#    rejected at admission with `violates PodSecurity`.
kubectl get ns -o custom-columns='NS:.metadata.name,ENFORCE:.metadata.labels.pod-security\.kubernetes\.io/enforce'

# 3. The build is what the cluster sees, so assert the build. 92 assertions,
#    including that the dataset stays read-only in the training pod.
python3 g5/eks/validate-nvme.py
kubectl apply -k g5/single-node/ --dry-run=server   # proves admission accepts it
```

Both components deliberately use `type: Directory` rather than
`DirectoryOrCreate`, and `validate-nvme.py` fails if that is ever weakened. The
reason is the quiet failure: on a node with no instance store mounted there,
`DirectoryOrCreate` makes an empty directory on the **root** volume instead and
the pod starts normally, filling the disk containerd needs while appearing to
use 6.5 TiB of NVMe. With `type: Directory` the kubelet refuses to start the pod
and names the path. An absent disk must not render as a working one.

Two consequences of a hostPath worth knowing up front. **It does not fix the
image pull** — containerd's data root is on the root volume, so a 100 GB image
still fails on a 107 GB root volume no matter how much NVMe the node has; a
hostPath moves the *job's* bytes, not the *image's*. And **nothing is reclaimed
when you delete the namespace**: the data stays at
`/opt/dlami/nvme/qwen3-pretrain/` on whichever node ran the pod, which is why a
re-run on the same node reuses the dataset instead of re-downloading it, and
also how you leak disk no Kubernetes object accounts for. Delete that one
directory on the node to reclaim.

### Scenario 1 — single node, ~1B parameters, 85% of one A10G

```bash
# 1. Check the Kubernetes and EC2 definitions of the scenario still agree
./g5/scenario-check.sh single-node

# 2. Create the namespace, config, PVC, Service and both Jobs
kubectl apply -k g5/single-node/

# 3. CHECK IT STARTED, before you wait on anything. Both pods must reach
#    Running, and the PVC must be Bound. This takes seconds; step 4 takes
#    half an hour and cannot tell you what step 3 tells you now.
kubectl -n qwen3-pretrain get pods,pvc

# 4. Only once step 3 looks right: wait for the CPU-only dataset build
#    (~10 min; 30 min cold, because of the 77 GB image pull)
kubectl -n qwen3-pretrain wait --for=condition=complete job/c4-prep --timeout=30m

# 5. Watch the training run
kubectl -n qwen3-pretrain logs -f job/qwen3-pretrain -c train

# 6. Teardown — everything is namespaced
kubectl delete namespace qwen3-pretrain
```

Measured: **1,009,385,472 parameters, 19.08 GiB peak allocated (84.8% of
22.49 GiB), 7,085 tok/s**, 0 skipped and 0 NaN iterations. `SEQ_LENGTH=1024` is a
measured ceiling, not a cautious default — 2048 at this geometry OOMed after one
iteration with 38 MiB free. EFA is deliberately off: at one node there is no
inter-node traffic for it to carry.

### When nothing happens — read this before waiting 30 minutes

Everything in this section fails *quietly*. These are the three traps, all three
observed on a real cluster:

**`wait` tells you nothing.** After its full timeout it prints exactly:

```
error: timed out waiting for the condition on jobs/c4-prep
```

No mention of the pod, the volume or the scheduler. That is why step 3 above
comes first.

**`logs` is silent and exits 0.** Against a pod that has not started — and will
never start — every form of the command returns **zero bytes with exit status
0**: `logs job/...`, `logs -f job/...`, and the label-selector form. You cannot
tell "not started yet" from "wrong command" from "finished, printed nothing".
Silence here is not reassurance.

**So ask the pod directly.** This is the command that actually answers the
question, and it is the one a reader is most likely not to think of:

```bash
kubectl -n qwen3-pretrain get pods
kubectl -n qwen3-pretrain describe pod -l app=qwen3-pretrain | sed -n '/Events:/,$p'
```

| what `describe` says | what it means |
|---|---|
| `pod has unbound immediate PersistentVolumeClaims` | the PVC never bound — check `kubectl -n qwen3-pretrain get pvc`; almost always a missing `efs-sc` StorageClass |
| `didn't match Pod's node affinity/selector` | the hardcoded `nodeSelector` — your nodes are not `g5.8xlarge`; see Step 1 for the override |
| `Init:ErrImagePull` + `node was low on resource: ephemeral-storage` | not enough node disk for the ~100 GB image; see the node-disk prerequisite |
| `Insufficient nvidia.com/gpu` | no GPU capacity free, or the device plugin is not running |
| `Insufficient vpc.amazonaws.com/efa` | scenario 2 on nodes without EFA — see its prerequisites |
| `0/N nodes are available` + taint messages | the g5 nodes carry taints your pod does not tolerate |

**Re-applying after ANY change needs the Jobs deleted first.** A Job's pod
template is immutable, so `kubectl apply -k` can create one but never update it.
Change the geometry, the StorageClass or the `nodeSelector` and re-apply, and you
get thousands of characters of Go struct dump ending in the only words that
matter — `spec.template: Invalid value: core.PodTemplateSpec{...}: field is
immutable`. The fix:

```bash
kubectl -n qwen3-pretrain delete job c4-prep qwen3-pretrain
kubectl apply -k g5/single-node/
```

**Read logs by label, not by Job.** `kubectl logs job/<name>` resolves to an
arbitrary pod of that Job, so after any retry it is as likely to pick the dead
pod as the live one — it says `Found 2 pods, using pod/...` and then
`container "prep" in pod "..." is terminated`. Use the selector instead:

```bash
kubectl -n qwen3-pretrain logs -l app=qwen3-c4-prep -c prep --tail=20
kubectl -n qwen3-pretrain logs -l app=qwen3-pretrain -c train --tail=20
```

**Do not use `--dry-run=server` as a pre-flight.** It looks like the careful
thing to do and produces five alarming errors that are pure artefact — the
namespace is not really created in a dry run, so every namespaced object reports
`namespaces "qwen3-pretrain" not found`. The manifest is fine. Use
`./g5/scenario-check.sh` and `kubectl kustomize g5/single-node/` to inspect it
instead.

**If teardown is refused or hangs.** `kubectl delete namespace` can block for
minutes on finalizers with no output, and some environments block the command
outright. Delete the objects individually instead — this leaves only an empty
namespace behind:

```bash
kubectl -n qwen3-pretrain delete job c4-prep qwen3-pretrain --ignore-not-found
kubectl -n qwen3-pretrain delete pvc qwen3-data --ignore-not-found
kubectl -n qwen3-pretrain delete svc qwen3-rdzv --ignore-not-found
kubectl -n qwen3-pretrain delete configmap qwen3-config --ignore-not-found
```


### Scenario 2 — two nodes over EFA, DP=2, ~1.5B parameters

Two further cluster prerequisites, and the second is the one that usually bites:

```bash
# 1. The EFA device plugin
helm repo add eks https://aws.github.io/eks-charts
helm install aws-efa-k8s-device-plugin --namespace kube-system \
  eks/aws-efa-k8s-device-plugin

# 2. Confirm the nodes actually advertise the resource
kubectl get nodes -o custom-columns=NAME:.metadata.name,EFA:.status.allocatable.'vpc\.amazonaws\.com/efa'
```

If that column is empty the pods stay `Pending` with
`Insufficient vpc.amazonaws.com/efa`. Two distinct causes, and the plugin being
installed rules out neither:

- **The node group was not created with EFA on its launch template.** An EFA
  interface cannot be added to a running node, so such a node group has to be
  replaced. Neither the plugin nor any script here can do it for you.
- **The nodes are not labelled.** The Helm chart's DaemonSet selects
  `efa=true`, so on an unlabelled cluster it sits at `DESIRED 0` and advertises
  nothing while looking installed. Confirm with
  `kubectl get ds -n kube-system aws-efa-k8s-device-plugin` — a `DESIRED` of 0
  means no node matched, not that the chart failed.

The node security group must also allow **all traffic both ways to and from
itself**. Not inbound only, and not a CIDR rule: EFA is not IP, so the default
`0.0.0.0/0` egress rule cannot match it. Missing that outbound rule cost a run on
2026-10-04 — libfabric selected the `efa` provider and the handshake then failed
with `Unresponsive receiver (reachable by EFA device but handshake failed)`.

```bash
./g5/scenario-check.sh multi-node
kubectl apply -k g5/multi-node/

# CHECK IT STARTED FIRST. Both pods must be Running and the PVC Bound. At two
# nodes there are TWO pods, and seeing only one is itself the symptom — most
# often `Insufficient vpc.amazonaws.com/efa` on the second node.
kubectl -n qwen3-pretrain get pods,pvc -o wide

kubectl -n qwen3-pretrain wait --for=condition=complete job/c4-prep --timeout=30m

# Both ranks at once. `kubectl logs -f job/...` follows only ONE pod, so a
# 2-pod job needs the label selector and a raised request cap. Rank 0 carries
# the geometry and the memory summary; rank 1 carries the per-iteration
# timings, because Megatron's print_rank_last writes there.
kubectl -n qwen3-pretrain logs -f -l app=qwen3-pretrain -c train \
  --prefix --max-log-requests 2

kubectl delete namespace qwen3-pretrain
```

Everything in [When nothing happens](#when-nothing-happens--read-this-before-waiting-30-minutes)
applies here too, and more so: this scenario can also hang at the rendezvous with
both pods `Running` and no error anywhere. If rank 1 never appears, check that
the two pods landed in the **same availability zone** (the Step 1 command shows
`ZONE`) and that both nodes advertise the EFA resource.

Measured: **1,490,550,784 parameters, 19.27 GiB peak allocated (85.7%), 5,753
tok/s** over 1000 iterations, loss 11.3351 → 5.4236, 0 skipped and 0 NaN.
`2048/18 = 113.8` matches Qwen3-8B's `4096/36 = 113.8` exactly, so this is the
best-proportioned proxy available at this memory budget.

The second node buys **+48% model capacity at lower throughput**, not speed. It
is the only honest summary of the 2-node result: `use_distributed_optimizer`
shards 12 of the 18 bytes/parameter of static state, so per-rank static memory
drops and a larger model fits — while the step time stays pinned to the gradient
all-reduce.

### What the overlays change, and why all three values move together

The multi-node overlay sets **three** things, because any one of them alone is a
*silent* misconfiguration rather than an error:

| | set alone, you get |
|---|---|
| `NNODES: "2"` in the ConfigMap | a hang at the rendezvous, waiting for a pod that was never created |
| `parallelism`/`completions: 2` on the Job | two pods each training alone at `WORLD_SIZE=1`, both looking healthy |
| `vpc.amazonaws.com/efa: 1` on the container | silent fallback to TCP — a **3.1× measured** throughput loss |

`./g5/scenario-check.sh` asserts these against the **built** manifest rather than
the patch text, so a patch that silently fails to apply is caught, and it
cross-checks every key against the `scenario.env` the direct-EC2 path reads. Two
copies of a geometry drift invisibly: both files stay valid and both paths keep
running while measuring different models.

`python3 g5/eks/validate.py` additionally checks the base manifest's own
invariants — the four node-count values, the Service cross-references, and that
the rendezvous stays headless with no NodePort, LoadBalancer or `hostNetwork`.

`python3 g5/eks/validate-nvme.py` covers the two optional `local-nvme`
components, which `validate.py` cannot: it reads the base manifest directly, and
`local-nvme-data` removes the PVC its checks assume. The nvme validator instead
*builds* each overlay-plus-component pair and asserts the result — 92
assertions, including that the dataset stays read-only in the training pod and
that `type: Directory` is never weakened to `DirectoryOrCreate`. It also carries
one deliberate negative case: pairing `local-nvme-data` with `multi-node` is the
unsupported combination, so the suite asserts that it is *detectable* and counts
the detection as a pass.

> **Do not change the rendezvous Service to `type: NodePort` or
> `type: LoadBalancer` to make debugging easier.** It is headless
> (`clusterIP: None`) on purpose; publishing it would expose an unauthenticated
> PyTorch rendezvous store on the node or the internet. Use `kubectl logs` or
> `kubectl port-forward`, which tunnel through the API server.

Each scenario's own README has the full detail, including which figures are
measured and which are still predicted:
[`g5/single-node/README.md`](g5/single-node/README.md),
[`g5/multi-node/README.md`](g5/multi-node/README.md).
[`g5/PREREQUISITES.md`](g5/PREREQUISITES.md) covers the AWS-account side —
credentials, the GPU vCPU quota, cost — and applies to **both** this path and the
direct-EC2 one. [`g5/results/eks-human-run-notes.md`](g5/results/eks-human-run-notes.md)
records a step-by-step run of this section on a real cluster, including the
eleven things that went wrong and which of them these instructions now cover.

## Model Architecture: Qwen3-8B



| Parameter | Value |
|-----------|-------|
| Layers | 36 |
| Hidden dim (d_model) | 4096 |
| Q-heads | 32 |
| KV-heads | 8 (GQA) |
| FFN dim | 12288 (SwiGLU) |
| Vocab size | 151,936 |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Sequence length | 4096 |
| Precision | BF16 |
| Total params | 8.2B |

## Parallelism Strategy

**Pure Data Parallelism (DP=16)** — the model fits entirely on a single GPU.

| Component | Setting |
|-----------|---------|
| Tensor Parallel | 1 |
| Pipeline Parallel | 1 |
| Data Parallel | 16 |
| Distributed Optimizer | Yes (shards Adam states across DP ranks) |
| Overlap Grad Reduce | Yes |
| Overlap Param Gather | Yes |

**Why TP=1 is optimal:** At 8.2B params, the model + optimizer states fit on one GPU with distributed optimizer. Adding tensor parallelism introduces all-reduce communication for every transformer layer — validated experimentally: TP=2 was 11% slower (868 vs 976 TFLOP/s on B300).

## Best Configuration Per Cluster

| Parameter | H200 (p5en.48xlarge) | B300 (p6-b300.48xlarge) |
|-----------|---------------------|------------------------|
| GPUs | 16× H200 (141 GB HBM3) | 16× B300 (288 GB HBM3e) |
| Parallelism | TP=1, PP=1, DP=16 | TP=1, PP=1, DP=16 |
| **Micro-batch size** | **2** | **4** |
| Global batch size | 128 (grad_accum=4) | 128 (grad_accum=2) |
| Sequence length | 4096 | 4096 |
| Precision | BF16 | BF16 |
| **Gradient checkpointing** | Selective (core_attn only) | Selective (core_attn only) | 
| Distributed optimizer | Yes (sharded Adam) | Yes (sharded Adam) |
| Overlap grad reduce | Yes | Yes |
| Overlap param gather| Yes | Yes |
| Framework | Megatron-Bridge (NeMo 26.04) | Megatron-Bridge (NeMo 26.04) |

## Key Findings

1. **Both clusters are compute-saturated with perfect communication overlap.** AllReduce and AllGather are fully hidden behind compute — verified by single-GPU benchmarks showing lower TFLOP/s due to reduced batch arithmetic intensity.

2. **Both clusters use the Megatron-Bridge recipe API.** NeMo 26.04 for both H200 and B300.

3. **Pure data parallelism is optimal** when the model fits in single-GPU memory. Distributed optimizer + overlapped grad reduce eliminate the memory penalty.

4. **Selective gradient checkpointing used on both clusters:** lightweight core_attn recompute is Megatron-Core's standard behavior, keeping H200 peak at ~114 GB (MBS=2) and B300 at ~173 GB (MBS=4).

## Hardware

| | H200 Cluster | B300 Cluster |
|---|---|---|
| Instance | p5en.48xlarge | p6-b300.48xlarge |
| Nodes | 2 | 2 |
| GPUs per node | 8× H200 | 8× B300 |
| GPU Memory | 141 GB HBM3 | 288 GB HBM3e |
| Interconnect | EFA GDRDMA (3200 Gbps) | EFA GDRDMA (6400 Gbps) |
| Intra-node | NVLink (900 GB/s) | NVLink (1800 GB/s) |

## Project Structure

```
├── README.md              ← You are here
│   Dockerfile             ← NeMo 26.04 + EFA container
├── h200/
│   ├── train.py           ← Megatron-Bridge training script
│   └── slurm/
│       └── run.sh         ← Slurm submission script
├── b300/
│   ├── train.py           ← Megatron-Bridge training script
│   └── slurm/
│       └── run.sh         ← Slurm submission script
└── g5/                    ← single-GPU validation on cheap hardware, no Slurm
    ├── README.md          ← measured results + the memory model
    ├── PREREQUISITES.md   ← START HERE: tools, credentials, quota, cost
    ├── predict.py         ← calibrated memory predictor (--self-check)
    ├── launch-instance.sh ← provision, terminate-instance.sh tears down
    ├── prepare-c4.sh      ← tokenize real c4 on the instance
    ├── finish-run.sh      ← 1-node driver; finish-run-2node.sh for 2
    ├── single-node/       ← scenario 1: scenario.env + kustomization.yaml
    ├── multi-node/        ← scenario 2: + max-model.py sizing search
    ├── eks/               ← shared kustomize base for both scenarios
    └── results/           ← logs from every run on record
```

## License

MIT-0
