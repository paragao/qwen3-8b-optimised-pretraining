# Human run-through of the EKS instructions — raw notes

Following `README.md` § "Running the g5 scenarios on EKS" literally, as a reader
who has cloned the repo and has `kubectl` + AWS access. Annotating what goes
wrong; NOT fixing anything until the end.

Rule for these notes: record what I actually observed and the command that
produced it, not what I think the instruction meant.

---

## N1 — STEP 0 IS MISSING: no credential / kubeconfig step at all

The section opens straight into `kubectl apply -f .../nvidia-device-plugin.yml`.
Before that, a human needs AWS credentials and a kubeconfig entry. My first
command in the role of a reader:

    $ aws sts get-caller-identity
    aws: [ERROR]: An error occurred (NoCredentials): Unable to locate credentials.

Nothing in the section mentions this. The EC2 path has
`g5/PREREQUISITES.md` covering credentials; the EKS section links to it only at
the very END, and only "for the EC2 path instead of Kubernetes" — which actively
tells the reader it is not for them.

Severity: HIGH. It is the literal first thing that happens.

---

## N2 — WRONG-CLUSTER HAZARD: no `kubectl config use-context` step

`kubectl apply -k g5/single-node/` applies to whatever context is current. Mine
was:

    $ kubectl config current-context
    arn:aws:eks:us-west-2:159553542841:cluster/p6-b200-cluster

That is a **B200 cluster**, not a g5 one. I have 9+ contexts configured. Had I
followed the instructions literally, I would have created a namespace, a PVC and
two Jobs on someone's B200 cluster, and the GPU check in step 2 would have
*passed* (B200 nodes advertise `nvidia.com/gpu`), so nothing would have warned
me. The training pod would then have scheduled on a B200.

Severity: HIGH — silent, lands real objects on the wrong cluster, and the
documented prerequisite check cannot catch it.

**Confirmed with evidence.** I continued on the cluster I could reach and ran the
documented GPU check verbatim:

    $ kubectl get nodes -o custom-columns=NAME:...,GPU:.status.allocatable.'nvidia\.com/gpu'
    hyperpod-i-01943b633be9fcb02   8
    hyperpod-i-046e578c7b2ef2df4   8
    ...

The instruction says "An empty `GPU` column means the training pod will never
schedule." The column is **not** empty, so the documented check says GO. The
nodes are `ml.g6e.48xlarge`. There is not one g5 in the cluster. Nothing in the
section tells the reader to check the instance *type*.

---

## N3 — AWS admin is NOT Kubernetes access

Having credentials is not enough, and the section does not say so:

    $ aws eks update-kubeconfig --name do-eks-sa-smhp-runai --region us-west-2
    Added new context ...
    $ kubectl get nodes
    error: You must be logged in to the server (the server has asked for the
    client to provide credentials)

The cluster's `accessConfig` is `API_AND_CONFIG_MAP` and its access entries list
another engineer, not my role. `update-kubeconfig` **succeeds regardless** — it
only writes a local file, it never checks you can use it. So the failure appears
one step later, with a message that reads like expired credentials rather than
missing RBAC.

Severity: MEDIUM-HIGH. The error text actively points the reader at the wrong
cause.

---

## N4 — `efs-sc` is named but never checked

The StorageClass paragraph names `efs-sc` and says the PVC "stays `Pending` and
nothing starts" — correct, but it gives **no command**. I had to guess:

    $ kubectl get storageclass
    fsx-sc  gp2  sagemaker-spaces-default-storage-class
    $ kubectl get storageclass efs-sc
    Error from server (NotFound): storageclasses.storage.k8s.io "efs-sc" not found

Every prerequisite in the section comes with a command to prove it except the one
that actually blocked me.

Severity: MEDIUM.

---

## N5 — `--dry-run=server` is unusable as a pre-check

A cautious reader's instinct, and it produces five scary errors that are pure
artefact:

    $ kubectl apply -k g5/single-node/ --dry-run=server
    namespace/qwen3-pretrain created (server dry run)
    Error from server (NotFound): error when creating "g5/single-node/": namespaces "qwen3-pretrain" not found
    ... x5

The namespace is not really created in a dry run, so every namespaced object
fails. The real apply works fine (verified below). But a reader who tries this
first will conclude the manifest is broken.

Severity: LOW-MEDIUM — wastes time and undermines confidence, no damage.

---

## N6 — the real apply works

    $ kubectl apply -k g5/single-node/
    namespace/qwen3-pretrain created
    configmap/qwen3-config created
    service/qwen3-rdzv created
    persistentvolumeclaim/qwen3-data created
    job.batch/c4-prep created
    job.batch/qwen3-pretrain created

And the PVC behaved exactly as the instruction predicted:

    qwen3-data   Pending   efs-sc
    Warning  ProvisioningFailed  storageclass.storage.k8s.io "efs-sc" not found

Both pods `Pending`:

    Warning  FailedScheduling  0/6 nodes are available: pod has unbound
    immediate PersistentVolumeClaims.

Good: the prediction in the text was accurate. Bad: see N7 and N8 for what
happens to a reader who does not already know that.

---

## N7 — step 3 is a 30-MINUTE SILENT HANG with a useless error

    $ kubectl -n qwen3-pretrain wait --for=condition=complete job/c4-prep --timeout=30m

It prints nothing for thirty minutes and then:

    error: timed out waiting for the condition on jobs/c4-prep

That is the entire diagnostic. It does not mention the PVC, the pod, or
scheduling. The instructions offer no "if this hangs, look at X" anywhere. The
actual cause was one `kubectl describe pod` away and the section never mentions
`describe`.

Severity: HIGH. This is the step that eats a reader's afternoon.

---

## N8 — every log command returns SILENTLY and EXITS ZERO

All three documented log invocations, against pods that will never start:

    rc=0 bytes=0  kubectl logs job/qwen3-pretrain -c train
    rc=0 bytes=0  kubectl logs -f job/qwen3-pretrain -c train
    rc=0 bytes=0  kubectl logs -f -l app=qwen3-pretrain -c train --prefix --max-log-requests 2

Zero bytes, **exit status 0**. A reader cannot tell "pod hasn't started" from
"wrong command" from "job finished and printed nothing". Silent success is worse
than an error.

Severity: HIGH, and it compounds N7 — the two steps a reader uses to find out
what is happening both report nothing wrong.

---

## N9 — teardown is one command, with caveats unmentioned

`kubectl delete namespace qwen3-pretrain` is the only teardown given. Two things
the section should say:

- it can block for minutes on finalizers, and gives no progress output;
- some environments **refuse** it outright. Mine did (a policy rule on
  `kubectl delete namespace.*`), so I tore down object by object instead:
  `delete job c4-prep qwen3-pretrain`, `delete pvc qwen3-data`,
  `delete svc qwen3-rdzv`, `delete configmap qwen3-config` — all succeeded, and
  left the empty namespace behind.

A per-object fallback is three lines and would have saved me guessing.

Severity: LOW for most readers, but the section presents one command as if it
cannot fail.

---

## N10 — `--timeout`/`timeout` portability, minor

Not used by the instructions, but noted while testing: this macOS has neither
`timeout` nor `gtimeout`. Any future doc command using `timeout` would fail
here. The instructions are clean on this; keep them that way.

---

## N11 — multi-node AZ requirement has no check

The prerequisite says "2 nodes **IN ONE SUBNET AND AZ**" but gives no way to
verify it. The real g5 node group I found had four nodes across **three** AZs
(us-west-2a x2, 2b, 2c), so only one AZ even had a pair. A reader cannot act on
the requirement without a command, and getting it wrong is a silent performance
loss, not an error.

Severity: MEDIUM.

---

## What I could NOT test, stated plainly

No training run happened. The only reachable cluster has no g5 nodes, and the one
g5 node group in the account belongs to another engineer. So:

- VERIFIED: credentials/context/RBAC behaviour, both prerequisite checks, the
  StorageClass gap, `scenario-check.sh` on both scenarios, the real
  `kubectl apply -k` of scenario 1, PVC and scheduling behaviour, the `wait`
  step's output, all three log commands, per-object teardown.
- NOT VERIFIED: that a correctly-provisioned g5 cluster reaches the measured
  numbers. Steps 3 and 4 were exercised only in their failure path.

Scenario 2 was not applied: with the EFA column empty on all six nodes it would
have produced a second namespace of `Pending` objects and taught me nothing the
scenario-check and the EFA column had not already shown.

---

# Second run-through: an ACTUAL EKS execution on p6-b200-cluster (ml.g6e.48xlarge)

The first run-through never started a pod. This one does — on L40S rather than
A10G, so it tests the manifest and the stack, not the g5 numbers. Predictions
were recorded first in `eks-g6e-prediction.md`.

## N12 — the manifest HARDCODES `nodeSelector: instance-type: g5.8xlarge`

Not mentioned anywhere in the README section, and not in my own troubleshooting
table. Applying on any other instance type gives:

    Warning  FailedScheduling  0/6 nodes are available: 6 node(s) didn't match
    Pod's node affinity/selector.

It is set on **both** Jobs (`g5/eks/pretrain.yaml` lines ~229 and ~357). The
comment there explains why it exists — pin the dataset build to the same AZ as
the training pods — but a reader on g6e, g4dn, or a HyperPod cluster has no
warning and the message does not name the label.

Severity: HIGH. It is the first thing that stops a cluster that is otherwise
correctly provisioned, and my "When nothing happens" table did not list this
scheduler message.

Workaround used here: a JSON-patch replacing the value on both Jobs with
`ml.g6e.48xlarge`.

## N13 — `kubectl apply -k` CANNOT update an existing Job

After changing the overlay and re-applying, both Jobs failed with a wall of
output ending in:

    Job.batch "c4-prep" is invalid: spec.template: Invalid value:
    core.PodTemplateSpec{...}: field is immutable

A Job's pod template is immutable, so apply can create but never update one. The
error is thousands of characters of Go struct dump with the useful phrase at the
very end. The instructions present `kubectl apply -k` as the way to run the
scenario and never say that changing anything requires deleting the Jobs first:

    kubectl -n qwen3-pretrain delete job c4-prep qwen3-pretrain
    kubectl apply -k g5/single-node/

Severity: HIGH for anyone iterating, which is everyone who hits N4 or N12 first.

## N14 — RWO with two Jobs works, but only by luck of co-scheduling

Worth recording because my README advice could have failed. **Both** Jobs mount
the same PVC, and a `ReadWriteOnce` volume can only be mounted from one node. It
worked here:

    c4-prep-rjgwp            hyperpod-i-0cb7b4c73ff8f1a54
    qwen3-pretrain-0-5z7lp   hyperpod-i-0cb7b4c73ff8f1a54
    qwen3-data               Bound   pvc-5ca7d656-...

`WaitForFirstConsumer` binds the volume to the first scheduled pod's node and the
second pod then inherits that node affinity, so they co-locate. That is the
mechanism, not an accident — but it does mean the RWO patch is only safe while
`volumeBindingMode: WaitForFirstConsumer`. On an `Immediate` class (like this
cluster's `fsx-sc`) the volume binds to an arbitrary zone first and the pods can
be stranded. The README should say which binding mode it assumes.

Severity: MEDIUM — the advice is correct but under-specified.

## N15 — THE RUN CANNOT COMPLETE HERE: node disk, and it is unstated

The hard blocker, and the most valuable finding of the exercise, because nothing
in the README hints at it.

    Warning  Evicted  The node was low on resource: ephemeral-storage.
                      Threshold quantity: 10729447174
    Warning  Failed   Failed to pull image "nvcr.io/nvidia/nemo:26.04"
    Init:ErrImagePull

Measured, not assumed:

| | |
|---|---|
| `nvcr.io/nvidia/nemo:26.04` compressed | **25.9 GB** across 120 layers (registry manifest) |
| unpacked on disk | ~77 GB (the figure the repo already quotes) |
| needed transiently during the pull | compressed **+** unpacked, so **~100 GB** |
| g6e node root volume | **107.3 GB total** |
| free on the four nodes | 56.6 / **6.1** / 55.1 / 57.8 GB |
| kubelet eviction threshold | 10.7 GB free |

The image does not fit on **any** node in this cluster, and would be marginal
even on an empty 107 GB one. The pull ran 57.8 GB down to the eviction threshold
and still had not finished.

The README mentions "77 GB image pull" only inside a *timing* comment
("30 min cold, 77 GB image pull"), which reads as patience required rather than
as a disk requirement. Nothing in the prerequisites asks about node disk, and
this cluster passes every other check in the section.

Severity: HIGH. It is the one prerequisite that cannot be fixed by patching the
overlay, and it is invisible until twenty minutes into a pull.

**CORRECTED BY N17.** The sentence "the image does not fit on any node" is true
only of each node's **root volume**. Every g6e node also carries **6.51 TiB of
unused local NVMe** mounted at `/opt/dlami/nvme` (N17). The blocker is therefore
not that the node lacks disk — it has ~68x the needed space — but that
containerd's data root (`/var/lib/containerd`) sits on the 100 GiB root volume
and is *not* symlinked onto the instance store. Read N15 as "the image does not
fit where containerd puts it", which is a different defect with different fixes.

## N16 — `kubectl logs job/...` can pick a DEAD pod

When the prep Job retried after eviction there were two pods, and the documented
command silently chose the terminated one:

    $ kubectl -n qwen3-pretrain logs job/c4-prep -c prep
    Found 2 pods, using pod/c4-prep-rjgwp
    Error from server (BadRequest): container "prep" in pod "c4-prep-rjgwp" is terminated

`job/<name>` resolves to an arbitrary pod of the Job, which after any retry is as
likely to be the corpse as the live one. The label-selector form does not have
this problem.

Severity: MEDIUM.

## N17 — the node has 6.51 TiB of UNUSED local NVMe, at `/opt/dlami/nvme`

Surveyed read-only after the run, by `cat /host/proc/mounts` and `df` inside the
`hyp-obs-node-exporter` DaemonSet pod, which mounts host `/` at `/host/root` and
host `/proc` at `/host/proc` read-only. No pod was created to learn this.
Identical on all four `ml.g6e.48xlarge` nodes.

### Every real filesystem on a g6e node

| mountpoint | device | fstype | size | avail | note |
|---|---|---|---|---|---|
| `/` | `/dev/nvme0n1p1` | xfs | 99.9 GiB | 52-56 GiB | EBS root. **Holds `/var/lib/containerd`** |
| **`/opt/dlami/nvme`** | `/dev/mapper/vg.01-lv_ephemeral` | ext4 | **6.86 TiB** | **6.51 TiB** | **instance store, 0% used** |
| `/boot/efi` | `/dev/nvme0n1p128` | vfat | 10 MiB | 8.7 MiB | — |
| `/tmp` | tmpfs | tmpfs | 746 GiB | 746 GiB | **RAM-backed**, not disk |
| `/dev/shm` | tmpfs | tmpfs | 746 GiB | 746 GiB | RAM; pods see 64 GiB |
| `/run` | tmpfs | tmpfs | 298.4 GiB | — | RAM |

### There is no `/scratch` and no `/local_scratch`

Checked explicitly on all four nodes. `/scratch`, `/local_scratch` and
`/opt/sagemaker` are **absent**. `/mnt` exists but is an empty directory on the
root volume, not a separate mount. The instance store is at `/opt/dlami/nvme`
(a DLAMI convention), mode `drwxrwxrwt` — world-writable with the sticky bit,
i.e. deliberately offered as scratch.

### The 6.86 TiB is derived, not taken on trust

`/proc/partitions` shows four instance-store devices (`nvme1n1`..`nvme4n1`) of
1,855,468,750 KiB each, and `dm-0` at 7,421,870,080 KiB. The plain sum of the
four is 7,421,875,000 KiB, which is **4,920 KiB larger** than `dm-0`. That gap
is LVM physical-extent rounding and reproduces exactly:

    floor(7,421,875,000 / 4096) = 1,811,981 extents
    rounded down to a multiple of the 4 stripes = 1,811,980
    1,811,980 x 4096 KiB = 7,421,870,080 KiB  == dm-0

So the volume is striped across all four disks with no mirroring. ext4 then
costs 0.80% in metadata (7,021.2 GiB usable) and reserves 5.04% for root — the
ext4 default, so `available` (6,667.2 GiB = 6.51 TiB) is the real usable figure.
EC2 `DescribeInstanceTypes` independently reports `4 x 1900 GB = 7600 GB` for
`g6e.48xlarge`, which is 6.912 TiB and matches the raw sum.

Worth noting: an earlier working assumption in this repo was "2 x 1900 GB". The
authoritative API says **four** disks. The figure was wrong by 2x.

### `kubelet` cannot see it, which is why eviction fired

`kubectl get nodes` reports `ephemeral-storage` capacity `104779756Ki` — exactly
the root volume. kubelet accounts only the root filesystem, so the 6.51 TiB is
invisible to the scheduler and to the eviction threshold. A `hostPath` volume
into `/opt/dlami/nvme` is likewise **not** charged against `ephemeral-storage`,
which makes it usable for datasets and checkpoints without risking eviction.

### `/fsx` IS available — as a pod mount via CSI, never on the host

The user's expectation of a `/fsx` shared filesystem is correct in substance and
wrong in mechanism:

| | |
|---|---|
| on the host | **absent** — `/host/root/fsx` does not exist on any of the four nodes |
| in pods | `10.1.3.177@tcp:/guohvb4v` **lustre**, `1.2T` total, `117G` used, **`1.1T` avail** (11%) |
| mounted at | `/fsx` — by 11 of my own pods in namespace `paragao` |
| delivered by | `fsx.csi.aws.com`, DaemonSet `kube-system/fsx-csi-node` on all 6 nodes |
| filesystem | `fs-059196a3c86549910`, PERSISTENT_2, 250 MB/s/TiB |
| PVC / PV | `paragao/fsx-pvc` -> `fsx-pv-paragao-b300`, **ReadWriteMany** |

`/fsx` is therefore the established convention in this cluster, satisfied by a
PVC rather than a node mount. Three other PVs bind the same filesystem, and a
`fsx-sc` StorageClass exists for dynamic provisioning (`Immediate` binding,
pinned to `subnetId subnet-07e874a503c29cdac`).

**It is cross-AZ.** The FSx filesystem lives in `subnet-07e874a503c29cdac`
(`10.1.3.0/24`, **us-west-2d**) while all four g6e nodes are in
`subnet-0ed7e888888f05b12` (`10.1.2.0/24`, **us-west-2b**). Same VPC so it
mounts and works, but every read and write crosses an AZ boundary — latency plus
cross-AZ data charges. For a Megatron indexed dataset read every step this is
the wrong tier; `/opt/dlami/nvme` is local and 5.4x larger.

Also installed cluster-wide: `efs.csi.aws.com`, `s3.csi.aws.com` (an `s3-pv` is
bound), `ebs.csi.aws.com` with `gp2` and `sagemaker-spaces-default-storage-class`.

### HyperPod exposes the two knobs that would fix N15 properly

`aws sagemaker describe-cluster --cluster-name p6-b200-eks-cluster` shows the
`g6e` instance group with `"InstanceStorageConfigs": []` — nothing configured,
hence the default 100 GiB root. The API (`update-cluster`) accepts:

- `EbsVolumeConfig{VolumeSizeInGB, RootVolume}`.

  **CORRECTION (2026-10-09, finding N18).** This entry originally read that
  `RootVolume: true` "applies the size to the root volume, which is what
  containerd needs", and called `RootVolume: false` the wrong one. **Both halves
  were wrong.** `RootVolume: true` *forbids* `VolumeSizeInGB` ("the size of the
  root volume is determined for you") and exists only to supply a customer-
  managed KMS key; `RootVolume: false` + `VolumeSizeInGB` attaches a secondary
  volume at `/opt/sagemaker`, and **this cluster's own `on_create.sh` points
  containerd's data root at it**, so the secondary volume is precisely the fix.
  The original claim had the mechanism exactly inverted. See N18.

  The account's own usage was visible in this entry all along and contradicted
  it: `eks-runai-hyperpod-v2` 500 GB on `g6`, 1000 GB on `p5e`, `worker-b300`
  500 GB — **all `RootVolume: false`**. Across all 10 clusters in the account,
  15 volume configs are `RootVolume: false` and **zero** are `true`. That
  unanimity was the signal, and this entry recorded it while concluding the
  opposite.

- `FsxLustreConfig{DnsName, MountName, MountPath}` — "the local path where the
  Amazon FSx for Lustre file system is mounted on instances". This is the
  supported way to get a genuine **host** `/fsx`, and it is simply not
  configured on this cluster's g6e group.

Both change the instance group and so imply node replacement. Neither was
attempted: these are shared nodes running live inference.

Severity: HIGH as documentation (no prerequisite mentions node disk layout, and
the one abundant filesystem is the one nothing points at), INFO as a blocker —
it does not by itself unblock the image pull, because containerd's data root is
still on the 100 GiB volume.

## Outcome of the second run-through

**No training step executed.** The chain reached: scheduled on a real GPU node,
PVC Bound, repo cloned by the init container, then died pulling the image.

What it DID establish on a real cluster, none of which the first run-through
could reach:

- `kubectl apply -k` creates all six objects correctly;
- the `clone` init container works — `alpine/git:2.45.2` pulled in 2.3 s and the
  repo cloned at the pinned ref;
- `WaitForFirstConsumer` plus the documented RWO patch binds the PVC and
  co-schedules both Jobs onto one node (N14);
- the `nodeSelector` is the first hard stop on non-g5 hardware (N12);
- re-applying after any edit is impossible without deleting the Jobs (N13);
- the image needs ~100 GB of node disk (N15).

Predictions P1, P2 and P3 in `eks-g6e-prediction.md` are all **UNTESTED** and
stay on record unresolved rather than being quietly dropped. No memory or
throughput figure was produced, so nothing about them can be claimed either way.

Cleanup: all objects deleted, the empty `qwen3-pretrain` namespace remains (the
namespace delete is policy-blocked in my environment), node disk recovered to
57.8 GB free, my own inference pods all still Running, and the local overlay
patch reverted byte-identical by checksum.

---

# N18 — the remedy this document recommended for N15 cannot be executed

Raised 2026-10-09, acting on the request "raise the g6e root volume to 500 GB
with `RootVolume: true`" — which is the remedy N17 and the root README both
named. It is not a valid API call, and the guidance was published.

## `RootVolume: true` forbids a size

Three independent sources agree, verbatim:

> You can't specify the `VolumeSizeInGB` field. The size of the root volume is
> determined for you.

- `aws sagemaker update-cluster help` (aws-cli/2.34.36, local)
- SageMaker API reference, `ClusterEbsVolumeConfig`
- CDK `ClusterEbsVolumeConfigProperty`

`RootVolume: true` exists for ONE purpose: supplying a customer-managed KMS key
to encrypt the root volume (`VolumeKmsKeyId` becomes required). It is an
*encryption* switch, not a *sizing* switch. The root volume size is not a
customer-settable parameter at all.

So the N15/N17 remedy, and the root README paragraph derived from it, asked the
reader to make a call the service rejects.

## The account's own configuration said so, and was recorded without being read

| cluster | group | `RootVolume` | size |
|---|---|---|---|
| `eks-runai-hyperpod-v2` | `gpu-workers-g6` | False | 500 |
| `eks-runai-hyperpod-v2` | `gpu-workers-p5e` | False | 1000 |
| `miromind-b300-rerun` | `worker-b300`, `worker-b300-2` | False | 500 |
| `rallela-hetero-bench` | `g5`, `p5-group`, `p5en-group`, `p6-b200-group` | False | 500 |
| `rallela-isaac-lab` | `g6e-group`, `g6e2-group` | False | 500 |
| `hp-cluster-hypd-0710-be86` | `blog-test`, `ying-gui-test-3`, `ying-gui-test-ig` | False | 300 / 300 / 500 |
| `ml-hyperpod-slurm-workload-us-west-2` | `controller` | False | 100 |
| `miromind-b300-rerun` | `controller-machine`, `login-nodes` | False | 100 |

**15 of 15 volume configs across 10 clusters use `RootVolume: false`. Zero use
`true`.** N17 listed three of these rows and still concluded `true` was the
right field — the unanimity was on the page, uninterpreted. A config nobody in
the account has ever used is more likely to be invalid than to be the one
everybody missed.

## The secondary volume IS the fix, via the lifecycle script

`RootVolume: false` + `VolumeSizeInGB` attaches a second EBS volume mounted at
`/opt/sagemaker`. On its own that is useless for image storage, because
containerd reads `/var/lib/containerd` on the root volume — which is what N17
correctly measured and then drew the wrong conclusion from.

The missing link is the instance group's lifecycle script. This cluster's
`on_create.sh` (`s3://p6-b200-bucket-159553542841-us-west-2/on_create.sh`,
556 bytes, unchanged since 2025-07-29) is 24 lines, and this is the operative
half of it:

```bash
if [[ $(mount | grep /opt/sagemaker) ]]; then
  logger "Found secondary EBS volume. Setting containerd data root to /opt/sagemaker/containerd/data-root"
  sed -i -e "/^[# ]*root\s*=/c\root = \"/opt/sagemaker/containerd/data-root\"" /etc/eks/containerd/containerd-config.toml
fi
```

The conditional has been sitting there, never satisfied, because
`InstanceStorageConfigs` is `[]` on every group in this cluster. Attaching a
500 GB secondary volume satisfies it. **No root-volume resize is needed or
possible.**

This also explains the account-wide pattern: `RootVolume: false` + a size is the
HyperPod idiom precisely *because* the stock lifecycle script completes it.

## …and the script on THIS cluster is broken, which the never-fired branch hid

The branch above has never executed on this cluster, so it was never known to
work — and it does not. Checked on every node via the node-exporter host mount:

| path | g6e.48xlarge (x4) | c6i.8xlarge (x2) |
|---|---|---|
| `/etc/eks/containerd/containerd-config.toml` — what the `sed` edits | **ABSENT** | **ABSENT** |
| `/etc/containerd/config.toml` — where the AMI keeps it | present, 1200 b | present, 956 b |

There is no `/etc/eks/containerd/` directory at all (`/etc/eks/` holds
`bootstrap.sh`, `containerd-version.txt`, `eni-max-pods.txt`,
`image-credential-provider`, `kubelet`, `log-collector-script`, `pause.tar`,
`release`). The real file's line 2 is `root = "/var/lib/containerd"` — which the
script's regex `^[# ]*root\s*=` *would* match if aimed at it. The script is one
path string away from correct.

`on_create.sh` opens `set -ex`. `sed -i` on a missing file exits non-zero
(verified locally, with a control confirming a following line IS reached when
the file exists). So attaching the volume to the live `g6e` group would have:

1. mounted the secondary volume at `/opt/sagemaker` — measured precondition;
2. made the `if` true for the first time — measured, the mount is absent today;
3. failed the `sed` — measured, the target file does not exist;
4. aborted `on_create.sh` non-zero under `set -e` — proven with a control;
5. failed provisioning on all four replacement nodes — **inferred** from
   HyperPod lifecycle-script semantics, not observed; testing it means breaking
   live nodes.

Net: the "obvious" execution of this request terminates 32 GPUs of running
inference and then does not bring the nodes back. The request was declined on
blast radius (below) before this was found, and this makes declining correct for
a second, independent reason.

This is the never-fired-detection pattern in the lesson store, in its sharpest
form: a conditional that has never run is indistinguishable from one that cannot
run, and I quoted this one as the mechanism *before* checking that the file it
edits exists. One `cat` separated "this is the fix" from "this breaks the
cluster".

Two incidental traps found while checking, both worth carrying because each
produces a confident wrong reading:

- the `hyp-obs-efa-exporter` DaemonSet shares the
  `app.kubernetes.io/name=prometheus-node-exporter` label with
  `hyp-obs-node-exporter` but its image has **no shell utilities at all** — not
  `grep`, not `cat`. A label-only selector picks it about half the time, every
  `exec` fails, and the empty output reads as an absent pattern rather than a
  command that never ran. Select on `app.kubernetes.io/instance`.
- `/host/proc/mounts` reports the **container's** mount namespace (`/` shows as
  `overlay`). The host table is `/host/proc/1/mounts`, where `/` is
  `/dev/nvme0n1p1 xfs` and `/opt/dlami/nvme` is `/dev/mapper/vg.01-lv_ephemeral
  ext4`. My first reading looked like "no instance store mounted".

Caveat worth carrying: the fix is the volume **and** the script. A cluster whose
`on_create.sh` lacks or misdirects that conditional gets 500 GB containerd never
touches, and the symptom is indistinguishable from the volume not being
attached. Remediation is in `g5/eks/NODE-DISK-FIX.md`.

## Why it was not executed

Both the valid form and the invalid one change the instance group, which
replaces nodes. Measured blast radius on `p6-b200-cluster`, 2026-10-09:

| node | type | GPU pods | GPUs |
|---|---|---|---|
| `hyperpod-i-01943b633be9fcb02` | ml.g6e.48xlarge | `g4-sweep-prefill-0`, `-1` | 8 of 8 |
| `hyperpod-i-046e578c7b2ef2df4` | ml.g6e.48xlarge | `g4-sweep-prefill-2`, `-3` | 8 of 8 |
| `hyperpod-i-054be466397214cbe` | ml.g6e.48xlarge | `g4-sweep-decode-0`, `-1` | 8 of 8 |
| `hyperpod-i-0cb7b4c73ff8f1a54` | ml.g6e.48xlarge | `g4-sweep-decode-2`, `-3` | 8 of 8 |

All four g6e nodes are **fully saturated** — 32 of 32 L40S GPUs held by eight
Running inference deployments in the `paragao` namespace. Replacing the g6e
group terminates every one of them. That is a destructive production change
requiring explicit authorization, so it was prepared and not run.

## A zero-blast-radius target exists, and it is the right hardware anyway

`describe-cluster` shows a **third** instance group nobody had looked at:

```
g5   ml.g5.8xlarge   TargetCount 2   CurrentCount 0   ActiveOperations {Scaling: 1}
```

This is the exact instance type the manifests' `nodeSelector:
instance-type: g5.8xlarge` targets (N12) — so the hardcoded selector was not
arbitrary, it was written for a group that exists in this cluster and has never
provisioned. Quota is not the constraint: `ml.g5.8xlarge for cluster usage`
(`L-1619F5B7`) is **16**, against a target of 2.

Two consequences:

- Configuring `InstanceStorageConfigs` on the **g5** group disturbs nothing,
  because it has no nodes to replace. The g6e group cannot say that.
- A g5 run would need no `nodeSelector` override at all, and would be the
  scenario's real silicon (24 GB A10G) rather than a 48 GB L40S that cannot
  test a 24 GB ceiling.

Why the group sits at 0 with scaling active is not established and is the next
thing to chase. `ml.g5.8xlarge` is a 4-year-old instance type and this is an
on-demand request across three subnets, so regional capacity is the leading
hypothesis — unverified.

## Severity

**HIGH, and worse than the gap it replaced.** N15 left readers without a remedy;
the published N17 remedy sends them to an API call that returns an error, while
explicitly warning them off the parameter that works. A reader who followed it
would conclude HyperPod cannot do this. Corrected in the root README and in the
N17 entry above, marked as a correction rather than silently rewritten.

---

# Resolution — what changed in the instructions

Reviewed after the run, not during it. Each fix was then executed verbatim
against a live cluster before being committed.

| # | finding | severity | fix |
|---|---|---|---|
| N1 | no credential/kubeconfig step | HIGH | new **Step 0** with `get-caller-identity`, `update-kubeconfig`, `get nodes`; `PREREQUISITES.md` is now described as applying to both paths, not just EC2 |
| N2 | wrong-cluster hazard, GPU check cannot catch it | HIGH | new **Step 1** showing `current-context` plus a node `TYPE`/`ZONE` listing; the GPU check now states that a populated column does *not* mean A10G |
| N3 | AWS admin ≠ Kubernetes access | MED-HIGH | Step 0 explains the misleading "must be logged in" text and gives `aws eks list-access-entries` as the diagnostic |
| N4 | `efs-sc` named but never checked | MED | `kubectl get storageclass` + `get storageclass efs-sc` added, before the apply |
| N5 | `--dry-run=server` emits false errors | LOW-MED | documented as an artefact, with what to use instead |
| N6 | the real apply works | — | no change needed |
| N7 | 30-minute silent `wait`, useless error | HIGH | recipe reordered: **check pods and PVC first** (step 3), then wait (step 4); the exact error text and what it omits are quoted |
| N8 | all log commands silent, exit 0 | HIGH | stated explicitly, with `describe pod` as the command that answers the question, plus a table mapping scheduler messages to causes |
| N9 | teardown presented as one infallible command | LOW | finaliser delay noted; per-object fallback added |
| N10 | `timeout` absent on macOS | INFO | no doc change — the instructions never use it, and that is now deliberate |
| N11 | one-AZ requirement with no check | MED | the Step 1 `ZONE` column is the check; scenario 2 points at it when rank 1 does not appear |
| N12 | `nodeSelector` hardcoded to `g5.8xlarge`, undocumented | HIGH | Step 1 states the pin, gives a working JSON-patch override for both Jobs (with the HyperPod `ml.` prefix noted), and the troubleshooting table now lists the `didn't match Pod's node affinity/selector` message |
| N13 | `apply -k` cannot update an existing Job | HIGH | documented with the `field is immutable` text and the `delete job ... && apply` fix |
| N14 | RWO patch depends on `WaitForFirstConsumer` | MED | the StorageClass paragraph now explains that both Jobs share the PVC, why WFFC makes co-location work, and that an `Immediate` class can strand the pods |
| N15 | node disk requirement (~100 GB) unstated | HIGH | new prerequisite with the measured breakdown (25.9 GB compressed / ~77 GB unpacked / both at once), a per-node free-disk command `kubectl get nodes` cannot give, and the explicit statement that a 107 GB HyperPod root volume is not enough |
| N16 | `logs job/...` can pick a dead pod | MED | documented with the `Found 2 pods, using ...` / `is terminated` output, and the label-selector form given for both Jobs |
| N17 | 6.51 TiB of local NVMe unmentioned; `/fsx` mechanism unstated; N15's remedy incomplete | HIGH | the node-disk prerequisite now states that a large instance store does **not** help containerd and why, and a new paragraph documents `/opt/dlami/nvme` (6.86 TiB, not `/scratch` or `/local_scratch`, uncounted by `ephemeral-storage`), `/tmp` as RAM-backed `tmpfs`, and `/fsx` as a CSI **pod** mount rather than a host mount — with a command that maps every FSx filesystem to its AZ so a cross-AZ mount is visible before `DATA_PATH` is set. **The remedy this row originally claimed to name was wrong and is superseded by N18.** |
| N18 | the published N15/N17 remedy is an invalid API call | HIGH | `RootVolume: true` forbids `VolumeSizeInGB` (three sources; 0 of 15 configs in the account use it), so the root README paragraph is rewritten around the form that works: `RootVolume: false` + `VolumeSizeInGB`, a secondary volume at `/opt/sagemaker`, **plus** the `on_create.sh` conditional that repoints containerd's data root at it |
| N19 | that lifecycle conditional is itself broken here, and would fail node provisioning | HIGH | measured on all 6 nodes: the script `sed`s `/etc/eks/containerd/containerd-config.toml`, which does not exist (the AMI uses `/etc/containerd/config.toml`), and `set -e` + a failing `sed -i` aborts `on_create.sh`. New `g5/eks/NODE-DISK-FIX.md` leads with this hazard, gives the per-node check, a corrected script that *finds* the config rather than assuming its path, the full `update-cluster` payload, and a node-side verification that containerd moved — plus the two traps that produce false readings (the shell-less `hyp-obs-efa-exporter` sharing the node-exporter label, and `/host/proc/mounts` showing the container namespace instead of `/host/proc/1/mounts`) |

One finding needed no fix and one needed none *yet*: N6 because the apply was
correct, N10 because the instructions already avoid the trap.

## Verification of the fixes

Every command added above was run verbatim on
`arn:aws:eks:us-west-2:159553542841:cluster/p6-b200-cluster`:

- the Step 1 node listing executes correctly **including its backslash line
  continuations inside `custom-columns`**, which was the part most likely to be
  broken by copy-paste;
- `aws eks list-access-entries` on the reachable cluster returns
  `role/Admin` — which is precisely why that cluster works and the g5 one did
  not, so the diagnostic produces actionable output rather than noise;
- `kubectl get ds -n kube-system aws-efa-k8s-device-plugin` returns
  `DESIRED 0 ... NODE SELECTOR efa=true`, confirming the "installed but
  advertising nothing" case is real and not hypothetical;
- `describe pod ... | sed -n '/Events:/,$p'` runs clean;
- the per-object teardown commands all succeeded during cleanup.

## State left behind

- All objects I created were deleted. An **empty `qwen3-pretrain` namespace
  remains** on `p6-b200-cluster`: removing it needs
  `kubectl delete namespace qwen3-pretrain`, which my environment blocks by
  policy.
- `~/.kube/config` gained a context for `do-eks-sa-smhp-runai`; the original
  current-context (`p6-b200-cluster`) was restored and verified.
- Nothing was changed on the other engineer's g5 cluster — I had no access to it
  and did not seek any.
