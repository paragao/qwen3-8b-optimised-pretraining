# Fixing HyperPod node disk for the image pull (cluster-admin)

The training image needs ~100 GB free on the node's **root** volume and the
SageMaker HyperPod default root volume is ~107 GB, so the pull fails on any node
that is not nearly empty. See the node-disk prerequisite in the repo root
`README.md` for the measurement. This file is the operational fix.

Read all of it before running anything. The naive version of this change can
terminate your nodes **and fail to bring them back**, and that is not a
hypothetical — it is the measured state of `p6-b200-eks-cluster` (2026-10-09).

## What does not work, and why it looks like it should

`ClusterEbsVolumeConfig.RootVolume: true` *forbids* `VolumeSizeInGB` — "the size
of the root volume is determined for you" — and exists only to supply a
customer-managed KMS key. **The root volume is not resizable.** An
`update-cluster` call pairing the two is rejected by the service.

Across the 10 HyperPod clusters in account 159553542841, 15 of 15 volume configs
use `RootVolume: false` and none use `true`. That is the idiom, and the next
section is why.

## How the fix actually works — two halves, and the second one is easy to miss

`RootVolume: false` + `VolumeSizeInGB` attaches a **secondary** EBS volume, which
HyperPod mounts at `/opt/sagemaker`. On its own that does nothing for image
storage, because containerd reads its data root from a config file pointing at
`/var/lib/containerd` on the root volume.

The link is the instance group's **lifecycle script**. AWS's sample
`on_create.sh` carries this conditional:

```bash
if [[ $(mount | grep /opt/sagemaker) ]]; then
  sed -i -e "/^[# ]*root\s*=/c\root = \"/opt/sagemaker/containerd/data-root\"" \
    /etc/eks/containerd/containerd-config.toml
fi
```

So the volume is the capacity and the script is what makes containerd use it.
Both are required. A volume without the script is storage containerd never
touches, and the symptom is indistinguishable from the volume not being
attached.

## STOP — verify the script's target path exists on your nodes

**This conditional has never fired on any cluster whose
`InstanceStorageConfigs` is empty**, so it has never been exercised, and a
branch that has never run is not known to work. On `p6-b200-eks-cluster` it is
broken:

| path | g6e.48xlarge (x4) | c6i.8xlarge (x2) |
|---|---|---|
| `/etc/eks/containerd/containerd-config.toml` — what the script edits | **ABSENT** | **ABSENT** |
| `/etc/containerd/config.toml` — where the AMI actually keeps it | present, 1200 b | present, 956 b |

There is no `/etc/eks/containerd/` directory at all; `/etc/eks/` holds
`bootstrap.sh`, `kubelet`, `release` and friends. The real file's line 2 is
`root = "/var/lib/containerd"`, which the script's regex *would* match — if it
were pointed at it.

`on_create.sh` begins `set -ex`, and `sed -i` on a missing file exits non-zero
(verified, with a control proving a later line is reached when the file does
exist). So attaching the volume without fixing the script gives you this:

1. the secondary volume mounts at `/opt/sagemaker`;
2. the `if` becomes true for the first time ever;
3. the `sed` fails — no such file;
4. `set -e` aborts `on_create.sh` with a non-zero exit;
5. node provisioning fails, on every node in the group at once.

Steps 1-4 are measured or proven. Step 5 is the documented consequence of a
failed lifecycle script, inferred rather than observed — testing it means
breaking live nodes. Either way, do not attach the volume before fixing the
script.

Check your own nodes. Substitute a node name and a node-exporter pod on it:

```bash
NS=hyperpod-observability
NODE=<your-node>
POD=$(kubectl get pods -n $NS -l app.kubernetes.io/instance=hyp-obs-node-exporter \
  --field-selector "spec.nodeName=$NODE" -o jsonpath='{.items[0].metadata.name}')

# rc=0 means the script's target exists; rc!=0 means the script is broken
kubectl exec -n $NS "$POD" -- cat /host/root/etc/eks/containerd/containerd-config.toml >/dev/null; echo "target rc=$?"
kubectl exec -n $NS "$POD" -- cat /host/root/etc/containerd/config.toml | grep -nE '^\s*root\s*='
```

Two gotchas in that snippet, both of which cost a round to find. The
`hyp-obs-efa-exporter` DaemonSet shares the
`app.kubernetes.io/name=prometheus-node-exporter` label but its image has **no
shell utilities at all** — not even `cat` — so select on
`app.kubernetes.io/instance=hyp-obs-node-exporter`. And filter output locally:
piping `grep` *inside* the container fails the same way, and a command that
could not run returns nothing, which reads exactly like a pattern that is
absent.

## Step 1 — fix the lifecycle script if its target is missing

Make the script find the file instead of assuming a path, and make a genuinely
missing config an explicit failure rather than a silent one:

```bash
if mountpoint -q /opt/sagemaker; then
  CFG=""
  for c in /etc/containerd/config.toml /etc/eks/containerd/containerd-config.toml; do
    [[ -f "$c" ]] && { CFG="$c"; break; }
  done
  if [[ -z "$CFG" ]]; then
    logger "FATAL: secondary volume present but no containerd config found"
    exit 1
  fi
  logger "Pointing containerd data root at /opt/sagemaker (config: $CFG)"
  sed -i -e "/^[# ]*root[[:space:]]*=/c\\root = \"/opt/sagemaker/containerd/data-root\"" "$CFG"
fi
```

`mountpoint -q` also replaces `$(mount | grep /opt/sagemaker)`, which matches any
line *containing* the string. Upload it to the group's lifecycle location:

```bash
CLUSTER=p6-b200-eks-cluster
REGION=us-west-2
URI=$(aws sagemaker describe-cluster --cluster-name "$CLUSTER" --region "$REGION" \
  --query 'InstanceGroups[0].LifeCycleConfig.SourceS3Uri' --output text)
aws s3 cp "$URI/on_create.sh" "./on_create.sh.bak-$(date +%Y%m%d)"   # keep the original
aws s3 cp ./on_create.sh "$URI/on_create.sh"
```

## Step 2 — capture the group's CURRENT spec

`update-cluster` takes a whole instance-group specification, not a patch. Fields
you omit are not left alone, so read them first and put them all back.

```bash
GROUP=g6e
aws sagemaker describe-cluster --cluster-name "$CLUSTER" --region "$REGION" \
  --query "InstanceGroups[?InstanceGroupName=='$GROUP'].{Name:InstanceGroupName,\
Type:InstanceType,Count:TargetCount,Role:ExecutionRole,Threads:ThreadsPerCore,\
LC:LifeCycleConfig,Storage:InstanceStorageConfigs}" --output json
```

## Step 3 — check what you are about to terminate

The storage config takes effect at node creation, so applying it **replaces
every node in the group**:

```bash
kubectl get pods -A -o custom-columns=\
'NS:.metadata.namespace,POD:.metadata.name,NODE:.spec.nodeName,\
GPU:.spec.containers[*].resources.limits.nvidia\.com/gpu' --no-headers \
  | awk '$4!="<none>" && $4!=""'
```

A group with `CurrentCount: 0` is free to reconfigure. One carrying live work is
a maintenance window, and `NodeRecovery: Automatic` does not make node
replacement non-disruptive. On `p6-b200-eks-cluster` the `g6e` group was at 32
of 32 GPUs in use when this was written, and the `g5` group at 0 nodes — so the
two groups are not remotely equivalent targets.

## Step 4 — apply

Edit every value to match what Step 2 printed. `RootVolume: false` is correct
and deliberate.

```bash
aws sagemaker update-cluster \
  --cluster-name "$CLUSTER" --region "$REGION" \
  --instance-groups '[{
    "InstanceGroupName": "g6e",
    "InstanceType": "ml.g6e.48xlarge",
    "InstanceCount": 4,
    "ExecutionRole": "arn:aws:iam::ACCOUNT:role/YOUR-ExecutionRole",
    "ThreadsPerCore": 2,
    "LifeCycleConfig": {
      "SourceS3Uri": "s3://YOUR-BUCKET",
      "OnCreate": "on_create.sh"
    },
    "InstanceStorageConfigs": [
      { "EbsVolumeConfig": { "VolumeSizeInGB": 500, "RootVolume": false } }
    ]
  }]'
```

Two documented failure modes: the call "will not function as expected" while
deep health checks are running, and if `MinInstanceCount` is not met within
three hours the group rolls back to its previous settings.

## Step 5 — verify containerd actually moved

A mounted volume with containerd still on the root volume is the failure this
runbook exists to prevent, and it is silent. Check the node, not the API. Note
the host mount table is at `/host/proc/1/mounts` — `/host/proc/mounts` reflects
the *container's* namespace and will show `/` as `overlay`:

```bash
POD=$(kubectl get pods -n $NS -l app.kubernetes.io/instance=hyp-obs-node-exporter \
  --field-selector "spec.nodeName=$NODE" -o jsonpath='{.items[0].metadata.name}')

# 1. the volume is mounted
kubectl exec -n $NS "$POD" -- cat /host/proc/1/mounts \
  | awk '$2=="/opt/sagemaker" || $2=="/" {print $2, $1, $3}'

# 2. containerd points into it  (expect /opt/sagemaker/containerd/data-root)
kubectl exec -n $NS "$POD" -- cat /host/root/etc/containerd/config.toml \
  | grep -nE '^\s*root\s*='
```

Then confirm the headroom moved, using the per-node free-disk command in the
root `README.md` — `kubectl get node` reports `ephemeral-storage` for the root
filesystem only and will not show you the new volume.

## If you cannot change the cluster

The instance store is a dead end for this specific problem: `/opt/dlami/nvme`
holds 6.5 TiB on a `g6e.48xlarge` and containerd does not use it, because its
data root is on the root volume. See `g5/eks/local-nvme/` for what that storage
*is* good for — the job's own bytes.

What remains: pre-pull the image onto the nodes, build a smaller image, or run
the scenario on direct EC2, which is the path every measured number in this repo
actually came from.
