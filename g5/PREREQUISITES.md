<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Prerequisites for the `g5/` scenarios

Everything you need before running either
[`single-node/`](single-node/README.md) or
[`multi-node/`](multi-node/README.md), with a command to prove each one.

Shared by both scenarios deliberately: the requirements are identical, and two
copies would drift. Read this once, then go to the scenario you want.

## 0. What this costs, and how to stop paying

`g5.8xlarge` on-demand in `us-west-2` is about **$2.45/hour**. The scenarios run
for roughly these times end to end, including instance boot, the 77 GB container
pull and dataset preparation:

| | instances | wall clock | approx cost |
|---|---|---|---|
| `single-node` | 1 | ~50 min | **~$2** |
| `multi-node` | 2 | ~90 min | **~$7** |

**The instance keeps billing until you terminate it.** Nothing in these scripts
stops it for you, and an idle GPU costs exactly the same as a busy one:

```bash
./g5/terminate-instance.sh                      # whatever you last launched
./g5/terminate-instance.sh i-0abc i-0def        # or name them explicitly
```

Every argument is an instance id, so **a 2-node cluster is torn down in one
call**. Passing just one id when you launched two leaves the second one running
and billing.

Then confirm with your own eyes rather than trusting the script:

```bash
aws ec2 describe-instances --region us-west-2 \
  --filters Name=instance-state-name,Values=running,pending \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime]' \
  --output table
```

The script verifies termination itself and refuses to report success while any
target is still alive, then deletes `g5/.last-instance-id`. Root EBS volumes are
`DeleteOnTermination=true`, so they go with the instance.

**It deliberately leaves two things behind**: the IAM role/instance profile and
the security group. Both cost nothing, and keeping them makes a re-run faster
and avoids needing IAM write a second time. The script prints the exact commands
to remove them if you want a completely clean account:

```bash
aws iam remove-role-from-instance-profile \
  --instance-profile-name qwen3-g5-validation-ssm --role-name qwen3-g5-validation-ssm
aws iam delete-instance-profile --instance-profile-name qwen3-g5-validation-ssm
aws iam detach-role-policy --role-name qwen3-g5-validation-ssm \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam delete-role --role-name qwen3-g5-validation-ssm
aws ec2 delete-security-group --region us-west-2 --group-name qwen3-g5-validation-sg
```

> **If you launched more than once**, `g5/.last-instance-id` holds only the most
> recent launch, so a bare `terminate-instance.sh` will leave earlier instances
> running and billing. Use the `describe-instances` command above and pass every
> id explicitly.

## 1. Local tools

Run these on your own machine, not on the instance. All four must be present:

```bash
aws --version                 # need v2.x
session-manager-plugin        # need: "The Session Manager plugin is installed"
python3 --version             # need 3.8+
ssh -V                        # any OpenSSH
```

- **`aws` CLI v2** — [install
  guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
- **`session-manager-plugin`** — the scripts reach the instance by tunnelling
  SSH over AWS Systems Manager, so **this is required, not optional**, and it is
  a separate install from the CLI. [Install
  guide](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html).
  Without it every connection fails at the proxy step.
- **`python3`** — for `predict.py`, `throughput.py` and `max-model.py`. No
  packages needed: they use only the standard library, so there is no `pip
  install` and no virtualenv.
- **`ssh` / `scp`** — present by default on macOS and Linux.

Not needed locally: **`jq`** (the scripts use `aws --query`), **`docker`** (it
runs on the instance, not here) and **`kubectl`** (only for the Kubernetes path,
see each scenario's README).

Scripts are POSIX/bash-3.2 compatible, so stock **macOS bash 3.2** works.
`./g5/lint-portability.sh` enforces that and takes a second to run.

## 2. AWS credentials

```bash
aws sts get-caller-identity          # must print your account and role
```

If this fails, configure credentials first — `aws configure`, or
`aws sso login` for IAM Identity Center. Every script calls this and stops early
with the reason if it fails.

The scripts use **whatever credentials your environment already resolves** — the
standard chain of environment variables, SSO, your default profile, or an
instance role. You do not need a particular profile name. If you keep several
profiles and want a specific one:

```bash
aws configure list-profiles          # see what you have
export AWS_PROFILE=my-profile        # the scripts honour this
```

Pick your region once and keep it consistent; it defaults to `us-west-2`:

```bash
export REGION=us-west-2              # read by every g5/ script
```

## 3. Permissions — including IAM write, which is the one that usually bites

The scripts need more than EC2 read. The full set of API calls they make:

| service | why |
|---|---|
| `sts:GetCallerIdentity` | the preflight check above |
| `ec2:RunInstances`, `TerminateInstances`, `Describe*` | launch and tear down |
| `ec2:*SecurityGroup*` | create the self-referencing SG the ranks need |
| `ssm:GetParameter` | resolve the Deep Learning AMI id (no hardcoded AMI) |
| `ssm:StartSession`, `ssm:DescribeInstanceInformation` | the SSH tunnel |
| **`iam:CreateRole`, `CreateInstanceProfile`, `AttachRolePolicy`** | **see below** |

**The IAM calls are the common blocker.** The launcher creates a small IAM role
with the AWS-managed `AmazonSSMManagedInstanceCore` policy and attaches it to the
instance, because without it Systems Manager cannot reach the instance and no
connection is possible. Many corporate accounts deny IAM writes to individual
engineers. Check before you launch:

```bash
aws iam get-role --role-name qwen3-g5-validation-ssm >/dev/null 2>&1 \
  && echo "role exists -- the launcher reuses it, no IAM write needed" \
  || echo "launcher must CREATE it -- you need iam:CreateRole"
```

`AdministratorAccess` covers everything here. If you cannot get IAM write, ask
an administrator to create that role and instance profile once; the launcher is
idempotent and reuses them, making no IAM write calls on later runs.

## 4. GPU instance quota — check this first, it is the most likely hard stop

**`g5.8xlarge` is 32 vCPUs.** The relevant quota counts vCPUs, not instances:

| scenario | instances | vCPUs needed |
|---|---|---|
| `single-node` | 1 | **32** |
| `multi-node` | 2 | **64** |

```bash
aws service-quotas get-service-quota --region us-west-2 \
  --service-code ec2 --quota-code L-DB2E81BA \
  --query '[QuotaName,Value]' --output text
```

That is `Running On-Demand G and VT instances`. **A fresh AWS account commonly
has 0 here**, in which case `run-instances` fails with
`VcpuLimitExceeded` no matter how correct everything else is. Request an
increase in the Service Quotas console — approval is usually hours, sometimes a
day or two, so do this before anything else.

Quotas are **per region**. An increase in `us-east-1` does nothing for a run in
`us-west-2`.

## 5. Container image access

The run pulls **`nvcr.io/nvidia/nemo:26.04`** (77 GB) onto the instance. This
usually works unauthenticated. If the pull fails, `run.sh` stops and prints the
exact fix, which you run **on the instance**:

```bash
docker login nvcr.io -u '$oauthtoken' -p <YOUR_NGC_API_KEY>
```

A free NGC API key comes from [ngc.nvidia.com](https://ngc.nvidia.com) →
*Setup* → *Generate API Key*. No custom image is built: the stock NeMo container
is used as-is.

Budget **10-15 minutes** for that first pull. It is cached afterwards, so a
second run on the same instance skips it.

## 6. How your code reaches the instance

Worth knowing, because it determines whether your edits take effect.

Both scenarios copy **five files from your local working tree** onto the
instance over the SSH tunnel every run:

```
g5/train.py  g5/run.sh  g5/throughput.py  g5/prepare_c4.py  g5/prepare-c4.sh
```

So **local edits do take effect** — you do not need to commit or push anything
to try a change. Nothing is read from GitHub at run time.

The 2-node path additionally does a `git clone --depth 1` on each node first,
but only to create the directory layout on a bare instance; the five files above
are then copied over the top. If you work in a fork, or the clone cannot reach
GitHub, point it elsewhere:

```bash
REPO_URL=https://github.com/you/your-fork.git ./g5/finish-run-2node.sh
```

The 2-node script also refuses to start if your local `g5/train.py` lacks the
data-parallel branch, so a stale checkout fails immediately with the reason
rather than silently running the wrong model.

## 7. What you do NOT need

- **No Hugging Face token.** `allenai/c4` is public and the Qwen3 tokenizer
  downloads unauthenticated. `prepare-c4.sh` uses `HF_TOKEN` if it happens to be
  set and works fine without it.
- **No GPU, Docker or NVIDIA driver on your own machine.** Everything that needs
  a GPU runs on the instance. The AMI already carries the driver and Docker.
- **No model weights.** Both scenarios pre-train from random initialization, so
  there is no Qwen3-8B checkpoint to download and no gated-repo access needed.
- **No Kubernetes cluster**, unless you choose the Kubernetes path.

## 8. Check everything at once

Paste this before you launch. It only reads:

```bash
cd /path/to/qwen3-g5-validate
REGION=${REGION:-us-west-2}
ok=1
echo "== local tools =="
for c in aws session-manager-plugin python3 ssh; do
  if command -v "$c" >/dev/null 2>&1; then echo "  OK    $c"
  else echo "  MISSING $c"; ok=0; fi
done
echo "== credentials =="
acct=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
if [ -n "$acct" ]; then echo "  OK    credentials valid (account ${acct})"
else echo "  FAIL  no usable credentials -- run 'aws configure' or 'aws sso login'"; ok=0; fi
echo "== GPU vCPU quota in ${REGION} (need 32 for 1 node, 64 for 2) =="
q=$(aws service-quotas get-service-quota --region "$REGION" \
      --service-code ec2 --quota-code L-DB2E81BA \
      --query Value --output text 2>/dev/null)
case "$q" in
  ''|None) echo "  WARN  could not read quota -- needs servicequotas:GetServiceQuota" ;;
  *) awk -v v="$q" 'BEGIN{printf "  %s  %d vCPUs available\n", (v>=64?"OK   ":"LOW  "), v}' ;;
esac
echo "== repo self-checks (no AWS calls, no cost) =="
python3 g5/predict.py --self-check >/dev/null 2>&1 \
  && echo "  OK    predict.py" || { echo "  FAIL  predict.py"; ok=0; }
./g5/scenario-check.sh >/dev/null 2>&1 \
  && echo "  OK    scenario-check.sh" || { echo "  FAIL  scenario-check.sh"; ok=0; }
./g5/lint-portability.sh >/dev/null 2>&1 \
  && echo "  OK    lint-portability.sh" || { echo "  FAIL  lint-portability.sh"; ok=0; }
[ "$ok" = 1 ] && echo "READY" || echo "NOT READY -- fix the items above first"
```

The last three cost nothing and need no AWS access, so you can run them straight
after cloning to confirm the repo is intact.

## Next

- **[`single-node/README.md`](single-node/README.md)** — one A10G, 1.01 B
  parameters, 19.08 GiB. Start here: it is cheaper, faster, and has no fabric to
  go wrong.
- **[`multi-node/README.md`](multi-node/README.md)** — two nodes over EFA, DP=2,
  1.49 B parameters, 19.27 GiB.
- **[`README.md`](README.md)** — the measured results, the memory model, and the
  six setup failures this work had to get through first.
