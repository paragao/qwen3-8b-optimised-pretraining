#!/usr/bin/env python3
"""Validate g5/eks/pretrain.yaml before applying it to a cluster.

Checks three classes of defect that `kubectl apply` accepts happily and that
then fail at runtime in confusing ways:

  EXPOSURE      the rendezvous Service must stay headless with no NodePort,
                LoadBalancer, externalIP, hostPort or hostNetwork, and
                MASTER_ADDR must be cluster-internal DNS.
  CROSS-REFS    subdomain/Service name, Service selector/pod labels, and
                MASTER_ADDR against <job>-0.<svc>.<ns>.svc.cluster.local --
                a mismatch makes the rendezvous hang rather than error.
  NODE COUNT    parallelism, completions, NNODES and GLOBAL_BATCH_SIZE must
                agree. If parallelism is 2 but NNODES is 1 you get two
                independent WORLD_SIZE=1 runs that look fine; the reverse
                waits forever for a rank that never joins.

Run after g5/eks/set-nodes.sh and before kubectl apply:

    python3 g5/eks/validate.py

Needs pyyaml only. Exits non-zero on any failure.
"""
import os
import sys

import yaml

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrain.yaml")

docs = [d for d in yaml.safe_load_all(open(MANIFEST)) if d]
by = {(d["kind"], d["metadata"]["name"]): d for d in docs}
fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


svc = by[("Service", "qwen3-rdzv")]
job = by[("Job", "qwen3-pretrain")]
prep = by[("Job", "c4-prep")]
cm = by[("ConfigMap", "qwen3-config")]
pvc = by[("PersistentVolumeClaim", "qwen3-data")]
pod = job["spec"]["template"]["spec"]
train = next(c for c in pod["containers"] if c["name"] == "train")


def resolve_env(container, key):
    """Effective value of an env var, following envFrom into the ConfigMap.

    A container can get a value either from an explicit `env` entry or from
    `envFrom: configMapRef`. Reading only `env` silently misses the second,
    which is how this validator initially reported a false failure after
    NNODES moved into the ConfigMap.
    """
    for e in container.get("env", []):
        if e["name"] == key and "value" in e:
            return e["value"]
    for src in container.get("envFrom", []):
        ref = src.get("configMapRef", {}).get("name")
        if ref == cm["metadata"]["name"] and key in cm["data"]:
            return cm["data"][key]
    return None


all_train_containers = pod["containers"] + pod.get("initContainers", [])


def uses(container, key):
    """True if the container's command text references $KEY / ${KEY}.

    Scoping the env checks this way avoids a false failure on the `clone` init
    container, which has no envFrom and legitimately needs neither MASTER_ADDR
    nor NNODES -- while still catching a container that USES a variable it
    cannot resolve.
    """
    text = " ".join(container.get("command", []) or [])
    return f"${{{key}}}" in text or f"${key}" in text


def consumers(key):
    return [c for c in all_train_containers if uses(c, key)]

print("=== EXPOSURE (builder-security) ===")
check("Service is headless (clusterIP: None)",
      svc["spec"].get("clusterIP") is None or svc["spec"].get("clusterIP") == "None",
      f"clusterIP={svc['spec'].get('clusterIP')!r}")
check("Service type is not NodePort/LoadBalancer",
      svc["spec"].get("type") in (None, "ClusterIP"),
      f"type={svc['spec'].get('type')!r}")
check("no nodePort on any Service port",
      all("nodePort" not in p for p in svc["spec"]["ports"]))
check("no externalIPs", "externalIPs" not in svc["spec"])
check("no hostPort on any container",
      all("hostPort" not in p
          for c in pod["containers"] + pod.get("initContainers", [])
          for p in c.get("ports", [])))
check("no hostNetwork", pod.get("hostNetwork") in (None, False))
_mc = consumers("MASTER_ADDR")
_masters = [resolve_env(c, "MASTER_ADDR") for c in _mc]
check("MASTER_ADDR resolves in every container that uses it",
      len(_mc) >= 2 and all(m is not None for m in _masters),
      f"{[c['name'] for c in _mc]} -> {_masters}")
check("MASTER_ADDR is cluster-internal DNS",
      bool(_masters) and all(m and m.endswith(".svc.cluster.local")
                             for m in _masters))
check("no 0.0.0.0 anywhere in the manifest",
      "0.0.0.0" not in open(MANIFEST).read())

print("\n=== CROSS-REFERENCES ===")
check("Job subdomain matches Service name",
      pod.get("subdomain") == svc["metadata"]["name"],
      f"{pod.get('subdomain')} == {svc['metadata']['name']}")
check("Service selector matches pod labels",
      svc["spec"]["selector"] == job["spec"]["template"]["metadata"]["labels"],
      f"{svc['spec']['selector']}")
check("publishNotReadyAddresses is true (rank 1 must resolve rank 0 early)",
      svc["spec"].get("publishNotReadyAddresses") is True)
ns = job["metadata"]["namespace"]
expect_master = f"{job['metadata']['name']}-0.{svc['metadata']['name']}.{ns}.svc.cluster.local"
masters = {resolve_env(c, "MASTER_ADDR") for c in consumers("MASTER_ADDR")}
check("MASTER_ADDR matches <job>-0.<svc>.<ns>.svc.cluster.local",
      masters == {expect_master}, f"{masters} vs {{{expect_master}}}")
check("all resources share one namespace",
      len({d["metadata"].get("namespace") for d in docs
           if d["kind"] != "Namespace"}) == 1)
check("Service port matches ConfigMap MASTER_PORT",
      str(svc["spec"]["ports"][0]["port"]) == cm["data"]["MASTER_PORT"],
      f"{svc['spec']['ports'][0]['port']} == {cm['data']['MASTER_PORT']}")
check("containerPort matches MASTER_PORT",
      str(train["ports"][0]["containerPort"]) == cm["data"]["MASTER_PORT"])

print("\n=== 1-NODE / 2-NODE CONSISTENCY ===")
par, comp = job["spec"]["parallelism"], job["spec"]["completions"]
check("parallelism == completions", par == comp, f"{par} == {comp}")
_nc = consumers("NNODES")
nnodes = {resolve_env(c, "NNODES") for c in _nc}
check("NNODES matches parallelism in every container that uses it",
      len(_nc) >= 2 and nnodes == {str(par)},
      f"{[c['name'] for c in _nc]} -> NNODES={nnodes} parallelism={par}")
gbs = int(cm["data"]["GLOBAL_BATCH_SIZE"])
mbs = int(cm["data"]["MICRO_BATCH_SIZE"])
check("GLOBAL_BATCH_SIZE divisible by nodes x MICRO_BATCH_SIZE",
      gbs % (par * mbs) == 0, f"{gbs} % ({par}*{mbs}) == {gbs % (par * mbs)}")
check("completionMode is Indexed (needed for stable hostnames)",
      job["spec"]["completionMode"] == "Indexed")
check("backoffLimit is 0 (no half-dead rendezvous retries)",
      job["spec"]["backoffLimit"] == 0)
check("anti-affinity on hostname (one pod per node)",
      pod["affinity"]["podAntiAffinity"]
      ["requiredDuringSchedulingIgnoredDuringExecution"][0]["topologyKey"]
      == "kubernetes.io/hostname")

print("\n=== GPU AND MEMORY ===")
check("GPU requested and limited at 1",
      train["resources"]["requests"]["nvidia.com/gpu"] == 1
      and train["resources"]["limits"]["nvidia.com/gpu"] == 1)
check("prep Job requests NO gpu",
      all("nvidia.com/gpu" not in c["resources"].get("requests", {})
          for c in prep["spec"]["template"]["spec"]["containers"]))
dshm = next(v for v in pod["volumes"] if v["name"] == "dshm")
check("/dev/shm is a Memory emptyDir", dshm["emptyDir"]["medium"] == "Memory")
shm_gi = int(dshm["emptyDir"]["sizeLimit"].rstrip("Gi"))
mem_gi = int(train["resources"]["limits"]["memory"].rstrip("Gi"))
check("pod memory limit exceeds /dev/shm size",
      mem_gi > shm_gi, f"{mem_gi}Gi limit > {shm_gi}Gi shm")
check("/dev/shm mounted at the right path",
      any(m["mountPath"] == "/dev/shm" for m in train["volumeMounts"]))
check("instance type pinned to g5.8xlarge",
      pod["nodeSelector"]["node.kubernetes.io/instance-type"] == "g5.8xlarge")

print("\n=== DATA ===")
check("PVC is ReadWriteMany (required for 2 nodes)",
      pvc["spec"]["accessModes"] == ["ReadWriteMany"])
check("dataset mounted read-only in the training pod",
      any(m["name"] == "data" and m.get("readOnly") for m in train["volumeMounts"]))
check("dataset mounted WRITABLE in the prep pod",
      any(m["name"] == "data" and not m.get("readOnly")
          for c in prep["spec"]["template"]["spec"]["containers"]
          for m in c["volumeMounts"]))
check("DATA_PATH points at the mounted volume",
      cm["data"]["DATA_PATH"].startswith("/data/"))
check("repo mounted read-only in training pod",
      any(m["name"] == "repo" and m.get("readOnly") for m in train["volumeMounts"]))

print("\n=== EFA CONSISTENCY ===")
# USE_EFA and the vpc.amazonaws.com/efa resource request must agree. The
# mismatch is quiet in the worst direction: USE_EFA=1 with no resource means
# the pod has no EFA device, NCCL falls back to TCP, and the only symptom is
# that the run is slower than the scaling tables predict.
_use_efa = cm["data"].get("USE_EFA", "0") == "1"
_efa_key = "vpc.amazonaws.com/efa"
_req = _efa_key in train["resources"].get("requests", {})
_lim = _efa_key in train["resources"].get("limits", {})
print(f"  USE_EFA={cm['data'].get('USE_EFA')!r}  resource in requests={_req}  limits={_lim}")
check("EFA resource requested exactly when USE_EFA=1",
      (_req and _lim) == _use_efa,
      "a mismatch silently falls back to TCP" if _use_efa != _req else "")
check("EFA resource present in BOTH requests and limits, or neither",
      _req == _lim)
if _use_efa and _req and _lim:
    check("EFA quantity is 1 (g5.8xlarge has one EFA interface)",
          train["resources"]["requests"][_efa_key] == 1
          and train["resources"]["limits"][_efa_key] == 1)
if _use_efa:
    check("libfabric provider is set for EFA",
          cm["data"].get("FI_PROVIDER") == "efa")

print()
if fails:
    print(f"MANIFEST VALIDATION FAILED: {len(fails)}: {', '.join(fails)}")
    sys.exit(1)
print("MANIFEST VALIDATION PASSED")
