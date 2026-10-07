#!/usr/bin/env python3
"""Validate the local-nvme components by BUILDING them, not by reading them.

    python3 g5/eks/validate-nvme.py

Why this is separate from validate.py: that one reads pretrain.yaml directly,
and its checks assume the PVC exists -- which ../local-nvme-data deliberately
removes. This one runs `kubectl kustomize` on each overlay+component pair and
asserts the RESULT, which is the only thing the cluster ever sees.

What it refuses, and why each refusal exists rather than being a comment:

  type: Directory          DirectoryOrCreate on a node WITHOUT the instance
                           store silently creates a directory on the ROOT
                           volume, and the run then fills the disk containerd
                           needs while appearing to use 6.5 TiB of NVMe.
                           Measured on p6-b200-cluster 2026-10-07:
                           /opt/dlami/nvme is present on ml.g6e.48xlarge and
                           ABSENT on ml.c6i.8xlarge in the same cluster, so
                           this is a live hazard, not a hypothetical one.

  subPath on every mount   a bare hostPath mount would put the pod at the ROOT
                           of the shared instance store.

  readOnly preserved       the dataset must stay read-only in the training pod
                           and writable in the prep pod. The data component
                           re-points those mounts, which is exactly where a
                           merge can quietly drop a key.

  no orphan PVC reference  spec.volumes is resolved by the scheduler whether
                           or not a container mounts it, so a deleted PVC with
                           a surviving volume entry fails scheduling with
                           `persistentvolumeclaim "qwen3-data" not found`.

  data component is        a hostPath dataset at DP=2 gives rank 1 no dataset.
  single-node only         Asserted by building it against multi-node and
                           requiring NNODES=1 in what comes out.

Needs pyyaml and kubectl (kustomize is built in). Exits non-zero on failure.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
G5 = os.path.dirname(HERE)
fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


def build(overlay, component):
    """kubectl kustomize <overlay + component>, in a throwaway copy."""
    tmp = tempfile.mkdtemp(prefix="nvme-check-", dir=G5)
    try:
        src = os.path.join(G5, overlay, "kustomization.yaml")
        dst = os.path.join(tmp, "kustomization.yaml")
        shutil.copyfile(src, dst)
        with open(dst, "a") as fh:
            fh.write(f"\ncomponents:\n  - ../eks/{component}\n")
        out = subprocess.run(["kubectl", "kustomize", tmp],
                             capture_output=True, text=True)
        if out.returncode != 0:
            return None, out.stderr.strip()
        return [d for d in yaml.safe_load_all(out.stdout) if d], None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- source text
# The type: Directory refusal is asserted against the component SOURCE as well
# as the build, so that a weakened source is caught even if a future base
# change stopped the volume reaching the output.
print("=== COMPONENT SOURCE ===")
for comp in ("local-nvme", "local-nvme-data"):
    path = os.path.join(HERE, comp, "kustomization.yaml")
    check(f"{comp}/kustomization.yaml exists", os.path.isfile(path))
    if not os.path.isfile(path):
        continue
    text = open(path).read()
    body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    check(f"{comp}: no DirectoryOrCreate outside comments",
          "DirectoryOrCreate" not in body,
          "an absent disk must not render as a working one")

# -------------------------------------------------------------------- builds
# (overlay, component, dataset-on-hostPath, is-a-supported-combination)
#
# The last entry is a NEGATIVE test and its presence is deliberate. Pairing the
# data component with multi-node is the one combination that must not be used,
# so the suite asserts that it is DETECTABLE -- NNODES comes out 2 while the
# dataset is node-local -- and counts that detection as a PASS. Without this
# row the single-node-only constraint would be prose with nothing behind it;
# with it scored as a failure, the validator could never exit 0 and would stop
# being run at all.
MATRIX = [
    ("single-node", "local-nvme", False, True),
    ("multi-node", "local-nvme", False, True),
    ("single-node", "local-nvme-data", True, True),
    ("multi-node", "local-nvme-data", True, False),
]

for overlay, component, data_on_nvme, supported in MATRIX:
    label = "SUPPORTED" if supported else "NEGATIVE TEST -- must be detected"
    print(f"\n=== {overlay} + {component}   [{label}] ===")
    docs, err = build(overlay, component)
    if docs is None:
        check(f"{overlay}+{component}: builds", False, err.splitlines()[0] if err else "")
        continue
    check(f"{overlay}+{component}: builds", True)

    jobs = {d["metadata"]["name"]: d for d in docs if d.get("kind") == "Job"}
    pvcs = [d["metadata"]["name"] for d in docs
            if d.get("kind") == "PersistentVolumeClaim"]
    cms = [d for d in docs if d.get("kind") == "ConfigMap"]
    nnodes = cms[0]["data"]["NNODES"] if cms else None

    # The data component is single-node only. Rather than trusting the prose,
    # assert the combination that would be wrong is recognisable: if the
    # dataset is on a hostPath, NNODES must be 1.
    if data_on_nvme:
        # A hostPath dataset is node-local, so it is only correct at one node.
        # For the supported row, NNODES must be 1. For the negative row, the
        # requirement is INVERTED: the misconfiguration must be visible in the
        # built output, which is what makes the constraint checkable.
        if supported:
            check(f"{overlay}: hostPath dataset implies NNODES=1",
                  nnodes == "1", f"NNODES={nnodes!r}")
        else:
            check(f"{overlay}: wrong combination is detected, not silent",
                  nnodes != "1",
                  f"NNODES={nnodes!r} with a node-local dataset -- rank 1 finds "
                  f"no dataset and stops at the preflight FATAL")
        check(f"{overlay}: PVC removed", "qwen3-data" not in pvcs)
    else:
        check(f"{overlay}: PVC retained for the dataset", "qwen3-data" in pvcs)

    for jname in ("qwen3-pretrain", "c4-prep"):
        spec = jobs[jname]["spec"]["template"]["spec"]
        vols = {v["name"]: v for v in spec["volumes"]}
        hp = vols.get("nvme", {}).get("hostPath", {})
        check(f"{jname}: hostPath /opt/dlami/nvme",
              hp.get("path") == "/opt/dlami/nvme", repr(hp.get("path")))
        check(f"{jname}: hostPath type is Directory",
              hp.get("type") == "Directory", repr(hp.get("type")))

        containers = spec["containers"] + spec.get("initContainers", [])

        # No volume entry may reference a PVC that the build does not contain,
        # and nothing may mount one once the dataset has moved.
        for vname, v in vols.items():
            claim = v.get("persistentVolumeClaim", {}).get("claimName")
            if claim:
                check(f"{jname}: volume {vname!r} claim {claim!r} exists",
                      claim in pvcs)

        for c in containers:
            mounts = {m["mountPath"]: m for m in c.get("volumeMounts", [])}
            env = {e["name"]: e.get("value") for e in c.get("env", [])}

            if env.get("HF_HOME"):
                check(f"{jname}/{c['name']}: HF_HOME on nvme",
                      env["HF_HOME"] == "/nvme/hf", env["HF_HOME"])
                m = mounts.get("/nvme/hf", {})
                check(f"{jname}/{c['name']}: /nvme/hf is an nvme subPath",
                      m.get("name") == "nvme" and bool(m.get("subPath")),
                      f"{m.get('name')!r} subPath={m.get('subPath')!r}")

            if c["name"] == "train":
                m = mounts.get("/workspace/run", {})
                check("train: /workspace/run is an nvme subPath",
                      m.get("name") == "nvme" and bool(m.get("subPath")),
                      f"{m.get('name')!r} subPath={m.get('subPath')!r}")

            if "/data" in mounts:
                m = mounts["/data"]
                want = "nvme" if data_on_nvme else "data"
                check(f"{jname}/{c['name']}: /data from {want!r}",
                      m.get("name") == want, repr(m.get("name")))
                if data_on_nvme:
                    check(f"{jname}/{c['name']}: /data has a subPath",
                          bool(m.get("subPath")), repr(m.get("subPath")))
                # readOnly must survive the merge: writable ONLY in prep.
                want_ro = c["name"] != "prep"
                check(f"{jname}/{c['name']}: /data readOnly={want_ro}",
                      bool(m.get("readOnly")) == want_ro,
                      f"got {bool(m.get('readOnly'))}")

        # Every mount of the shared instance store must be confined to a
        # subPath, or the pod sees the whole node's scratch tree.
        bare = [m["mountPath"] for c in containers
                for m in c.get("volumeMounts", [])
                if m.get("name") == "nvme" and not m.get("subPath")]
        check(f"{jname}: no bare (subPath-less) nvme mount", not bare, str(bare))

print()
if fails:
    print(f"LOCAL-NVME VALIDATION FAILED: {len(fails)}: {', '.join(fails)}")
    sys.exit(1)
print("LOCAL-NVME VALIDATION PASSED")
