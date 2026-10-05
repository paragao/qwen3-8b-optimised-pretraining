# EKS run on p6-b200-cluster (ml.g6e.48xlarge / L40S) — predictions recorded BEFORE the run

Written before applying anything, so the comparison afterwards is a test and not
a rationalisation.

## What this run can and cannot validate

The hardware is **wrong for the g5 scenario on purpose** — it is the cluster I
can actually reach. `ml.g6e.48xlarge` carries 8x **L40S (48 GB)**; the scenario
is sized for 1x **A10G (24 GB)**. The pod requests `nvidia.com/gpu: 1`, so it
gets one L40S.

- VALIDATES: the EKS manifest end to end — init-container clone, the c4 dataset
  Job, the PVC, the headless rendezvous, torchrun, Megatron, real `c4` data, and
  the documented StorageClass patch. Nothing in this chain has ever actually
  executed on Kubernetes; every measured number on record came from the direct
  EC2 path.
- DOES NOT VALIDATE: the g5 memory ceiling (19.76 GiB on a 22.49 GiB card) or
  any g5 throughput figure. A 48 GB card cannot test a 24 GB ceiling, and an
  L40S is far faster than an A10G.

## P1 — peak allocated should be ~UNCHANGED at 19.08 GiB

Peak *allocated* is a property of the model, batch and optimizer, not of the
card. Same geometry, same framework, same `GLOBAL_BATCH_SIZE=8`, so:

    predicted peak allocated : 19.08 GiB  (the measured g5 `1b` figure)
    accept if within         : +/- 1.0 GiB

Wider than the memory model's own 0.0787 GiB worst error on purpose, because two
things genuinely differ: cuDNN/cuBLAS pick different algorithms on L40S
(Ada, sm_89) than on A10G (Ampere, sm_86), and workspace sizes come with them.

**Peak RESERVED is expected to differ more than allocated**, because the
allocator will happily cache against 48 GB instead of 22.49 GiB. A large
reserved figure is NOT a defect here.

REFUTES the "peak allocated is card-independent" assumption if it lands outside
+/- 1.0 GiB. That would mean the memory model carries a hardware term nobody has
identified.

## P2 — throughput should be 2-3x the g5 figure

A10G BF16 dense peak is 125 TFLOP/s; L40S is ~362 TFLOP/s, ~2.9x. The g5 `1b`
run achieved 34.3 TFLOP/s, i.e. MFU 0.27.

    g5 measured        :  7,085 tok/s @ 1.156 s/step
    predicted band     : 14,000 - 21,000 tok/s  (2.0x - 3.0x)
    central            : ~20,000 tok/s if MFU holds at 0.27

If it lands BELOW 14,000 the bottleneck is not compute, and the first thing to
check is the dataset path on a network volume rather than local NVMe.

## P3 — no OOM, with ~27 GiB spare

`SEQ_LENGTH=1024` is the measured ceiling on a 24 GB A10G. On 48 GB it should be
nowhere near a limit. If this OOMs, something is wrong with the manifest rather
than with the sizing.

## Cluster state at submit time, for the record

    4x ml.g6e.48xlarge, all us-west-2b, no taints, all Ready, 8 GPU each
    21 GPUs held by my own g4-* inference pods; 15 free
    per-node free: 3, 4, 4, 4   -> a 1-GPU job is placeable on any of them
    default/gemma4-unified-g6e is Pending for 8 GPUs and was ALREADY
      unschedulable (max free on any single node is 4) before this run
    nvcr.io/nvidia/nemo:26.04 is NOT cached on any node -> full ~77 GB pull
    no node advertises vpc.amazonaws.com/efa -> single-node scenario only
    StorageClasses: fsx-sc (Immediate), gp2 (in-tree legacy),
      sagemaker-spaces-default-storage-class (ebs.csi.aws.com, WFFC)
    -> using sagemaker-spaces-default-storage-class: the real EBS CSI driver,
       WaitForFirstConsumer, ReadWriteOnce. Correct for ONE node only.
