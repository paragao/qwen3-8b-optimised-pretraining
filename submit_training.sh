#!/bin/bash
#SBATCH --job-name=qwen3-8b-pretrain
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --output=/fsx/paragao/qwen3-8b/logs/qwen3-8b-%j.out
#SBATCH --error=/fsx/paragao/qwen3-8b/logs/qwen3-8b-%j.err
#SBATCH --time=30-00:00:00
#SBATCH --partition=p5en

# ============ Container & Mount ============
CONTAINER_IMAGE="/fsx/paragao/qwen3-8b/containers/qwen3-8b-training.sqsh"
MOUNT="/fsx:/fsx"

# ============ NCCL / EFA ============
export NCCL_DEBUG=INFO
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo,veth

# ============ Launch ============
srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${MOUNT}" \
     --no-container-mount-home \
     torchrun \
     --nnodes=${SLURM_NNODES} \
     --nproc-per-node=8 \
     --rdzv-id=${SLURM_JOB_ID} \
     --rdzv-backend=c10d \
     --rdzv-endpoint=$(scontrol show hostname $SLURM_NODELIST | head -n1):29500 \
     /fsx/paragao/qwen3-8b/code/train_qwen3_8b.py \
     --model-name Qwen/Qwen3-8B \
     --dataset allenai/c4 \
     --dataset-subset en \
     --seq-length 4096 \
     --micro-batch-size 12 \
     --grad-accum-steps 4 \
     --lr 3e-4 \
     --min-lr 3e-5 \
     --warmup-steps 2000 \
     --max-steps 100 \
     --weight-decay 0.1 \
     --grad-clip 1.0 \
     --checkpoint-dir /fsx/paragao/qwen3-8b/checkpoints \
     --checkpoint-interval 50 \
     --log-interval 10 \
     --wandb-project qwen3-8b-pretrain
