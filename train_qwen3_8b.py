#!/usr/bin/env python3
"""Qwen3-8B Pre-Training — HF Accelerate + DeepSpeed ZeRO-1"""
import os, argparse, time, math
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-8B Pre-Training")
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--dataset", default="allenai/c4")
    p.add_argument("--dataset-subset", default="en")
    p.add_argument("--seq-length", type=int, default=4096)
    p.add_argument("--micro-batch-size", type=int, default=12)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=317891)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--checkpoint-dir", default="/fsx/paragao/qwen3-8b/checkpoints")
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--wandb-project", default="qwen3-8b-pretrain")
    p.add_argument("--resume-from", default=None)
    return p.parse_args()


def cosine_with_min_lr(optimizer, warmup, total, min_ratio):
    """Cosine schedule with warmup and minimum learning rate."""
    from torch.optim.lr_scheduler import LambdaLR

    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(min_ratio, 0.5 * (1 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, fn)


class StreamingDataset(IterableDataset):
    """Wrap HF streaming dataset as a PyTorch IterableDataset."""

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __iter__(self):
        return iter(self.dataset)


def main():
    args = parse_args()

    # DeepSpeed ZeRO-1 config
    ds_config = {
        "zero_optimization": {
            "stage": 1,
            "overlap_comm": True,
            "reduce_bucket_size": 5e8,
        },
        "bf16": {"enabled": True},
        "gradient_clipping": args.grad_clip,
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
    }

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum_steps,
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=ds_config),
    )

    # Model — init from config with random weights (pre-training from scratch)
    accelerator.print(f"Loading config: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    accelerator.print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Dataset — streaming C4
    accelerator.print(f"Loading dataset: {args.dataset}/{args.dataset_subset} (streaming)")
    dataset = load_dataset(args.dataset, args.dataset_subset, split="train", streaming=True)

    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=args.seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc["labels"] = enc["input_ids"].clone()
        return enc

    dataloader = DataLoader(
        StreamingDataset(dataset),
        batch_size=args.micro_batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = cosine_with_min_lr(
        optimizer, args.warmup_steps, args.max_steps, args.min_lr / args.lr
    )

    # Prepare
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Resume from checkpoint
    global_step = 0
    if args.resume_from:
        accelerator.print(f"Resuming from: {args.resume_from}")
        accelerator.load_state(args.resume_from)
        global_step = int(args.resume_from.split("step-")[-1])

    # Training loop
    accelerator.print(f"Training from step {global_step}, max {args.max_steps}")
    model.train()
    t0 = time.time()
    running_loss = 0.0

    for batch in dataloader:
        if global_step >= args.max_steps:
            break

        with accelerator.accumulate(model):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            accelerator.backward(out.loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += out.loss.detach().float()
        global_step += 1

        # Logging
        if global_step % args.log_interval == 0:
            avg = running_loss / args.log_interval
            elapsed = time.time() - t0
            tps = (
                args.micro_batch_size
                * args.seq_length
                * args.grad_accum_steps
                * accelerator.num_processes
                * args.log_interval
                / elapsed
            )
            accelerator.print(
                f"[Step {global_step}/{args.max_steps}] "
                f"loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e} tok/s={tps:.0f}"
            )
            running_loss = 0.0
            t0 = time.time()

        # Checkpointing
        if global_step % args.checkpoint_interval == 0:
            ckpt_path = f"{args.checkpoint_dir}/step-{global_step}"
            accelerator.print(f"Saving checkpoint: {ckpt_path}")
            accelerator.save_state(ckpt_path)

    # Final checkpoint
    ckpt_path = f"{args.checkpoint_dir}/step-{global_step}-final"
    accelerator.save_state(ckpt_path)
    accelerator.print(f"Training complete! Final step: {global_step}")


if __name__ == "__main__":
    main()
