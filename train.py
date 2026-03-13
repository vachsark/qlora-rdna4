#!/usr/bin/env python3
"""
QLoRA fine-tuning with HQQ — works on AMD RDNA4 (and any PyTorch-supported GPU).

Usage:
    python train.py --model Qwen/Qwen2.5-7B-Instruct --data training_data.jsonl

The training data should be JSONL with a "messages" field in chat format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

HQQ provides 4-bit quantization via pure PyTorch ops — no custom CUDA/HIP kernels needed.
This is the key enabler for RDNA4 (gfx1200/gfx1201) where bitsandbytes fails.
"""
import argparse
import gc
import json
import os
import statistics
import sys
import time

import torch

# Patch torch.compile for Python 3.14+ (HQQ uses @torch.compile() decorator)
if sys.version_info >= (3, 14):
    _orig_compile = torch.compile
    def _patched_compile(fn=None, *args, **kwargs):
        if fn is None:
            return lambda f: f
        if callable(fn):
            return fn
        return _orig_compile(fn, *args, **kwargs)
    torch.compile = _patched_compile

from transformers import AutoTokenizer, AutoModelForCausalLM
from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_and_quantize(model_name: str, device: str = "cuda:0") -> AutoModelForCausalLM:
    """Load a model in bf16 and quantize target layers with HQQ 4-bit."""
    print(f"Loading {model_name} to CPU in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cpu",
    )

    print("Quantizing linear layers with HQQ 4-bit (group_size=64)...")
    quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
    quantized_count = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(t in name for t in TARGET_MODULES):
                hqq_layer = HQQLinear(
                    module, quant_config,
                    compute_dtype=torch.bfloat16, device=device,
                )
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], hqq_layer)
                quantized_count += 1

    print(f"Quantized {quantized_count} layers")

    model = model.to(device)
    gc.collect()
    torch.cuda.empty_cache()

    HQQLinear.set_backend(HQQBackend.PYTORCH)
    print(f"VRAM after load: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    return model


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning with HQQ on AMD RDNA4")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--data", required=True, help="Training data JSONL (chat format)")
    parser.add_argument("--val-data", default=None, help="Validation data JSONL (optional)")
    parser.add_argument("--output", default="./output", help="Output directory for LoRA + merged model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--max-length", type=int, default=2560, help="Max sequence length (default: 2560)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--no-merge", action="store_true", help="Skip merging LoRA into base model")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No GPU detected.")
        sys.exit(1)

    print("=" * 60)
    print("QLoRA Fine-Tuning with HQQ")
    print("=" * 60)
    print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(f"VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch:    {torch.__version__}")
    print(f"Model:      {args.model}")
    print(f"Data:       {args.data}")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-flight token length check
    print("Pre-flight: checking token lengths...")
    all_lengths = []
    with open(args.data) as f:
        for line in f:
            messages = json.loads(line)["messages"]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            all_lengths.append(len(tokenizer.encode(text)))

    train_count = len(all_lengths)
    print(f"  Examples: {train_count}")
    print(f"  Token lengths: min={min(all_lengths)}, max={max(all_lengths)}, "
          f"mean={statistics.mean(all_lengths):.0f}, median={statistics.median(all_lengths):.0f}")

    truncated = sum(1 for l in all_lengths if l > args.max_length)
    if truncated > 0:
        print(f"  WARNING: {truncated}/{train_count} ({truncated/train_count*100:.1f}%) exceed max_length")
    print()

    # Load and quantize model
    model = load_and_quantize(args.model)
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    data_files = {"train": args.data}
    if args.val_data:
        data_files["val"] = args.val_data
    dataset = load_dataset("json", data_files=data_files)

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )

    # Training config
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, train_count // effective_batch)
    eval_steps = max(10, min(100, steps_per_epoch // 5))

    lora_dir = os.path.join(args.output, "lora")
    training_args = SFTConfig(
        output_dir=lora_dir,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5,
        eval_strategy="steps" if args.val_data else "no",
        eval_steps=eval_steps if args.val_data else None,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=42,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "formatting_func": formatting_func,
    }
    if args.val_data:
        trainer_kwargs["eval_dataset"] = dataset["val"]

    trainer = SFTTrainer(**trainer_kwargs)

    print(f"\nStarting training ({args.epochs} epochs, {steps_per_epoch} steps/epoch)...")
    start_time = time.time()
    trainer.train()
    wall_clock = time.time() - start_time

    # Save LoRA
    print(f"\nSaving LoRA adapter to {lora_dir}")
    trainer.save_model(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    # Merge
    if not args.no_merge:
        merged_dir = os.path.join(args.output, "merged")
        print(f"Merging LoRA into base model -> {merged_dir}")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

    # Summary
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    train_loss = trainer.state.log_history[-2].get("train_loss", "?") if len(trainer.state.log_history) >= 2 else "?"

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Model:      {args.model}")
    print(f"  Train loss:  {train_loss}")
    print(f"  Peak VRAM:   {peak_vram:.2f} GB")
    print(f"  Wall clock:  {wall_clock:.0f}s ({wall_clock/60:.1f}m)")
    print(f"  Quantization: HQQ 4-bit (group_size=64, pure PyTorch)")
    print(f"  Output:      {args.output}")
    print("=" * 60)

    # Append to training log
    log_record = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
        "hyperparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_length": args.max_length,
            "lr": args.lr,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "quantization": "hqq-4bit",
        },
        "train_loss": train_loss,
        "peak_vram_gb": round(peak_vram, 2),
        "train_count": train_count,
        "wall_clock_s": round(wall_clock, 1),
    }
    log_path = os.path.join(args.output, "training-log.jsonl")
    os.makedirs(args.output, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(log_record) + "\n")
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
