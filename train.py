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
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import Dataset as HFDataset


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


def load_prompt_completion(data_file: str, tokenizer, max_length: int):
    """Convert messages JSONL to prompt/completion format with smart truncation.

    Pre-truncates prompts so prompt+completion fits within max_length.
    Preserves system prompt and chat template structure — only truncates
    the user message content from the MIDDLE (keeps start for task framing
    and end for recent content).

    Required for completion_only_loss=True which masks all prompt tokens.
    """
    max_completion_tokens = max_length // 2

    records = []
    truncated = 0
    for line in open(data_file):
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        messages = entry["messages"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        completion_msgs = [m for m in messages if m["role"] == "assistant"]
        completion_text = (completion_msgs[0]["content"] + "<|im_end|>\n"
                           if completion_msgs else "")

        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
        if len(completion_ids) > max_completion_tokens:
            completion_ids = completion_ids[:max_completion_tokens]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)

        skeleton_msgs = system_msgs + [{"role": "user", "content": ""}]
        skeleton_text = tokenizer.apply_chat_template(
            skeleton_msgs, tokenize=False, add_generation_prompt=True,
        )
        skeleton_ids = tokenizer.encode(skeleton_text, add_special_tokens=False)

        available_for_user = max_length - len(skeleton_ids) - len(completion_ids)

        user_content = user_msgs[0]["content"] if user_msgs else ""
        user_ids = tokenizer.encode(user_content, add_special_tokens=False)

        if len(user_ids) > available_for_user and available_for_user > 0:
            truncated += 1
            keep_start = available_for_user // 3
            keep_end = available_for_user - keep_start
            user_ids = user_ids[:keep_start] + user_ids[-keep_end:]
            user_content = tokenizer.decode(user_ids, skip_special_tokens=False)

        final_msgs = system_msgs + [{"role": "user", "content": user_content}]
        prompt_text = tokenizer.apply_chat_template(
            final_msgs, tokenize=False, add_generation_prompt=True,
        )

        records.append({"prompt": prompt_text, "completion": completion_text})

    if truncated:
        print(f"  Truncated {truncated}/{len(records)} user messages (middle-truncation)")
    return HFDataset.from_list(records)


def merge_hqq_safe(model_name: str, lora_dir: str, merged_dir: str, tokenizer):
    """Merge LoRA adapter into base model without HQQ tensor names.

    HQQ models have quantized tensor names (W_q, meta) that llama.cpp
    can't convert. The fix: reload base model in bf16, apply LoRA adapter,
    then merge. This produces clean tensor names.
    """
    print(f"Merging LoRA into base model -> {merged_dir}")
    print("  HQQ path: reloading base model in bf16 for clean merge...")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base_model, lora_dir)
    merged = merged.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"  Saved merged model to {merged_dir}")


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning with HQQ on AMD RDNA4")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--data", required=True, help="Training data JSONL (chat format)")
    parser.add_argument("--val-data", default=None, help="Validation data JSONL (optional)")
    parser.add_argument("--output", default="./output", help="Output directory for LoRA + merged model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length (default: 1024)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--neftune-alpha", type=float, default=5.0, help="NEFTune noise alpha (default: 5, 0 to disable)")
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
    completion_lengths = []
    with open(args.data) as f:
        for line in f:
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            all_lengths.append(len(tokenizer.encode(text)))
            for m in messages:
                if m["role"] == "assistant":
                    completion_lengths.append(len(tokenizer.encode(m["content"], add_special_tokens=False)))

    train_count = len(all_lengths)
    print(f"  Examples: {train_count}")
    print(f"  Token lengths: min={min(all_lengths)}, max={max(all_lengths)}, "
          f"mean={statistics.mean(all_lengths):.0f}, median={statistics.median(all_lengths):.0f}")

    if completion_lengths:
        avg_input = statistics.mean(all_lengths) - statistics.mean(completion_lengths)
        avg_comp = statistics.mean(completion_lengths)
        ratio = avg_input / avg_comp if avg_comp > 0 else float("inf")
        print(f"  Completion tokens: mean={avg_comp:.0f}, median={statistics.median(completion_lengths):.0f}")
        print(f"  Input/output ratio: {ratio:.1f}x")
        if ratio > 5:
            print(f"  -> Using completion_only_loss (ratio > 5x, {ratio/(1+ratio)*100:.0f}% of loss would be wasted on input)")
        else:
            print(f"  -> Ratio OK for standard training, but completion_only_loss still recommended")

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

    # Load dataset in prompt/completion format for completion_only_loss
    print("Loading training data (prompt/completion format)...")
    train_ds = load_prompt_completion(args.data, tokenizer, args.max_length)
    val_ds = None
    if args.val_data:
        print("Loading validation data...")
        val_ds = load_prompt_completion(args.val_data, tokenizer, args.max_length)

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
        eval_strategy="steps" if val_ds else "no",
        eval_steps=eval_steps if val_ds else None,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=42,
        completion_only_loss=True,
        neftune_noise_alpha=args.neftune_alpha if args.neftune_alpha > 0 else None,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
    }
    if val_ds:
        trainer_kwargs["eval_dataset"] = val_ds

    trainer = SFTTrainer(**trainer_kwargs)

    print(f"\nStarting training ({args.epochs} epochs, {steps_per_epoch} steps/epoch)...")
    start_time = time.time()
    trainer.train()
    wall_clock = time.time() - start_time

    # Save LoRA
    print(f"\nSaving LoRA adapter to {lora_dir}")
    trainer.save_model(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    # Merge (HQQ-safe: reload base in bf16 -> apply LoRA -> merge)
    if not args.no_merge:
        merged_dir = os.path.join(args.output, "merged")
        merge_hqq_safe(args.model, lora_dir, merged_dir, tokenizer)

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
            "neftune_alpha": args.neftune_alpha,
            "quantization": "hqq-4bit",
            "completion_only_loss": True,
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
