#!/usr/bin/env python3
"""
7B QLoRA Smoke Test — bitsandbytes approach.

This script attempts to use bitsandbytes NF4 quantization for QLoRA.
On AMD RDNA4 (gfx1200/gfx1201), this FAILS with either:
  - hipErrorNoBinaryForGpu (PyPI wheel)
  - undefined symbol: hsa_amd_memory_get_preferred_copy_engine (source build)

Kept for reference. Use smoke_test.py (HQQ approach) instead.
"""
import sys
import time


def main():
    print("=" * 60)
    print("7B QLoRA Smoke Test — bitsandbytes (EXPECTED TO FAIL ON RDNA4)")
    print("=" * 60)
    print()

    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes: {bnb.__version__}")
    except ImportError:
        print("FAIL: bitsandbytes not installed")
        sys.exit(1)

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("FAIL: No GPU detected")
        sys.exit(1)

    print()
    print("[1/4] Loading Qwen2.5-7B-Instruct in 4-bit...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        vram_after_load = torch.cuda.max_memory_allocated() / 1e9
        print(f"  OK — loaded. VRAM: {vram_after_load:.2f} GB")
    except Exception as e:
        print(f"  FAIL: {e}")
        print()
        print("This is the expected failure on RDNA4. Use smoke_test.py (HQQ) instead.")
        sys.exit(1)

    # If we somehow get here, continue with LoRA + training
    print("[2/4] Applying LoRA adapter...")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[3/4] Creating dummy data...")
    from datasets import Dataset
    dummy_texts = ["Test sentence for training."] * 20
    dataset = Dataset.from_dict({"text": dummy_texts})
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=128, padding="max_length"), batched=True, remove_columns=["text"])

    print("[4/4] Training 10 steps...")
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    training_args = TrainingArguments(
        output_dir="/tmp/bnb-qlora-test", max_steps=10,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=2e-4, bf16=True, logging_steps=5, report_to="none",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nSUCCESS (unexpected on RDNA4!): {elapsed:.1f}s, {peak_vram:.2f} GB peak")

    import shutil
    shutil.rmtree("/tmp/bnb-qlora-test", ignore_errors=True)


if __name__ == "__main__":
    main()
