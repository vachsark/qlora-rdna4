#!/usr/bin/env python3
"""
7B QLoRA via HQQ — Pure PyTorch backend, no custom CUDA/HIP kernels.
Tests 4-bit quantized loading + LoRA training on AMD RDNA4.

Usage:
    python smoke_test.py [--model MODEL_NAME]

Default model: Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import sys
import time

# Patch torch.compile for Python 3.14+ (not supported, but HQQ uses it as decorator)
import torch
if sys.version_info >= (3, 14):
    _orig_compile = torch.compile
    def _patched_compile(fn=None, *args, **kwargs):
        if fn is None:
            return lambda f: f
        if callable(fn):
            return fn
        return _orig_compile(fn, *args, **kwargs)
    torch.compile = _patched_compile


def main():
    parser = argparse.ArgumentParser(description="7B QLoRA smoke test on AMD RDNA4")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model to test (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--steps", type=int, default=10, help="Training steps (default: 10)")
    args = parser.parse_args()

    print("=" * 60)
    print("7B QLoRA via HQQ — RDNA4 Smoke Test")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"Python:  {sys.version}")

    if not torch.cuda.is_available():
        print("ERROR: No GPU detected. Ensure ROCm is installed and HSA_OVERRIDE_GFX_VERSION is set.")
        sys.exit(1)

    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Step 1: Load model in bf16, then quantize with HQQ post-load
    print(f"[1/5] Loading {args.model} in bf16, then HQQ 4-bit quantize...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
    import gc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in bf16 to CPU first, then quantize and move to GPU
    print("  Loading to CPU in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cpu",
    )

    # Quantize linear layers with HQQ
    print("  Quantizing linear layers with HQQ 4-bit...")
    quant_config = BaseQuantizeConfig(nbits=4, group_size=64)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    quantized_count = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(t in name for t in target_modules):
                hqq_layer = HQQLinear(
                    module,
                    quant_config,
                    compute_dtype=torch.bfloat16,
                    device="cuda:0",
                )
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], hqq_layer)
                quantized_count += 1

    print(f"  Quantized {quantized_count} layers")

    # Move remaining parameters to GPU
    print("  Moving remaining parameters to GPU...")
    model = model.to("cuda:0")

    gc.collect()
    torch.cuda.empty_cache()

    vram_after_load = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VRAM after load: {vram_after_load:.2f} GB")

    # Step 2: Force pure PyTorch backend
    print("[2/5] Setting HQQ to pure PyTorch backend...")
    HQQLinear.set_backend(HQQBackend.PYTORCH)
    print("  Backend: PYTORCH (no custom kernels)")

    # Step 3: Apply LoRA
    print("[3/5] Applying LoRA adapter...")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Step 4: Quick forward + backward pass
    print("[4/5] Testing forward + backward pass...")
    inputs = tokenizer("The quick brown fox", return_tensors="pt").to(model.device)
    inputs["labels"] = inputs["input_ids"].clone()

    output = model(**inputs)
    loss = output.loss
    print(f"  Forward pass OK. Loss: {loss.item():.4f}")

    loss.backward()
    print("  Backward pass OK.")

    vram_after_backward = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VRAM after backward: {vram_after_backward:.2f} GB")

    # Step 5: Run training steps with SFTTrainer
    print(f"[5/5] Running {args.steps} training steps...")
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    dummy_data = Dataset.from_dict({
        "messages": [
            [{"role": "user", "content": f"Question {i}"}, {"role": "assistant", "content": f"Answer {i} with some detail."}]
            for i in range(20)
        ]
    })

    training_args = SFTConfig(
        output_dir="/tmp/hqq-qlora-test",
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=5,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=256,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dummy_data,
        processing_class=tokenizer,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    print()
    print("=" * 60)
    print("SUCCESS: 7B QLoRA via HQQ works!")
    print(f"  GPU:           {torch.cuda.get_device_name(0)}")
    print(f"  Model:         {args.model}")
    print(f"  Peak VRAM:     {peak_vram:.2f} GB")
    print(f"  Training:      {elapsed:.1f}s for {args.steps} steps ({elapsed/args.steps:.1f}s/step)")
    print(f"  Quantization:  HQQ 4-bit (group_size=64)")
    print(f"  Backend:       Pure PyTorch (no custom CUDA/HIP kernels)")
    print("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree("/tmp/hqq-qlora-test", ignore_errors=True)


if __name__ == "__main__":
    main()
