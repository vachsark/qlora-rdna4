# QLoRA Fine-Tuning on AMD RDNA4 (RX 9070 XT)

**7B model fine-tuning on AMD's newest GPU architecture — no custom CUDA/HIP kernels required.**

AMD's RX 9070 XT (RDNA4, gfx1200/gfx1201) has 16GB VRAM — more than enough for 7B QLoRA training. But the standard quantization library `bitsandbytes` doesn't work on RDNA4. This repo documents the problem, the solution, and provides ready-to-use training scripts.

**We used this to train 5 production models** (Qwen2.5-7B-Instruct) that run daily automated tasks — knowledge evaluation, goal planning, research synthesis, and more. All trained and deployed on a single consumer GPU.

## The Problem

Two failure modes prevent bitsandbytes from working on RDNA4:

1. **PyPI wheels** are compiled for older ROCm/CUDA targets. The HIP kernel crashes on gfx1200/gfx1201 with `hipErrorNoBinaryForGpu`.
2. **Building from source** against ROCm 7.2 produces a binary that links against ROCm 7.2's HSA runtime, but PyTorch ships with ROCm 6.3 bundled. The symbol `hsa_amd_memory_get_preferred_copy_engine` exists in 7.2 but not 6.3, causing a load-time crash.

Without 4-bit quantization, a 7B model in bf16 needs ~14GB just for weights, leaving no room for activations and gradients.

## The Solution: HQQ (Half-Quadratic Quantization)

[HQQ](https://github.com/mobiusml/hqq) provides 4-bit quantization with a **pure PyTorch backend** — no custom CUDA/HIP kernels. It works on any device PyTorch supports, including RDNA4.

### How It Works

Instead of using the `transformers` `HqqConfig` integration (which is broken in transformers 5.x), we:

1. Load the model in bf16 to CPU
2. Manually replace target linear layers with `HQQLinear` quantized versions on GPU
3. Set `HQQBackend.PYTORCH` to use pure PyTorch ops (no custom kernels)
4. Apply LoRA adapters on top of the quantized layers

This gives us 4-bit weights with bf16 compute — same effective setup as bitsandbytes NF4, but without any platform-specific kernels.

## Production Results

We trained 5 task-specific models from Qwen2.5-7B-Instruct using this pipeline:

**Hardware**: AMD RX 9070 XT (RDNA4, gfx1201, 16GB VRAM)
**Software**: PyTorch 2.9.1+rocm6.3, Python 3.14
**Date**: 2026-03-13

| Model                | Task                                  | Examples | Peak VRAM | Training Time | max_length |
| -------------------- | ------------------------------------- | -------- | --------- | ------------- | ---------- |
| vault-judge-7b       | Knowledge quality evaluation          | 67       | 14.50 GB  | 21 min        | 2048       |
| vault-planner-7b     | Goal planning & prioritization        | 99       | 12.21 GB  | 14 min        | 1024       |
| vault-seeder-7b      | Research note synthesis               | 64       | 12.21 GB  | 9 min         | 1024       |
| vault-analyst-7b     | Technical analysis & recommendations  | 77       | 12.21 GB  | 11 min        | 1024       |
| vault-reflector-3b\* | Session reflection & pattern analysis | 42       | 11.45 GB  | 11 min        | 2048       |

_\*Reflector uses 3B (Qwen2.5-3B-Instruct, full LoRA bf16) because its outputs are short enough to fit without quantization._

### Smoke Test Results (Synthetic Data)

| Metric               | HQQ 4-bit QLoRA (7B)     |
| -------------------- | ------------------------ |
| **Model VRAM**       | 5.85 GB                  |
| **Peak VRAM**        | 10.09 GB                 |
| **Speed**            | 3.0 s/step               |
| **Quantization**     | HQQ 4-bit, group_size=64 |
| **Trainable params** | 40.4M (0.92%)            |
| **Backend**          | Pure PyTorch             |

### VRAM Budget Guide

The main constraint is `max_length` — longer sequences need more activation memory:

| max_length | Peak VRAM (7B HQQ) | Status                  |
| ---------- | ------------------ | ----------------------- |
| 256        | ~10 GB             | Comfortable             |
| 1024       | ~12.2 GB           | Good (3.7 GB headroom)  |
| 2048       | ~14.5 GB           | Tight (1.4 GB headroom) |
| 4096       | OOM                | Exceeds 16 GB           |

For 16GB GPUs, **max_length=1024 is the sweet spot** for 7B QLoRA training. Use gradient checkpointing (enabled by default in our scripts).

## Quick Start

### Prerequisites

- AMD GPU with ROCm support (tested: RX 9070 XT, gfx1200/gfx1201)
- PyTorch 2.9.1+rocm6.3 (or newer)
- Python 3.12+ (3.14 requires torch.compile patch — handled automatically)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate

# Install PyTorch with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# Install training dependencies
pip install transformers peft trl datasets accelerate

# Install HQQ (disable CUDA build)
DISABLE_CUDA=1 pip install hqq
```

### Patch HQQ for dtype compatibility

HQQ's `matmul()` doesn't cast input tensors to match the dequantized weight dtype. When `prepare_model_for_kbit_training` upcasts some layers to float32, the matmul fails with a dtype mismatch. Apply the patch:

```bash
python patches/apply_hqq_patch.py
```

Or manually edit `.venv/lib/pythonX.Y/site-packages/hqq/core/quantize.py`:

**In `matmul()` (~line 880)**:

```python
def matmul(self, x: Tensor, transpose: bool = True) -> Tensor:
    weight = self.dequantize()
    x = x.to(weight.dtype)  # <-- add this line
    return torch.matmul(x, weight.t() if (transpose) else weight)
```

**In `forward_pytorch()` (~line 894)**:

```python
def forward_pytorch(self, x: Tensor) -> Tensor:
    w = self.dequantize()
    out = torch.matmul(x.to(w.dtype), w.t())  # <-- cast x
    if self.bias is not None:
        out += self.bias
    return out
```

### Run the smoke test

```bash
# Set GFX version override if needed (for RDNA4)
export HSA_OVERRIDE_GFX_VERSION=12.0.0

python smoke_test.py
```

### Train a model

```bash
python train.py --model Qwen/Qwen2.5-7B-Instruct \
    --data training_data.jsonl \
    --output ./my-model \
    --max-length 1024 \
    --epochs 3 \
    --lr 5e-5 \
    --neftune-alpha 5
```

The script automatically:

- Converts data to prompt/completion format for `completion_only_loss`
- Applies smart middle-truncation for long inputs
- Adds NEFTune embedding noise for better generalization
- Uses HQQ-safe merge (reload base in bf16 → apply LoRA → merge) for clean GGUF conversion

## Gotchas (Lessons from Production)

These cost us hours of debugging. Save yourself the pain.

### 1. Ollama Modelfile MUST include chat template (silent, catastrophic)

When registering a custom GGUF model with `ollama create`, you **must** include the model's chat template in the Modelfile. Without it, Ollama sends raw text — the model can't parse system/user/assistant roles and generates input continuations instead of following instructions.

**Symptom**: Model outputs look like note content or Q&A instead of task-specific output. Looks like a training bug, but it's a deployment bug.

**Fix**: Add the full ChatML template for Qwen2.5 models:

```
FROM your-model.Q8_0.gguf
TEMPLATE """{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}{{ if not $last }}<|im_end|>
{{ end }}
{{- end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- end }}"""
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
```

### 2. completion_only_loss is mandatory for high input/output ratios

If your training data has long inputs and short outputs (ratio > 5:1), you **must** use `completion_only_loss=True`. Without it, 85%+ of the gradient signal teaches the model to predict input tokens — the model learns "continue the input" instead of "follow instructions and produce output."

Our training data had ratios from 6:1 to 28:1. All models trained without loss masking were completely broken (scored 0/5 on format compliance). The base model with no fine-tuning outperformed them all.

**This is now the default in our `train.py` script.** The data is automatically converted to prompt/completion format with smart middle-truncation.

### 3. Smart truncation: keep start + end, drop middle

When inputs exceed `max_length`, don't use TRL's default right-truncation (which cuts off the completion) or simple left-truncation (which removes task framing headers). Instead, truncate from the **middle** of the user content:

- **Keep the start (1/3)**: Contains task framing headers (e.g., `=== KNOWLEDGE JUDGE DATA ===`)
- **Keep the end (2/3)**: Contains the most recent/relevant content
- **Remove the middle**: Redundant intermediate content

This preserves both the task instructions and the most relevant data.

## Patches We Discovered

Three patches were needed beyond the standard HQQ + PEFT setup. None of these are documented elsewhere:

### 1. HQQ dtype mismatch (critical)

`prepare_model_for_kbit_training` upcasts some layers to float32, but HQQ's `matmul()` doesn't cast inputs to match. See `patches/apply_hqq_patch.py`.

### 2. Python 3.14 torch.compile (if applicable)

HQQ uses `@torch.compile()` as a class method decorator. Python 3.14 raises `RuntimeError: torch.compile is not supported on Python 3.14+`. Our scripts monkey-patch `torch.compile` to return an identity decorator.

### 3. Clean LoRA merge for HQQ models

After training, `model.merge_and_unload()` on an HQQ model produces tensors with HQQ-specific names (`W_q`, `meta`) that llama.cpp can't convert. The fix: save the LoRA adapter, reload the base model in bf16 (no HQQ), apply the adapter, then merge. See `train.py` for the implementation.

## Repository Structure

```
.
├── README.md                     # This file
├── smoke_test.py                 # Quick validation — loads, quantizes, trains 10 steps
├── train.py                      # Full training script with HQQ + LoRA
├── patches/
│   └── apply_hqq_patch.py        # Auto-patches HQQ dtype issue
├── docs/
│   └── bitsandbytes-failure.md   # Detailed bitsandbytes failure analysis
└── failed_approaches/
    └── bitsandbytes_test.py      # bitsandbytes attempt (for reference)
```

## Why Not bitsandbytes?

We tried building bitsandbytes from source with `cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1200"`. It compiled successfully, but the resulting `libbitsandbytes_rocm72.so` links against ROCm 7.2's HSA runtime, while PyTorch 2.9.1+rocm6.3 bundles ROCm 6.3's runtime. See [`docs/bitsandbytes-failure.md`](docs/bitsandbytes-failure.md) for the full analysis.

**Fix path**: When PyTorch releases a wheel built against ROCm 7.2, bitsandbytes compiled from source should work. Until then, HQQ is the reliable alternative.

## Environment Variables

| Variable                   | Purpose                                          | Example                    |
| -------------------------- | ------------------------------------------------ | -------------------------- |
| `HSA_OVERRIDE_GFX_VERSION` | Tell ROCm to treat your GPU as a known target    | `12.0.0`                   |
| `DISABLE_CUDA`             | Skip CUDA kernel compilation during HQQ install  | `1`                        |
| `PYTORCH_ALLOC_CONF`       | Tune memory allocator (helps with fragmentation) | `expandable_segments:True` |

## Related Links

- [HQQ GitHub](https://github.com/mobiusml/hqq) — Half-Quadratic Quantization
- [PEFT HQQ support](https://huggingface.co/docs/peft/developer_guides/quantization#hqq) — LoRA on HQQ layers
- [ROCm PyTorch compatibility matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/3rd-party-support-matrix.html)
- [Blog post: QLoRA on RDNA4](https://blog.vachsark.com/qlora-rdna4) — Extended writeup with context

## Contributing

Found a fix for bitsandbytes on RDNA4? Got it working on a different AMD GPU? Open an issue or PR.

## License

MIT
