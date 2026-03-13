# QLoRA Fine-Tuning on AMD RDNA4 (RX 9070 XT)

**7B model fine-tuning on AMD's newest GPU architecture — no custom CUDA/HIP kernels required.**

AMD's RX 9070 XT (RDNA4, gfx1200/gfx1201) has 16GB VRAM — more than enough for 7B QLoRA training. But the standard quantization library `bitsandbytes` doesn't work on RDNA4. This repo documents the problem, the solution, and provides ready-to-use training scripts.

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

## Benchmark Results

**Hardware**: AMD RX 9070 XT (RDNA4, gfx1201, 16GB VRAM)
**Model**: Qwen2.5-7B-Instruct
**Software**: PyTorch 2.9.1+rocm6.3, Python 3.14
**Date**: 2026-03-12

| Metric               | HQQ 4-bit QLoRA (7B)     |
| -------------------- | ------------------------ |
| **Peak VRAM**        | 10.09 GB                 |
| **Speed**            | 3.0 s/step               |
| **Quantization**     | HQQ 4-bit, group_size=64 |
| **Trainable params** | 40.4M (0.92%)            |
| **Model capacity**   | 7.6B total               |
| **Backend**          | Pure PyTorch             |

Key findings:

- **VRAM**: 5.85 GB after loading quantized model, 10.09 GB peak during training with gradient checkpointing
- **Speed**: 3.0 seconds/step (batch_size=1, grad_accum=4, max_length=256)
- **Headroom**: ~6 GB VRAM free at peak — room for longer sequences or larger batch sizes

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
python train.py judge-7b
```

## Repository Structure

```
.
├── README.md                # This file
├── smoke_test.py            # Quick validation — loads, quantizes, trains 10 steps
├── train.py                 # Full training script with HQQ + LoRA
├── patches/
│   └── apply_hqq_patch.py   # Auto-patches HQQ dtype issue
├── docs/
│   └── bitsandbytes-failure.md   # Detailed bitsandbytes failure analysis
└── failed_approaches/
    └── bitsandbytes_test.py      # bitsandbytes attempt (for reference)
```

## Why Not bitsandbytes?

We tried building bitsandbytes from source with `cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1200"`. It compiled successfully, but the resulting `libbitsandbytes_rocm72.so` links against ROCm 7.2's HSA runtime, while PyTorch 2.9.1+rocm6.3 bundles ROCm 6.3's runtime. See [`docs/bitsandbytes-failure.md`](docs/bitsandbytes-failure.md) for the full analysis.

**Fix path**: When PyTorch releases a wheel built against ROCm 7.2, bitsandbytes compiled from source should work. Until then, HQQ is the reliable alternative.

## Additional Patches

**Python 3.14+ compatibility**: HQQ uses `@torch.compile()` as a class method decorator, but `torch.compile` is not supported on Python 3.14+. The training scripts automatically monkey-patch `torch.compile` to return an identity decorator when running on Python 3.14+.

## Environment Variables

| Variable                   | Purpose                                         | Example                    |
| -------------------------- | ----------------------------------------------- | -------------------------- |
| `HSA_OVERRIDE_GFX_VERSION` | Tell ROCm to treat your GPU as a known target   | `12.0.0`                   |
| `DISABLE_CUDA`             | Skip CUDA kernel compilation during HQQ install | `1`                        |
| `PYTORCH_HIP_ALLOC_CONF`   | Optional: tune HIP memory allocator             | `expandable_segments:True` |

## Related Links

- [HQQ GitHub](https://github.com/mobiusml/hqq) — Half-Quadratic Quantization
- [PEFT HQQ support](https://huggingface.co/docs/peft/developer_guides/quantization#hqq) — LoRA on HQQ layers
- [ROCm PyTorch compatibility matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/3rd-party-support-matrix.html)

## Contributing

Found a fix for bitsandbytes on RDNA4? Got it working on a different AMD GPU? Open an issue or PR.

## License

MIT
