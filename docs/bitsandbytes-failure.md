# Why bitsandbytes Fails on AMD RDNA4

## Summary

As of March 2026, bitsandbytes cannot be used for QLoRA on AMD RDNA4 GPUs (RX 9070 series, gfx1200/gfx1201). There are two independent failure paths, and both are currently unsolvable without upstream changes.

## Failure Path 1: PyPI Wheel

The pre-built bitsandbytes wheel on PyPI includes HIP kernels compiled for older AMD GPU architectures. RDNA4 introduces new ISA targets (gfx1200, gfx1201) that are not included in the build matrix.

**Error**:

```
hipErrorNoBinaryForGpu: Unable to find code object for all current devices!
Error invalid device function at line 68 in file /src/csrc/ops.hip
```

**Root cause**: The PyPI wheel's `.so` binary doesn't contain compiled kernels for gfx1200/gfx1201.

## Failure Path 2: Building from Source

Building bitsandbytes from source with the correct architecture target:

```bash
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1200" ..
make -j$(nproc)
```

This compiles successfully and produces `libbitsandbytes_rocm72.so`. But it links against ROCm 7.2's HSA runtime (the system-installed ROCm), while PyTorch 2.9.1+rocm6.3 bundles ROCm 6.3's runtime libraries.

**Error**:

```
ImportError: /path/to/libbitsandbytes_rocm72.so: undefined symbol: hsa_amd_memory_get_preferred_copy_engine
```

**Root cause**: `hsa_amd_memory_get_preferred_copy_engine` was added in ROCm 7.2. PyTorch's bundled ROCm 6.3 runtime doesn't have it. The binary compiled against 7.2 headers requires 7.2 symbols at load time.

## Why Can't We Just Use ROCm 6.3 Headers?

ROCm 6.3 didn't include gfx1200/gfx1201 as supported targets. Even if you try to build bitsandbytes against ROCm 6.3, the HIP compiler won't recognize `--offload-arch=gfx1200` and the build fails.

## The Dependency Loop

```
gfx1200 support → requires ROCm 7.2+ headers
ROCm 7.2 headers → produces binary needing ROCm 7.2 runtime
PyTorch → ships with ROCm 6.3 runtime
```

There is no way to satisfy all three constraints simultaneously.

## Fix Path

When PyTorch releases a wheel built against ROCm 7.2 (or later), bitsandbytes compiled from source with `gfx1200` should work. The ROCm runtime versions will match.

Track:

- [PyTorch ROCm support matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/3rd-party-support-matrix.html)
- [bitsandbytes ROCm issues](https://github.com/bitsandbytes-foundation/bitsandbytes/issues)

## Alternative: HQQ

[HQQ (Half-Quadratic Quantization)](https://github.com/mobiusml/hqq) provides 4-bit quantization using pure PyTorch operations. No custom CUDA/HIP kernels means no architecture-specific build requirements. See the main README for usage.
