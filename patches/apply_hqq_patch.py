#!/usr/bin/env python3
"""
Patches HQQ's quantize.py to fix dtype mismatch with prepare_model_for_kbit_training.

Problem: When PEFT's prepare_model_for_kbit_training upcasts some layers to float32,
HQQ's matmul() receives float32 inputs but dequantizes weights to bfloat16.
The matmul fails with a dtype mismatch error.

Fix: Add x.to(weight.dtype) casts in matmul() and forward_pytorch().

Usage:
    python patches/apply_hqq_patch.py

This finds your installed HQQ package and patches it in-place.
"""
import importlib.util
import os
import re
import sys


def find_hqq_quantize():
    """Find the installed hqq/core/quantize.py path."""
    spec = importlib.util.find_spec("hqq.core.quantize")
    if spec is None or spec.origin is None:
        print("ERROR: hqq package not found. Install with: DISABLE_CUDA=1 pip install hqq")
        sys.exit(1)
    return spec.origin


def patch_file(filepath):
    """Apply dtype cast patches to quantize.py."""
    with open(filepath, "r") as f:
        content = f.read()

    changes = 0

    # Patch 1: matmul() — add x = x.to(weight.dtype) after weight = self.dequantize()
    old_matmul = "def matmul(self, x: Tensor, transpose: bool = True) -> Tensor:\n        weight = self.dequantize()\n        return torch.matmul(x, weight.t()"
    new_matmul = "def matmul(self, x: Tensor, transpose: bool = True) -> Tensor:\n        weight = self.dequantize()\n        x = x.to(weight.dtype)  # RDNA4 patch: cast to match dequantized weight dtype\n        return torch.matmul(x, weight.t()"

    if "x = x.to(weight.dtype)" not in content and old_matmul in content:
        content = content.replace(old_matmul, new_matmul)
        changes += 1
        print("  Patched matmul()")

    # Patch 2: forward_pytorch() — cast x in the matmul call
    old_forward = "def forward_pytorch(self, x: Tensor) -> Tensor:\n        w = self.dequantize()\n        out = torch.matmul(x, w.t())"
    new_forward = "def forward_pytorch(self, x: Tensor) -> Tensor:\n        w = self.dequantize()\n        out = torch.matmul(x.to(w.dtype), w.t())  # RDNA4 patch: cast to match weight dtype"

    if "x.to(w.dtype)" not in content and old_forward in content:
        content = content.replace(old_forward, new_forward)
        changes += 1
        print("  Patched forward_pytorch()")

    if changes == 0:
        # Check if already patched
        if "RDNA4 patch" in content or "x.to(weight.dtype)" in content:
            print("  Already patched!")
            return True
        else:
            print("  WARNING: Could not find expected code patterns to patch.")
            print("  You may need to apply the patch manually. See README.md for details.")
            return False

    # Write patched file
    with open(filepath, "w") as f:
        f.write(content)

    print(f"  Applied {changes} patch(es)")
    return True


def main():
    filepath = find_hqq_quantize()
    print(f"Found HQQ at: {filepath}")

    # Backup
    backup = filepath + ".bak"
    if not os.path.exists(backup):
        import shutil
        shutil.copy2(filepath, backup)
        print(f"Backup saved: {backup}")

    success = patch_file(filepath)
    if success:
        print("\nHQQ patched successfully. You can now use HQQ with prepare_model_for_kbit_training().")
    else:
        print("\nPatch failed. See README.md for manual patching instructions.")
        sys.exit(1)


if __name__ == "__main__":
    main()
