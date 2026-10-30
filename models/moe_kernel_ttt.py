"""
Kernel-DeltaNet Engine
Module: models.moe_kernel_ttt
=============================
Compatibility forwarding module.
Canonical implementation is now in `models.moe_kernel_deltanet`.
"""

from models.moe_kernel_deltanet import MoEKernelDeltaNetBlock, MoEKernelTTTBlock

__all__ = ["MoEKernelDeltaNetBlock", "MoEKernelTTTBlock"]
