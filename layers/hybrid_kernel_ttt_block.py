"""
Kernel-DeltaNet Engine
Module: layers.hybrid_kernel_ttt_block
======================================
Compatibility forwarding module.
Canonical implementation is now in `layers.hybrid_kernel_deltanet_block`.
"""

from layers.hybrid_kernel_deltanet_block import (
    RMSNorm,
    SwiGLUFFN,
    HybridKernelDeltaNetBlock,
    HybridKernelTTTBlock,
)

__all__ = [
    "RMSNorm",
    "SwiGLUFFN",
    "HybridKernelDeltaNetBlock",
    "HybridKernelTTTBlock",
]
