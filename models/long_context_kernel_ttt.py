"""
Kernel-DeltaNet Engine
Module: models.long_context_kernel_ttt
======================================
Compatibility forwarding module.
Canonical implementation is now in `models.long_context_kernel_deltanet`.
"""

from models.long_context_kernel_deltanet import (
    StreamingLongContextKernelDeltaNet,
    StreamingLongContextKernelTTT,
)

__all__ = [
    "StreamingLongContextKernelDeltaNet",
    "StreamingLongContextKernelTTT",
]
