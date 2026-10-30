"""
Kernel-DeltaNet Engine
Module: models.kernel_ttt_lm
============================
Compatibility forwarding module.
Canonical implementation is now in `models.kernel_deltanet_lm`.
"""

from models.kernel_deltanet_lm import KernelDeltaNetLM, KernelTTTLM

__all__ = ["KernelDeltaNetLM", "KernelTTTLM"]
