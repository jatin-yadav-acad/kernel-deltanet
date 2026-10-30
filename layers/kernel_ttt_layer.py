"""
Kernel-DeltaNet Engine
Module: layers.kernel_ttt_layer
===============================
Compatibility forwarding module.
Canonical implementation is now in `layers.kernel_deltanet_layer`.
"""

from layers.kernel_deltanet_layer import KernelDeltaNetLayer, KernelTTTLayer

__all__ = ["KernelDeltaNetLayer", "KernelTTTLayer"]
