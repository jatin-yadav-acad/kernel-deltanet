"""
Models module for Kernel-DeltaNet (MLSys 2026)
"""

from .kernel_deltanet_lm import KernelDeltaNetLM, KernelDeltaNetLM
from .long_context_kernel_deltanet import StreamingLongContextKernelDeltaNet, StreamingLongContextKernelDeltaNet
from .moe_kernel_deltanet import MoEKernelDeltaNetBlock, MoEKernelDeltaNetBlock
from .upcycling_llama_bridge import LlamaKernelDeltaNetBlock, construct_upcycled_1b_model

__all__ = [
    "KernelDeltaNetLM",
    "KernelDeltaNetLM",
    "StreamingLongContextKernelDeltaNet",
    "StreamingLongContextKernelDeltaNet",
    "MoEKernelDeltaNetBlock",
    "MoEKernelDeltaNetBlock",
    "LlamaKernelDeltaNetBlock",
    "construct_upcycled_1b_model",
]
