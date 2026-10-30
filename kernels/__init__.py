"""
Kernels package for Kernel-DeltaNet.
"""

from .fwht import FastWalshHadamardProjection, fast_walsh_hadamard_transform
from .fused_kernel_deltanet import fused_chunked_deltanet_forward, FusedKernelDeltaNet
from .triton_chunk_gemm import triton_fused_chunk_gemm
from .roofline_analyzer import HardwareRooflineProfiler
from .fused_triton_engine import fused_triton_chunk_scan
from .tiled_triton_engine import tiled_triton_chunk_scan
from .cuda_graph_decode_engine import FusedDecodeCell

__all__ = [
    "FastWalshHadamardProjection",
    "fast_walsh_hadamard_transform",
    "fused_chunked_deltanet_forward",
    "FusedKernelDeltaNet",
    "triton_fused_chunk_gemm",
    "HardwareRooflineProfiler",
    "fused_triton_chunk_scan",
    "tiled_triton_chunk_scan",
    "FusedDecodeCell",
]
