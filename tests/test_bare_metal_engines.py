"""
tests/test_bare_metal_engines.py
================================
Unit tests for bare-metal Triton fused chunk scan and tiled engines on CUDA GPU.
"""

import pytest
import torch
import torch.nn.functional as F

from kernels.fused_triton_engine import fused_triton_chunk_scan
from kernels.tiled_triton_engine import tiled_triton_chunk_scan
from kernels.cuda_graph_decode_engine import FusedDecodeCell

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bare-metal tests")
def test_fused_triton_chunk_scan_execution():
    device = torch.device("cuda:0")
    B, H, T, D, d_v = 2, 2, 256, 64, 64
    q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, d_v, device=device, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=torch.bfloat16))
    decay = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=torch.bfloat16))

    out = fused_triton_chunk_scan(q, k, v, beta, decay, chunk_size=64)
    assert out.shape == (B, H, T, d_v)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bare-metal tests")
def test_tiled_triton_chunk_scan_d512():
    device = torch.device("cuda:0")
    B, H, T, D, d_v = 2, 2, 128, 512, 64
    q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, d_v, device=device, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=torch.bfloat16))
    decay = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=torch.bfloat16))

    out = tiled_triton_chunk_scan(q, k, v, beta, decay, chunk_size=64, block_d=64)
    assert out.shape == (B, H, T, d_v)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bare-metal tests")
def test_cuda_graph_decode_cell():
    device = torch.device("cuda:0")
    batch_size = 4
    num_heads = 2
    d_v = 64
    D = 64
    
    cell = FusedDecodeCell(num_heads=num_heads, d_v=d_v, D=D).to(device)
    S = torch.zeros(batch_size, num_heads, d_v, D, device=device, dtype=torch.bfloat16)
    q = torch.randn(batch_size, num_heads, 1, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, 1, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, 1, d_v, device=device, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(batch_size, num_heads, 1, device=device, dtype=torch.bfloat16))
    decay = torch.sigmoid(torch.randn(batch_size, num_heads, 1, device=device, dtype=torch.bfloat16))
    
    out, S_next = cell(q, k, v, beta, decay, S)
    assert out.shape == (batch_size, num_heads, 1, d_v)
    assert S_next.shape == (batch_size, num_heads, d_v, D)
    assert not torch.isnan(out).any()
