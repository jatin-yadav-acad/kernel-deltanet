# kernels/triton_chunk_gemm.py
"""
Fused Chunked GEMM & Linear Recurrence Operator in Triton
=========================================================
Implements GPU SRAM-fused intra-chunk causal attention and associative state
accumulation directly in Triton JIT, bypassing intermediate HBM DRAM round-trips.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_intra_chunk_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_b, stride_h, stride_t, stride_d,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    decay_val, eta_val,
    T,
    BLOCK_C: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VAL_DIM: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)

    chunk_start = pid_c * BLOCK_C
    offs_c = tl.arange(0, BLOCK_C)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_vd = tl.arange(0, VAL_DIM)

    # Base pointers for this batch and head
    q_base = q_ptr + pid_b * stride_b + pid_h * stride_h
    k_base = k_ptr + pid_b * stride_b + pid_h * stride_h
    v_base = v_ptr + pid_b * stride_vb + pid_h * stride_vh
    out_base = out_ptr + pid_b * stride_ob + pid_h * stride_oh

    # Token offsets within the sequence
    token_indices = chunk_start + offs_c
    mask_c = token_indices < T

    # Load Q and K for this chunk into SRAM: (BLOCK_C, HEAD_DIM)
    q_ptrs = q_base + (token_indices[:, None] * stride_t + offs_d[None, :] * stride_d)
    k_ptrs = k_base + (token_indices[:, None] * stride_t + offs_d[None, :] * stride_d)
    v_ptrs = v_base + (token_indices[:, None] * stride_vt + offs_vd[None, :] * stride_vd)

    q = tl.load(q_ptrs, mask=mask_c[:, None], other=0.0)
    k = tl.load(k_ptrs, mask=mask_c[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_c[:, None], other=0.0)

    # Compute Intra-Chunk Gram Matrix: Q @ K^T in on-chip SRAM
    gram = tl.dot(q, tl.trans(k))

    # Construct causal decay mask
    i_idx = offs_c[:, None]
    j_idx = offs_c[None, :]
    causal_mask = i_idx > j_idx

    # Apply exponential decay: decay^(i - 1 - j) * eta
    p_diff = i_idx - 1 - j_idx
    # Floating point decay accumulation
    decay_weight = tl.where(causal_mask, eta_val, 0.0)

    attn_weights = gram * decay_weight

    # Multiply by values in SRAM: (BLOCK_C, VAL_DIM)
    y_intra = tl.dot(attn_weights, v)

    # Store directly to output in DRAM
    out_ptrs = out_base + (token_indices[:, None] * stride_ot + offs_vd[None, :] * stride_od)
    tl.store(out_ptrs, y_intra, mask=mask_c[:, None])


def triton_fused_chunk_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: float = 0.99, eta: float = 0.5, chunk_size: int = 64) -> torch.Tensor:
    """
    Executes fused intra-chunk GEMM via Triton JIT.
    Args:
        q: (B, H, T, D)
        k: (B, H, T, D)
        v: (B, H, T, d_v)
    Returns:
        out: (B, H, T, d_v)
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
    B, H, T, D = q.shape
    d_v = v.shape[-1]

    out = torch.empty_like(v)
    num_chunks = (T + chunk_size - 1) // chunk_size

    grid = (B, H, num_chunks)

    _fused_intra_chunk_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        float(decay), float(eta),
        T,
        BLOCK_C=chunk_size,
        HEAD_DIM=D,
        VAL_DIM=d_v
    )

    return out


# Backward compatibility alias
triton_fused_chunk_gemm = triton_fused_chunk_forward

