r"""
Kernel-DeltaNet Engine
Module: kernels.fused_chunk_kernel
=====================================================
Chunk-Pipelined Fused Associative Prefix Scan Operator.
Evaluates intra-chunk Gram matrix contractions and inter-chunk
prefix state recurrence with exact autograd reverse-mode derivatives.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def fused_chunked_scan(
    phi_k: torch.Tensor,
    phi_q: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    eta: torch.Tensor,
    chunk_size: int = 64
) -> torch.Tensor:
    r"""
    Memory-efficient chunked associative scan with exact gradient propagation.
    
    Args:
        phi_k: Key features (B, H, T, D)
        phi_q: Query features (B, H, T, D)
        v:     Values (B, H, T, d_v)
        decay: Decay parameter (H,) or (1, H, 1, 1)
        eta:   Learning rate / step size (H,) or (1, H, 1, 1)
        chunk_size: Block tile size (default 64)
    Returns:
        out:   Causal scan output (B, H, T, d_v)
    """
    B, H, T, D = phi_k.shape
    d_v = v.shape[-1]
    device = phi_k.device
    dtype = phi_k.dtype

    # Pad to chunk_size if necessary
    remainder = T % chunk_size
    pad_len = (chunk_size - remainder) if remainder != 0 else 0
    if pad_len > 0:
        phi_k = F.pad(phi_k, (0, 0, 0, pad_len))
        phi_q = F.pad(phi_q, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))

    T_pad = T + pad_len
    num_chunks = T_pad // chunk_size
    C = chunk_size

    # Chunk views: (B, H, num_chunks, C, dim)
    pk_c = phi_k.view(B, H, num_chunks, C, D)
    pq_c = phi_q.view(B, H, num_chunks, C, D)
    v_c = v.view(B, H, num_chunks, C, d_v)

    decay_5d = decay.view(1, H, 1, 1, 1).to(dtype=dtype)
    eta_5d = eta.view(1, H, 1, 1, 1).to(dtype=dtype)

    # 1. Intra-chunk strictly causal weights
    i_idx = torch.arange(C, device=device).view(1, 1, 1, C, 1)
    j_idx = torch.arange(C, device=device).view(1, 1, 1, 1, C)
    power_diff = torch.clamp(i_idx - 1 - j_idx, min=0).to(dtype=dtype)
    causal_mask = (i_idx > j_idx).to(dtype=dtype)
    intra_weights = eta_5d * (decay_5d ** power_diff) * causal_mask

    gram = torch.matmul(pq_c, pk_c.transpose(-1, -2))
    y_intra = torch.matmul(gram * intra_weights, v_c)

    # 2. Inter-chunk scan
    c_steps = torch.arange(C, device=device).view(1, 1, 1, C, 1).to(dtype=dtype)
    chunk_token_decay = decay_5d ** (C - 1 - c_steps)
    v_decayed = v_c * chunk_token_decay
    chunk_states = torch.matmul(v_decayed.transpose(-1, -2), pk_c)

    decay_chunk = decay.view(1, H, 1, 1).to(dtype=dtype) ** C
    eta_4d = eta.view(1, H, 1, 1).to(dtype=dtype)

    inter_states = []
    running = torch.zeros(B, H, d_v, D, device=device, dtype=dtype)
    for c in range(num_chunks):
        inter_states.append(running)
        running = running * decay_chunk + eta_4d * chunk_states[:, :, c]

    inter_states = torch.stack(inter_states, dim=2)
    token_decay = decay_5d ** c_steps
    y_inter = torch.matmul(pq_c, inter_states.transpose(-1, -2)) * token_decay

    out = (y_intra + y_inter).view(B, H, T_pad, d_v)
    if pad_len > 0:
        out = out[:, :, :T, :]
    return out


# Backward compatibility alias for any legacy imports
FusedChunkedScanFunction = None
