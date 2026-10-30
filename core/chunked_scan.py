"""
Kernel-DeltaNet Engine
Module: core.chunked_scan
===========================================
Hardware-Native Chunked Parallel Associative Prefix Scan.

Mathematical Derivation:
-------------------------
The recurrent test-time state update in RKHS feature space is:
  S_t = lambda * S_{t-1} + eta * (v_t @ phi_t^T) in R^{d_v x D}
  y_t = S_{t-1} @ phi_t in R^{d_v}

Notice:
For any token t in chunk c (chunk length C, offset i in {0, ..., C-1}):
  S_t = lambda^i S_{start_c} + sum_{j=0}^{i-1} eta * lambda^{i - 1 - j} (v_{c, j} @ phi_{c, j}^T)

Therefore, the causal output y_t decomposes into:
  y_{c, i} = y_inter(c, i) + y_intra(c, i)
where:
  1. y_inter(c, i) = lambda^i * (S_{start_c} @ phi_{c, i})
  2. y_intra(c, i) = sum_{j < i} eta * lambda^{i - 1 - j} * <phi_{c, i}, phi_{c, j}> * v_{c, j}

Intra-Chunk Computation:
  Gram matrix G_{c, i, j} = <phi_{c, i}, phi_{c, j}>.
  Intra-decay matrix W_{i, j} = eta * lambda^{i - 1 - j} * [i > j].
  y_intra = (G * W) @ V_c  <--- Evaluated as pure batched GEMM on Tensor Cores!

Inter-Chunk State Propagation:
  Each chunk computes its total state delta:
    Delta_S_c = sum_{j=0}^{C-1} lambda^{C - 1 - j} * (v_{c, j} @ phi_{c, j}^T) in R^{d_v x D}
  The prefix scan across chunks:
    S_{c+1} = lambda^C * S_c + eta * Delta_S_c
  Takes only num_chunks = T / C steps (e.g. 4096 / 64 = 64 steps, taking microseconds).

This matches recurrent inference down to float32 machine epsilon while achieving
massive GPU parallelization.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def chunked_affine_scan(
    phi_k: torch.Tensor,
    v: torch.Tensor,
    decay: Optional[torch.Tensor] = None,
    eta: Optional[torch.Tensor] = None,
    phi_q: Optional[torch.Tensor] = None,
    chunk_size: int = 64
) -> torch.Tensor:
    """
    Computes parallel test-time memory state accumulation via chunked associative scan.
    Update equation: S_t = lambda * S_{t-1} + eta * v_t * phi_k_t^T
    Query readout:   y_t = S_{t-1} @ phi_q_t
    Strict causal formulation matching sequential step_recurrent.
    
    Args:
        phi_k: Key RKHS features of shape (B, H, T, D)
        v: Value vectors of shape (B, H, T, d_v)
        phi_q: Query RKHS features of shape (B, H, T, D). If None, defaults to phi_k.
        decay: Retention factor lambda in (0, 1] of shape (1, H, 1, 1)
        eta: Step size of shape (1, H, 1, 1)
        chunk_size: Intra-chunk GEMM block size (default: 64)
    Returns:
        outputs: Retrieved values of shape (B, H, T, d_v)
    """
    if phi_q is None:
        phi_q = phi_k
    if decay is None:
        decay = torch.ones(1, phi_k.shape[1], 1, 1, device=phi_k.device) * 0.99
    if eta is None:
        eta = torch.ones(1, phi_k.shape[1], 1, 1, device=phi_k.device) * 0.5

    B, H, T, D = phi_k.shape
    _, _, _, d_v = v.shape
    device = phi_k.device

    # Pad sequence to integer multiple of chunk_size if necessary
    remainder = T % chunk_size
    if remainder != 0:
        pad_len = chunk_size - remainder
        phi_k = torch.cat([phi_k, torch.zeros(B, H, pad_len, D, device=device)], dim=2)
        phi_q = torch.cat([phi_q, torch.zeros(B, H, pad_len, D, device=device)], dim=2)
        v = torch.cat([v, torch.zeros(B, H, pad_len, d_v, device=device)], dim=2)
        T_padded = T + pad_len
    else:
        pad_len = 0
        T_padded = T

    num_chunks = T_padded // chunk_size
    C = chunk_size

    # Reshape into chunks: (B, H, num_chunks, C, dim)
    phi_k_c = phi_k.view(B, H, num_chunks, C, D)
    phi_q_c = phi_q.view(B, H, num_chunks, C, D)
    v_c = v.view(B, H, num_chunks, C, d_v)

    dtype = phi_k.dtype
    decay_5d = decay.view(1, H, 1, 1, 1).to(dtype=dtype)
    eta_5d = eta.view(1, H, 1, 1, 1).to(dtype=dtype)

    i_idx = torch.arange(C, device=device).view(1, 1, 1, C, 1)
    j_idx = torch.arange(C, device=device).view(1, 1, 1, 1, C)
    # Strictly non-negative powers: clamp min=0 to avoid negative powers when i <= j (prevents decay**(-k) -> Inf -> Inf*0 = NaN)
    power_diff = torch.clamp(i_idx - 1 - j_idx, min=0).to(dtype=dtype)
    causal_mask = (i_idx > j_idx).to(dtype=dtype)   # (1, 1, 1, C, C) strictly lower triangular

    decay_matrix = (decay_5d ** power_diff) * causal_mask # (1, H, 1, C, C)
    intra_weights = eta_5d * decay_matrix                 # (1, H, 1, C, C)

    # Gram matrix: G_{c, i, j} = phi_q_{c, i} @ phi_k_{c, j}^T
    gram = torch.matmul(phi_q_c, phi_k_c.transpose(-1, -2)) # (B, H, num_chunks, C, C)
    y_intra = torch.matmul(gram * intra_weights, v_c)       # (B, H, num_chunks, C, d_v)

    # 2. Inter-Chunk Parallel Scan
    c_steps = torch.arange(C, device=device).view(1, 1, 1, C, 1).to(dtype=dtype)
    chunk_token_decay = decay_5d ** (C - 1 - c_steps)
    v_c_decayed = v_c * chunk_token_decay

    chunk_states = torch.matmul(v_c_decayed.transpose(-1, -2), phi_k_c) # (B, H, num_chunks, d_v, D)

    decay_chunk = (decay.view(1, H, 1, 1).to(dtype=dtype) ** C)
    inter_states = torch.zeros(B, H, num_chunks, d_v, D, device=device, dtype=dtype)
    running_state = torch.zeros(B, H, d_v, D, device=device, dtype=dtype)
    eta_4d = eta.view(1, H, 1, 1).to(dtype=dtype)

    for c in range(num_chunks):
        inter_states[:, :, c] = running_state
        running_state = running_state * decay_chunk + eta_4d * chunk_states[:, :, c]

    # 3. Combine Inter-Chunk Historical Memory with Intra-Chunk Attention
    token_decay = decay_5d ** c_steps
    y_inter = torch.matmul(phi_q_c, inter_states.transpose(-1, -2)) * token_decay

    out = (y_inter + y_intra).view(B, H, T_padded, d_v)

    if pad_len > 0:
        out = out[:, :, :T, :]
    return out


def recurrent_step(
    phi_t: torch.Tensor,
    v_t: torch.Tensor,
    q_t: torch.Tensor,
    state: Optional[torch.Tensor],
    decay: torch.Tensor,
    eta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact O(1) Memory Recurrent Step for Autoregressive Generation.
    Args:
        phi_t: Key RKHS vector of shape (B, H, 1, D)
        v_t: Value vector of shape (B, H, 1, d_v)
        q_t: Query RKHS vector of shape (B, H, 1, D)
        state: Prior memory state of shape (B, H, d_v, D). If None, initialized to 0.
        decay: Decay parameter lambda of shape (1, H, 1, 1)
        eta: Learning rate eta of shape (1, H, 1, 1)
    Returns:
        y_t: Readout output of shape (B, H, 1, d_v)
        new_state: Updated memory state of shape (B, H, d_v, D)
    """
    B, H, _, D = phi_t.shape
    _, _, _, d_v = v_t.shape
    device = phi_t.device

    dtype = phi_t.dtype
    if state is None:
        state = torch.zeros(B, H, d_v, D, device=device, dtype=dtype)
    else:
        state = state.to(dtype=dtype)
    decay = decay.to(dtype=dtype)
    eta = eta.to(dtype=dtype)

    # 1. Readout from state BEFORE updating (strictly causal):
    # y_t = state @ q_t^T -> (B, H, d_v, 1)
    y_t = torch.matmul(state, q_t.transpose(-1, -2)).transpose(-1, -2) # (B, H, 1, d_v)

    # 2. State update: S_t = lambda * S_{t-1} + eta * (v_t @ phi_t^T)
    # delta_S = v_t^T @ phi_t -> shape (B, H, d_v, D)
    delta_S = torch.matmul(v_t.transpose(-1, -2), phi_t) # (B, H, d_v, D)
    new_state = decay * state + eta * delta_S

    return y_t, new_state
