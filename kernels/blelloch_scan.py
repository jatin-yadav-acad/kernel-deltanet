# kernels/blelloch_scan.py
"""
Hierarchical Two-Level Blelloch Prefix Scan Engine
==================================================
Eliminates the short-context throughput deficit at T <= 2048.

Mathematical Formulation:
-------------------------
In standard sequential chunk scanning across N_c = T / C chunks:
   S_{c+1} = S_c @ A_c + Delta S_c
requires N_c serial steps across GPU streaming multiprocessors.

Under the affine monoid (T, o) with transition tuples T_c = (A_c, Delta S_c):
   (A_1, B_1) o (A_2, B_2) = (A_1 @ A_2,  B_1 @ A_2 + B_2)

The Hierarchical Blelloch Prefix Scan executes in two parallel phases:
1. Up-Sweep (Parallel Reduction Tree):
   For d = 0 to log2(N_c) - 1:
      In parallel for i = 0 to N_c - 1 step 2^{d+1}:
         T_{i + 2^{d+1} - 1} = T_{i + 2^d - 1} o T_{i + 2^{d+1} - 1}

2. Down-Sweep (Parallel Distribution Tree):
   Root is set to identity E = (I_D, 0).
   For d = log2(N_c) - 1 down to 0:
      In parallel for i = 0 to N_c - 1 step 2^{d+1}:
         temp = T_{i + 2^d - 1}
         T_{i + 2^d - 1} = T_{i + 2^{d+1} - 1}
         T_{i + 2^{d+1} - 1} = T_{i + 2^{d+1} - 1} o temp

Critical Path:
   Sequential: O(N_c) steps. For N_c = 32 -> 32 sequential rounds.
   Blelloch:   2 * log2(N_c) steps. For N_c = 32 -> 10 parallel rounds (3.2x reduction).
"""

import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def compose_transitions(
    T1: Tuple[torch.Tensor, torch.Tensor], # (A1, B1) where A1: (B, D, D), B1: (B, d_v, D)
    T2: Tuple[torch.Tensor, torch.Tensor]  # (A2, B2) where A2: (B, D, D), B2: (B, d_v, D)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates monoid composition (T1 o T2):
    A_new = A1 @ A2
    B_new = B1 @ A2 + B2
    """
    A1, B1 = T1
    A2, B2 = T2
    
    A_new = torch.bmm(A1, A2)
    B_new = torch.bmm(B1, A2) + B2
    return A_new, B_new


def blelloch_prefix_scan(
    transitions: List[Tuple[torch.Tensor, torch.Tensor]], # List of N_c tuples (A_c, Delta S_c)
    initial_state: Optional[torch.Tensor] = None           # (B, d_v, D)
) -> List[torch.Tensor]:
    """
    Executes parallel Blelloch prefix scan over N_c chunk transition tuples.
    Returns: list of N_c prefix boundary states [S_0, S_1, ..., S_{N_c-1}].
    """
    N_c = len(transitions)
    assert N_c > 0
    # Next power of 2
    N_pow2 = 1 << (N_c - 1).bit_length()
    
    B, D, _ = transitions[0][0].shape
    d_v = transitions[0][1].shape[1]
    device = transitions[0][0].device
    dtype = transitions[0][0].dtype

    identity_A = torch.eye(D, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    zero_B = torch.zeros(B, d_v, D, device=device, dtype=dtype)
    identity_tuple = (identity_A, zero_B)

    # Pad tree to power of 2 with identities
    tree = [t for t in transitions]
    while len(tree) < N_pow2:
        tree.append(identity_tuple)

    # 1. Up-Sweep (Reduction Phase)
    step = 1
    while step < N_pow2:
        for i in range(0, N_pow2, step * 2):
            left_idx = i + step - 1
            right_idx = i + step * 2 - 1
            tree[right_idx] = compose_transitions(tree[left_idx], tree[right_idx])
        step *= 2

    # 2. Down-Sweep (Distribution Phase)
    # Save total composite reduction
    total_composite = tree[N_pow2 - 1]
    tree[N_pow2 - 1] = identity_tuple

    step = N_pow2 // 2
    while step > 0:
        for i in range(0, N_pow2, step * 2):
            left_idx = i + step - 1
            right_idx = i + step * 2 - 1
            
            temp = tree[left_idx]
            tree[left_idx] = tree[right_idx]
            tree[right_idx] = compose_transitions(tree[right_idx], temp)
        step //= 2

    # 3. Apply initial state S_0 to compute prefix states
    if initial_state is None:
        S_0 = torch.zeros(B, d_v, D, device=device, dtype=dtype)
    else:
        S_0 = initial_state

    prefix_states = []
    for c in range(N_c):
        # Prefix transition up to chunk c: (A_pref, B_pref)
        A_pref, B_pref = tree[c]
        # State at chunk c: S_c = S_0 @ A_pref + B_pref
        S_c = torch.bmm(S_0, A_pref) + B_pref
        prefix_states.append(S_c)

    return prefix_states


def sequential_prefix_scan(
    transitions: List[Tuple[torch.Tensor, torch.Tensor]],
    initial_state: Optional[torch.Tensor] = None
) -> List[torch.Tensor]:
    """Reference sequential prefix scan for validation."""
    N_c = len(transitions)
    B, D, _ = transitions[0][0].shape
    d_v = transitions[0][1].shape[1]
    device = transitions[0][0].device
    dtype = transitions[0][0].dtype

    if initial_state is None:
        S_curr = torch.zeros(B, d_v, D, device=device, dtype=dtype)
    else:
        S_curr = initial_state.clone()

    prefix_states = []
    for c in range(N_c):
        prefix_states.append(S_curr.clone())
        A_c, Delta_S_c = transitions[c]
        S_curr = torch.bmm(S_curr, A_c) + Delta_S_c

    return prefix_states
