# core/state_warmstart.py
"""
Closed-Form Stationary Prior State Warm-Start Engine
====================================================
Eliminates the Step 0 "Uniform Random Guessing" problem in student distillation.

Mathematical Derivation:
------------------------
In standard DeltaNet / linear recurrence, initializing S_0 = 0 means:
   y_1 = S_0 @ phi(q_1) = 0
producing completely uncalibrated zero logits at Step 0, with cross-entropy:
   L_uniform = ln(V) ~ 10.80 (for V = 49,152) or 11.76 (for V = 128,256).

We derive the Stationary Unigram Memory State S_0* under steady-state recurrence:
   E[S_inf] = E[S_inf] @ E[lambda_t * (I - beta_t * phi_t * phi_t^T)] + E[beta_t * v_t * phi_t^T]

Under empirical unigram distribution p = (p_1, ..., p_V):
   Sigma_phi = sum_{w=1}^V p_w * phi(k_w) * phi(k_w)^T in R^{D x D}
   M_v_phi   = sum_{w=1}^V p_w * v_w * phi(k_w)^T       in R^{d_v x D}

Setting S_0* as the fixed point under mean decay lambda_bar and learning rate beta_bar:
   S_0* @ [ (1 - lambda_bar) * I_D + lambda_bar * beta_bar * Sigma_phi ] = beta_bar * M_v_phi

Yielding the closed-form analytical warm-start state:
   S_0* = beta_bar * M_v_phi @ [ (1 - lambda_bar) * I_D + lambda_bar * beta_bar * Sigma_phi ]^(-1)

Alongside closed-form gate bias initializations:
   b_beta  = logit( 1 / sqrt(d_k) )
   b_decay = logit( 1 - 1 / T_context )
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_analytical_state_warmstart(
    W_k: torch.Tensor,               # (d_k, d_model)
    W_v: torch.Tensor,               # (d_v, d_model)
    embedding_table: torch.Tensor,   # (V, d_model)
    fwht_proj: nn.Module,            # R^{d_k} -> S^{D-1}
    unigram_probs: Optional[torch.Tensor] = None, # (V,)
    mean_decay: float = 0.98,
    mean_beta: Optional[float] = None,
    reg: float = 1e-4
) -> torch.Tensor:
    """
    Computes S_0* in R^{d_v x D} in closed form.
    """
    V, d_model = embedding_table.shape
    d_k = W_k.shape[0]
    d_v = W_v.shape[0]
    D = fwht_proj.D
    device = embedding_table.device

    if mean_beta is None:
        mean_beta = 1.0 / math.sqrt(d_k)

    if unigram_probs is None:
        # Uniform or Zipfian prior if token counts unavailable
        unigram_probs = torch.ones(V, device=device, dtype=torch.float64) / V
    else:
        unigram_probs = unigram_probs.double() / unigram_probs.sum().double()

    # Subsample top tokens if V is very large (e.g. top 4096 tokens account for >80% probability mass)
    if V > 4096:
        top_k = 4096
        top_probs, top_indices = torch.topk(unigram_probs, top_k)
        top_probs = top_probs / top_probs.sum()
        sub_embed = embedding_table[top_indices].double() # (top_k, d_model)
    else:
        top_probs = unigram_probs
        sub_embed = embedding_table.double()

    # Compute keys and values for top tokens
    # k_w = W_k @ embed_w
    k_tokens = torch.matmul(sub_embed, W_k.double().T) # (top_k, d_k)
    v_tokens = torch.matmul(sub_embed, W_v.double().T) # (top_k, d_v)

    # Project keys through FWHT
    phi_tokens = fwht_proj(k_tokens.float()).double()  # (top_k, D) on S^{D-1}

    # Weight by unigram probabilities
    # M_v_phi = sum_w p_w * v_w * phi_w^T in R^{d_v x D}
    weighted_v = top_probs.unsqueeze(-1) * v_tokens    # (top_k, d_v)
    M_v_phi = torch.matmul(weighted_v.T, phi_tokens)   # (d_v, D)

    # Sigma_phi = sum_w p_w * phi_w * phi_w^T in R^{D x D}
    weighted_phi = top_probs.unsqueeze(-1) * phi_tokens # (top_k, D)
    Sigma_phi = torch.matmul(weighted_phi.T, phi_tokens) # (D, D)

    # Invert [ (1 - lambda_bar) * I_D + lambda_bar * beta_bar * Sigma_phi + reg * I_D ]
    coeff_matrix = (1.0 - mean_decay + reg) * torch.eye(D, device=device, dtype=torch.float64) + \
                   (mean_decay * mean_beta) * Sigma_phi

    inv_coeff = torch.linalg.inv(coeff_matrix) # (D, D)
    S_0_star = mean_beta * torch.matmul(M_v_phi, inv_coeff) # (d_v, D)

    return S_0_star.float()


def initialize_analytical_gate_biases(d_k: int, context_length: int = 2048) -> Tuple[float, float]:
    """
    Returns closed-form initial bias values (b_beta, b_decay) for gate linear projections:
    b_beta  = logit( 1 / sqrt(d_k) )
    b_decay = logit( 1 - 1 / context_length )
    """
    target_beta = 1.0 / math.sqrt(d_k)
    target_decay = 1.0 - (1.0 / context_length)

    # Clamping target for logit stability
    target_beta = max(1e-4, min(1.0 - 1e-4, target_beta))
    target_decay = max(1e-4, min(1.0 - 1e-4, target_decay))

    b_beta = math.log(target_beta / (1.0 - target_beta))
    b_decay = math.log(target_decay / (1.0 - target_decay))

    return b_beta, b_decay


def compute_unigram_logit_bias(unigram_probs: torch.Tensor) -> torch.Tensor:
    """
    Computes analytical unigram logit bias to calibrate Step 0 token probabilities:
    b_w = ln(p_w + eps) - mean(ln(p_w + eps))
    """
    p = unigram_probs.float() / unigram_probs.sum().float()
    log_p = torch.log(torch.clamp(p, min=1e-12))
    return log_p - log_p.mean()

