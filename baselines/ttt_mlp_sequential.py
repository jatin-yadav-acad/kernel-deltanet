"""
Kernel-DeltaNet Engine
Module: baselines.ttt_mlp_sequential
======================================================
Classical TTT-MLP Baseline (Sun et al., 2024 / Titans, Google 2024).

The Parallelization Paradox:
Because non-linear activations (GELU) reside inside the inner model weights:
  f(x) = W_2 * GELU(W_1 * x)
The gradient update:
  W_1 <- W_1 - eta * grad_{W_1} ||f(k_t) - v_t||^2
is non-linear in W_1. Parallel prefix scan is mathematically impossible.
It forces sequential step-by-step inner loops that throttle GPU Tensor Cores.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.hybrid_kernel_deltanet_block import RMSNorm, SwiGLUFFN


class TTTMLPLayer(nn.Module):
    """
    Sequential TTT-MLP with 2-layer inner model and test-time gradient descent.
    """
    def __init__(
        self,
        d_model: int,
        inner_hidden: int = 128,
        lr: float = 0.05
    ):
        super().__init__()
        self.d_model = d_model
        self.inner_hidden = inner_hidden
        self.lr = lr

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

        # Base parameters from which test-time adaptation starts
        self.base_W1 = nn.Parameter(torch.randn(d_model, inner_hidden) * (1.0 / math.sqrt(d_model)))
        self.base_W2 = nn.Parameter(torch.randn(inner_hidden, d_model) * (1.0 / math.sqrt(inner_hidden)))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Sequential execution through time (The Parallelization Paradox).
        """
        B, T, D = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        outputs = []
        batch_states = []

        for b in range(B):
            if state is not None:
                W1, W2 = state[0][b].clone(), state[1][b].clone()
            else:
                W1 = self.base_W1.clone()
                W2 = self.base_W2.clone()

            b_outputs = []
            for t in range(T):
                k_t = k[b, t:t+1]
                v_t = v[b, t:t+1]
                q_t = q[b, t:t+1]

                # 1. Query prediction using current inner MLP state BEFORE update (causal)
                h_q = F.gelu(torch.matmul(q_t, W1))
                y_t = torch.matmul(h_q, W2)
                b_outputs.append(y_t)

                # 2. Test-time gradient step on inner reconstruction objective:
                # Loss = 0.5 * ||W_2 * GELU(k_t * W_1) - v_t||^2
                h_k = F.gelu(torch.matmul(k_t, W1))
                pred_v = torch.matmul(h_k, W2)
                err = pred_v - v_t

                # Gradients:
                grad_W2 = torch.matmul(h_k.t(), err)
                # GELU derivative approx: sigmoid(1.702 * x)
                act_deriv = torch.sigmoid(1.702 * torch.matmul(k_t, W1))
                grad_W1 = torch.matmul(k_t.t(), torch.matmul(err, W2.t()) * act_deriv)

                # Inner update
                W1 = W1 - self.lr * grad_W1
                W2 = W2 - self.lr * grad_W2

            outputs.append(torch.cat(b_outputs, dim=0))
            batch_states.append((W1, W2))

        out_stack = torch.stack(outputs, dim=0) # (B, T, D)
        final_state = (
            torch.stack([s[0] for s in batch_states], dim=0),
            torch.stack([s[1] for s in batch_states], dim=0)
        )
        return self.w_out(out_stack), final_state


class TTTMLPBlock(nn.Module):
    def __init__(self, d_model: int, inner_hidden: int = 128, d_ff: Optional[int] = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = TTTMLPLayer(d_model, inner_hidden=inner_hidden)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        norm_x = self.norm1(x)
        attn_out, new_state = self.attn(norm_x, state=state)
        h = x + attn_out
        out = h + self.ffn(self.norm2(h))
        return out, new_state


class TTTMLPLM(nn.Module):
    """
    TTT-MLP Language Model Baseline.
    Suffers severe GPU throttling due to sequential autograd unrolling.
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 256,
        num_layers: int = 6,
        inner_hidden: int = 128,
        d_ff: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TTTMLPBlock(d_model, inner_hidden=inner_hidden, d_ff=d_ff)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x, _ = layer(x, state=None)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss
