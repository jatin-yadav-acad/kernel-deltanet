"""
Kernel-DeltaNet Engine
Module: baselines.ttt_linear_deltanet
======================================================
TTT-Linear / DeltaNet Baseline (Linear Associative State W in R^{d x d}).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.hybrid_kernel_deltanet_block import RMSNorm, SwiGLUFFN


class TTTLinearLayer(nn.Module):
    """
    Classical TTT-Linear / DeltaNet.
    Affine update: W_t = lambda * W_{t-1} + eta * v_t * k_t^T.
    Parallelizable via chunked scan, but capacity is strictly bounded by rank d.
    """
    def __init__(
        self,
        d_model: int,
        chunk_size: int = 64,
        initial_decay: float = 0.99,
        initial_eta: float = 0.5
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

        init_logit = math.log(initial_decay / (1.0 - initial_decay + 1e-7))
        self.logit_decay = nn.Parameter(torch.tensor([float(init_logit)]))
        init_raw_eta = math.log(math.exp(initial_eta) - 1.0 + 1e-7)
        self.raw_eta = nn.Parameter(torch.tensor([float(init_raw_eta)]))

    def get_decay(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_decay)

    def get_eta(self) -> torch.Tensor:
        return F.softplus(self.raw_eta)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape
        q = self.w_q(x)
        k = F.normalize(self.w_k(x), dim=-1)
        v = self.w_v(x)

        decay = self.get_decay()
        eta = self.get_eta()

        if state is None and T > 1:
            # Parallel chunked scan across sequence
            C = self.chunk_size
            num_chunks = int(math.ceil(T / C))
            pad_len = (num_chunks * C) - T
            if pad_len > 0:
                q = torch.cat([q, torch.zeros(B, pad_len, D, device=x.device)], dim=1)
                k = torch.cat([k, torch.zeros(B, pad_len, D, device=x.device)], dim=1)
                v = torch.cat([v, torch.zeros(B, pad_len, D, device=x.device)], dim=1)

            q_c = q.view(B, num_chunks, C, D)
            k_c = k.view(B, num_chunks, C, D)
            v_c = v.view(B, num_chunks, C, D)

            # Intra-chunk strictly causal decay
            i_idx = torch.arange(C, device=x.device).view(1, 1, C, 1)
            j_idx = torch.arange(C, device=x.device).view(1, 1, 1, C)
            power_diff = (i_idx - 1 - j_idx).to(x.dtype)
            causal_mask = (i_idx > j_idx).to(x.dtype)
            intra_weights = (eta * (decay ** power_diff) * causal_mask).to(x.dtype)

            intra_attn = torch.matmul(q_c, k_c.transpose(-1, -2))
            out_intra = torch.matmul(intra_attn * intra_weights, v_c)

            # Inter-chunk scan
            c_steps = torch.arange(C, device=x.device).view(1, 1, C, 1)
            chunk_token_decay = (decay ** (C - 1 - c_steps)).to(x.dtype)
            v_c_decayed = v_c * chunk_token_decay
            chunk_states = torch.matmul(v_c_decayed.transpose(-1, -2), k_c) # (B, num_chunks, D, D)

            decay_chunk = (decay ** C).to(x.dtype)
            inter_states = torch.zeros(B, num_chunks, D, D, device=x.device, dtype=x.dtype)
            running_W = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)

            for c in range(num_chunks):
                inter_states[:, c] = running_W
                running_W = running_W * decay_chunk + eta.to(x.dtype) * chunk_states[:, c]

            token_decay = (decay ** c_steps).to(x.dtype)
            out_inter = torch.matmul(q_c, inter_states.transpose(-1, -2)) * token_decay
            out_flat = (out_intra + out_inter).view(B, num_chunks * C, D)[:, :T, :]
            return self.w_out(out_flat), None
        else:
            # Recurrent step
            current_state = state if state is not None else torch.zeros(B, D, D, device=x.device)
            outputs = []
            for t in range(T):
                q_t = q[:, t:t+1, :]
                k_t = k[:, t:t+1, :]
                v_t = v[:, t:t+1, :]
                
                # Query readout before update
                y_t = torch.matmul(current_state, q_t.transpose(-1, -2)).transpose(-1, -2)
                outputs.append(y_t)
                
                delta_W = torch.matmul(v_t.transpose(-1, -2), k_t)
                current_state = decay * current_state + eta * delta_W

            out_stack = torch.cat(outputs, dim=1)
            return self.w_out(out_stack), current_state


class TTTLinearBlock(nn.Module):
    def __init__(self, d_model: int, chunk_size: int = 64, d_ff: Optional[int] = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = TTTLinearLayer(d_model, chunk_size=chunk_size)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        norm_x = self.norm1(x)
        attn_out, new_state = self.attn(norm_x, state=state)
        h = x + attn_out
        out = h + self.ffn(self.norm2(h))
        return out, new_state


class TTTLinearLM(nn.Module):
    """
    TTT-Linear / DeltaNet Causal Language Model.
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 256,
        num_layers: int = 6,
        chunk_size: int = 64,
        d_ff: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TTTLinearBlock(d_model, chunk_size=chunk_size, d_ff=d_ff)
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
