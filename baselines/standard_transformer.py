"""
Kernel-DeltaNet Engine
Module: baselines.standard_transformer
========================================================
Reference Softmax Attention Transformer Baseline (LLaMA-style).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.hybrid_kernel_deltanet_block import RMSNorm, SwiGLUFFN


class CausalSoftmaxAttention(nn.Module):
    """
    Standard Full Causal Multi-Head Softmax Attention.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, D = x.shape
        q = self.w_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        new_kv_cache = (k, v)

        # Scaled dot-product attention
        # PyTorch SDPA uses FlashAttention where available on GPU
        is_causal = (kv_cache is None) and (T > 1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.w_out(out), new_kv_cache


class StandardTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: Optional[int] = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSoftmaxAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        norm_x = self.norm1(x)
        attn_out, new_cache = self.attn(norm_x, kv_cache=kv_cache)
        h = x + attn_out
        out = h + self.ffn(self.norm2(h))
        return out, new_cache


class StandardTransformerLM(nn.Module):
    """
    Standard Full Causal Softmax Attention Transformer Language Model.
    Reference baseline for perplexity, throughput, and KV cache memory scaling.
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        d_ff: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            StandardTransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embed(input_ids)
        for layer in self.layers:
            x, _ = layer(x, kv_cache=None)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed.weight.numel()
        return n_params

