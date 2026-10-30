r"""
Kernel-DeltaNet Engine
Module: models.recurrent_memory_transformer
=============================================================
Recurrent Memory Transformer (RMT) Baseline Implementation (Bulatov et al., 2022).
Passes m special memory tokens across sequential segments of length L.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.hybrid_kernel_deltanet_block import RMSNorm
from baselines.standard_transformer import CausalSoftmaxAttention


class RMTBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSoftmaxAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.norm1(x))[0]
        out = h + self.ffn(self.norm2(h))
        return out


class RecurrentMemoryTransformerLM(nn.Module):
    r"""
    RMT Baseline with persistent memory tokens prepended to each chunk.
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        num_mem_tokens: int = 8,
        chunk_size: int = 128
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_mem_tokens = num_mem_tokens
        self.chunk_size = chunk_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.initial_memory = nn.Parameter(torch.randn(1, num_mem_tokens, d_model) * 0.02)

        self.layers = nn.ModuleList([
            RMTBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_segment(
        self,
        segment_tokens: torch.Tensor,
        memory_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Concatenates memory tokens with segment tokens:
          x = [memory_tokens, segment_tokens]
        Passes through transformer blocks, and extracts updated memory tokens from output.
        """
        B, L = segment_tokens.shape
        x_tok = self.embed(segment_tokens) # (B, L, D)
        # Prepend memory: (B, M + L, D)
        x_full = torch.cat([memory_tokens, x_tok], dim=1)

        for layer in self.layers:
            x_full = layer(x_full)

        x_full = self.final_norm(x_full)
        # Extract new memory tokens from prefix
        new_memory = x_full[:, :self.num_mem_tokens, :]
        token_logits = self.lm_head(x_full[:, self.num_mem_tokens:, :])

        return token_logits, new_memory

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        device = input_ids.device
        mem = self.initial_memory.expand(B, -1, -1)

        logits_list = []
        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            seg_ids = input_ids[:, start:end]
            logits_seg, mem = self.forward_segment(seg_ids, mem)
            logits_list.append(logits_seg)

        logits = torch.cat(logits_list, dim=1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss
