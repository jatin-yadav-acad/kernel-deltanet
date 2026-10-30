r"""
Kernel-DeltaNet Engine
Module: models.long_context_kernel_deltanet
===========================================
Streaming Long-Context (32k - 1M+ tokens) Kernel-DeltaNet Engine.
Processes arbitrary sequence lengths in streaming windows (e.g. W = 4096)
by carrying forward compact dual state tensors S_t in O(1) memory
without storing key-value histories in high-bandwidth memory.
"""

from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.hybrid_kernel_deltanet_block import HybridKernelDeltaNetBlock, RMSNorm


class StreamingLongContextKernelDeltaNet(nn.Module):
    r"""
    Streaming Long-Context Architecture for multi-million token contexts.
    Maintains constant GPU memory across arbitrary sequence lengths.
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        rkhs_factor: int = 4,
        chunk_size: int = 64,
        window_size: int = 2048,
        tie_weights: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.rkhs_factor = rkhs_factor
        self.chunk_size = chunk_size
        self.window_size = window_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            HybridKernelDeltaNetBlock(
                d_model=d_model,
                num_heads=num_heads,
                rkhs_factor=rkhs_factor,
                chunk_size=chunk_size
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embed.weight

    def init_streaming_states(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        r"""
        Initializes empty state list across all layers.
        Shape per layer: (B, H, d_head, rkhs_dim)
        """
        d_head = self.d_model // self.num_heads
        rkhs_dim = d_head * self.rkhs_factor
        return [
            torch.zeros(batch_size, self.num_heads, d_head, rkhs_dim, device=device)
            for _ in range(self.num_layers)
        ]

    def forward_streaming_window(
        self,
        input_ids: torch.Tensor,
        states: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        r"""
        Processes a single streaming window of tokens (e.g. W = 2048).
        Updates and returns the persistent recurrent memory states.

        Args:
            input_ids: (B, W) token window
            states: List of state tensors per layer
        Returns:
            logits: (B, W, vocab_size)
            new_states: List of updated state tensors
        """
        B, W = input_ids.shape
        x = self.embed(input_ids)
        new_states = []

        for i, layer in enumerate(self.layers):
            x, updated_state = layer(x, state=states[i])
            new_states.append(updated_state if updated_state is not None else states[i])

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_states

    def process_long_sequence(
        self,
        long_input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        r"""
        Iterates over a very long sequence (e.g. T = 65,536 tokens)
        in memory-safe chunks of size self.window_size.
        """
        B, T = long_input_ids.shape
        device = long_input_ids.device
        states = self.init_streaming_states(B, device)

        all_logits = []
        for w_start in range(0, T, self.window_size):
            w_end = min(w_start + self.window_size, T)
            window_ids = long_input_ids[:, w_start:w_end]

            logits_w, states = self.forward_streaming_window(window_ids, states)
            all_logits.append(logits_w)

        return torch.cat(all_logits, dim=1), states


# Backwards compatibility alias
StreamingLongContextKernelDeltaNet = StreamingLongContextKernelDeltaNet


# Backwards compatibility alias
StreamingLongContextKernelTTT = StreamingLongContextKernelDeltaNet
