"""
Kernel-DeltaNet Engine
Module: models.kernel_deltanet_lm
=================================
Scalable Causal Language Model Powered by Multi-Head Kernel-DeltaNet.
"""

import math
from typing import Optional, List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.hybrid_kernel_deltanet_block import HybridKernelDeltaNetBlock, RMSNorm


class KernelDeltaNetLM(nn.Module):
    """
    Complete Causal Autoregressive Language Model Architecture.
    - Training: Evaluated via O(N) chunked parallel scan across all layers.
    - Generation: Evaluated token-by-token with strictly O(1) constant memory (Zero KV Cache).
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        rkhs_factor: int = 4,
        chunk_size: int = 64,
        d_ff: Optional[int] = None,
        drop_rate: float = 0.0,
        tie_weights: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.rkhs_factor = rkhs_factor
        self.chunk_size = chunk_size
        self.tie_weights = tie_weights

        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        # Stack of Transformer blocks with Kernel-DeltaNet attention
        self.layers = nn.ModuleList([
            HybridKernelDeltaNetBlock(
                d_model=d_model,
                num_heads=num_heads,
                rkhs_factor=rkhs_factor,
                chunk_size=chunk_size,
                d_ff=d_ff,
                drop_rate=drop_rate
            )
            for _ in range(num_layers)
        ])

        # Final RMS Normalization
        self.final_norm = RMSNorm(d_model)

        # Unembedding prediction head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed.weight

        # Parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Counts total trainable parameters."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding and not self.tie_weights:
            n_params -= self.embed.weight.numel()
        return n_params

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parallel Training Forward Pass.
        Args:
            input_ids: (B, T) token indices
            targets: (B, T) target next-token indices (optional)
        Returns:
            logits: (B, T, vocab_size)
            loss: Cross-entropy scalar (if targets provided)
        """
        B, T = input_ids.shape
        x = self.dropout(self.embed(input_ids))

        # Pass through all layers using parallel chunked scan
        for layer in self.layers:
            x, _ = layer(x, state=None)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def prefill_and_init_states(
        self,
        prompt_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Processes prompt sequence and returns the initial states for all layers.
        """
        B, T = prompt_ids.shape

        # Step through prompt to construct initial recurrent states
        x = self.embed(prompt_ids)
        states: List[Optional[torch.Tensor]] = [None] * self.num_layers

        for t in range(T):
            x_t = x[:, t:t+1, :]
            for l, layer in enumerate(self.layers):
                x_t, states[l] = layer(x_t, state=states[l])

        x_last = self.final_norm(x_t)
        next_token_logits = self.lm_head(x_last)
        return next_token_logits, states

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95
    ) -> torch.Tensor:
        """
        O(1) Constant-Memory Autoregressive Generation.
        No expanding KV cache is ever allocated. Memory footprint is strictly flat.
        """
        B, T = prompt_ids.shape

        # 1. Prefill prompt and initialize constant states
        logits, states = self.prefill_and_init_states(prompt_ids)

        generated = []
        curr_token = prompt_ids[:, -1:]

        for step in range(max_new_tokens):
            scaled_logits = logits[:, -1, :]

            if temperature > 0.0:
                scaled_logits = scaled_logits / temperature

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = float('-inf')

                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    scaled_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(scaled_logits, dim=-1, keepdim=True)

            generated.append(next_token)
            curr_token = next_token

            # Single recurrent step through all layers: O(1) time and memory!
            x_step = self.embed(curr_token)
            for l, layer in enumerate(self.layers):
                x_step, states[l] = layer(x_step, state=states[l])

            x_norm = self.final_norm(x_step)
            logits = self.lm_head(x_norm)

        gen_tokens = torch.cat(generated, dim=1)
        return torch.cat([prompt_ids, gen_tokens], dim=1)


# Backwards compatibility alias
KernelDeltaNetLM = KernelDeltaNetLM


# Backwards compatibility alias
KernelTTTLM = KernelDeltaNetLM
