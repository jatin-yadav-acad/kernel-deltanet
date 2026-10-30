"""
Kernel-DeltaNet Engine
Module: data.token_corpus_streaming
=====================================================
Streaming Synthetic Pre-training Corpus with Zipfian Power-Law Token Distributions.
"""

from typing import Iterator, Tuple, Optional
import math
import torch


class StreamingTokenCorpus:
    """
    Simulates high-throughput streaming pre-training corpora (e.g., FineWeb-Edu / SlimPajama)
    following Zipf's law: P(rank r) proportional to 1 / r^alpha.
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        alpha: float = 1.05,
        device: str = "cpu"
    ):
        self.vocab_size = vocab_size
        self.device = device
        
        # Precompute Zipfian probability distribution
        ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32)
        probs = 1.0 / (ranks ** alpha)
        self.zipf_probs = probs / probs.sum()

    def sample_batch(
        self,
        batch_size: int = 16,
        seq_len: int = 512,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a batch of causal language modeling tokens:
        input_ids = tokens[:, :-1], targets = tokens[:, 1:].
        """
        if seed is not None:
            torch.manual_seed(seed)

        token_indices = torch.multinomial(
            self.zipf_probs,
            num_samples=batch_size * seq_len,
            replacement=True
        ).view(batch_size, seq_len).to(self.device)

        inputs = token_indices[:, :-1].contiguous()
        targets = token_indices[:, 1:].contiguous()
        return inputs, targets
