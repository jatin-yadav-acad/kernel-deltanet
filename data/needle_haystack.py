"""
Kernel-DeltaNet Engine
Module: data.needle_haystack
==============================================
Needle-In-A-Haystack (NIAH) Task Generator for Long-Context Evaluation.
"""

from typing import Tuple, Dict, List, Optional
import torch


class NeedleInAHaystackGenerator:
    """
    Generates synthetic Needle-In-A-Haystack contexts of arbitrary length T (up to 128k tokens).
    Places a distinct needle key-value association at a specific depth fraction (0.0 to 1.0),
    and queries for the needle at the end of the context.
    """
    def __init__(self, vocab_size: int = 8192, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.device = device
        
        # Special tokens
        self.needle_key = 42
        self.needle_value = 1337

    def generate_needle_task(
        self,
        context_length: int = 4096,
        depth_percent: float = 0.5,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context_length: Total sequence length T
            depth_percent: Float in [0.0, 1.0] specifying location of needle
        Returns:
            input_ids: (1, T) tensor
            target_id: Scalar target value
        """
        if seed is not None:
            torch.manual_seed(seed)

        # 1. Generate background haystack tokens (avoiding needle tokens)
        haystack = torch.randint(100, self.vocab_size, (1, context_length), device=self.device)

        # 2. Insert needle [KEY, VALUE] at target depth
        insert_idx = int(context_length * depth_percent)
        insert_idx = max(0, min(insert_idx, context_length - 4))

        haystack[0, insert_idx] = self.needle_key
        haystack[0, insert_idx + 1] = self.needle_value

        # 3. Query at final token: [..., NEEDLE_KEY] -> Target is NEEDLE_VALUE
        haystack[0, -1] = self.needle_key
        target = torch.tensor([self.needle_value], device=self.device)

        return haystack, target
