"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: data.algorithmic_tasks
================================================
Synthetic Algorithmic & In-Context Learning Tasks.
"""

from typing import Tuple, Dict, List, Optional
import torch


class AlgorithmicTaskGenerator:
    """
    Generates synthetic algorithmic and in-context learning tasks.
    Supports any vocabulary size >= 16.
    """
    def __init__(self, vocab_size: int = 128, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.device = device
        self.key_min = 1
        self.key_max = max(2, vocab_size // 4)
        self.val_min = self.key_max
        self.val_max = max(self.val_min + 1, vocab_size // 2)
        self.filler_min = self.val_max
        self.filler_max = vocab_size

    def generate_induction_batch(
        self,
        batch_size: int = 8,
        seq_len: int = 512,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Induction Head Task: [..., A, B, ..., A] -> Next token is B.
        Tests whether test-time memory state successfully captures associative bigrams.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Fill background tokens
        seq = torch.randint(self.filler_min, self.filler_max, (batch_size, seq_len), device=self.device)

        for b in range(batch_size):
            A = int(torch.randint(self.key_min, self.key_max, (1,)).item())
            B = int(torch.randint(self.val_min, self.val_max, (1,)).item())

            # Place A, B early in the sequence
            pos1 = int(torch.randint(2, max(3, seq_len // 2), (1,)).item())
            seq[b, pos1] = A
            seq[b, pos1 + 1] = B

            # Query A at T-2
            seq[b, -2] = A
            seq[b, -1] = B # Next token prediction target

        inputs = seq[:, :-1].contiguous()
        targets = seq[:, 1:].contiguous()
        return inputs, targets

    def generate_associative_copy(
        self,
        batch_size: int = 8,
        seq_len: int = 512,
        num_associations: int = 8,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Associative Copy Task: stores multi-span associations and queries one at the end.
        """
        if seed is not None:
            torch.manual_seed(seed)

        seq = torch.randint(self.filler_min, self.filler_max, (batch_size, seq_len), device=self.device)

        for b in range(batch_size):
            keys = torch.randint(self.key_min, self.key_max, (num_associations,), device=self.device)
            vals = torch.randint(self.val_min, self.val_max, (num_associations,), device=self.device)

            positions = torch.randperm(int(seq_len * 0.75))[:num_associations * 2].sort()[0]
            for i in range(num_associations):
                seq[b, positions[2 * i]] = keys[i]
                seq[b, positions[2 * i + 1]] = vals[i]

            query_idx = int(torch.randint(0, num_associations, (1,)).item())
            seq[b, -2] = keys[query_idx]
            seq[b, -1] = vals[query_idx]

        inputs = seq[:, :-1].contiguous()
        targets = seq[:, 1:].contiguous()
        return inputs, targets
