"""
Kernel-DeltaNet Engine
Module: data.mqar_generator
=============================================
Multi-Query Associative Recall (MQAR) Task Generator.

Mathematical Purpose:
---------------------
Tests associative memory capacity under linear rank saturation:
- Key dimension: d_key = d_model // 2
- When K > d_key, the number of stored associations strictly exceeds the rank
  of any linear associative matrix W in R^{d x d}.
- Tests whether Kernel-DeltaNet's high-dimensional RKHS expansion phi(k) in R^D (D = M * d)
  prevents linear rank collapse.
"""

from typing import Tuple, Dict, List, Optional
import torch
import numpy as np


class MQARGenerator:
    """
    Generates multi-query associative recall sequences with embedded key-value pairs
    and intermediate distractor tokens.
    """
    def __init__(self, d_model: int = 64, device: str = "cpu"):
        self.d_model = d_model
        self.d_key = d_model // 2
        self.d_val = d_model // 2
        self.device = device

    def generate_batch(
        self,
        batch_size: int = 1,
        seq_len: int = 1024,
        num_kv_pairs: int = 64,
        target_turn: str = "earliest",
        noise_scale: float = 0.05,
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generates a batch of continuous MQAR sequences.
        Args:
            batch_size: Number of sequences in batch
            seq_len: Total sequence length T
            num_kv_pairs: Number of distinct associations K to store
            target_turn: 'earliest' (hardest, longest retention distance), 'middle', or 'latest'
            noise_scale: Standard deviation of background distractor tokens
        Returns:
            dict containing:
              - 'sequences': (B, T, d_model)
              - 'target_keys': (B, d_key)
              - 'target_values': (B, d_val)
              - 'all_keys': (B, K, d_key)
              - 'all_values': (B, K, d_val)
        """
        if seed is not None:
            torch.manual_seed(seed)

        # 1. Initialize sequence with Gaussian distractor tokens
        seq = torch.randn(batch_size, seq_len, self.d_model, device=self.device) * noise_scale

        all_keys_list = []
        all_vals_list = []
        target_vals_list = []
        target_keys_list = []

        max_insert_pos = int(seq_len * 0.85)

        for b in range(batch_size):
            # Generate K distinct normalized keys and values
            keys = torch.randn(num_kv_pairs, self.d_key, device=self.device)
            keys = torch.nn.functional.normalize(keys, p=2, dim=-1)

            values = torch.randn(num_kv_pairs, self.d_val, device=self.device)
            values = torch.nn.functional.normalize(values, p=2, dim=-1)

            # Insert positions
            insert_positions = torch.randperm(max_insert_pos)[:num_kv_pairs].sort()[0]

            for i, pos in enumerate(insert_positions):
                # First half is Key, Second half is Value
                seq[b, pos, :self.d_key] = keys[i]
                seq[b, pos, self.d_key:] = values[i]

            # Select target key to query at final token
            if target_turn == "earliest":
                target_idx = 0 # Longest retention distance
            elif target_turn == "middle":
                target_idx = num_kv_pairs // 2
            elif target_turn == "latest":
                target_idx = num_kv_pairs - 1
            else:
                target_idx = int(torch.randint(0, num_kv_pairs, (1,)).item())

            # Query token at T-1
            seq[b, -1, :self.d_key] = keys[target_idx]
            seq[b, -1, self.d_key:] = 0.0

            all_keys_list.append(keys)
            all_vals_list.append(values)
            target_keys_list.append(keys[target_idx])
            target_vals_list.append(values[target_idx])

        return {
            "sequences": seq,
            "target_keys": torch.stack(target_keys_list, dim=0),
            "target_values": torch.stack(target_vals_list, dim=0),
            "all_keys": torch.stack(all_keys_list, dim=0),
            "all_values": torch.stack(all_vals_list, dim=0)
        }
