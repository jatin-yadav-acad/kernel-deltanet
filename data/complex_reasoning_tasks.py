r"""
Kernel-DeltaNet Engine
Module: data.complex_reasoning_tasks
======================================================
Complex Algorithmic Reasoning and Dynamic Memory Synthetic Generators:
1. Multi-Hop Associative Recall (graph traversal: A -> B -> C -> D)
2. Dynamic Memory Overwrite & Selective Forgetting (testing update vs stale state)
3. Dyck-k Context-Free Grammar Language (bracket matching and stack depth)
"""

import random
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np


class MultiHopRecallGenerator:
    r"""
    Multi-Hop Associative Recall (MHAR) Generator.
    Encodes directed chain relations: (k_1, k_2), (k_2, k_3), ..., (k_{m-1}, k_m).
    The query provides k_1 and tests if the model can traverse m hops to predict k_m.
    """
    def __init__(self, d_model: int = 64, device: torch.device = torch.device("cpu")):
        self.d_model = d_model
        self.d_sub = d_model // 2
        self.device = device

    def generate_task(
        self,
        num_hops: int = 3,
        num_distractors: int = 16,
        seed: int = 42
    ) -> Dict[str, torch.Tensor]:
        torch.manual_seed(seed)
        random.seed(seed)

        # Generate orthogonal unit vectors for entities in chain
        total_nodes = num_hops + 1
        nodes = torch.randn(total_nodes, self.d_sub, device=self.device)
        nodes = F.normalize(nodes, p=2, dim=-1)

        # Form hops: (nodes[i], nodes[i+1])
        kv_pairs = []
        for i in range(num_hops):
            pair_vec = torch.zeros(self.d_model, device=self.device)
            pair_vec[:self.d_sub] = nodes[i]
            pair_vec[self.d_sub:] = nodes[i+1]
            kv_pairs.append(pair_vec)

        # Add random distractor pairs
        for _ in range(num_distractors):
            d_k = F.normalize(torch.randn(self.d_sub, device=self.device), p=2, dim=-1)
            d_v = F.normalize(torch.randn(self.d_sub, device=self.device), p=2, dim=-1)
            d_vec = torch.zeros(self.d_model, device=self.device)
            d_vec[:self.d_sub] = d_k
            d_vec[self.d_sub:] = d_v
            kv_pairs.append(d_vec)

        # Shuffle pairs so hops are scattered across the context
        random.shuffle(kv_pairs)

        # Final query token: contains start node in key slot
        query_vec = torch.zeros(self.d_model, device=self.device)
        query_vec[:self.d_sub] = nodes[0]
        kv_pairs.append(query_vec)

        target_node = nodes[-1] # final destination

        sequence = torch.stack(kv_pairs, dim=0).unsqueeze(0) # (1, T, d_model)
        return {
            "sequence": sequence,
            "start_node": nodes[0],
            "target_node": target_node,
            "all_nodes": nodes,
            "num_hops": num_hops
        }


class DynamicMemoryOverwriteGenerator:
    r"""
    Dynamic Memory Overwrite Generator.
    Tests selective forgetting: a key k is initially paired with v_old.
    Later in the sequence, k is updated with v_new.
    The model must retrieve v_new and extinguish stale interference from v_old.
    """
    def __init__(self, d_model: int = 64, device: torch.device = torch.device("cpu")):
        self.d_model = d_model
        self.d_sub = d_model // 2
        self.device = device

    def generate_task(
        self,
        seq_len: int = 256,
        overwrite_gap: int = 64,
        seed: int = 42
    ) -> Dict[str, torch.Tensor]:
        torch.manual_seed(seed)

        target_key = F.normalize(torch.randn(self.d_sub, device=self.device), p=2, dim=-1)
        val_old = F.normalize(torch.randn(self.d_sub, device=self.device), p=2, dim=-1)
        val_new = F.normalize(torch.randn(self.d_sub, device=self.device), p=2, dim=-1)

        tokens = torch.randn(seq_len, self.d_model, device=self.device) * 0.05

        # Place initial association at t_old
        t_old = 20
        tokens[t_old, :self.d_sub] = target_key
        tokens[t_old, self.d_sub:] = val_old

        # Place overwritten association at t_new
        t_new = t_old + overwrite_gap
        tokens[t_new, :self.d_sub] = target_key
        tokens[t_new, self.d_sub:] = val_new

        # Place query at end of sequence
        tokens[-1, :self.d_sub] = target_key
        tokens[-1, self.d_sub:] = 0.0

        return {
            "sequence": tokens.unsqueeze(0),
            "val_old": val_old,
            "val_new": val_new,
            "target_key": target_key,
            "t_old": t_old,
            "t_new": t_new
        }


class DyckLanguageGenerator:
    r"""
    Dyck-(k) Context-Free Grammar Language Generator.
    Tests stack maintenance, bracket pairing, and nested state tracking.
    Alphabet consists of k distinct bracket types: ( ), [ ], { }, < >, etc.
    """
    def __init__(self, num_types: int = 3):
        self.num_types = num_types
        # 1-indexed tokens: open = 1..k, close = k+1..2k
        self.open_tokens = list(range(1, num_types + 1))
        self.close_tokens = list(range(num_types + 1, 2 * num_types + 1))
        self.vocab_size = 2 * num_types + 1

    def generate_well_nested_sequence(self, max_depth: int = 8, length: int = 64) -> torch.Tensor:
        seq = []
        stack = []

        while len(seq) < length - len(stack):
            # Decide to open or close
            can_open = len(stack) < max_depth
            can_close = len(stack) > 0

            if can_open and can_close:
                action = random.choice(["open", "close"])
            elif can_open:
                action = "open"
            else:
                action = "close"

            if action == "open":
                b_type = random.randint(0, self.num_types - 1)
                seq.append(self.open_tokens[b_type])
                stack.append(b_type)
            else:
                b_type = stack.pop()
                seq.append(self.close_tokens[b_type])

        # Close all remaining open brackets
        while stack:
            b_type = stack.pop()
            seq.append(self.close_tokens[b_type])

        return torch.tensor(seq, dtype=torch.long)
