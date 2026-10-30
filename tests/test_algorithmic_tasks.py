r"""
Kernel-DeltaNet Engine
Test: test_algorithmic_tasks.py
===============================
Verifies algorithmic task generators:
1. Multi-Hop Associative Recall (MHAR) graph generation
2. Dynamic Memory Overwrite sequence validity
3. Dyck-k Context-Free Grammar balanced bracket nesting
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from data.complex_reasoning_tasks import (
    MultiHopRecallGenerator,
    DynamicMemoryOverwriteGenerator,
    DyckLanguageGenerator
)
from data.algorithmic_tasks import AlgorithmicTaskGenerator


def test_mhar_generator():
    print("[*] Running TEST-1.11: Multi-Hop Associative Recall Generator...")
    gen = MultiHopRecallGenerator(d_model=64)
    batch = gen.generate_task(num_hops=3, num_distractors=10, seed=42)

    seq = batch["sequence"]
    assert seq.shape == (1, 14, 64), f"Unexpected sequence shape: {seq.shape}"
    assert batch["target_node"].shape == (32,), f"Unexpected target shape: {batch['target_node'].shape}"
    print("    Multi-Hop 3-hop chain + 10 distractors successfully generated.")
    print("[+] TEST-1.11 PASSED.\n")
    pass


def test_dynamic_overwrite_generator():
    print("[*] Running TEST-1.12: Dynamic Memory Overwrite Generator...")
    gen = DynamicMemoryOverwriteGenerator(d_model=64)
    batch = gen.generate_task(seq_len=128, overwrite_gap=32, seed=42)

    assert batch["sequence"].shape == (1, 128, 64)
    assert batch["t_new"] == batch["t_old"] + 32
    print("    Dynamic Overwrite task successfully generated with verified timestamps.")
    print("[+] TEST-1.12 PASSED.\n")
    pass


def test_dyck_language_nesting():
    print("[*] Running TEST-1.13: Dyck-3 Context-Free Grammar Generator...")
    gen = DyckLanguageGenerator(num_types=3)
    seq = gen.generate_well_nested_sequence(max_depth=6, length=64)

    # Verify that the sequence is strictly balanced
    stack = []
    for token in seq.tolist():
        t = token
        if t in gen.open_tokens:
            stack.append(t)
        elif t in gen.close_tokens:
            expected_open = t - gen.num_types
            assert len(stack) > 0, "Closing bracket encountered on empty stack!"
            popped = stack.pop()
            assert popped == expected_open, f"Mismatched bracket! Expected {expected_open}, got {popped}"

    assert len(stack) == 0, f"Unbalanced brackets remain on stack: {stack}"
    print(f"    Dyck-3 sequence of length {len(seq)} is strictly balanced.")
    print("[+] TEST-1.13 PASSED.\n")
    pass


if __name__ == "__main__":
    test_mhar_generator()
    test_dynamic_overwrite_generator()
    test_dyck_language_nesting()
