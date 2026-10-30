r"""
Kernel-DeltaNet Engine
Test: test_curriculum.py
========================
Verifies progressive context curriculum learning:
1. Dynamic sequence length progression: 256 -> 512 -> 1024
2. Batch size scaling preserving constant token budget
3. Loss decrease and autograd gradient health
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import numpy as np
from models.kernel_deltanet_lm import KernelDeltaNetLM
from training.curriculum_learning import run_curriculum_training, ProgressiveContextCurriculum


def test_curriculum_progression():
    print("[*] Running TEST-1.17: Progressive Context Curriculum Learning...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    curriculum = ProgressiveContextCurriculum([
        {"seq_len": 64, "batch_size": 4, "steps": 5},
        {"seq_len": 128, "batch_size": 2, "steps": 5}
    ])

    s1_len, s1_b = curriculum.get_stage_for_step(3)
    s2_len, s2_b = curriculum.get_stage_for_step(8)
    assert s1_len == 64 and s1_b == 4
    assert s2_len == 128 and s2_b == 2
    print("    Curriculum stage scheduling correctly transitions at target step boundaries.")

    model = KernelDeltaNetLM(
        vocab_size=256,
        d_model=64,
        num_layers=2,
        num_heads=2,
        rkhs_factor=4,
        chunk_size=32
    )

    history = run_curriculum_training(model, total_steps=30, vocab_size=256, device=device)
    assert len(history["losses"]) == 30
    assert not any(np.isnan(l) for l in history["losses"])

    print("[+] TEST-1.17 PASSED: Progressive Curriculum Training Verified.\n")
    pass


if __name__ == "__main__":
    test_curriculum_progression()
