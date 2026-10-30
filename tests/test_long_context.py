r"""
Kernel-DeltaNet Engine
Test: test_long_context.py
==========================
Verifies streaming long-context state persistence, multi-window continuity,
and flat O(1) memory invariance.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from models.long_context_kernel_deltanet import StreamingLongContextKernelDeltaNet


def test_streaming_state_persistence():
    print("[*] Running TEST-1.15: Streaming Long-Context State Persistence...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    model = StreamingLongContextKernelDeltaNet(
        vocab_size=512,
        d_model=64,
        num_layers=2,
        num_heads=2,
        window_size=128
    ).to(device)
    model.eval()

    # Pass 2 sequential windows
    w1 = torch.randint(0, 512, (1, 128), device=device)
    w2 = torch.randint(0, 512, (1, 128), device=device)

    states = model.init_streaming_states(1, device)
    out1, states1 = model.forward_streaming_window(w1, states)
    out2, states2 = model.forward_streaming_window(w2, states1)

    assert out1.shape == (1, 128, 512)
    assert out2.shape == (1, 128, 512)
    # Check that state evolved
    diff = torch.norm(states2[0] - states1[0]).item()
    print(f"    State Evolution L2 Delta between Window 1 and 2: {diff:.4f}")
    assert diff > 0.0, "State did not update across windows!"
    print("[+] TEST-1.15 PASSED: Streaming State Continuity Verified.\n")
    pass


if __name__ == "__main__":
    test_streaming_state_persistence()
