"""
Kernel-DeltaNet Engine
Test: test_autograd_gradients.py
================================
Automated verification of Autograd Flow and Spectral Parameter Trainability.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from layers.kernel_deltanet_layer import KernelDeltaNetLayer
from models.kernel_deltanet_lm import KernelDeltaNetLM


def test_autograd_flow():
    print("[*] Running TEST-1.4: Autograd Flow & Spectral Parameter Trainability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    model = KernelDeltaNetLM(vocab_size=1024, d_model=128, num_layers=2, num_heads=4).to(device)
    inputs = torch.randint(0, 1024, (2, 64), device=device)
    targets = torch.roll(inputs, shifts=-1, dims=1)

    logits, loss = model(inputs, targets=targets)
    loss.backward()

    # Check that key parameters received non-zero, finite gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive a gradient!"
        grad_norm = param.grad.norm().item()
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients!"
        assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients!"
        assert grad_norm > 0.0, f"Parameter {name} has zero gradient!"

    print("    All layer parameters (weights, biases, spectral log_gamma, logit_decay) have active, finite gradients.")
    print(f"    Loss: {loss.item():.4f}")
    print("[+] TEST-1.4 PASSED: Autograd Flow and Spectral Trainability Verified.\n")
    pass


if __name__ == "__main__":
    test_autograd_flow()
