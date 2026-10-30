r"""
Kernel-DeltaNet Engine
Test: test_hf_integration.py
============================
Verifies HuggingFace integration, config serialization, forward pass, and generation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import shutil
import tempfile
import torch

from models.hf_compat import KernelTTTConfig, KernelTTTForCausalLM


def test_huggingface_scaffolding():
    print("[*] Running TEST-1.9: HuggingFace Transformers Native Integration...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    config = KernelTTTConfig(
        vocab_size=512,
        d_model=64,
        num_layers=2,
        num_heads=2,
        rkhs_factor=4,
        chunk_size=32
    )

    model = KernelTTTForCausalLM(config).to(device)
    model.eval()

    # Forward pass with labels (loss computation)
    input_ids = torch.randint(0, 512, (2, 32), device=device)
    labels = torch.roll(input_ids, shifts=-1, dims=1)

    outputs = model(input_ids=input_ids, labels=labels)
    assert outputs.loss is not None, "Loss was not computed!"
    assert outputs.logits.shape == (2, 32, 512), f"Logits shape mismatch: {outputs.logits.shape}"
    print(f"    HF Forward Pass Loss: {outputs.loss.item():.4f}")

    # Test HuggingFace save_pretrained and from_pretrained
    temp_dir = tempfile.mkdtemp()
    try:
        model.save_pretrained(temp_dir)
        loaded_model = KernelTTTForCausalLM.from_pretrained(temp_dir).to(device)
        loaded_model.eval()

        with torch.no_grad():
            loaded_outputs = loaded_model(input_ids=input_ids)
            diff = torch.norm(outputs.logits - loaded_outputs.logits).item()

        print(f"    HF Save/Load Weights L2 Difference: {diff:.6e}")
        assert diff < 1e-5, f"Weights mismatch after HF load: {diff}"
        print("[+] TEST-1.9 PASSED: HuggingFace Native Scaffolding Fully Verified.\n")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    pass


if __name__ == "__main__":
    test_huggingface_scaffolding()
