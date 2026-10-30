"""
Kernel-DeltaNet Engine
Test: test_associativity.py
===========================
Automated verification of the Affine Semigroup Associativity Invariant.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import sys
import os
import torch

from core.associative_algebra import AffineSemigroupVerifier


def test_semigroup_associativity():
    print("[*] Running TEST-1.1: Semigroup Associativity Invariant...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = AffineSemigroupVerifier.verify_associativity(D=64, d_v=32, batch_size=4, device=device)
    
    print(f"    Error A (relative): {result['error_A_relative']:.2e}")
    print(f"    Error B (relative): {result['error_B_relative']:.2e}")
    print(f"    Is Associative:     {result['is_associative']}")
    
    assert result["is_associative"], f"Fatal: Semigroup Associativity Violated! {result}"
    print("[+] TEST-1.1 PASSED: Strict Semigroup Associativity Verified.\n")
    pass


if __name__ == "__main__":
    test_semigroup_associativity()
