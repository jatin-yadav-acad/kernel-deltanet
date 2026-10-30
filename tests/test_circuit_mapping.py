import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# tests/test_circuit_mapping.py
"""
Test Suite: Mechanistic Circuit Mapping & IOI Duplicate Suppression
====================================================================
Verifies:
1. Spectral power analysis tracks frequency distributions across RKHS bands.
2. Subtractive lateral inhibition actively suppresses duplicate tokens in IOI tasks.
"""

import pytest
import torch

from analysis.circuit_mapping import MechanisticCircuitProbe


def test_frequency_spectrum_analysis():
    """Verify frequency spectrum decomposition across low/mid/high bands."""
    torch.manual_seed(42)
    probe = MechanisticCircuitProbe(d_k=64, D=512, seed=42)

    keys = torch.randn(2, 64, 64)
    spectrum = probe.analyze_frequency_spectrum(keys)

    assert "low_band_pct" in spectrum
    assert "mid_band_pct" in spectrum
    assert "high_band_pct" in spectrum
    assert "spectral_entropy" in spectrum

    total_pct = spectrum["low_band_pct"] + spectrum["mid_band_pct"] + spectrum["high_band_pct"]
    assert abs(total_pct - 100.0) < 1e-3


def test_ioi_duplicate_suppression():
    """Verify subtractive lateral inhibition resolves IOI duplicate suppression."""
    torch.manual_seed(42)
    probe = MechanisticCircuitProbe(d_k=64, D=512, seed=42)

    # Generate orthogonal or distinct embeddings for two entities Mary and John
    name_Mary = torch.randn(64)
    name_Mary = name_Mary / torch.norm(name_Mary)

    name_John = torch.randn(64)
    name_John = name_John / torch.norm(name_John)

    results = probe.evaluate_ioi_suppression(name_Mary, name_John, beta=0.9, decay=0.99)

    print("\nIOI Circuit Mapping Results:")
    print(f"Target (Mary) Retrieval Similarity:      {results['sim_target_Mary']:.4f}")
    print(f"Distractor (John) Retrieval Similarity:  {results['sim_distractor_John']:.4f}")
    print(f"Logit Difference (Mary - John):          {results['logit_diff']:.4f}")

    assert results["logit_diff"] > 0.0, "Expected target Mary to have higher readout than distractor John"
    assert results["sim_target_Mary"] > results["sim_distractor_John"]
    print("[PASS] Mechanistic IOI Suppression Confirmed Successfully!")
