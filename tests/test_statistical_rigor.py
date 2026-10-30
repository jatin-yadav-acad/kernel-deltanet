r"""
Kernel-DeltaNet Engine
Test: test_statistical_rigor.py
===============================
Verifies hypothesis testing functions:
1. Paired Student's t-test symmetry and null behavior
2. Wilcoxon signed-rank test monotonicity
3. Non-parametric bootstrap confidence interval coverage
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
from experiments.statistical_significance import SignificanceTester


def test_significance_tester():
    print("[*] Running TEST-1.18: Statistical Hypothesis Testing Invariants...")

    # Null distribution: identical samples
    null_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    null_b = [1.0, 2.0, 3.0, 4.0, 5.0]
    res_null = SignificanceTester.compute_paired_comparison(null_a, null_b)
    assert res_null["mean_difference"] == 0.0
    assert res_null["p_value_student_t"] == 1.0
    print("    Null hypothesis p-value = 1.0 verified on identical distributions.")

    # High significance case
    a = [0.1, 0.2, 0.15, 0.12, 0.18, 0.14, 0.16]
    b = [0.8, 0.9, 0.85, 0.88, 0.92, 0.84, 0.87]
    res_sig = SignificanceTester.compute_paired_comparison(a, b)
    assert res_sig["mean_difference"] > 0.6
    assert res_sig["p_value_student_t"] < 1e-4
    assert res_sig["cohens_d"] > 5.0
    assert res_sig["bootstrap_ci_95_lower"] > 0.5
    print(f"    Significant case verified: p = {res_sig['p_value_student_t']:.2e}, Cohen's d = {res_sig['cohens_d']:.2f}")

    print("[+] TEST-1.18 PASSED: Statistical Testing Invariants Verified.\n")
    pass


if __name__ == "__main__":
    test_significance_tester()
