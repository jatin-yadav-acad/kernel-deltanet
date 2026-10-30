"""
MLSys 2026: Kernel-DeltaNet Sequence Engine
Master Test Runner: test_all_mlsys.py
======================================
Executes all automated invariant tests and validates the entire Kernel-DeltaNet suite.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import sys
import os
import json
import time

from tests.test_associativity import test_semigroup_associativity
from tests.test_parallel_vs_recurrent import test_chunked_scan_vs_recurrence, test_layer_dual_mode_equivalence
from tests.test_crosstalk_decoupling import test_rkhs_crosstalk_decoupling
from tests.test_autograd_gradients import test_autograd_flow
from tests.test_advanced_kernels import test_cauchy_kernel_properties, test_fastfood_transform_properties
from tests.test_hf_integration import test_huggingface_scaffolding
from tests.test_fused_chunk_kernel import test_fused_kernel_equivalence
from tests.test_algorithmic_tasks import test_mhar_generator, test_dynamic_overwrite_generator, test_dyck_language_nesting
from tests.test_long_context import test_streaming_state_persistence
from tests.test_curriculum import test_curriculum_progression
from tests.test_statistical_rigor import test_significance_tester
from tests.test_inter_woodbury import test_inter_woodbury_equivalence


def run_master_test_suite(save_log: bool = True):
    print("=" * 90)
    print("KERNEL-DELTANET MLSYS 2026 — MASTER ARTIFACT & INVARIANT TEST SUITE")
    print("=" * 90)

    t0 = time.time()
    results = {}

    try:
        test_semigroup_associativity()
        results["TEST-1.1_Semigroup_Associativity"] = True
        test_chunked_scan_vs_recurrence()
        results["TEST-1.2_Scan_Recurrent_Equivalence"] = True
        test_layer_dual_mode_equivalence()
        results["TEST-1.2b_Layer_Dual_Mode"] = True
        test_rkhs_crosstalk_decoupling()
        results["TEST-1.3_Crosstalk_Decoupling"] = True
        test_autograd_flow()
        results["TEST-1.4_Autograd_Trainability"] = True
        test_cauchy_kernel_properties()
        results["TEST-1.5_Cauchy_Kernel_Properties"] = True
        test_fastfood_transform_properties()
        results["TEST-1.6_Fastfood_Transform"] = True
        test_huggingface_scaffolding()
        results["TEST-1.7_HuggingFace_Native_Scaffolding"] = True
        test_fused_kernel_equivalence()
        results["TEST-1.8_Fused_Chunk_Kernel"] = True
        test_mhar_generator()
        results["TEST-1.9_MHAR_Generator"] = True
        test_dynamic_overwrite_generator()
        results["TEST-1.10_Dynamic_Overwrite"] = True
        test_dyck_language_nesting()
        results["TEST-1.11_Dyck_Grammar"] = True
        test_streaming_state_persistence()
        results["TEST-1.12_Long_Context_Streaming"] = True
        test_curriculum_progression()
        results["TEST-1.13_Curriculum_Learning"] = True
        test_significance_tester()
        results["TEST-1.14_Statistical_Significance"] = True
        test_inter_woodbury_equivalence()
        results["TEST-1.15_Woodbury_Machine_Precision"] = True
        all_passed = True
    except Exception as e:
        print(f"\n[-] TEST SUITE FAILED WITH ERROR: {e}")
        all_passed = False
        raise e

    elapsed = time.time() - t0
    print("=" * 90)
    print(f"[+] ALL KERNEL-DELTANET INVARIANT TESTS PASSED IN {elapsed:.2f}s!")
    print("=" * 90)

    if save_log:
        log_path = "results/unit_tests_report.json"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({
                "all_passed": all_passed,
                "elapsed_seconds": elapsed,
                "tests_executed": results
            }, f, indent=2)
        print(f"[*] Test results logged to {log_path}\n")

    return all_passed


if __name__ == "__main__":
    run_master_test_suite()
