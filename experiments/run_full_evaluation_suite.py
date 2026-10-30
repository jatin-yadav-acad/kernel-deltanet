r"""
Kernel-DeltaNet Engine
Module: experiments.run_full_evaluation_suite
=============================================
Master Experiment Orchestrator and Full Evaluation Suite Runner.
Executes the comprehensive empirical verification of Kernel-DeltaNet:
1. Master Invariant Tests & Standalone Tests
2. Fair Hardware Step Latency & Throughput Benchmark
3. Deep-Scale MQAR Capacity Saturation Stress Testing
4. Calibrated Long-Context Passkey Retrieval Benchmark
5. 50-Trial Statistical Significance Hypothesis Testing & Bootstrap CIs
6. Constant O(1) Memory Autoregressive Generation Profiling
7. Systematic 5-Axis Architectural Ablation Studies on RTX 5070 Ti
8. Downstream Perplexity Evaluation on Converged FineWeb-Edu Checkpoints
9. High-Resolution Publication Figure Generation
10. Camera-Ready MLSys LaTeX Table Generation
"""

import os
import json
import time
from typing import Dict


def run_full_evaluation_pipeline(
    save_path: str = "results/master_evaluation_report.json"
) -> Dict:
    from tests.test_all_mlsys import run_master_test_suite
    from tests.test_kernel_deltanet_suite import run_test_2_suite
    from evaluation.eval_fair_hardware_benchmark import run_fair_hardware_benchmark
    from evaluation.eval_mqar_capacity import run_mqar_capacity_benchmark
    from evaluation.eval_calibrated_retrieval import run_calibrated_retrieval_benchmark
    from experiments.statistical_significance import run_statistical_significance_suite
    from evaluation.eval_generation_memory import run_generation_memory_benchmark
    from experiments.ablation_studies import run_all_ablations
    from evaluation.eval_downstream_perplexity import run_perplexity_benchmark
    from visualization.generate_all_figures import generate_all_figures
    from experiments.latex_table_generator import generate_all_latex_tables

    print("=" * 105)
    print("KERNEL-DELTANET: MASTER EMPIRICAL EXPERIMENTAL SUITE")
    print("=" * 105)
    t_start = time.time()
    report = {
        "suite_name": "Kernel-DeltaNet Master Evaluation Suite",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phases": {}
    }

    # Step 1: Unit Tests
    print("\n>>> STEP 1: Executing Master & Standalone Unit Tests...")
    t0 = time.time()
    t1_ok = run_master_test_suite(save_log=True)
    t2_ok = run_test_2_suite(save_log=True)
    report["phases"]["step1_unit_tests"] = {
        "master_suite_passed": t1_ok,
        "test_2_suite_passed": t2_ok,
        "elapsed_seconds": time.time() - t0
    }

    # Step 2: Fair Hardware Throughput Benchmark
    print("\n>>> STEP 2: Fair Hardware Throughput Benchmark (B=8, 16)...")
    t0 = time.time()
    hw_res = run_fair_hardware_benchmark(batch_sizes=[8, 16], sequence_lengths=[512, 1024, 2048, 4096])
    report["phases"]["step2_fair_hardware"] = {
        "results": hw_res,
        "elapsed_seconds": time.time() - t0
    }

    # Step 3: Deep-Scale MQAR Capacity Saturation
    print("\n>>> STEP 3: Deep-Scale MQAR Capacity Saturation Benchmark (up to 16x)...")
    t0 = time.time()
    mqar_res = run_mqar_capacity_benchmark(saturation_factors=[1, 2, 4, 8, 16], sequence_length=2048)
    report["phases"]["step3_mqar_capacity"] = {
        "results": mqar_res,
        "elapsed_seconds": time.time() - t0
    }

    # Step 4: Calibrated Long-Context Passkey Retrieval
    print("\n>>> STEP 4: Calibrated Long-Context Passkey Retrieval Benchmark (up to 32k)...")
    t0 = time.time()
    retrieval_res = run_calibrated_retrieval_benchmark(context_lengths=[1024, 2048, 4096, 8192, 16384, 32768])
    report["phases"]["step4_long_context_retrieval"] = {
        "results": retrieval_res,
        "elapsed_seconds": time.time() - t0
    }

    # Step 5: Statistical Significance
    print("\n>>> STEP 5: 50-Trial Statistical Significance Hypothesis Testing...")
    t0 = time.time()
    stat_res = run_statistical_significance_suite(num_trials=50, sequence_length=2048)
    report["phases"]["step5_statistical_significance"] = {
        "results": stat_res,
        "elapsed_seconds": time.time() - t0
    }

    # Step 6: Constant Memory Profiling
    print("\n>>> STEP 6: Autoregressive Generation Constant Memory Profiling...")
    t0 = time.time()
    mem_res = run_generation_memory_benchmark(sequence_lengths=[512, 1024, 2048, 4096, 8192, 16384])
    report["phases"]["step6_memory_profiling"] = {
        "results": mem_res,
        "elapsed_seconds": time.time() - t0
    }

    # Step 7: Ablation Studies
    print("\n>>> STEP 7: Systematic 5-Axis Architectural Ablations...")
    t0 = time.time()
    ablation_res = run_all_ablations()
    report["phases"]["step7_ablations"] = {
        "results": ablation_res,
        "elapsed_seconds": time.time() - t0
    }

    # Step 8: Downstream Perplexity
    print("\n>>> STEP 8: Downstream Perplexity on Converged Checkpoints...")
    t0 = time.time()
    ppl_res = run_perplexity_benchmark()
    report["phases"]["step8_perplexity"] = {
        "results": ppl_res,
        "elapsed_seconds": time.time() - t0
    }

    # Step 9: Publication Figures
    print("\n>>> STEP 9: Generating Publication-Ready Figures (300 DPI)...")
    t0 = time.time()
    fig_res = generate_all_figures()
    report["phases"]["step9_figures"] = {
        "figures_manifest": fig_res,
        "elapsed_seconds": time.time() - t0
    }

    # Step 10: Camera-Ready LaTeX Tables
    print("\n>>> STEP 10: Generating Camera-Ready LaTeX Tables...")
    t0 = time.time()
    tab_res = generate_all_latex_tables()
    report["phases"]["step10_latex_tables"] = {
        "tables_manifest": tab_res,
        "elapsed_seconds": time.time() - t0
    }

    total_time = time.time() - t_start
    report["total_runtime_seconds"] = total_time
    report["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 105)
    print(f"MASTER EMPIRICAL SUITE COMPLETED SUCCESSFULLY IN {total_time:.2f}s!")
    print(f"Master Evaluation Report saved to: {save_path}")
    print("=" * 105 + "\n")

    return report


if __name__ == "__main__":
    run_full_evaluation_pipeline()
