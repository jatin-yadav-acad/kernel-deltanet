"""
Kernel-DeltaNet Engine
High-Power Offline Downstream Reasoning Evaluation Suite
=========================================================
Standardized Downstream Evaluation Suite across Canonical Benchmark Tasks.

Models Evaluated:
  1. Base Foundation Model: TinyLlama-1.1B-Chat-v1.0 (22 Attention Layers, 100% KV cache)
  2. Pure Upcycled Kernel-DeltaNet: 1.1B (22 Kernel-DeltaNet Layers, 0% KV cache)
  3. Hybrid Gauge Topology: 1.1B (20 Kernel-DeltaNet + 2 Attention Layers at [10, 21], 9.1% KV cache)

Tasks:
  - PIQA (Physical Interaction QA, N = 500)
  - ARC-Challenge (AI2 Reasoning Challenge, N = 500)
  - HellaSwag (Commonsense Situational Reasoning, N = 500)
  - LAMBADA (Target Word Disambiguation, N = 500)
"""

import os
import sys
import gc
import time
import math
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Repository path insertion
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.canonical_nlp_eval_loader import load_canonical_eval_datasets
from models.upcycling_llama_bridge import construct_upcycled_1b_model


@torch.no_grad()
def evaluate_model_on_benchmark(
    model: nn.Module,
    tokenizer: Any,
    samples: List[Dict[str, Any]],
    device: str = "cuda"
) -> Tuple[int, int, List[bool]]:
    """
    Evaluates model accuracy on a multiple-choice benchmark using exact
    prompt-conditioned log-likelihood scoring. Batches choices per sample.
    """
    correct = 0
    total = len(samples)
    per_sample_correct = []

    for item in samples:
        prompt = item["prompt"]
        choices = item["choices"]
        label = item["label"]

        choice_losses = []
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_ids)

        # Batch all candidate choices for this question
        encoded_candidates = []
        for choice in choices:
            full_ids = tokenizer.encode(prompt + " " + choice, add_special_tokens=True)
            encoded_candidates.append(full_ids)

        max_len = max(len(c) for c in encoded_candidates)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Build padded batch [K, max_len]
        batch_input_ids = []
        batch_target_masks = []
        for c in encoded_candidates:
            padded = c + [pad_id] * (max_len - len(c))
            mask = [0] * (prompt_len - 1) + [1] * (len(c) - prompt_len) + [0] * (max_len - len(c) + 1)
            batch_input_ids.append(padded)
            batch_target_masks.append(mask[:max_len])

        input_tensor = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
        logits = model(input_tensor).logits  # [K, max_len, vocab_size]

        # Shift logits and compute cross entropy only on choice tokens
        shift_logits = logits[:, :-1, :].contiguous()  # [K, max_len - 1, V]
        shift_labels = input_tensor[:, 1:].contiguous()  # [K, max_len - 1]

        K, L_seq, V = shift_logits.shape
        loss_matrix = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none"
        ).view(K, L_seq)

        # Mask only the continuation tokens (choice tokens)
        for k in range(K):
            c_len = len(encoded_candidates[k])
            # continuation token positions in shifted sequence: from (prompt_len - 1) to (c_len - 1)
            choice_loss = loss_matrix[k, prompt_len - 1: c_len - 1].sum().item()
            num_tokens = max(1, c_len - prompt_len)
            choice_losses.append(choice_loss / num_tokens)

        predicted_choice = int(np.argmin(choice_losses))
        is_correct = (predicted_choice == label)
        if is_correct:
            correct += 1
        per_sample_correct.append(is_correct)

    return correct, total, per_sample_correct


def calculate_metrics(correct: int, total: int, base_acc: Optional[float] = None) -> Dict[str, float]:
    p = correct / total if total > 0 else 0.0
    acc = p * 100.0
    se = math.sqrt(p * (1.0 - p) / total) * 100.0 if total > 0 else 0.0
    ci_lower = max(0.0, acc - 1.96 * se)
    ci_upper = min(100.0, acc + 1.96 * se)
    retention = (acc / max(base_acc, 1e-4)) * 100.0 if base_acc is not None else 100.0

    return {
        "correct": correct,
        "total": total,
        "accuracy": round(acc, 2),
        "standard_error": round(se, 2),
        "ci_95_lower": round(ci_lower, 2),
        "ci_95_upper": round(ci_upper, 2),
        "retention_percentage": round(retention, 2)
    }


def run_full_scale_evaluation():
    print("=" * 85)
    print("  KERNEL-DELTANET: HIGH-POWER DOWNSTREAM EVALUATION (N = 500 SAMPLES)")
    print("  Comparing Base Transformer vs Pure Kernel-DeltaNet vs Hybrid Gauge Topology")
    print(f"  Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 85)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    # Find latest checkpoint
    ckpt_dir = REPO_ROOT / "checkpoints_distill_1b"
    ckpts = sorted(ckpt_dir.glob("llama1b_kernel_deltanet_step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0)
    if not ckpts:
        raise FileNotFoundError("No 1.1B distillation checkpoint found!")
    latest_ckpt_path = str(ckpts[-1])
    print(f"[*] Selected Trained Checkpoint: {Path(latest_ckpt_path).name}")

    # 1. Load Datasets (N = 500 samples per task)
    print("\n[*] Loading Canonical Benchmark Datasets (N = 500 samples per task)...")
    benchmarks = load_canonical_eval_datasets(num_samples_per_task=500)
    for name, data in benchmarks.items():
        print(f"  - {name:<15}: {len(data)} validated test questions.")

    model_backbone = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results_master = {}

    # =========================================================================
    # Stage 1: Evaluate Base TinyLlama-1.1B (Transformer Reference)
    # =========================================================================
    print("\n" + "=" * 85)
    print("  EVALUATING MODEL 1/3: Base Foundation Model (TinyLlama-1.1B Transformer)")
    print("  Architecture: 22 Full Attention Layers | 100% KV-Cache Footprint")
    print("=" * 85)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_backbone,
        torch_dtype=dtype,
        device_map=device
    )
    base_model.eval()

    base_results = {}
    for task_name in ["PIQA", "ARC-Challenge", "HellaSwag", "LAMBADA"]:
        if task_name in benchmarks:
            t0 = time.time()
            c, t, _ = evaluate_model_on_benchmark(base_model, tokenizer, benchmarks[task_name], device=device)
            dt = time.time() - t0
            metrics = calculate_metrics(c, t, base_acc=None)
            metrics["latency_sec"] = round(dt, 2)
            base_results[task_name] = metrics
            print(f"  [Base Transformer] {task_name:<15}: Acc = {metrics['accuracy']:>5.2f}% ± {metrics['standard_error']:>4.2f}% (N={t}, {dt:.1f}s)")

    results_master["Base_TinyLlama_1.1B"] = base_results

    # Free base model from GPU memory
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    print("[+] Freed Base Model from VRAM.")

    # =========================================================================
    # Stage 2: Evaluate Pure Upcycled Kernel-DeltaNet 1.1B
    # =========================================================================
    print("\n" + "=" * 85)
    print("  EVALUATING MODEL 2/3: Pure Upcycled Kernel-DeltaNet 1.1B")
    print("  Architecture: 22 Kernel-DeltaNet Layers | 0% KV-Cache (100% Reduction)")
    print("=" * 85)

    pure_model = construct_upcycled_1b_model(
        model_name_or_path=model_backbone,
        chunk_size=128,
        device=device,
        dtype=dtype,
        freeze_backbone=True,
        hybrid_attn_layers=None
    )
    ckpt_data = torch.load(latest_ckpt_path, map_location=device)
    trainable_sd = ckpt_data.get("trainable_state_dict", ckpt_data)
    pure_model.load_state_dict(trainable_sd, strict=False)
    pure_model.eval()

    pure_results = {}
    for task_name in ["PIQA", "ARC-Challenge", "HellaSwag", "LAMBADA"]:
        if task_name in benchmarks:
            t0 = time.time()
            c, t, _ = evaluate_model_on_benchmark(pure_model, tokenizer, benchmarks[task_name], device=device)
            dt = time.time() - t0
            base_acc = base_results[task_name]["accuracy"]
            metrics = calculate_metrics(c, t, base_acc=base_acc)
            metrics["latency_sec"] = round(dt, 2)
            pure_results[task_name] = metrics
            print(f"  [Pure Kernel-DeltaNet] {task_name:<15}: Acc = {metrics['accuracy']:>5.2f}% ± {metrics['standard_error']:>4.2f}% | Ret = {metrics['retention_percentage']:>5.1f}% (N={t}, {dt:.1f}s)")

    results_master["Pure_Kernel_DeltaNet_1.1B"] = pure_results

    # Free pure model from GPU memory
    del pure_model
    gc.collect()
    torch.cuda.empty_cache()
    print("[+] Freed Pure Kernel-DeltaNet Model from VRAM.")

    # =========================================================================
    # Stage 3: Evaluate Hybrid Gauge Topology 1.1B (20 Kernel-DeltaNet : 2 Attention)
    # =========================================================================
    print("\n" + "=" * 85)
    print("  EVALUATING MODEL 3/3: Hybrid Gauge Topology 1.1B (20:2 Topology)")
    print("  Architecture: 20 Kernel-DeltaNet + 2 Full Attention Layers [10, 21]")
    print("  Memory Reduction: 90.9% KV-Cache Savings (Preserves 9.1% Global Anchors)")
    print("=" * 85)

    hybrid_model = construct_upcycled_1b_model(
        model_name_or_path=model_backbone,
        chunk_size=128,
        device=device,
        dtype=dtype,
        freeze_backbone=True,
        hybrid_attn_layers=[10, 21]
    )
    hybrid_model.load_state_dict(trainable_sd, strict=False)
    hybrid_model.eval()

    hybrid_results = {}
    for task_name in ["PIQA", "ARC-Challenge", "HellaSwag", "LAMBADA"]:
        if task_name in benchmarks:
            t0 = time.time()
            c, t, _ = evaluate_model_on_benchmark(hybrid_model, tokenizer, benchmarks[task_name], device=device)
            dt = time.time() - t0
            base_acc = base_results[task_name]["accuracy"]
            metrics = calculate_metrics(c, t, base_acc=base_acc)
            metrics["latency_sec"] = round(dt, 2)
            hybrid_results[task_name] = metrics
            print(f"  [Hybrid Gauge Topology] {task_name:<15}: Acc = {metrics['accuracy']:>5.2f}% ± {metrics['standard_error']:>4.2f}% | Ret = {metrics['retention_percentage']:>5.1f}% (N={t}, {dt:.1f}s)")

    results_master["Hybrid_Gauge_Topology_1.1B"] = hybrid_results

    # Free hybrid model from GPU memory
    del hybrid_model
    gc.collect()
    torch.cuda.empty_cache()
    print("[+] Freed Hybrid Model from VRAM.")

    # =========================================================================
    # Master Comparison Table & Synthesis
    # =========================================================================
    print("\n" + "=" * 92)
    print("  FULL-SCALE EMPIRICAL EVALUATION SUMMARY (N = 500 SAMPLES PER BENCHMARK)")
    print("=" * 92)
    print(f"{'Benchmark Task':<16} | {'Base (Transformer)':<20} | {'Pure Kernel-DeltaNet':<22} | {'Hybrid Gauge Topology':<22}")
    print("-" * 92)

    tasks = ["PIQA", "ARC-Challenge", "HellaSwag", "LAMBADA"]
    mean_base = np.mean([base_results[t]["accuracy"] for t in tasks])
    mean_pure = np.mean([pure_results[t]["accuracy"] for t in tasks])
    mean_pure_ret = np.mean([pure_results[t]["retention_percentage"] for t in tasks])
    mean_hybrid = np.mean([hybrid_results[t]["accuracy"] for t in tasks])
    mean_hybrid_ret = np.mean([hybrid_results[t]["retention_percentage"] for t in tasks])

    for t in tasks:
        b_str = f"{base_results[t]['accuracy']:.1f}% ± {base_results[t]['standard_error']:.1f}%"
        p_str = f"{pure_results[t]['accuracy']:.1f}% ({pure_results[t]['retention_percentage']:.1f}%)"
        h_str = f"{hybrid_results[t]['accuracy']:.1f}% ({hybrid_results[t]['retention_percentage']:.1f}%)"
        print(f"{t:<16} | {b_str:<20} | {p_str:<22} | {h_str:<22}")

    print("-" * 92)
    print(f"{'MEAN ACCURACY':<16} | {mean_base:>6.2f}%               | {mean_pure:>6.2f}% (Ret: {mean_pure_ret:>5.1f}%) | {mean_hybrid:>6.2f}% (Ret: {mean_hybrid_ret:>5.1f}%)")
    print(f"{'KV-CACHE MEMORY':<16} | {'100.0% (Baseline)':<20} | {'0.0% (-100.0% flat)':<22} | {'9.1% (-90.9% saved)':<22}")
    print("=" * 92)

    # Save JSON report
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "full_scale_downstream_eval.json"

    master_payload = {
        "evaluation_protocol": "Standardized Zero-Shot Prompt-Conditioned Log-Likelihood Scoring",
        "sample_size_per_task": 500,
        "error_bars_formula": "Standard Error SE = sqrt(p*(1-p)/N) * 100%, 95% CI = [Acc - 1.96*SE, Acc + 1.96*SE]",
        "checkpoint_evaluated": latest_ckpt_path,
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "results": results_master,
        "summary": {
            "mean_base_accuracy": round(float(mean_base), 2),
            "mean_pure_accuracy": round(float(mean_pure), 2),
            "mean_pure_retention": round(float(mean_pure_ret), 2),
            "mean_hybrid_accuracy": round(float(mean_hybrid), 2),
            "mean_hybrid_retention": round(float(mean_hybrid_ret), 2),
            "pure_memory_reduction": "100.0%",
            "hybrid_memory_reduction": "90.9%"
        },
        "scientific_conclusion": (
            "Empirical evaluation across N=500 official validation samples confirms the "
            "Hybrid Gauge Topology (20 Kernel-DeltaNet : 2 Full Attention layers) recovers "
            f"{mean_hybrid_ret:.2f}% of teacher capability "
            "while slashing 90.9% of KV cache memory."
        )
    }
    }

    with open(json_path, "w") as f:
        json.dump(master_payload, f, indent=2)
    print(f"\n[+] Master Downstream JSON saved to: {json_path}")

    # =========================================================================
    # Publication Visualization: hybrid_gauge_reasoning_recovery.png
    # =========================================================================
    print("\n[*] Generating publication-grade comparison figure: hybrid_gauge_reasoning_recovery.png...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot 1: Downstream Benchmark Accuracy with Error Bars
    ax1 = axes[0]
    x = np.arange(len(tasks))
    width = 0.26

    acc_base = [base_results[t]["accuracy"] for t in tasks]
    se_base = [base_results[t]["standard_error"] for t in tasks]

    acc_pure = [pure_results[t]["accuracy"] for t in tasks]
    se_pure = [pure_results[t]["standard_error"] for t in tasks]

    acc_hyb = [hybrid_results[t]["accuracy"] for t in tasks]
    se_hyb = [hybrid_results[t]["standard_error"] for t in tasks]

    b1 = ax1.bar(x - width, acc_base, width, yerr=se_base, capsize=4, label="Base Transformer (100% KV)", color="#34495e", alpha=0.9)
    b2 = ax1.bar(x, acc_pure, width, yerr=se_pure, capsize=4, label="Pure Kernel-DeltaNet (0% KV)", color="#e74c3c", alpha=0.9)
    b3 = ax1.bar(x + width, acc_hyb, width, yerr=se_hyb, capsize=4, label="Hybrid Gauge (20:2, 9.1% KV)", color="#2ecc71", alpha=0.9)

    ax1.set_title("A. Zero-Shot Downstream Accuracy (N = 500 with ±1σ SE)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 95)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Annotate bars
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Subplot 2: Retention vs KV Cache Tradeoff Frontier
    ax2 = axes[1]
    models = ["Base Transformer", "Pure Kernel-DeltaNet", "Hybrid Gauge Topology"]
    retentions = [100.0, mean_pure_ret, mean_hybrid_ret]
    mem_savings = [0.0, 100.0, 90.9]
    colors = ["#34495e", "#e74c3c", "#2ecc71"]

    for i in range(len(models)):
        ax2.scatter(mem_savings[i], retentions[i], s=250, color=colors[i], label=models[i], zorder=5, edgecolors="black", linewidth=1.5)
        offset_y = 3 if i != 1 else -5
        ax2.annotate(f"{models[i]}\n(Ret: {retentions[i]:.1f}%, -{mem_savings[i]:.1f}% Mem)",
                     (mem_savings[i], retentions[i]),
                     xytext=(0, offset_y * 3), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i], alpha=0.85))

    ax2.plot([mem_savings[1], mem_savings[2], mem_savings[0]], [retentions[1], retentions[2], retentions[0]],
             linestyle="--", color="gray", alpha=0.6, zorder=3)

    ax2.set_title("B. Reasoning Capability Retention vs Memory Reduction", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("KV-Cache Memory Footprint Reduction (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Reasoning Retention Relative to Base (%)", fontsize=11, fontweight="bold")
    ax2.set_xlim(-10, 115)
    ax2.set_ylim(40, 110)
    ax2.axhline(90.0, color="green", linestyle=":", alpha=0.7, label="90% Retention Threshold")
    ax2.legend(loc="lower left", frameon=True, fontsize=9)

    plt.tight_layout()

    fig_workspace = REPO_ROOT / "hybrid_gauge_reasoning_recovery.png"
    plt.savefig(fig_workspace, dpi=300, bbox_inches="tight")
    print(f"[+] Saved figure to workspace: {fig_workspace}")
    print("\n[+] HIGH-POWER DOWNSTREAM EVALUATION COMPLETE.")


if __name__ == "__main__":
    run_full_scale_evaluation()
