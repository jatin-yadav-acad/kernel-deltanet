"""
Kernel-DeltaNet Engine
Experiment: experiments.run_1b_upcycling_experiment
=====================================================================
Executes the full 1B Upcycling & Knowledge Distillation Pipeline:
1. Model Surgery & Semantic Weight Inheritance
2. Parameter Partitioning Audit (Frozen MLPs vs Trainable Scan)
3. Real-Token Distillation Training (Feature Distillation + Cross-Entropy)
4. Downstream Zero-Shot Evaluation (PIQA, ARC, HellaSwag, LAMBADA)
5. 32k Autoregressive Decoding Memory Profile (Flat O(1) State)
6. Results Export & Metrics Logging
"""

import os
import sys
import time
import math
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# Repository path insertion
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from models.upcycling_llama_bridge import (
    construct_upcycled_1b_model,
    audit_llama_1b_parameters
)
from training.distill_trainer_1b import DistillTrainer1B
from evaluation.eval_llama_1b_upcycled import Llama1BUpcyclingEvaluator


def load_real_token_batches(
    tokenizer,
    seq_len: int = 512,
    max_batches: int = 60
) -> List[torch.Tensor]:
    """
    Loads real natural language texts (PG-19, Sherlock, Time Machine),
    tokenizes them, and chunks into batches.
    """
    data_dir = repo_root / "data"
    text_files = ["pg19_sample.txt", "sherlock.txt", "time_machine.txt"]
    all_text = ""
    for fname in text_files:
        fpath = data_dir / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n\n"

    tokens = tokenizer(all_text, return_tensors="pt").input_ids[0]
    num_tokens = len(tokens)
    print(f"[Data] Tokenized {num_tokens:,} real natural language tokens with LLaMA tokenizer.")

    # Slice into batches
    batches = []
    total_samples = num_tokens // seq_len
    num_to_take = min(total_samples, max_batches)
    for i in range(num_to_take):
        chunk = tokens[i * seq_len : (i + 1) * seq_len].unsqueeze(0)  # Shape: (1, seq_len)
        batches.append(chunk)

    print(f"[Data] Created {len(batches)} batches of shape (1, {seq_len}).")
    return batches


def run_experiment(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_train_steps: int = 250,
    seq_len: int = 512,
    chunk_size: int = 128
):
    print("=" * 80)
    print("  KERNEL-DELTANET: BILLION-PARAMETER UPCYCLING & DISTILLATION PIPELINE")
    print("  Model Target: 1.1B LLaMA Foundation Architecture")
    print("  Sequence Engine: Kernel-DeltaNet (Chunked Scan C=128, RFF D=256)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Hardware] Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Initialize Distillation Trainer
    trainer = DistillTrainer1B(
        base_model_name=model_name,
        fallback_model_name=model_name,
        chunk_size=chunk_size,
        rkhs_factor=4,
        alpha=0.5,
        beta=0.1,
        learning_rate=1e-4,
        device=device,
        output_dir=str(repo_root / "checkpoints_distill_1b"),
        results_path=str(repo_root / "results" / "distillation_1b_training.json")
    )

    # 2. Prepare real token batches
    batches = load_real_token_batches(trainer.tokenizer, seq_len=seq_len, max_batches=max_train_steps)

    # 3. Execute Distillation Training
    train_history = trainer.train_epoch(
        token_batches=batches,
        max_steps=max_train_steps,
        log_interval=25,
        checkpoint_interval=50
    )

    # Latest checkpoint path
    latest_ckpt = repo_root / "checkpoints_distill_1b" / f"llama1b_kernel_deltanet_step_{len(train_history)}.pt"

    # 4. Zero-Shot Reasoning & Memory Profiling
    print("\n" + "=" * 80)
    print("  DOWNSTREAM ZERO-SHOT EVALUATION & MEMORY PROFILING")
    print("=" * 80)

    evaluator = Llama1BUpcyclingEvaluator(
        base_model_name=model_name,
        fallback_model_name=model_name,
        checkpoint_path=str(latest_ckpt) if latest_ckpt.exists() else None,
        chunk_size=chunk_size,
        device=device,
        results_path=str(repo_root / "results" / "llama_1b_upcycling_eval.json")
    )

    eval_results = evaluator.run_full_evaluation()

    # 5. Combine and export unified experiment record
    combined_record = {
        "title": "Billion-Parameter Upcycling & Distillation of Kernel-DeltaNet into LLaMA-1B",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0,
            "peak_vram_allocated_gb": max(h["vram_allocated_gb"] for h in train_history) if train_history else 0.0
        },
        "model_architecture": {
            "base_backbone": model_name,
            "total_parameters": trainer.audit["total_parameters"],
            "trainable_parameters": trainer.audit["trainable_parameters"],
            "frozen_parameters": trainer.audit["frozen_parameters"],
            "chunk_size": chunk_size,
            "rkhs_dimension": 256
        },
        "distillation_convergence": train_history,
        "evaluation": eval_results
    }

    final_json = repo_root / "results" / "llama_1b_distillation_convergence.json"
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(combined_record, f, indent=2)

    print(f"\n[Master] Complete experiment results written to: {final_json}")
    return combined_record


if __name__ == "__main__":
    run_experiment(max_train_steps=250, seq_len=512, chunk_size=128)
