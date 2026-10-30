r"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: evaluation.eval_in_context_induction
==============================================================
In-Context Induction Head and Length Extrapolation Stress Test on NVIDIA RTX 5070 Ti.
Uses query-focused training on controlled vocabulary (vocab_size=64) to trace
the emergence and retention of induction circuits across delay horizons.
"""

import os
import json
import math
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.kernel_deltanet_lm import KernelDeltaNetLM
from baselines.standard_transformer import StandardTransformerLM
from baselines.ttt_linear_deltanet import TTTLinearLM
from data.algorithmic_tasks import AlgorithmicTaskGenerator


def train_and_eval_induction_task(
    vocab_size: int = 64,
    d_model: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
    train_steps: int = 250,
    batch_size: int = 16,
    delays: List[int] = [32, 64, 128, 256, 512],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Dict]:
    r"""
    Trains models on the Induction Head task using query-focused loss and tests recall across delays.
    """
    print("\n" + "=" * 90)
    print(f"[*] IN-CONTEXT INDUCTION HEAD RECALL EXPERIMENT ({device})")
    print("=" * 90)

    task_gen = AlgorithmicTaskGenerator(vocab_size=vocab_size, device=device)

    models = {
        "Standard Transformer": StandardTransformerLM(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, num_heads=num_heads).to(device),
        "Linear DeltaNet":      TTTLinearLM(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers).to(device),
        "Kernel-DeltaNet (Ours)": KernelDeltaNetLM(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, num_heads=num_heads, rkhs_factor=4, chunk_size=32).to(device)
    }

    results = {}
    for name, model in models.items():
        print(f"\n--- Training {name} on Induction Heads ({train_steps} steps) ---")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=0.01)
        model.train()

        t0 = time.time()
        for step in range(train_steps):
            inputs, targets = task_gen.generate_induction_batch(batch_size=batch_size, seq_len=128, seed=2000 + step)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(inputs)
                # Query-focused cross-entropy loss specifically on the induction target token at T-1
                loss = F.cross_entropy(logits[:, -1, :], targets[:, -1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (step + 1) % 50 == 0:
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == targets[:, -1]).float().mean().item() * 100.0
                print(f"    Step {step+1:>3}/{train_steps} | Query Loss: {loss.item():.4f} | Train Acc: {acc:>5.1f}%")

        elapsed_train = time.time() - t0
        print(f"    [+] Training completed in {elapsed_train:.2f}s")

        # Evaluate across delay horizons
        model.eval()
        delay_results = {}
        for delay in delays:
            seq_len = delay + 48
            acc_list = []
            with torch.no_grad():
                for trial in range(25):
                    inputs, targets = task_gen.generate_induction_batch(batch_size=16, seq_len=seq_len, seed=5000 + trial)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = model(inputs)
                    pred = logits[:, -1, :].argmax(dim=-1)
                    target_b = targets[:, -1]
                    acc = (pred == target_b).float().mean().item() * 100.0
                    acc_list.append(acc)

            mean_acc = float(np.mean(acc_list))
            delay_results[f"delay_{delay}"] = {
                "delay_tokens": delay,
                "seq_len": seq_len,
                "induction_accuracy_pct": mean_acc
            }
            print(f"    Delay: {delay:>3} tokens (SeqLen {seq_len:>4}) | Induction Recall: {mean_acc:>5.1f}%")

        results[name] = delay_results

    return results


def train_and_eval_associative_copy(
    vocab_size: int = 64,
    d_model: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
    train_steps: int = 250,
    batch_size: int = 16,
    k_associations: List[int] = [2, 4, 8, 12, 16],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Dict]:
    r"""
    Trains models on Associative Copy and evaluates retrieval accuracy as K increases.
    """
    print("\n" + "=" * 90)
    print(f"[*] ASSOCIATIVE MULTI-SPAN COPYING CAPACITY EXPERIMENT ({device})")
    print("=" * 90)

    task_gen = AlgorithmicTaskGenerator(vocab_size=vocab_size, device=device)

    models = {
        "Standard Transformer": StandardTransformerLM(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, num_heads=num_heads).to(device),
        "Linear DeltaNet":      TTTLinearLM(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers).to(device),
        "Kernel-DeltaNet (Ours)": KernelDeltaNetLM(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, num_heads=num_heads, rkhs_factor=4, chunk_size=32).to(device)
    }

    results = {}
    for name, model in models.items():
        print(f"\n--- Training {name} on Associative Copy ({train_steps} steps) ---")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=0.01)
        model.train()

        t0 = time.time()
        for step in range(train_steps):
            inputs, targets = task_gen.generate_associative_copy(batch_size=batch_size, seq_len=128, num_associations=4, seed=3000 + step)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(inputs)
                loss = F.cross_entropy(logits[:, -1, :], targets[:, -1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (step + 1) % 50 == 0:
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == targets[:, -1]).float().mean().item() * 100.0
                print(f"    Step {step+1:>3}/{train_steps} | Query Loss: {loss.item():.4f} | Train Acc: {acc:>5.1f}%")

        elapsed_train = time.time() - t0
        print(f"    [+] Training completed in {elapsed_train:.2f}s")

        # Evaluate across number of associations K
        model.eval()
        k_results = {}
        for k in k_associations:
            acc_list = []
            with torch.no_grad():
                for trial in range(25):
                    inputs, targets = task_gen.generate_associative_copy(batch_size=16, seq_len=192, num_associations=k, seed=6000 + trial)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = model(inputs)
                    pred = logits[:, -1, :].argmax(dim=-1)
                    target_v = targets[:, -1]
                    acc = (pred == target_v).float().mean().item() * 100.0
                    acc_list.append(acc)

            mean_acc = float(np.mean(acc_list))
            k_results[f"K_{k}"] = {
                "num_associations": k,
                "copy_accuracy_pct": mean_acc
            }
            print(f"    K = {k:>2} Stored Pairs | Recall Accuracy: {mean_acc:>5.1f}%")

        results[name] = k_results

    return results


def run_in_context_suite(
    save_path: str = "results/in_context_induction_benchmark.json"
) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 100)
    print(f"IN-CONTEXT INDUCTION HEAD & ASSOCIATIVE COPY SUITE ON {device}")
    print("=" * 100)

    t0 = time.time()
    ind_res = train_and_eval_induction_task(delays=[32, 64, 128, 256, 512], device=device)
    copy_res = train_and_eval_associative_copy(k_associations=[2, 4, 8, 12, 16], device=device)

    all_results = {
        "induction_head_task": ind_res,
        "associative_copy_task": copy_res,
        "total_elapsed_seconds": time.time() - t0
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[+] In-context induction & copy results successfully saved to {save_path}")
    return all_results


if __name__ == "__main__":
    run_in_context_suite()
