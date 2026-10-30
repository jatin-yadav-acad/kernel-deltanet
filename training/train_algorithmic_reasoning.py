"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: training.train_algorithmic_reasoning
=============================================================
Algorithmic Fine-Tuning & Learning Dynamics Benchmark on RTX 5070 Ti.
Trains and compares Kernel-DeltaNet vs Linear DeltaNet vs Standard Transformer on:
1. Multi-Hop Associative Recall (2-hop, 3-hop reasoning chains with distractors)
2. Dynamic Memory Variable Overwrite (Tracking temporal state mutations)
3. Dyck-3 Formal Grammar Stack Tracking (Hierarchical bracket matching)

Measures:
- Sample efficiency (steps to 80% accuracy)
- Final validation accuracy (%)
- Final cross-entropy loss
- Convergence speed (tokens/sec during training)
"""

import os
import json
import time
import math
import random
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.kernel_deltanet_lm import KernelDeltaNetLM
from baselines.ttt_linear_deltanet import TTTLinearLM as LinearDeltaNetLM
from baselines.standard_transformer import StandardTransformerLM
from .optimizer_utils import configure_optimizers, CosineWarmupLRScheduler


# -------------------------------------------------------------
# Synthetic Algorithmic Task Generators
# -------------------------------------------------------------

def generate_multihop_batch(
    batch_size: int,
    seq_len: int,
    hops: int = 2,
    num_entities: int = 64,
    vocab_offset: int = 10,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates multi-hop associative chains:
    e.g., A -> B, B -> C. Query: A? Target: C.
    Interspersed with random distractors.
    """
    inputs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Generate random distinct entities
        entities = torch.randperm(num_entities)[:hops + 1] + vocab_offset
        
        # Place random distractors
        inputs[b] = torch.randint(vocab_offset + num_entities, vocab_offset + num_entities + 100, (seq_len,), device=device)

        # Place the multi-hop chain at random positions
        positions = sorted(random.sample(range(0, seq_len - 10), hops * 2))
        for h in range(hops):
            inputs[b, positions[h*2]] = entities[h]
            inputs[b, positions[h*2 + 1]] = entities[h+1]

        # Query token at the end
        inputs[b, -2] = entities[0]
        # Target is the terminal entity
        targets[b, -2] = entities[-1]

    return inputs, targets


def generate_dynamic_overwrite_batch(
    batch_size: int,
    seq_len: int,
    num_vars: int = 8,
    num_vals: int = 32,
    vocab_offset: int = 10,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates dynamic variable overwrite sequences:
    x = v1, y = v2, ..., x = v_new. Query: x? Target: v_new.
    """
    inputs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    var_base = vocab_offset
    val_base = vocab_offset + num_vars + 2

    for b in range(batch_size):
        # Fill with neutral background tokens
        inputs[b] = torch.randint(val_base + num_vals, val_base + num_vals + 50, (seq_len,), device=device)

        # Track variable states
        var_state = {}
        target_var = random.randint(0, num_vars - 1)
        
        # Place assignments
        num_writes = seq_len // 8
        positions = sorted(random.sample(range(0, seq_len - 6), num_writes * 2))
        for w in range(num_writes):
            v_idx = random.randint(0, num_vars - 1)
            val = random.randint(0, num_vals - 1)
            var_state[v_idx] = val

            inputs[b, positions[w*2]] = var_base + v_idx
            inputs[b, positions[w*2 + 1]] = val_base + val

        # Query at terminal position
        inputs[b, -2] = var_base + target_var
        targets[b, -2] = val_base + var_state.get(target_var, 0)

    return inputs, targets


def generate_dyck3_batch(
    batch_size: int,
    seq_len: int,
    max_depth: int = 10,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates Dyck-3 formal grammar sequences with matching brackets:
    Pairs: (0, 1), (2, 3), (4, 5).
    Evaluates predicting the closing bracket corresponding to the deepest open bracket.
    """
    inputs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    open_tokens = [10, 12, 14]
    close_tokens = [11, 13, 15]

    for b in range(batch_size):
        stack = []
        for t in range(seq_len - 1):
            can_open = len(stack) < max_depth
            can_close = len(stack) > 0

            if can_open and (not can_close or random.random() < 0.55):
                b_type = random.randint(0, 2)
                inputs[b, t] = open_tokens[b_type]
                stack.append(close_tokens[b_type])
            elif can_close:
                expected_close = stack.pop()
                inputs[b, t] = expected_close
                targets[b, t-1] = expected_close
            else:
                b_type = random.randint(0, 2)
                inputs[b, t] = open_tokens[b_type]
                stack.append(close_tokens[b_type])

        # Fill final target
        if stack:
            targets[b, seq_len - 2] = stack[-1]

    return inputs, targets


# -------------------------------------------------------------
# Training & Fine-Tuning Engine
# -------------------------------------------------------------

def fine_tune_single_model_on_task(
    model: nn.Module,
    task_name: str,
    steps: int = 100,
    batch_size: int = 16,
    seq_len: int = 256,
    lr: float = 3e-4,
    device: str = "cuda"
) -> Dict:
    """
    Fine-tunes a model on an algorithmic task and logs learning progression.
    """
    model.train()
    optimizer = configure_optimizers(model, weight_decay=0.01, learning_rate=lr)
    scheduler = CosineWarmupLRScheduler(optimizer, warmup_steps=10, max_steps=steps, min_lr_ratio=0.05)

    loss_history = []
    acc_history = []
    step_times = []
    step_to_80_acc = None

    t0_start = time.perf_counter()

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        
        # Batch generation
        if task_name == "multihop_2hop":
            x, y = generate_multihop_batch(batch_size, seq_len, hops=2, device=device)
        elif task_name == "multihop_3hop":
            x, y = generate_multihop_batch(batch_size, seq_len, hops=3, device=device)
        elif task_name == "dynamic_overwrite":
            x, y = generate_dynamic_overwrite_batch(batch_size, seq_len, device=device)
        elif task_name == "dyck3_stack":
            x, y = generate_dyck3_batch(batch_size, seq_len, device=device)
        else:
            raise ValueError(f"Unknown task: {task_name}")

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            
            # Masked cross entropy on supervised query tokens
            loss_mask = (y != -100)
            if loss_mask.sum() > 0:
                active_logits = logits[loss_mask]
                active_targets = y[loss_mask]
                loss = F.cross_entropy(active_logits, active_targets)
                
                # Compute accuracy
                preds = torch.argmax(active_logits, dim=-1)
                acc = (preds == active_targets).float().mean().item() * 100.0
            else:
                loss = F.cross_entropy(logits[:, -1, :], torch.zeros(batch_size, dtype=torch.long, device=device))
                acc = 0.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_times.append(dt)
        loss_history.append(float(loss.item()))
        acc_history.append(float(acc))

        if acc >= 80.0 and step_to_80_acc is None:
            step_to_80_acc = step

    total_time = time.perf_counter() - t0_start
    final_acc = float(np.mean(acc_history[-10:]))
    final_loss = float(np.mean(loss_history[-10:]))
    throughput = (steps * batch_size * seq_len) / total_time

    return {
        "final_accuracy": round(final_acc, 2),
        "final_loss": round(final_loss, 4),
        "step_to_80_acc": step_to_80_acc,
        "total_time_sec": round(total_time, 2),
        "training_throughput_tok_per_sec": round(throughput, 1),
        "loss_curve": [round(l, 4) for l in loss_history[::10]],
        "acc_curve": [round(a, 1) for a in acc_history[::10]]
    }


import numpy as np


def run_algorithmic_finetuning_suite(
    tasks: List[str] = ["multihop_2hop", "multihop_3hop", "dynamic_overwrite", "dyck3_stack"],
    d_model: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    steps: int = 100,
    output_path: str = "results/algorithmic_finetuning_results.json"
) -> Dict:
    """
    Executes algorithmic fine-tuning suite across all 3 models on RTX 5070 Ti silicon.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 90)
    print("KERNEL-DELTANET: ALGORITHMIC FINE-TUNING SUITE (ON RTX 5070 Ti)")
    print("=" * 90)
    print(f"[*] Tasks: {tasks}")
    print(f"[*] Steps per Task: {steps} | Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    model_constructors = {
        "Kernel-DeltaNet (Ours)": lambda: KernelDeltaNetLM(
            vocab_size=256, d_model=d_model, num_layers=num_layers, num_heads=num_heads, rkhs_factor=4, chunk_size=64
        ).to(device),
        "Linear DeltaNet": lambda: LinearDeltaNetLM(
            vocab_size=256, d_model=d_model, num_layers=num_layers, chunk_size=64
        ).to(device),
        "Standard Transformer": lambda: StandardTransformerLM(
            vocab_size=256, d_model=d_model, num_layers=num_layers, num_heads=num_heads
        ).to(device)
    }

    results = {}

    for task in tasks:
        print(f"\n[*] =================== Task: {task.upper()} ===================")
        results[task] = {}

        for model_name, constructor in model_constructors.items():
            print(f"  -> Fine-Tuning {model_name}...", end="", flush=True)
            # Fresh instance with controlled seed for fair comparison
            torch.manual_seed(1337)
            model = constructor()

            task_report = fine_tune_single_model_on_task(
                model=model,
                task_name=task,
                steps=steps,
                batch_size=16,
                seq_len=256,
                lr=3e-4,
                device=device
            )

            results[task][model_name] = task_report
            step_str = f"Step {task_report['step_to_80_acc']}" if task_report['step_to_80_acc'] else "Did not reach"
            print(f" Final Acc: {task_report['final_accuracy']}% | Loss: {task_report['final_loss']:.4f} | "
                  f"80% Milestone: {step_str} | Time: {task_report['total_time_sec']}s")

    master_finetune_report = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "steps_per_task": steps,
        "tasks": tasks,
        "results": results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(master_finetune_report, f, indent=2)

    print(f"\n[+] Algorithmic fine-tuning benchmark logged to {output_path}")
    return master_finetune_report


if __name__ == "__main__":
    run_algorithmic_finetuning_suite()
