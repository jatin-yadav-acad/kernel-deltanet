r"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: evaluation.eval_algorithmic_reasoning
===============================================================
Comprehensive Algorithmic Reasoning and Dynamic Memory Benchmark on NVIDIA Silicon.
Evaluates:
1. Dynamic Memory Overwrite & Selective Forgetting (v_new vs v_old)
2. Multi-Hop Associative Recall (MHAR) across 2-hop, 3-hop, and 4-hop chain traversal
3. Dyck-3 Context-Free Grammar LIFO Stack Tracking (matching bracket prediction)
"""

import os
import json
import time
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.kernel_deltanet_layer import KernelDeltaNetLayer
from baselines.standard_transformer import CausalSoftmaxAttention
from baselines.ttt_linear_deltanet import TTTLinearLayer
from models.kernel_deltanet_lm import KernelDeltaNetLM
from baselines.standard_transformer import StandardTransformerLM
from baselines.ttt_linear_deltanet import TTTLinearLM
from data.complex_reasoning_tasks import (
    MultiHopRecallGenerator,
    DynamicMemoryOverwriteGenerator,
    DyckLanguageGenerator
)


def evaluate_dynamic_overwrite(
    models: Dict[str, nn.Module],
    d_model: int = 64,
    num_trials: int = 30,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Dict[str, float]]:
    r"""
    Tests whether models retrieve newly updated values over stale old values.
    Accuracy is 1.0 if retrieved vector is closer to v_new than v_old.
    """
    gen = DynamicMemoryOverwriteGenerator(d_model=d_model, device=device)
    results = {}
    d_sub = d_model // 2

    for name, m in models.items():
        m.eval()
        new_accs = []
        cos_news = []
        cos_olds = []

        for trial in range(num_trials):
            batch = gen.generate_task(seq_len=256, overwrite_gap=64, seed=500 + trial)
            seq = batch["sequence"].to(device)
            v_new = batch["val_new"].to(device)
            v_old = batch["val_old"].to(device)

            with torch.no_grad():
                out, _ = m(seq)

            pred_val = out[0, -1, d_sub:]
            cos_new = torch.cosine_similarity(pred_val.unsqueeze(0), v_new.unsqueeze(0)).item()
            cos_old = torch.cosine_similarity(pred_val.unsqueeze(0), v_old.unsqueeze(0)).item()

            cos_news.append(cos_new)
            cos_olds.append(cos_old)
            new_accs.append(1.0 if cos_new > cos_old else 0.0)

        results[name] = {
            "overwrite_success_rate": float(np.mean(new_accs) * 100.0),
            "mean_cos_sim_new": float(np.mean(cos_news)),
            "mean_cos_sim_old": float(np.mean(cos_olds))
        }

    return results


class MultiLayerStack(nn.Module):
    """Stack of sequence layers for multi-hop reasoning."""
    def __init__(self, layer_type: str, d_model: int, num_layers: int, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if layer_type == "kernel":
                self.layers.append(KernelDeltaNetLayer(d_model=d_model, num_heads=1, rkhs_factor=4, chunk_size=32, initial_decay=1.0, initial_eta=1.0, **kwargs))
            elif layer_type == "linear":
                self.layers.append(TTTLinearLayer(d_model=d_model, chunk_size=32, initial_decay=1.0, initial_eta=1.0, **kwargs))
            elif layer_type == "transformer":
                self.layers.append(CausalSoftmaxAttention(d_model=d_model, num_heads=1, **kwargs))
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            out, _ = layer(x)
            x = norm(x + out)
        return x


def evaluate_multihop_recall(
    d_model: int = 64,
    hops_list: List[int] = [2, 3],
    num_trials: int = 25,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Dict]:
    r"""
    Evaluates multi-hop graph chain traversal across multi-layer models.
    """
    gen = MultiHopRecallGenerator(d_model=d_model, device=device)
    d_sub = d_model // 2
    results = {}

    for hops in hops_list:
        results[f"{hops}_hops"] = {}
        # Multi-layer networks with depth equal to number of hops
        stack_models = {
            "Standard Softmax Attention": MultiLayerStack("transformer", d_model, num_layers=hops).to(device),
            "Linear DeltaNet":           MultiLayerStack("linear", d_model, num_layers=hops).to(device),
            "Kernel-DeltaNet (Ours)":    MultiLayerStack("kernel", d_model, num_layers=hops).to(device)
        }

        # Initialize aligned projection weights for clear algebraic transmission
        for m in stack_models.values():
            for layer in m.layers:
                with torch.no_grad():
                    if hasattr(layer, "w_q"):
                        W = torch.eye(d_model, device=device)
                        layer.w_q.weight.copy_(W)
                        layer.w_k.weight.copy_(W)
                        layer.w_v.weight.copy_(W)
                        layer.w_out.weight.copy_(W)

        for name, m in stack_models.items():
            m.eval()
            top1_list = []
            cos_list = []

            for trial in range(num_trials):
                batch = gen.generate_task(num_hops=hops, num_distractors=12, seed=600 + trial)
                seq = batch["sequence"].to(device)
                target = batch["target_node"].to(device)
                all_nodes = batch["all_nodes"].to(device)

                with torch.no_grad():
                    out = m(seq)

                pred_val = out[0, -1, d_sub:]
                cos = torch.cosine_similarity(pred_val.unsqueeze(0), target.unsqueeze(0)).item()
                cos_list.append(cos)

                sims = torch.matmul(all_nodes, pred_val)
                top1 = bool(sims.argmax().item() == (hops))
                top1_list.append(1.0 if top1 else 0.0)

            results[f"{hops}_hops"][name] = {
                "top1_accuracy": float(np.mean(top1_list) * 100.0),
                "mean_cos_sim": float(np.mean(cos_list))
            }

    return results


def evaluate_dyck_language(
    d_model: int = 64,
    num_sequences: int = 50,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Dict]:
    r"""
    Dyck-3 Context-Free Grammar Stack Tracking.
    Tests closing bracket prediction accuracy on well-nested bracket sequences.
    """
    gen = DyckLanguageGenerator(num_types=3)
    vocab_size = gen.vocab_size

    models = {
        "Standard Transformer LM": StandardTransformerLM(vocab_size=vocab_size, d_model=d_model, num_layers=3, num_heads=2).to(device),
        "Linear DeltaNet LM":      TTTLinearLM(vocab_size=vocab_size, d_model=d_model, num_layers=3).to(device),
        "Kernel-DeltaNet LM":      KernelDeltaNetLM(vocab_size=vocab_size, d_model=d_model, num_layers=3, num_heads=2, rkhs_factor=4).to(device)
    }

    # Generate well-nested sequences
    sequences = [gen.generate_well_nested_sequence(max_depth=6, length=64).to(device) for _ in range(num_sequences)]

    results = {}
    for name, m in models.items():
        m.eval()
        correct_closes = 0
        total_closes = 0

        with torch.no_grad():
            for seq in sequences:
                inputs = seq[:-1].unsqueeze(0)
                targets = seq[1:].unsqueeze(0)
                logits, _ = m(inputs)
                preds = logits[0].argmax(dim=-1)

                for t in range(len(preds)):
                    target_tok = targets[0, t].item()
                    # Only evaluate accuracy on closing bracket tokens
                    if target_tok in gen.close_tokens:
                        total_closes += 1
                        if preds[t].item() == target_tok:
                            correct_closes += 1

        acc = (correct_closes / total_closes * 100.0) if total_closes > 0 else 0.0
        results[name] = {
            "bracket_matching_accuracy": float(acc),
            "total_evaluated_tokens": total_closes
        }

    return results


def run_algorithmic_suite(
    d_model: int = 64,
    save_path: str = "results/algorithmic_reasoning_benchmark.json"
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print(f"RIGOROUS ALGORITHMIC REASONING & DYNAMIC MEMORY BENCHMARK ON {device}")
    print("=" * 95)

    single_layer_models = {
        "Standard Softmax Attention": CausalSoftmaxAttention(d_model=d_model, num_heads=1).to(device),
        "Linear DeltaNet":           TTTLinearLayer(d_model=d_model, chunk_size=32, initial_decay=0.95, initial_eta=1.0).to(device),
        "Kernel-DeltaNet (Ours)":    KernelDeltaNetLayer(d_model=d_model, num_heads=1, rkhs_factor=8, chunk_size=32, initial_decay=0.95, initial_eta=1.0).to(device)
    }

    d_sub = d_model // 2
    W_q = torch.zeros(d_model, d_model, device=device)
    W_q[:d_sub, :d_sub] = torch.eye(d_sub, device=device)
    W_k = torch.zeros(d_model, d_model, device=device)
    W_k[:d_sub, :d_sub] = torch.eye(d_sub, device=device)
    W_v = torch.zeros(d_model, d_model, device=device)
    W_v[d_sub:, d_sub:] = torch.eye(d_sub, device=device)
    W_out = torch.eye(d_model, device=device)

    for m in single_layer_models.values():
        with torch.no_grad():
            if hasattr(m, "w_q"):
                m.w_q.weight.copy_(W_q)
                m.w_k.weight.copy_(W_k)
                m.w_v.weight.copy_(W_v)
                m.w_out.weight.copy_(W_out)

    print("\n--- Benchmark 1: Dynamic Memory Overwrite & Selective Forgetting ---")
    overwrite_results = evaluate_dynamic_overwrite(single_layer_models, d_model=d_model, num_trials=30, device=device)
    for name, r in overwrite_results.items():
        print(f"  -> {name:<32} | Overwrite Success: {r['overwrite_success_rate']:>5.1f}% | CosSim(v_new): {r['mean_cos_sim_new']:>+.3f} | CosSim(v_old): {r['mean_cos_sim_old']:>+.3f}")

    print("\n--- Benchmark 2: Multi-Hop Associative Recall (MHAR Graph Traversal) ---")
    multihop_results = evaluate_multihop_recall(d_model=d_model, hops_list=[2, 3], num_trials=25, device=device)
    for hop_k, h_res in multihop_results.items():
        print(f"  Configuration: {hop_k}")
        for name, r in h_res.items():
            print(f"    -> {name:<30} | Top-1 Recall: {r['top1_accuracy']:>5.1f}% | Mean CosSim: {r['mean_cos_sim']:>+.3f}")

    print("\n--- Benchmark 3: Dyck-3 Context-Free Grammar LIFO Stack Tracking ---")
    dyck_results = evaluate_dyck_language(d_model=d_model, num_sequences=50, device=device)
    for name, r in dyck_results.items():
        print(f"  -> {name:<30} | Bracket Match Acc: {r['bracket_matching_accuracy']:>5.1f}% ({r['total_evaluated_tokens']} tokens)")

    all_results = {
        "dynamic_overwrite": overwrite_results,
        "multihop_recall": multihop_results,
        "dyck_stack_tracking": dyck_results
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[+] Rigorous algorithmic reasoning results saved to {save_path}")
    return all_results


if __name__ == "__main__":
    run_algorithmic_suite()
