"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: evaluation.eval_downstream_nlp_benchmarks
===================================================================
Standard Zero-Shot Downstream NLP Benchmarks on RTX 5070 Ti Silicon:
Evaluates converged 35M and 70M checkpoints against Transformer baselines on:
1. LAMBADA: Long-range target word prediction (context retention)
2. HellaSwag: Common-sense reasoning continuation (4-way choice)
3. PIQA: Physical interaction question answering (2-way choice)
4. ARC-Easy: Elementary grade science reasoning (4-way choice)

Evaluates via length-normalized unconditional / conditional log-likelihood scoring
matching standard LM-Evaluation-Harness conventions.
"""

import os
import json
import math
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from models.kernel_deltanet_lm import KernelDeltaNetLM
from baselines.standard_transformer import StandardTransformerLM


def compute_sequence_log_likelihood(
    model: nn.Module,
    context_ids: List[int],
    continuation_ids: List[int],
    device: str = "cuda"
) -> float:
    """
    Computes length-normalized log-likelihood: log P(continuation | context) / len(continuation).
    """
    model.eval()
    full_ids = torch.tensor([context_ids + continuation_ids], dtype=torch.long, device=device)
    cont_len = len(continuation_ids)
    if cont_len == 0:
        return 0.0

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(full_ids) # (1, T, vocab_size)
        
        # Logits predicting continuation start at len(context_ids) - 1
        cont_logits = logits[0, len(context_ids) - 1 : len(full_ids[0]) - 1, :]
        log_probs = F.log_softmax(cont_logits, dim=-1)
        
        cont_targets = torch.tensor(continuation_ids, dtype=torch.long, device=device)
        token_log_probs = log_probs.gather(dim=-1, index=cont_targets.unsqueeze(-1)).squeeze(-1)
        
        # Length-normalized log-likelihood
        norm_log_likelihood = float(token_log_probs.sum().item()) / cont_len
    return norm_log_likelihood


def evaluate_hellaswag_sample(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    sample: Dict,
    device: str = "cuda"
) -> bool:
    ctx = sample["ctx"]
    endings = sample["endings"]
    label = int(sample["label"])

    ctx_ids = tokenizer.encode(ctx)
    scores = []
    for ending in endings:
        ending_ids = tokenizer.encode(" " + ending)
        score = compute_sequence_log_likelihood(model, ctx_ids, ending_ids, device=device)
        scores.append(score)

    pred = int(torch.argmax(torch.tensor(scores)).item())
    return pred == label


def evaluate_piqa_sample(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    sample: Dict,
    device: str = "cuda"
) -> bool:
    goal = sample["goal"]
    sol1 = sample["sol1"]
    sol2 = sample["sol2"]
    label = int(sample["label"])

    goal_ids = tokenizer.encode(goal)
    s1_ids = tokenizer.encode(" " + sol1)
    s2_ids = tokenizer.encode(" " + sol2)

    score1 = compute_sequence_log_likelihood(model, goal_ids, s1_ids, device=device)
    score2 = compute_sequence_log_likelihood(model, goal_ids, s2_ids, device=device)

    pred = 0 if score1 > score2 else 1
    return pred == label


def evaluate_arc_easy_sample(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    sample: Dict,
    device: str = "cuda"
) -> bool:
    question = sample["question"]
    choices = sample["choices"]
    texts = choices["text"]
    labels = choices["label"]
    target_label = sample["answerKey"]

    q_ids = tokenizer.encode(question)
    scores = []
    for opt_text in texts:
        opt_ids = tokenizer.encode(" " + opt_text)
        score = compute_sequence_log_likelihood(model, q_ids, opt_ids, device=device)
        scores.append(score)

    pred_idx = int(torch.argmax(torch.tensor(scores)).item())
    pred_label = labels[pred_idx]
    return pred_label == target_label


def evaluate_lambada_sample(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    sample: Dict,
    device: str = "cuda"
) -> bool:
    text = sample["text"]
    words = text.strip().split()
    if len(words) < 2:
        return False
    context = " ".join(words[:-1])
    target_word = words[-1]

    ctx_ids = tokenizer.encode(context)
    target_ids = tokenizer.encode(" " + target_word)
    if not target_ids:
        return False

    model.eval()
    input_ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(input_ids)
        pred_token = int(torch.argmax(logits[0, -1, :]).item())
    
    return pred_token == target_ids[0]


def run_downstream_nlp_benchmarks(
    num_samples: int = 150,
    output_path: str = "results/downstream_nlp_benchmarks.json"
) -> Dict:
    """
    Executes standard zero-shot evaluations across 35M and 70M checkpoints on RTX 5070 Ti.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 95)
    print("KERNEL-DELTANET: STANDARD ZERO-SHOT DOWNSTREAM NLP BENCHMARKS (RTX 5070 Ti)")
    print("=" * 95)
    print(f"[*] Tasks: LAMBADA, HellaSwag, PIQA, ARC-Easy (Zero-Shot)")
    print(f"[*] Number of Evaluated Samples per Task: {num_samples}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load test datasets
    print("[*] Loading benchmark validation slices...")
    ds_lambada = list(load_dataset("cimec/lambada", split=f"validation[:{num_samples}]"))
    ds_hellaswag = list(load_dataset("Rowan/hellaswag", split=f"validation[:{num_samples}]"))
    ds_piqa = list(load_dataset("lighteval/piqa", split=f"validation[:{num_samples}]"))
    ds_arc = list(load_dataset("allenai/ai2_arc", "ARC-Easy", split=f"test[:{num_samples}]"))
    print(f"[+] Loaded: LAMBADA ({len(ds_lambada)}), HellaSwag ({len(ds_hellaswag)}), PIQA ({len(ds_piqa)}), ARC-Easy ({len(ds_arc)})")

    # Models to evaluate
    model_configs = [
        ("35m", "Kernel-DeltaNet", "kernel_deltanet_35m_bpe_converged.pt", 384, 8, 6, 4),
        ("35m", "Standard Transformer", "transformer_35m_bpe_converged.pt", 384, 8, 6, 4),
        ("70m", "Kernel-DeltaNet", "kernel_deltanet_70m_bpe_converged.pt", 512, 10, 8, 4),
        ("70m", "Standard Transformer", "transformer_70m_bpe_converged.pt", 512, 10, 8, 4)
    ]

    benchmark_results = {}

    for scale, model_family, ckpt_name, d_model, num_layers, num_heads, rkhs_factor in model_configs:
        cfg_key = f"{model_family}_{scale}"
        print(f"\n[*] Evaluating: {cfg_key} (d={d_model}, L={num_layers}, heads={num_heads})...")
        ckpt_path = os.path.join("checkpoints", ckpt_name)

        if "Kernel" in model_family:
            model = KernelDeltaNetLM(
                vocab_size=50257,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                rkhs_factor=rkhs_factor,
                chunk_size=128
            ).to(device)
        else:
            model = StandardTransformerLM(
                vocab_size=50257,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads
            ).to(device)

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded weights from {ckpt_path}")
        else:
            print(f"  [!] Checkpoint not found at {ckpt_path}, evaluating base initialization.")

        # 1. LAMBADA
        t0 = time.time()
        lambada_correct = sum(1 for s in ds_lambada if evaluate_lambada_sample(model, tokenizer, s, device=device))
        lambada_acc = (lambada_correct / len(ds_lambada)) * 100.0

        # 2. HellaSwag
        hs_correct = sum(1 for s in ds_hellaswag if evaluate_hellaswag_sample(model, tokenizer, s, device=device))
        hs_acc = (hs_correct / len(ds_hellaswag)) * 100.0

        # 3. PIQA
        piqa_correct = sum(1 for s in ds_piqa if evaluate_piqa_sample(model, tokenizer, s, device=device))
        piqa_acc = (piqa_correct / len(ds_piqa)) * 100.0

        # 4. ARC-Easy
        arc_correct = sum(1 for s in ds_arc if evaluate_arc_easy_sample(model, tokenizer, s, device=device))
        arc_acc = (arc_correct / len(ds_arc)) * 100.0

        eval_time = time.time() - t0
        print(f"  -> LAMBADA:   {lambada_acc:5.2f}% ({lambada_correct}/{len(ds_lambada)})")
        print(f"  -> HellaSwag: {hs_acc:5.2f}% ({hs_correct}/{len(ds_hellaswag)})")
        print(f"  -> PIQA:      {piqa_acc:5.2f}% ({piqa_correct}/{len(ds_piqa)})")
        print(f"  -> ARC-Easy:  {arc_acc:5.2f}% ({arc_correct}/{len(ds_arc)})")
        print(f"  -> Completed evaluation in {eval_time:.1f}s")

        benchmark_results[cfg_key] = {
            "scale": scale,
            "architecture": model_family,
            "lambada_acc": round(lambada_acc, 2),
            "hellaswag_acc": round(hs_acc, 2),
            "piqa_acc": round(piqa_acc, 2),
            "arc_easy_acc": round(arc_acc, 2),
            "mean_acc": round((lambada_acc + hs_acc + piqa_acc + arc_acc) / 4.0, 2),
            "eval_time_sec": round(eval_time, 2)
        }

    master_report = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "num_samples_per_task": num_samples,
        "results": benchmark_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    print(f"\n[+] Downstream NLP benchmarks logged to {output_path}")
    return master_report


if __name__ == "__main__":
    run_downstream_nlp_benchmarks()
