"""
Run full-scale standard benchmark suite across all available validation/test items:
- PIQA: N = 1,838 (validation)
- ARC-Challenge: N = 1,172 (test)
- HellaSwag: N = 10,042 (validation)
- LAMBADA: N = 5,153 (test)

Evaluates on physical silicon (RTX 5070 Ti) using batched candidate evaluation for high throughput.
Models evaluated:
1. Base_TinyLlama_1.1B (Softmax Teacher)
2. Pure_Kernel_DeltaNet_1.1B (Pure 22 layers)
3. Hybrid_Gauge_Topology_1.1B (20 Kernel-DeltaNet : 2 Softmax anchors)
"""

import os
import sys
import gc
import time
import math
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.upcycling_llama_bridge import construct_upcycled_1b_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32


def load_full_benchmarks() -> Dict[str, List[Dict[str, Any]]]:
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    benchmarks = {}

    # 1. PIQA (Full validation = 1,838)
    p_piqa = list((hub / "datasets--lighteval--piqa").rglob("*validation*.parquet"))[0]
    df_piqa = pd.read_parquet(p_piqa)
    piqa_samples = []
    for _, row in df_piqa.iterrows():
        piqa_samples.append({
            "prompt": row["goal"].strip(),
            "choices": [row["sol1"].strip(), row["sol2"].strip()],
            "label": int(row["label"])
        })
    benchmarks["PIQA"] = piqa_samples
    print(f"Loaded FULL PIQA validation set: {len(piqa_samples)} samples.")

    # 2. ARC-Challenge (Full test = 1,172)
    p_arc = hub / "datasets--ai2_arc" / "snapshots" / "210d026faf9955653af8916fad021475a3f00453" / "ARC-Challenge" / "test-00000-of-00001.parquet"
    df_arc = pd.read_parquet(p_arc)
    arc_samples = []
    for _, row in df_arc.iterrows():
        labels = list(row["choices"]["label"])
        key = row["answerKey"]
        if key in labels:
            arc_samples.append({
                "prompt": row["question"].strip(),
                "choices": [c.strip() for c in row["choices"]["text"]],
                "label": labels.index(key)
            })
    benchmarks["ARC-Challenge"] = arc_samples
    print(f"Loaded FULL ARC-Challenge test set: {len(arc_samples)} samples.")

    # 3. LAMBADA (Full test = 5,153)
    p_lam = list((hub / "datasets--cimec--lambada").rglob("*test*.parquet"))[0]
    df_lam = pd.read_parquet(p_lam)
    raw_lam = []
    all_targets = []
    for _, row in df_lam.iterrows():
        words = row["text"].strip().split()
        if len(words) > 10:
            raw_lam.append((" ".join(words[:-1]), words[-1]))
            all_targets.append(words[-1])
    import random
    rng = random.Random(42)
    lam_samples = []
    for prompt, target in raw_lam:
        distractors = rng.sample([w for w in all_targets if w.lower() != target.lower()], 3)
        choices = [target] + distractors
        order = list(range(4))
        rng.shuffle(order)
        lam_samples.append({
            "prompt": prompt,
            "choices": [choices[idx] for idx in order],
            "label": order.index(0)
        })
    benchmarks["LAMBADA"] = lam_samples
    print(f"Loaded FULL LAMBADA test set: {len(lam_samples)} samples.")

    # 4. HellaSwag (Full validation = 10,042)
    p_hs = list((hub / "datasets--Rowan--hellaswag").rglob("*validation*.parquet"))[0]
    df_hs = pd.read_parquet(p_hs)
    hs_samples = []
    for _, row in df_hs.iterrows():
        hs_samples.append({
            "prompt": row["ctx"].strip(),
            "choices": [e.strip() for e in row["endings"]],
            "label": int(row["label"])
        })
    benchmarks["HellaSwag"] = hs_samples
    print(f"Loaded FULL HellaSwag validation set: {len(hs_samples)} samples.")

    return benchmarks


@torch.no_grad()
def evaluate_model_batched(
    model: torch.nn.Module,
    tokenizer: Any,
    samples: List[Dict[str, Any]],
    bsz_questions: int = 16
) -> Tuple[int, int, float, float, Tuple[float, float], float]:
    correct = 0
    total = len(samples)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    t0 = time.time()

    for i in range(0, total, bsz_questions):
        batch = samples[i:i + bsz_questions]
        all_seqs = []
        meta = []  # (prompt_len, seq_len, sample_idx, choice_idx)

        for s_idx, item in enumerate(batch):
            p_ids = tokenizer.encode(item["prompt"], add_special_tokens=True)
            p_len = len(p_ids)
            for c_idx, c in enumerate(item["choices"]):
                full_ids = tokenizer.encode(item["prompt"] + " " + c, add_special_tokens=True)
                all_seqs.append(full_ids)
                meta.append((p_len, len(full_ids), s_idx, c_idx))

        max_len = max(len(s) for s in all_seqs)
        padded = [s + [pad_id] * (max_len - len(s)) for s in all_seqs]
        inp = torch.tensor(padded, dtype=torch.long, device=DEVICE)

        logits = model(inp).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = inp[:, 1:]

        sample_losses = {s_idx: {} for s_idx in range(len(batch))}
        for seq_i, (p_len, s_len, s_idx, c_idx) in enumerate(meta):
            cand_logits = shift_logits[seq_i, p_len - 1:s_len - 1]
            cand_labels = shift_labels[seq_i, p_len - 1:s_len - 1]
            loss = F.cross_entropy(cand_logits, cand_labels, reduction="mean").item()
            sample_losses[s_idx][c_idx] = loss

        for s_idx, item in enumerate(batch):
            pred = min(sample_losses[s_idx].keys(), key=lambda k: sample_losses[s_idx][k])
            if pred == item["label"]:
                correct += 1

    dt = time.time() - t0
    acc = (correct / total) * 100.0 if total > 0 else 0.0
    se = math.sqrt((acc / 100.0) * (1.0 - acc / 100.0) / total) * 100.0 if total > 0 else 0.0
    ci_lower = max(0.0, acc - 1.96 * se)
    ci_upper = min(100.0, acc + 1.96 * se)

    return correct, total, acc, se, (ci_lower, ci_upper), dt


def run_full_suite():
    print("=" * 80)
    print("  PHYSICAL SILICON FULL STANDARD BENCHMARK EVALUATION (RTX 5070 Ti)")
    print("  Zero Subsampling: Full Sets (PIQA=1838, ARC=1172, LAMBADA=5153, HellaSwag=10042)")
    print("=" * 80)

    benchmarks = load_full_benchmarks()
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # 1. Base Model (Softmax Teacher)
    print("\n[Stage 1/3] Evaluating Teacher: TinyLlama-1.1B (Softmax Attention)...")
    teacher = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE, device_map=DEVICE)
    teacher.eval()

    results["Base_TinyLlama_1.1B"] = {}
    for task_name in ["PIQA", "ARC-Challenge", "LAMBADA", "HellaSwag"]:
        samples = benchmarks[task_name]
        bsz = 16 if task_name != "HellaSwag" else 8
        c, t, acc, se, ci, dt = evaluate_model_batched(teacher, tokenizer, samples, bsz_questions=bsz)
        results["Base_TinyLlama_1.1B"][task_name] = {
            "correct": c, "total": t, "accuracy": round(acc, 2), "se": round(se, 2),
            "ci_95": [round(ci[0], 2), round(ci[1], 2)], "latency_sec": round(dt, 2),
            "retention_pct": 100.0
        }
        print(f"  Teacher {task_name:<15}: {acc:.2f}% +/- {se:.2f}% (N={t}, {dt:.1f}s)")

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # Find trained student checkpoint
    ckpt_dir = REPO_ROOT / "checkpoints_distill_1b"
    ckpts = sorted(ckpt_dir.glob("llama1b_kernel_deltanet_step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0)
    latest_ckpt_path = str(ckpts[-1])
    print(f"\n[*] Loaded checkpoint: {Path(latest_ckpt_path).name}")
    raw_ckpt = torch.load(latest_ckpt_path, map_location=DEVICE)
    trainable_sd = raw_ckpt.get("trainable_state_dict", raw_ckpt.get("state_dict", raw_ckpt))

    # 2. Pure Kernel-DeltaNet (22:0)
    print("\n[Stage 2/3] Evaluating Pure Kernel-DeltaNet 1.1B (22 recurrent layers)...")
    pure_student = construct_upcycled_1b_model(
        model_name_or_path=model_name,
        chunk_size=128,
        device=DEVICE,
        dtype=DTYPE,
        freeze_backbone=True,
        hybrid_attn_layers=None
    )
    pure_student.load_state_dict(trainable_sd, strict=False)
    pure_student.eval()

    results["Pure_Kernel_DeltaNet_1.1B"] = {}
    for task_name in ["PIQA", "ARC-Challenge", "LAMBADA", "HellaSwag"]:
        samples = benchmarks[task_name]
        bsz = 16 if task_name != "HellaSwag" else 8
        c, t, acc, se, ci, dt = evaluate_model_batched(pure_student, tokenizer, samples, bsz_questions=bsz)
        t_acc = results["Base_TinyLlama_1.1B"][task_name]["accuracy"]
        ret = (acc / t_acc) * 100.0 if t_acc > 0 else 0.0
        results["Pure_Kernel_DeltaNet_1.1B"][task_name] = {
            "correct": c, "total": t, "accuracy": round(acc, 2), "se": round(se, 2),
            "ci_95": [round(ci[0], 2), round(ci[1], 2)], "latency_sec": round(dt, 2),
            "retention_pct": round(ret, 2)
        }
        print(f"  Pure KDN {task_name:<15}: {acc:.2f}% +/- {se:.2f}% | Ret = {ret:.2f}% (N={t}, {dt:.1f}s)")

    del pure_student
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Hybrid Gauge Topology (20:2)
    print("\n[Stage 3/3] Evaluating Hybrid Gauge Topology 1.1B (20 recurrent : 2 Softmax anchors)...")
    hybrid_student = construct_upcycled_1b_model(
        model_name_or_path=model_name,
        chunk_size=128,
        device=DEVICE,
        dtype=DTYPE,
        freeze_backbone=True,
        hybrid_attn_layers=[10, 21]
    )
    hybrid_student.load_state_dict(trainable_sd, strict=False)
    hybrid_student.eval()

    results["Hybrid_Gauge_Topology_1.1B"] = {}
    for task_name in ["PIQA", "ARC-Challenge", "LAMBADA", "HellaSwag"]:
        samples = benchmarks[task_name]
        bsz = 16 if task_name != "HellaSwag" else 8
        c, t, acc, se, ci, dt = evaluate_model_batched(hybrid_student, tokenizer, samples, bsz_questions=bsz)
        t_acc = results["Base_TinyLlama_1.1B"][task_name]["accuracy"]
        ret = (acc / t_acc) * 100.0 if t_acc > 0 else 0.0
        results["Hybrid_Gauge_Topology_1.1B"][task_name] = {
            "correct": c, "total": t, "accuracy": round(acc, 2), "se": round(se, 2),
            "ci_95": [round(ci[0], 2), round(ci[1], 2)], "latency_sec": round(dt, 2),
            "retention_pct": round(ret, 2)
        }
        print(f"  Hybrid (20:2) {task_name:<15}: {acc:.2f}% +/- {se:.2f}% | Ret = {ret:.2f}% (N={t}, {dt:.1f}s)")

    del hybrid_student
    gc.collect()
    torch.cuda.empty_cache()

    out_file = REPO_ROOT / "results" / "full_standard_benchmark_suite.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Master results successfully saved to {out_file}")


if __name__ == "__main__":
    run_full_suite()
