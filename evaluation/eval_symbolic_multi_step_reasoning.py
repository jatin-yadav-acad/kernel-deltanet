"""
Kernel-DeltaNet Engine
Module: evaluation.eval_symbolic_multi_step_reasoning
========================================================================
High-Precision Evaluation on Multi-Step Reasoning & Symbolic Tasks:
1. GSM8k (Chain-of-thought multi-step arithmetic reasoning)
2. HumanEval (Code syntax, state scoping, and variable binding)

Empirical Comparison across:
- Teacher Baseline: TinyLlama-1.1B-Chat (Full Softmax Attention, 22 layers)
- Pure Kernel-DeltaNet: 100% Recurrent RKHS In-Register DeltaNet (22 layers)
- Hybrid Gauge Topology: 20:2 Topology (Layers [10, 21] Full Attention anchors)

Provides empirical grounding for the expressiveness boundaries analyzed in
Section 9 and Supplementary Section S.6.7.
"""

import os
import sys
import gc
import re
import time
import math
import json
import argparse
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Insert repository root to path
repo_root = Path(__file__).resolve().parent.parent
parent_root = repo_root.parent
for p in [str(repo_root), str(parent_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.upcycling_llama_bridge import construct_upcycled_1b_model

FEW_SHOT_GSM8K = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

"""


def load_gsm8k_dataset(num_samples: int = 500) -> List[Dict[str, Any]]:
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    for d_name in ["datasets--gsm8k", "datasets--openai--gsm8k"]:
        p = hub / d_name
        if p.exists():
            for f in p.rglob("*test*.parquet"):
                df = pd.read_parquet(f)
                samples = []
                for _, row in df.head(num_samples).iterrows():
                    samples.append({
                        "question": str(row["question"]).strip(),
                        "answer": str(row["answer"]).strip()
                    })
                print(f"[DataLoader] Loaded {len(samples)} GSM8k test samples from {f.name}.")
                return samples
    raise FileNotFoundError("Could not find cached GSM8k parquet dataset!")


def load_humaneval_dataset(num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    for d_name in ["datasets--openai_humaneval", "datasets--openai--openai_humaneval"]:
        p = hub / d_name
        if p.exists():
            for f in p.rglob("*.parquet"):
                df = pd.read_parquet(f)
                if num_samples is not None:
                    df = df.head(num_samples)
                samples = []
                for _, row in df.iterrows():
                    samples.append({
                        "task_id": str(row["task_id"]),
                        "prompt": str(row["prompt"]),
                        "canonical_solution": str(row["canonical_solution"]),
                        "test": str(row["test"]),
                        "entry_point": str(row["entry_point"])
                    })
                print(f"[DataLoader] Loaded {len(samples)} HumanEval test samples from {f.name}.")
                return samples
    raise FileNotFoundError("Could not find cached HumanEval parquet dataset!")


def extract_gsm8k_pred(text: str) -> Optional[str]:
    # Match standard answer formulations
    m = re.findall(r"(?:The answer is|the answer is|####)\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "").strip()
    m2 = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    if m2:
        return m2[-1].replace(",", "").strip()
    return None


def extract_gsm8k_ground_truth(answer: str) -> Optional[str]:
    m = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", answer)
    if m:
        return m[-1].replace(",", "").strip()
    m2 = re.findall(r"(-?\d+(?:\.\d+)?)", answer)
    if m2:
        return m2[-1].replace(",", "").strip()
    return None


def execute_humaneval_code(code_str: str, timeout: float = 3.0) -> bool:
    def _exec():
        exec_globals = {}
        exec(code_str, exec_globals)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_exec)
        try:
            return bool(future.result(timeout=timeout))
        except Exception:
            return False


def clean_code_completion(prompt: str, generated_text: str) -> str:
    # Remove prompt if model echoes prompt
    if generated_text.startswith(prompt):
        completion = generated_text[len(prompt):]
    else:
        completion = generated_text

    # Stop at def, class, if __name__, or markdown fences
    lines = []
    for line in completion.splitlines():
        if line.strip().startswith("def ") or line.strip().startswith("class "):
            break
        if line.strip().startswith("if __name__"):
            break
        if "```" in line:
            break
        lines.append(line)
    return "\n".join(lines)


@torch.no_grad()
def evaluate_gsm8k(
    model: nn.Module,
    tokenizer: Any,
    samples: List[Dict[str, Any]],
    device: str = "cuda",
    max_new_tokens: int = 96
) -> Dict[str, Any]:
    correct = 0
    total = len(samples)
    t0 = time.time()
    latencies = []

    for idx, item in enumerate(samples):
        q = item["question"]
        gt = extract_gsm8k_ground_truth(item["answer"])
        full_prompt = FEW_SHOT_GSM8K + f"Question: {q}\nAnswer:"

        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        t_sample = time.perf_counter()
        
        # Greedy decoding
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id
        )
        sample_lat = time.perf_counter() - t_sample
        latencies.append(sample_lat)

        gen_tokens = out[0, inputs.input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pred = extract_gsm8k_pred(gen_text)

        is_correct = (pred is not None and gt is not None and pred == gt)
        if is_correct:
            correct += 1

        if (idx + 1) % 25 == 0 or (idx + 1) == total:
            cur_acc = (correct / (idx + 1)) * 100.0
            print(f"  [GSM8k Progress] {idx+1}/{total} | Acc: {cur_acc:5.2f}% ({correct}/{idx+1}) | Lat: {sample_lat*1000:5.1f}ms", flush=True)

    dt = time.time() - t0
    acc = (correct / total) * 100.0 if total > 0 else 0.0
    stderr = math.sqrt(acc * (100.0 - acc) / total) if total > 0 else 0.0

    return {
        "benchmark": "GSM8k",
        "num_samples": total,
        "correct": correct,
        "accuracy_pct": round(acc, 2),
        "stderr_pct": round(stderr, 2),
        "total_time_sec": round(dt, 2),
        "avg_latency_ms": round(float(np.mean(latencies)) * 1000.0, 2)
    }


@torch.no_grad()
def evaluate_humaneval(
    model: nn.Module,
    tokenizer: Any,
    samples: List[Dict[str, Any]],
    device: str = "cuda",
    max_new_tokens: int = 128
) -> Dict[str, Any]:
    correct = 0
    total = len(samples)
    t0 = time.time()
    latencies = []

    for idx, item in enumerate(samples):
        prompt = item["prompt"]
        test_suite = item["test"]
        entry_point = item["entry_point"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t_sample = time.perf_counter()

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id
        )
        sample_lat = time.perf_counter() - t_sample
        latencies.append(sample_lat)

        gen_tokens = out[0, inputs.input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        cleaned_completion = clean_code_completion(prompt, gen_text)

        full_code = prompt + cleaned_completion + "\n" + test_suite + f"\ncheck({entry_point})\n"
        passed = execute_humaneval_code(full_code, timeout=3.0)

        if passed:
            correct += 1

        if (idx + 1) % 25 == 0 or (idx + 1) == total:
            cur_acc = (correct / (idx + 1)) * 100.0
            print(f"  [HumanEval Progress] {idx+1}/{total} | Pass@1: {cur_acc:5.2f}% ({correct}/{idx+1}) | Lat: {sample_lat*1000:5.1f}ms", flush=True)

    dt = time.time() - t0
    acc = (correct / total) * 100.0 if total > 0 else 0.0
    stderr = math.sqrt(acc * (100.0 - acc) / total) if total > 0 else 0.0

    return {
        "benchmark": "HumanEval",
        "num_samples": total,
        "correct": correct,
        "pass_at_1_pct": round(acc, 2),
        "stderr_pct": round(stderr, 2),
        "total_time_sec": round(dt, 2),
        "avg_latency_ms": round(float(np.mean(latencies)) * 1000.0, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Step Symbolic & Reasoning Evaluation on 1.1B Models")
    parser.add_argument("--num_gsm_samples", type=int, default=500, help="Number of GSM8k samples (default: 500)")
    parser.add_argument("--num_human_samples", type=int, default=None, help="Number of HumanEval samples (default: all 164)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_distill_1b/llama1b_kernel_deltanet_step_650401.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_file", type=str, default="results/symbolic_multi_step_reasoning_eval.json")
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print("=" * 90, flush=True)
    print("  KERNEL-DELTANET: EMPIRICAL EVALUATION ON MULTI-STEP REASONING & SYMBOLIC BINDING", flush=True)
    print(f"  Physical Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    print(f"  GSM8k Samples: {args.num_gsm_samples} | HumanEval Samples: {args.num_human_samples or 164} (Full)", flush=True)
    print("=" * 90, flush=True)

    # 1. Load Datasets
    gsm_samples = load_gsm8k_dataset(num_samples=args.num_gsm_samples)
    human_samples = load_humaneval_dataset(num_samples=args.num_human_samples)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results_master = {
        "metadata": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "foundation_backbone": model_name,
            "checkpoint": args.checkpoint,
            "dtype": str(dtype),
            "gsm_samples": len(gsm_samples),
            "humaneval_samples": len(human_samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "models": {}
    }

    # =========================================================================
    # Stage 1: Softmax Attention Teacher Baseline
    # =========================================================================
    print("\n" + "=" * 90, flush=True)
    print("  [Stage 1/3] Evaluating Teacher Baseline: TinyLlama-1.1B-Chat (Softmax Attention)", flush=True)
    print("=" * 90, flush=True)

    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device
    )
    teacher_model.eval()

    print("[*] Running GSM8k on Teacher Baseline...", flush=True)
    teacher_gsm = evaluate_gsm8k(teacher_model, tokenizer, gsm_samples, device=device)
    print(f"[+] Teacher Baseline GSM8k Acc: {teacher_gsm['accuracy_pct']}% ± {teacher_gsm['stderr_pct']}%\n", flush=True)

    print("[*] Running HumanEval on Teacher Baseline...", flush=True)
    teacher_human = evaluate_humaneval(teacher_model, tokenizer, human_samples, device=device)
    print(f"[+] Teacher Baseline HumanEval Pass@1: {teacher_human['pass_at_1_pct']}% ± {teacher_human['stderr_pct']}%\n", flush=True)

    results_master["models"]["Teacher_Softmax_1.1B"] = {
        "GSM8k": teacher_gsm,
        "HumanEval": teacher_human
    }

    del teacher_model
    gc.collect()
    torch.cuda.empty_cache()
    print("[+] Freed Teacher Model from VRAM.\n", flush=True)

    # Load trained checkpoint weights
    print(f"[*] Loading Trained Student Checkpoint: {args.checkpoint}...", flush=True)
    sd_data = torch.load(args.checkpoint, map_location=device)
    trainable_sd = sd_data.get("trainable_state_dict", sd_data)

    # =========================================================================
    # Stage 2: Pure Kernel-DeltaNet (22 Recurrent Layers)
    # =========================================================================
    print("=" * 90, flush=True)
    print("  [Stage 2/3] Evaluating Pure Kernel-DeltaNet-1.1B (22 Layers In-Register FWHT)", flush=True)
    print("=" * 90, flush=True)

    pure_model = construct_upcycled_1b_model(
        model_name_or_path=model_name,
        chunk_size=128,
        device=device,
        dtype=dtype,
        freeze_backbone=True,
        hybrid_attn_layers=None
    )
    pure_model.load_state_dict(trainable_sd, strict=False)
    pure_model.eval()

    print("[*] Running GSM8k on Pure Kernel-DeltaNet...", flush=True)
    pure_gsm = evaluate_gsm8k(pure_model, tokenizer, gsm_samples, device=device)
    print(f"[+] Pure Kernel-DeltaNet GSM8k Acc: {pure_gsm['accuracy_pct']}% ± {pure_gsm['stderr_pct']}%\n", flush=True)

    print("[*] Running HumanEval on Pure Kernel-DeltaNet...", flush=True)
    pure_human = evaluate_humaneval(pure_model, tokenizer, human_samples, device=device)
    print(f"[+] Pure Kernel-DeltaNet HumanEval Pass@1: {pure_human['pass_at_1_pct']}% ± {pure_human['stderr_pct']}%\n", flush=True)

    results_master["models"]["Pure_Kernel_DeltaNet_1.1B"] = {
        "GSM8k": pure_gsm,
        "HumanEval": pure_human
    }

    del pure_model
    gc.collect()
    torch.cuda.empty_cache()
    print("[+] Freed Pure Kernel-DeltaNet Model from VRAM.\n", flush=True)

    # =========================================================================
    # Stage 3: 20:2 Hybrid Gauge Topology (20 Recurrent + 2 Softmax Anchors)
    # =========================================================================
    print("=" * 90, flush=True)
    print("  [Stage 3/3] Evaluating 20:2 Hybrid Gauge Topology (Anchors at Layers [10, 21])", flush=True)
    print("=" * 90, flush=True)

    hybrid_model = construct_upcycled_1b_model(
        model_name_or_path=model_name,
        chunk_size=128,
        device=device,
        dtype=dtype,
        freeze_backbone=True,
        hybrid_attn_layers=[10, 21]
    )
    hybrid_model.load_state_dict(trainable_sd, strict=False)
    hybrid_model.eval()

    print("[*] Running GSM8k on 20:2 Hybrid Topology...", flush=True)
    hybrid_gsm = evaluate_gsm8k(hybrid_model, tokenizer, gsm_samples, device=device)
    print(f"[+] 20:2 Hybrid Topology GSM8k Acc: {hybrid_gsm['accuracy_pct']}% ± {hybrid_gsm['stderr_pct']}%\n", flush=True)

    print("[*] Running HumanEval on 20:2 Hybrid Topology...", flush=True)
    hybrid_human = evaluate_humaneval(hybrid_model, tokenizer, human_samples, device=device)
    print(f"[+] 20:2 Hybrid Topology HumanEval Pass@1: {hybrid_human['pass_at_1_pct']}% ± {hybrid_human['stderr_pct']}%\n", flush=True)

    results_master["models"]["Hybrid_Gauge_Topology_1.1B"] = {
        "GSM8k": hybrid_gsm,
        "HumanEval": hybrid_human
    }

    del hybrid_model
    gc.collect()
    torch.cuda.empty_cache()
    print("[+] Freed Hybrid Model from VRAM.\n", flush=True)

    # =========================================================================
    # SUMMARY & SAVING
    # =========================================================================
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_master, f, indent=2)

    print("=" * 90, flush=True)
    print("  EMPIRICAL EVALUATION SUMMARY (GSM8k & HumanEval)", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Model Architecture':<32} | {'GSM8k Acc (%)':<18} | {'HumanEval Pass@1 (%)':<22}", flush=True)
    print("-" * 90, flush=True)
    
    t_gsm = results_master["models"]["Teacher_Softmax_1.1B"]["GSM8k"]["accuracy_pct"]
    t_hum = results_master["models"]["Teacher_Softmax_1.1B"]["HumanEval"]["pass_at_1_pct"]
    p_gsm = results_master["models"]["Pure_Kernel_DeltaNet_1.1B"]["GSM8k"]["accuracy_pct"]
    p_hum = results_master["models"]["Pure_Kernel_DeltaNet_1.1B"]["HumanEval"]["pass_at_1_pct"]
    h_gsm = results_master["models"]["Hybrid_Gauge_Topology_1.1B"]["GSM8k"]["accuracy_pct"]
    h_hum = results_master["models"]["Hybrid_Gauge_Topology_1.1B"]["HumanEval"]["pass_at_1_pct"]

    print(f"{'TinyLlama-1.1B Teacher (Softmax)':<32} | {t_gsm:>6.2f}% ± {results_master['models']['Teacher_Softmax_1.1B']['GSM8k']['stderr_pct']:<5.2f}% | {t_hum:>7.2f}% ± {results_master['models']['Teacher_Softmax_1.1B']['HumanEval']['stderr_pct']:<5.2f}%", flush=True)
    print(f"{'Pure Kernel-DeltaNet (22:0)':<32} | {p_gsm:>6.2f}% ± {results_master['models']['Pure_Kernel_DeltaNet_1.1B']['GSM8k']['stderr_pct']:<5.2f}% | {p_hum:>7.2f}% ± {results_master['models']['Pure_Kernel_DeltaNet_1.1B']['HumanEval']['stderr_pct']:<5.2f}%", flush=True)
    print(f"{'Hybrid Gauge Topology (20:2)':<32} | {h_gsm:>6.2f}% ± {results_master['models']['Hybrid_Gauge_Topology_1.1B']['GSM8k']['stderr_pct']:<5.2f}% | {h_hum:>7.2f}% ± {results_master['models']['Hybrid_Gauge_Topology_1.1B']['HumanEval']['stderr_pct']:<5.2f}%", flush=True)
    print("=" * 90, flush=True)
    print(f"[+] Results saved to: {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
