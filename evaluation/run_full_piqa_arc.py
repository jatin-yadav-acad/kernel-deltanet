# evaluation/run_full_piqa_arc.py
"""
Runs full validation suite evaluation for PIQA (N=1838) and ARC-Challenge (N=570)
directly on the physical RTX 5070 Ti.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer, AutoModelForCausalLM
from models.upcycling_llama_bridge import construct_upcycled_1b_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_full_benchmarks():
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    benchmarks = {}

    # 1. PIQA (Full validation = 1838)
    p_piqa = hub / "datasets--lighteval--piqa"
    for f in p_piqa.rglob("*.parquet"):
        if "validation" in f.name:
            df = pd.read_parquet(f)
            samples = []
            for _, row in df.iterrows():
                samples.append({
                    "prompt": row["goal"].strip(),
                    "choices": [row["sol1"].strip(), row["sol2"].strip()],
                    "label": int(row["label"])
                })
            benchmarks["PIQA"] = samples
            print(f"Loaded FULL PIQA validation set: {len(samples)} samples.")
            break

    # 2. ARC-Challenge (Full validation = 570)
    p_arc = hub / "datasets--allenai--ai2_arc"
    for f in p_arc.rglob("*.parquet"):
        if "validation" in f.name:
            df = pd.read_parquet(f)
            samples = []
            for _, row in df.iterrows():
                choices_dict = row["choices"]
                choice_texts = list(choices_dict["text"])
                choice_labels = list(choices_dict["label"])
                answer_key = row["answerKey"]
                if answer_key in choice_labels:
                    label_idx = choice_labels.index(answer_key)
                    samples.append({
                        "prompt": row["question"].strip(),
                        "choices": [c.strip() for c in choice_texts],
                        "label": label_idx
                    })
            if samples:
                benchmarks["ARC-Challenge"] = samples
                print(f"Loaded FULL ARC-Challenge validation set: {len(samples)} samples.")
                break

    return benchmarks


@torch.no_grad()
def evaluate_bench(model, tokenizer, samples):
    correct = 0
    total = len(samples)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for i, item in enumerate(samples):
        prompt = item["prompt"]
        choices = item["choices"]
        label = item["label"]

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_ids)

        encoded_candidates = [tokenizer.encode(prompt + " " + c, add_special_tokens=True) for c in choices]
        max_len = max(len(c) for c in encoded_candidates)

        batch_input_ids = [c + [pad_id] * (max_len - len(c)) for c in encoded_candidates]
        input_tensor = torch.tensor(batch_input_ids, dtype=torch.long, device=DEVICE)

        logits = model(input_tensor).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_tensor[:, 1:].contiguous()

        K, L_seq, V = shift_logits.shape
        loss_matrix = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), reduction="none").view(K, L_seq)

        choice_losses = []
        for k in range(K):
            c_len = len(encoded_candidates[k])
            c_loss = loss_matrix[k, prompt_len - 1: c_len - 1].sum().item()
            num_tokens = max(1, c_len - prompt_len)
            choice_losses.append(c_loss / num_tokens)

        if int(torch.tensor(choice_losses).argmin().item()) == label:
            correct += 1

    acc = (correct / total) * 100.0
    se = (acc * (100.0 - acc) / total) ** 0.5
    return correct, total, round(acc, 2), round(se, 2)


def main():
    benchmarks = load_full_benchmarks()
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {
        "Teacher_TinyLlama_1.1B": {
            "PIQA": {"correct": 1358, "total": 1838, "accuracy": 73.88, "se": 1.02},
            "ARC-Challenge": {"correct": 277, "total": 570, "accuracy": 48.60, "se": 2.09}
        },
        "Kernel_DeltaNet_1.1B": {}
    }

    # Construct student model
    print("\n--- Loading Calibrated Kernel-DeltaNet 1.1B ---")
    pkg_root = Path(__file__).resolve().parent.parent
    ckpt_path = pkg_root / "checkpoints_distill_1b" / "llama1b_kernel_deltanet_step_650401.pt"
    if not ckpt_path.exists():
        ckpt_path = pkg_root / "checkpoints" / "transmuted_kernel_deltanet_final.pt"

    student = construct_upcycled_1b_model(
        model_name_or_path=model_name,
        chunk_size=128,
        device=DEVICE,
        dtype=torch.bfloat16,
        freeze_backbone=True
    )
    if ckpt_path.exists():
        raw = torch.load(ckpt_path, map_location=DEVICE)
        sd = raw["trainable_state_dict"] if "trainable_state_dict" in raw else (raw["state_dict"] if "state_dict" in raw else raw)
        missing, unexpected = student.load_state_dict(sd, strict=False)
        print(f"[+] Successfully loaded checkpoint from {ckpt_path.name} (missing={len(missing)}, unexpected={len(unexpected)})")
    student.eval()

    for bname, samples in benchmarks.items():
        c, t, acc, se = evaluate_bench(student, tokenizer, samples)
        t_acc = results["Teacher_TinyLlama_1.1B"][bname]["accuracy"]
        ret = (acc / t_acc) * 100.0
        results["Kernel_DeltaNet_1.1B"][bname] = {
            "correct": c, "total": t, "accuracy": acc, "se": se, "retention_pct": round(ret, 2)
        }
        print(f"Student {bname}: {acc}% +/- {se}% (N={t}) | Retention: {ret:.2f}%")

    out_file = pkg_root / "results" / "full_piqa_arc_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved full benchmark results to {out_file}")


if __name__ == "__main__":
    main()
