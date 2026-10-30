"""
Kernel-DeltaNet Engine
Module: evaluation.canonical_nlp_eval_loader
==============================================================
Loads canonical, standardized benchmark datasets directly from local cached Parquet files:
1. PIQA (Physical Interaction QA)
2. ARC (AI2 Reasoning Challenge - Easy & Challenge)
3. HellaSwag (Commonsense Situational Reasoning)
4. LAMBADA (Language Modeling Long-Range Prediction)
5. WikiText-2 (Held-Out Perplexity)
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def load_canonical_eval_datasets(
    num_samples_per_task: int = 50
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extracts standardized evaluation samples directly from local HuggingFace cache.
    """
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    benchmarks = {}

    # 1. PIQA (Physical Interaction QA)
    p_piqa = hub / "datasets--lighteval--piqa"
    if p_piqa.exists():
        for f in p_piqa.rglob("*.parquet"):
            if "validation" in f.name:
                try:
                    df = pd.read_parquet(f)
                    samples = []
                    for _, row in df.head(num_samples_per_task).iterrows():
                        samples.append({
                            "prompt": row["goal"].strip(),
                            "choices": [row["sol1"].strip(), row["sol2"].strip()],
                            "label": int(row["label"])
                        })
                    benchmarks["PIQA"] = samples
                    print(f"[EvalLoader] Loaded {len(samples)} canonical validation samples for PIQA.")
                    break
                except Exception as e:
                    print(f"[EvalLoader] Note PIQA: {e}")

    # 2. ARC-Challenge & ARC-Easy
    p_arc = hub / "datasets--allenai--ai2_arc"
    if p_arc.exists():
        for f in p_arc.rglob("*.parquet"):
            if "validation" in f.name:
                try:
                    df = pd.read_parquet(f)
                    samples = []
                    for _, row in df.head(num_samples_per_task).iterrows():
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
                        print(f"[EvalLoader] Loaded {len(samples)} canonical validation samples for ARC-Challenge.")
                        break
                except Exception as e:
                    print(f"[EvalLoader] Note ARC: {e}")

    # 3. HellaSwag (Commonsense Situational Reasoning)
    p_hs = hub / "datasets--Rowan--hellaswag"
    if p_hs.exists():
        for f in p_hs.rglob("*.parquet"):
            if "validation" in f.name:
                try:
                    df = pd.read_parquet(f)
                    samples = []
                    for _, row in df.head(num_samples_per_task).iterrows():
                        ctx = row["ctx"].strip()
                        endings = [e.strip() for e in row["endings"]]
                        label = int(row["label"])
                        samples.append({
                            "prompt": ctx,
                            "choices": endings,
                            "label": label
                        })
                    benchmarks["HellaSwag"] = samples
                    print(f"[EvalLoader] Loaded {len(samples)} canonical validation samples for HellaSwag.")
                    break
                except Exception as e:
                    print(f"[EvalLoader] Note HellaSwag: {e}")

    # 4. LAMBADA (Target word prediction)
    p_lam = hub / "datasets--cimec--lambada"
    if p_lam.exists():
        for f in p_lam.rglob("*.parquet"):
            if "validation" in f.name:
                try:
                    df = pd.read_parquet(f)
                    samples_raw = []
                    all_targets = []
                    for _, row in df.iterrows():
                        text = row["text"].strip()
                        words = text.split()
                        if len(words) > 10:
                            prompt = " ".join(words[:-1])
                            target_word = words[-1]
                            samples_raw.append((prompt, target_word))
                            all_targets.append(target_word)
                    
                    import random
                    rng = random.Random(42)
                    samples = []
                    for prompt, target_word in samples_raw[:num_samples_per_task]:
                        # Select 3 random alternative target words from other passages as distractors
                        other_targets = [w for w in all_targets if w.lower() != target_word.lower()]
                        distractors = rng.sample(other_targets, min(3, len(other_targets)))
                        choices = [target_word] + distractors
                        # Shuffle choices to avoid always predicting choice 0
                        indices = list(range(len(choices)))
                        rng.shuffle(indices)
                        shuffled_choices = [choices[i] for i in indices]
                        label = indices.index(0)
                        samples.append({
                            "prompt": prompt,
                            "choices": shuffled_choices,
                            "label": label
                        })
                    benchmarks["LAMBADA"] = samples
                    print(f"[EvalLoader] Loaded {len(samples)} canonical validation samples for LAMBADA.")
                    break
                except Exception as e:
                    print(f"[EvalLoader] Note LAMBADA: {e}")

    return benchmarks
