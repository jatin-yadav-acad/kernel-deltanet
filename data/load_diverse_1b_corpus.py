"""
Kernel-DeltaNet Engine
Module: data.load_diverse_1b_corpus
======================================================
Builds a diverse, high-entropy token corpus for 1B foundation model distillation.
Combines:
1. Alpaca instruction-tuning dialogues (~52,000 tasks)
2. WikiText encyclopedic prose (~6.2 MB)
3. Classic literature (PG-19, Sherlock Holmes, Time Machine)
"""

import os
from pathlib import Path
from typing import List
import pandas as pd
import torch
from transformers import AutoTokenizer


def build_diverse_training_stream(
    tokenizer: AutoTokenizer,
    seq_len: int = 512,
    max_tokens: int = 600_000
) -> List[torch.Tensor]:
    """
    Constructs a high-entropy, diverse training corpus for foundation model distillation.
    """
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    text_corpus = []

    # 1. Alpaca instruction dataset
    p_alpaca = hub / "datasets--tatsu-lab--alpaca"
    if p_alpaca.exists():
        for f in p_alpaca.rglob("*.parquet"):
            if "train" in f.name:
                try:
                    df = pd.read_parquet(f)
                    # Sample instruction-response pairs
                    sample_texts = df["text"].dropna().sample(n=min(len(df), 2500), random_state=42).tolist()
                    text_corpus.extend(sample_texts)
                    print(f"[Corpus] Loaded {len(sample_texts)} instruction dialogues from Alpaca.")
                    break
                except Exception as e:
                    print(f"[Corpus] Note loading Alpaca: {e}")

    # 2. WikiText encyclopedic prose
    p_wikitext = hub / "datasets--wikitext"
    if p_wikitext.exists():
        for f in p_wikitext.rglob("*.parquet"):
            if "train" in f.name:
                try:
                    df = pd.read_parquet(f)
                    wiki_texts = [t for t in df["text"].dropna().tolist() if len(t.strip()) > 100]
                    sample_wiki = wiki_texts[:1500]
                    text_corpus.extend(sample_wiki)
                    print(f"[Corpus] Loaded {len(sample_wiki)} article paragraphs from WikiText.")
                    break
                except Exception as e:
                    print(f"[Corpus] Note loading WikiText: {e}")

    # 3. Classic Literature
    data_dir = Path(__file__).resolve().parent
    for fname in ["pg19_sample.txt", "sherlock.txt", "time_machine.txt"]:
        fpath = data_dir / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                text_corpus.append(f.read())

    # Concatenate with document boundary separators
    full_text = "\n\n<|endoftext|>\n\n".join(text_corpus)
    print(f"[Corpus] Total raw characters collected: {len(full_text):,}")

    # Tokenize in chunks
    token_ids = tokenizer(full_text, return_tensors="pt", truncation=False).input_ids[0]
    total_tokens = len(token_ids)
    print(f"[Corpus] Total tokens generated with LLaMA tokenizer: {total_tokens:,}")

    # Slice into uniform batches
    batches = []
    num_batches = min(total_tokens // seq_len, max_tokens // seq_len)
    for i in range(num_batches):
        batch = token_ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0)  # (1, seq_len)
        batches.append(batch)

    print(f"[Corpus] Created {len(batches)} batches of length {seq_len} ({len(batches) * seq_len:,} total tokens).")
    return batches
