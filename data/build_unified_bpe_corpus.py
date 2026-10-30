"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: data.build_unified_bpe_corpus
========================================================
Builds a unified, high-density GPT-2 BPE token corpus combining:
- FineWeb-Edu pre-tokenized cache
- Natural literary prose from PG-19, Sherlock Holmes, and Time Machine
Splits into Train and Val sets for genuine language modeling pretraining.
"""

import os
import torch
from transformers import AutoTokenizer


def build_unified_corpus(save_dir: str = "data"):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_tokens = []
    val_tokens = []

    # 1. Load FineWeb-Edu tokens
    fw_train = os.path.join(save_dir, "fineweb_cache", "fineweb_train.pt")
    fw_val = os.path.join(save_dir, "fineweb_cache", "fineweb_val.pt")
    if os.path.exists(fw_train):
        t = torch.load(fw_train, weights_only=True)
        train_tokens.extend(t.tolist())
        print(f"[*] Loaded {len(t):,} tokens from FineWeb-Edu train cache.")
    if os.path.exists(fw_val):
        v = torch.load(fw_val, weights_only=True)
        val_tokens.extend(v.tolist())
        print(f"[*] Loaded {len(v):,} tokens from FineWeb-Edu val cache.")

    # 2. Tokenize local natural English text files
    text_files = ["pg19_sample.txt", "sherlock.txt", "time_machine.txt"]
    for fname in text_files:
        path = os.path.join(save_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            tokens = tokenizer.encode(text)
            split_idx = int(len(tokens) * 0.90)
            train_tokens.extend(tokens[:split_idx])
            val_tokens.extend(tokens[split_idx:])
            print(f"[*] Tokenized {fname}: {len(tokens):,} tokens ({split_idx:,} train / {len(tokens)-split_idx:,} val)")

    train_tensor = torch.tensor(train_tokens, dtype=torch.long)
    val_tensor = torch.tensor(val_tokens, dtype=torch.long)

    train_out = os.path.join(save_dir, "bpe_train_tokens.pt")
    val_out = os.path.join(save_dir, "bpe_val_tokens.pt")

    torch.save(train_tensor, train_out)
    torch.save(val_tensor, val_out)

    print("\n" + "=" * 70)
    print(f"[+] Unified BPE Corpus Created:")
    print(f"    Train Tokens: {train_tensor.numel():,} -> {train_out}")
    print(f"    Val Tokens:   {val_tensor.numel():,} -> {val_out}")
    print(f"    Total Tokens: {train_tensor.numel() + val_tensor.numel():,}")
    print("=" * 70)
    return train_tensor, val_tensor


if __name__ == "__main__":
    build_unified_corpus()
