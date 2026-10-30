"""
Kernel-DeltaNet Engine
Module: data.prepare_fineweb_edu
==================================================
Streams and pre-tokenizes a clean slice of HuggingFaceFW/fineweb-edu
into memory-mapped / torch tensors for zero-overhead GPU pretraining.
"""

import os
import sys
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_fineweb_slice(
    target_train_tokens: int = 200_000,
    target_val_tokens: int = 50_000,
    save_dir: str = "data/fineweb_cache"
):
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "fineweb_train.pt")
    val_path = os.path.join(save_dir, "fineweb_val.pt")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"[*] Pre-tokenized FineWeb-Edu cache already exists at {save_dir}")
        train_tokens = torch.load(train_path, weights_only=True)
        val_tokens = torch.load(val_path, weights_only=True)
        print(f"    Train tokens: {train_tokens.numel():,}, Val tokens: {val_tokens.numel():,}")
        return train_tokens, val_tokens

    print(f"[*] Initializing FineWeb-Edu stream from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    all_tokens = []
    total_needed = target_train_tokens + target_val_tokens
    t0 = time.time()
    doc_count = 0

    print(f"[*] Streaming and tokenizing {total_needed:,} tokens from FineWeb-Edu...")
    for item in ds:
        text = item.get("text", "")
        if not text.strip():
            continue
        ids = tokenizer.encode(text)
        all_tokens.extend(ids)
        doc_count += 1
        if len(all_tokens) % 25000 < len(ids):
            print(f"    Progress: {len(all_tokens):,}/{total_needed:,} tokens ({doc_count} docs) in {time.time()-t0:.1f}s")
        if len(all_tokens) >= total_needed:
            break

    dt = time.time() - t0
    print(f"[+] Tokenized {len(all_tokens):,} tokens in {dt:.1f}s ({len(all_tokens)/dt:.1f} tok/s) across {doc_count} docs.")

    tokens_tensor = torch.tensor(all_tokens[:total_needed], dtype=torch.long)
    train_tokens = tokens_tensor[:target_train_tokens]
    val_tokens = tokens_tensor[target_train_tokens:]

    torch.save(train_tokens, train_path)
    torch.save(val_tokens, val_path)
    print(f"[+] Successfully saved {train_tokens.numel():,} train tokens -> {train_path}")
    print(f"[+] Successfully saved {val_tokens.numel():,} val tokens -> {val_path}")

    return train_tokens, val_tokens


if __name__ == "__main__":
    prepare_fineweb_slice()
