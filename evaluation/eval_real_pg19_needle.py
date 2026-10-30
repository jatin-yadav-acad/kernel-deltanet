"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: evaluation.eval_real_pg19_needle
==========================================================
Real-Document Long-Context Needle Retrieval (RULER / PG-19 Standard).
Evaluates retrieval of a factual needle planted inside natural English prose:
- Haystack: 191k tokens of natural literary English from PG-19
- Factual Needle: "The secret access code to the vault hidden in the cellar is 948291."
- Query: "\nQuestion: What is the secret access code to the vault hidden in the cellar?\nAnswer: The secret access code is "
- Evaluates Context Horizons: T in [2048, 4096, 8192, 16384, 32768]
- Evaluates Needle Depths: 10%, 30%, 50%, 70%, 90%
- Models: Kernel-DeltaNet vs Linear DeltaNet vs Standard Attention
"""

import os
import gc
import json
import math
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.kernel_deltanet_lm import KernelDeltaNetLM
from baselines.ttt_linear_deltanet import TTTLinearLM as LinearDeltaNetLM
from baselines.standard_transformer import StandardTransformerLM


def load_pg19_tokens(path: str = "data/pg19_sample.txt") -> torch.Tensor:
    """Loads and tokenizes natural PG-19 text."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    return torch.tensor(tokens, dtype=torch.long)


def build_pg19_needle_context(
    corpus_tokens: torch.Tensor,
    seq_len: int,
    depth_pct: float,
    tokenizer: AutoTokenizer
) -> Tuple[torch.Tensor, List[int], int]:
    """
    Constructs a context of length seq_len from natural PG-19 prose,
    inserting the factual needle at depth_pct and the query at the end.
    """
    needle_text = " The secret access code to the vault hidden in the cellar is 948291."
    query_text = "\nQuestion: What is the secret access code to the vault hidden in the cellar?\nAnswer: The secret access code is"
    
    needle_ids = tokenizer.encode(needle_text)
    query_ids = tokenizer.encode(query_text)
    answer_ids = tokenizer.encode(" 948291")

    # Target sequence length minus needle and query
    haystack_len = seq_len - len(needle_ids) - len(query_ids)
    assert haystack_len > 0, f"seq_len {seq_len} too short for prompt"

    # Slice a deterministic portion of the book
    start_offset = 1000
    haystack = corpus_tokens[start_offset : start_offset + haystack_len].tolist()

    insert_idx = int(haystack_len * (depth_pct / 100.0))
    full_tokens = haystack[:insert_idx] + needle_ids + haystack[insert_idx:] + query_ids
    
    # Ensure exact length
    full_tokens = full_tokens[:seq_len]
    return torch.tensor([full_tokens], dtype=torch.long), answer_ids, len(full_tokens) - 1


def evaluate_needle_response(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_tokens: List[int],
    device: str = "cuda"
) -> Dict:
    """
    Evaluates model generation/readout on the target needle tokens.
    """
    model.eval()
    input_ids = input_ids.to(device)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(input_ids)
        last_logits = logits[0, -1, :] # (vocab_size,)
        
        target_token = target_tokens[0]
        probs = F.softmax(last_logits, dim=-1)
        target_prob = float(probs[target_token].item())
        top1_token = int(torch.argmax(last_logits).item())
        
        # Calculate SNR in dB
        target_logit = float(last_logits[target_token].item())
        mean_logit = float(last_logits.mean().item())
        std_logit = float(last_logits.std().item() + 1e-6)
        snr_db = float(10.0 * math.log10(max(1e-4, (target_logit - mean_logit) / std_logit + 1e-3)))

        is_recalled = (top1_token == target_token)

    return {
        "is_recalled": is_recalled,
        "target_prob": target_prob,
        "top1_token": top1_token,
        "target_token": target_token,
        "snr_db": round(snr_db, 2)
    }


def run_pg19_needle_benchmark(
    sequence_lengths: List[int] = [2048, 4096, 8192, 16384, 32768],
    depths: List[int] = [10, 30, 50, 70, 90],
    d_model: int = 384,
    num_layers: int = 6,
    num_heads: int = 6,
    rkhs_factor: int = 4,
    chunk_size: int = 128,
    output_path: str = "results/pg19_needle_benchmark.json"
) -> Dict:
    """
    Executes the PG-19 Natural Narrative Needle Retrieval Benchmark on RTX 5070 Ti.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 95)
    print("KERNEL-DELTANET: REAL-DOCUMENT LONG-CONTEXT NEEDLE RETRIEVAL (PG-19 CORPUS)")
    print("=" * 95)
    print(f"[*] Hardware Silicon: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[*] Natural Text: PG-19 Pride and Prejudice Prose | Tokenizer: GPT-2 BPE")
    print(f"[*] Tested Sequence Horizons: {sequence_lengths}")
    print(f"[*] Tested Needle Depths: {depths}%")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    corpus_tokens = load_pg19_tokens()
    vocab_size = tokenizer.vocab_size

    models = {
        "Kernel-DeltaNet (Ours)": KernelDeltaNetLM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            rkhs_factor=rkhs_factor,
            chunk_size=chunk_size
        ).to(device),
        "Linear DeltaNet": LinearDeltaNetLM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            chunk_size=chunk_size
        ).to(device),
        "Standard Attention": StandardTransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads
        ).to(device)
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n[*] Evaluating Architecture: {model_name}...")
        results[model_name] = {}

        for T in sequence_lengths:
            t_key = f"T_{T}"
            results[model_name][t_key] = {}
            print(f"  -> Testing Context Horizon T = {T:,} tokens...")

            for depth in depths:
                depth_key = f"depth_{depth}%"
                input_ids, target_ids, _ = build_pg19_needle_context(
                    corpus_tokens=corpus_tokens,
                    seq_len=T,
                    depth_pct=depth,
                    tokenizer=tokenizer
                )

                try:
                    eval_res = evaluate_needle_response(model, input_ids, target_ids, device=device)
                    results[model_name][t_key][depth_key] = eval_res
                    rec_str = "RECALLED" if eval_res["is_recalled"] else "MISSED"
                    print(f"     Depth {depth:2d}%: [{rec_str}] | Target Prob: {eval_res['target_prob']:.2e} | SNR: {eval_res['snr_db']:+5.1f} dB")
                except torch.cuda.OutOfMemoryError:
                    print(f"     Depth {depth:2d}%: [!] OOM")
                    results[model_name][t_key][depth_key] = {
                        "is_recalled": False, "target_prob": 0.0, "snr_db": -99.9, "status": "OOM"
                    }
                    torch.cuda.empty_cache()
                    gc.collect()

    master_needle_report = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "corpus": "PG-19 Natural English Prose",
        "sequence_lengths": sequence_lengths,
        "depths": depths,
        "needle": "The secret access code to the vault hidden in the cellar is 948291.",
        "results": results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(master_needle_report, f, indent=2)

    print(f"\n[+] PG-19 Natural Needle benchmark logged to {output_path}")
    return master_needle_report


if __name__ == "__main__":
    run_pg19_needle_benchmark()
