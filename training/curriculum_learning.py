r"""
Kernel-DeltaNet Engine
Module: training.curriculum_learning
======================================================
Progressive Context-Length Curriculum Trainer.
Gradually expands sequence length: 512 -> 1024 -> 2048 -> 4096 tokens
while dynamically maintaining constant batch token volume B * T = const,
maximizing Tensor Core utilization and preventing early divergence.
"""

import time
from typing import Dict, List, Tuple, Callable
import torch
import torch.nn as nn
import numpy as np

from .optimizer_utils import configure_optimizers, CosineWarmupLRScheduler


class ProgressiveContextCurriculum:
    r"""
    Manages progressive schedule of context lengths and batch sizes.
    """
    def __init__(
        self,
        stages: List[Dict[str, int]] = None,
        target_tokens_per_batch: int = 16384
    ):
        if stages is None:
            self.stages = [
                {"seq_len": 512,  "batch_size": 32, "steps": 50},
                {"seq_len": 1024, "batch_size": 16, "steps": 50},
                {"seq_len": 2048, "batch_size": 8,  "steps": 50},
                {"seq_len": 4096, "batch_size": 4,  "steps": 50}
            ]
        else:
            self.stages = stages
        self.target_tokens = target_tokens_per_batch

    def get_stage_for_step(self, global_step: int) -> Tuple[int, int]:
        cum_steps = 0
        for s in self.stages:
            cum_steps += s["steps"]
            if global_step <= cum_steps:
                return s["seq_len"], s["batch_size"]
        last = self.stages[-1]
        return last["seq_len"], last["batch_size"]


def run_curriculum_training(
    model: nn.Module,
    total_steps: int = 100,
    vocab_size: int = 1024,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, List[float]]:
    r"""
    Runs progressive curriculum training across multi-stage context expansions.
    """
    model = model.to(device)
    model.train()
    optimizer = configure_optimizers(model, weight_decay=0.01, learning_rate=1e-3)
    scheduler = CosineWarmupLRScheduler(optimizer, warmup_steps=10, max_steps=total_steps)

    curriculum = ProgressiveContextCurriculum([
        {"seq_len": 256, "batch_size": 16, "steps": 30},
        {"seq_len": 512, "batch_size": 8,  "steps": 40},
        {"seq_len": 1024, "batch_size": 4, "steps": 30}
    ])

    losses = []
    stage_markers = []

    print(f"[*] Starting Progressive Curriculum Training on {device} ({total_steps} steps)")
    t0 = time.time()

    for step in range(1, total_steps + 1):
        seq_len, batch_size = curriculum.get_stage_for_step(step)

        # Generate batch on the fly
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.roll(inputs, shifts=-1, dims=1)

        optimizer.zero_grad()
        logits, loss = model(inputs, targets=targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % 25 == 0 or step == total_steps:
            elapsed = time.time() - t0
            print(f"  Step {step:03d}/{total_steps} | Stage: T={seq_len:>4}, B={batch_size:>2} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.2f}s")
            stage_markers.append((step, seq_len, loss.item()))

    print(f"[+] Curriculum Training Completed in {time.time() - t0:.2f}s!\n")
    return {
        "losses": losses,
        "stage_markers": stage_markers
    }
