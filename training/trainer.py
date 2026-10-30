"""
Kernel-DeltaNet Engine
Module: training.trainer
=========================================
Production-grade Trainer for Causal Language Modeling.
"""

import os
import json
import time
from typing import Dict, List, Optional, Callable, Tuple
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from .optimizer_utils import configure_optimizers, CosineWarmupLRScheduler


class Trainer:
    """
    Handles training, evaluation, validation, checkpointing, and JSON logging.
    """
    def __init__(
        self,
        model: nn.Module,
        train_dataloader_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
        val_dataloader_fn: Optional[Callable[[], Tuple[torch.Tensor, torch.Tensor]]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        max_steps: int = 200,
        warmup_steps: int = 20,
        eval_interval: int = 50,
        grad_clip: float = 1.0,
        mixed_precision: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: str = "results"
    ):
        self.model = model.to(device)
        self.train_fn = train_dataloader_fn
        self.val_fn = val_dataloader_fn
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision and (device == "cuda")
        self.device = device
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.optimizer = configure_optimizers(model, weight_decay=weight_decay, learning_rate=learning_rate)
        self.scheduler = CosineWarmupLRScheduler(self.optimizer, warmup_steps=warmup_steps, max_steps=max_steps)
        self.scaler = GradScaler(enabled=self.mixed_precision)

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.step_latencies: List[float] = []

    def train(self) -> Dict:
        """
        Executes complete training loop.
        """
        print(f"[*] Starting Training on {self.device} (Max Steps: {self.max_steps}, Mixed Precision: {self.mixed_precision})")
        self.model.train()
        t0 = time.time()

        for step in range(1, self.max_steps + 1):
            t_step_start = time.perf_counter()
            inputs, targets = self.train_fn()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.mixed_precision):
                logits, loss = self.model(inputs, targets=targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            if self.device == "cuda":
                torch.cuda.synchronize()

            t_step = (time.perf_counter() - t_step_start) * 1000.0 # ms
            self.step_latencies.append(t_step)
            self.train_losses.append(loss.item())

            if step % self.eval_interval == 0 or step == self.max_steps:
                val_loss = self.evaluate() if self.val_fn is not None else loss.item()
                self.val_losses.append(val_loss)
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Step {step:04d}/{self.max_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Latency: {t_step:.1f}ms | LR: {lr:.2e}")

        total_time = time.time() - t0
        summary = {
            "max_steps": self.max_steps,
            "total_time_seconds": float(total_time),
            "mean_step_ms": float(np.mean(self.step_latencies)),
            "final_train_loss": float(self.train_losses[-1]),
            "final_val_loss": float(self.val_losses[-1]) if self.val_losses else float(self.train_losses[-1]),
            "train_loss_history": self.train_losses[::max(1, len(self.train_losses)//20)]
        }

        log_path = os.path.join(self.log_dir, "trainer_summary.json")
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[+] Training Complete in {total_time:.2f}s. Summary logged to {log_path}")
        return summary

    @torch.no_grad()
    def evaluate(self, num_eval_batches: int = 5) -> float:
        """
        Runs validation evaluation.
        """
        self.model.eval()
        losses = []
        for _ in range(num_eval_batches):
            inputs, targets = self.val_fn()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with autocast(enabled=self.mixed_precision):
                _, loss = self.model(inputs, targets=targets)
            losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses))
