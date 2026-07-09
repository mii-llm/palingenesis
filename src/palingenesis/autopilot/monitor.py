"""Model behavior monitor: tracks collapse, forgetting, and distribution shifts.

This is the brain of autopilot -- it watches the model during training and
detects problems BEFORE they become catastrophic:

  - KL divergence from base model (forgetting signal)
  - Output entropy (collapse = entropy -> 0)
  - Weight drift per layer (which layers are changing most)
  - Loss spike detection (gradient explosion precursor)
  - Validation loss trend (overfitting/underfitting)

The monitor makes GO/STOP/ADJUST decisions that autopilot acts on.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class Signal(Enum):
    """Autopilot control signals."""

    HEALTHY = "healthy"  # All good, continue
    REDUCE_LR = "reduce_lr"  # Loss spike or oscillation detected
    EARLY_STOP = "early_stop"  # Val loss plateaued or model collapsed
    COLLAPSE = "collapse"  # Entropy/rank collapse -- abort this config


@dataclass(slots=True)
class ModelSnapshot:
    """Frozen statistics of model at a point in time."""

    weight_norms: dict[str, float] = field(default_factory=dict)
    output_entropy: float = 0.0
    val_loss: float = float("inf")
    train_loss: float = float("inf")
    step: int = 0


class BehaviorMonitor:
    """Watches model behavior and emits control signals.

    Tracks:
      - Train loss: moving average + spike detection
      - Val loss: trend detection (improving, plateau, degrading)
      - Weight drift: how far from init (forgetting proxy)
      - Output entropy: collapse detection

    Emits Signal when intervention is needed.
    """

    def __init__(
        self,
        patience: int = 5,  # val evals without improvement before stopping
        spike_threshold: float = 3.0,  # loss spike = 3x rolling avg
        drift_threshold: float = 0.3,  # 30% weight drift = forgetting concern
        entropy_floor: float = 1.0,  # below this = collapse
    ):
        self.patience = patience
        self.spike_threshold = spike_threshold
        self.drift_threshold = drift_threshold
        self.entropy_floor = entropy_floor

        # State
        self._train_losses: deque[float] = deque(maxlen=100)
        self._val_losses: list[float] = []
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._init_norms: dict[str, float] | None = None
        self._spike_count = 0

    def record_train_loss(self, loss: float) -> Signal:
        """Record a training loss value. Returns signal if spike detected."""
        if not math.isfinite(loss):
            self._spike_count += 1
            if self._spike_count >= 3:
                return Signal.COLLAPSE
            return Signal.REDUCE_LR

        self._train_losses.append(loss)

        # Spike detection: current loss > 3x rolling average
        if len(self._train_losses) > 20:
            avg = sum(list(self._train_losses)[-20:]) / 20
            if loss > self.spike_threshold * avg:
                self._spike_count += 1
                if self._spike_count >= 5:
                    return Signal.REDUCE_LR
        else:
            self._spike_count = max(0, self._spike_count - 1)

        return Signal.HEALTHY

    def record_val_loss(self, val_loss: float) -> Signal:
        """Record a validation loss. Returns signal for training control."""
        self._val_losses.append(val_loss)

        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._patience_counter = 0
            return Signal.HEALTHY
        else:
            self._patience_counter += 1
            if self._patience_counter >= self.patience:
                return Signal.EARLY_STOP
            return Signal.HEALTHY

    @torch.no_grad()
    def check_weight_drift(self, model: nn.Module) -> tuple[float, Signal]:
        """Check how far model has drifted from initialization.

        Returns (mean_drift, signal).
        """
        current_norms = {}
        for name, p in model.named_parameters():
            if p.ndim >= 2 and p.numel() > 1000:
                current_norms[name] = p.data.float().norm().item()

        if self._init_norms is None:
            self._init_norms = current_norms.copy()
            return 0.0, Signal.HEALTHY

        drifts = []
        for name, init_norm in self._init_norms.items():
            if name in current_norms and init_norm > 0:
                drift = abs(current_norms[name] - init_norm) / init_norm
                drifts.append(drift)

        mean_drift = sum(drifts) / max(len(drifts), 1)

        if mean_drift > self.drift_threshold:
            return mean_drift, Signal.REDUCE_LR
        return mean_drift, Signal.HEALTHY

    @torch.no_grad()
    def check_output_entropy(
        self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[float, Signal]:
        """Measure output distribution entropy (collapse detection).

        Low entropy = model is becoming deterministic = mode collapse.
        """
        model.eval()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids[:1, :512], attention_mask=attention_mask[:1, :512])
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Compute entropy of output distribution (last position)
        probs = F.softmax(logits[:, -1, :].float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()

        model.train()

        if entropy < self.entropy_floor:
            return entropy, Signal.COLLAPSE
        return entropy, Signal.HEALTHY

    def get_summary(self) -> dict:
        """Get current monitoring state as a dict."""
        return {
            "train_loss_avg": sum(self._train_losses) / max(len(self._train_losses), 1),
            "val_losses": self._val_losses[-5:],
            "best_val_loss": self._best_val_loss,
            "patience_remaining": self.patience - self._patience_counter,
            "spike_count": self._spike_count,
        }
