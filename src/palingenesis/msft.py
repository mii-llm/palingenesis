"""Adaptive Multi-Source Fine-Tuning with per-source weight scheduling.

Inspired by MSFT (arxiv:2603.21606) but improved for our use case:

MSFT paper: binary exclusion of overfitting sources + checkpoint rollback.
Our approach: CONTINUOUS weight decay toward a floor (never zero).

Why not hard exclusion?
  1. Catastrophic forgetting: excluded sources' capabilities degrade rapidly
  2. Loss of regularization: even "overfitting" sources provide gradient diversity
  3. Pretraining replay effect (arxiv:2603.04964): redundant data still helps target task
  4. Small data regime: excluding 1 of 3 sources = losing 33% of diversity
  5. J-shaped preprocessing already handled quality — double penalty is wasteful

Our approach: Exponential Weight Decay with Floor (EWD-F)
  - When a source's val loss increases: weight *= decay_factor (e.g., 0.7)
  - Weight has a FLOOR = 0.1 × original_weight (never reaches zero)
  - When a source's val loss improves: weight *= recovery_factor (e.g., 1.1)
  - Weight has a CEILING = original_weight (never exceeds initial allocation)

This is analogous to:
  - AdaGC (per-tensor, continuous) vs global clip (binary)
  - EMA (exponential smoothing) vs hard thresholding
  - The cosine LR schedule (gradual decay) vs step function

The floor ensures:
  - Continued exposure for anti-forgetting
  - Gradient diversity for optimization health
  - The source can still contribute if model state changes make it useful again

Integration:
  - Works with `data.sources` multi-dataset mode
  - Tracker adjusts mixing weights every `msft_eval_every` steps
  - MixedDataset's `probs` are updated in-place (no dataloader rebuild needed)
  - Metrics logged as `msft/{source}/val_loss`, `msft/{source}/weight`, `msft/{source}/trend`
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from palingenesis.loss import shift_labels

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass(slots=True)
class SourceState:
    """Tracking state for a single data source."""

    name: str
    weight: float  # Current mixing weight (dynamic)
    original_weight: float  # Original weight (ceiling)
    floor_weight: float  # Minimum weight (10% of original)
    val_losses: deque = field(default_factory=lambda: deque(maxlen=20))
    best_val_loss: float = float("inf")
    consecutive_increases: int = 0  # How many consecutive val loss increases
    total_decays: int = 0  # Total times weight was decayed


class AdaptiveSourceTracker:
    """Per-source adaptive weight scheduler for multi-task SFT.

    Monitors validation loss per source and adjusts mixing weights continuously.
    Overfitting sources get REDUCED weight (never zero). Improving sources get
    INCREASED weight (up to original). This provides soft pressure toward optimal
    compute allocation without the risks of hard exclusion.

    The algorithm:
      1. Every `eval_every` steps, compute per-source val loss
      2. For each source:
         - If val_loss < best → weight = min(weight × recovery, original)  [improving]
         - If val_loss > best → weight = max(weight × decay, floor)        [overfitting]
         - If val_loss ≈ best → weight unchanged                           [plateau]
      3. Normalize all weights so they sum to their original total
      4. Update the MixedDataset's sampling probabilities

    Args:
        source_configs: List of source dicts with 'name' and 'weight'
        eval_every: Steps between per-source evaluations
        decay_factor: Multiplicative decay when overfitting (default 0.7)
        recovery_factor: Multiplicative recovery when improving (default 1.15)
        floor_ratio: Minimum weight as fraction of original (default 0.1)
        plateau_threshold: Relative improvement threshold to count as "improving" (default 0.01)
    """

    def __init__(
        self,
        source_configs: list[dict],
        eval_every: int = 50,
        decay_factor: float = 0.7,
        recovery_factor: float = 1.15,
        floor_ratio: float = 0.1,
        plateau_threshold: float = 0.01,
    ):
        self.eval_every = eval_every
        self.decay_factor = decay_factor
        self.recovery_factor = recovery_factor
        self.floor_ratio = floor_ratio
        self.plateau_threshold = plateau_threshold
        self.sources: dict[str, SourceState] = {}

        for src in source_configs:
            name = src.get("name", src.get("dataset", "unknown"))
            weight = src.get("weight", 1.0)
            self.sources[name] = SourceState(
                name=name,
                weight=weight,
                original_weight=weight,
                floor_weight=weight * floor_ratio,
            )

        self._eval_batches: dict[str, list[dict[str, torch.Tensor]]] = {}
        self._original_total = sum(s.original_weight for s in self.sources.values())

        logger.info(
            f"Adaptive source tracker: {len(self.sources)} sources, "
            f"eval_every={eval_every}, decay={decay_factor}, "
            f"recovery={recovery_factor}, floor={floor_ratio}"
        )
        for name, state in self.sources.items():
            logger.info(f"  {name}: weight={state.weight:.3f}, floor={state.floor_weight:.3f}")

    def set_eval_batches(self, name: str, batches: list[dict[str, torch.Tensor]]):
        """Set pre-collected evaluation batches for a source."""
        self._eval_batches[name] = batches
        logger.info(f"  Eval set for '{name}': {len(batches)} batches")

    def should_eval(self, step: int) -> bool:
        """Check if we should evaluate at this step."""
        return step > 0 and step % self.eval_every == 0

    @torch.no_grad()
    def evaluate_and_adjust(
        self,
        model: nn.Module,
        step: int,
        device: torch.device,
        dtype: torch.dtype,
        bf16: bool = True,
    ) -> dict[str, float]:
        """Evaluate all sources, adjust weights, return metrics.

        Returns:
            Dict of metrics: {msft/{name}/val_loss, msft/{name}/weight, msft/{name}/trend}
            trend: +1 improving, 0 plateau, -1 overfitting
        """
        model.eval()
        metrics: dict[str, float] = {}

        for name, state in self.sources.items():
            batches = self._eval_batches.get(name, [])
            if not batches:
                continue

            # Compute validation loss for this source
            val_loss = self._compute_val_loss(model, batches, device, dtype, bf16)
            state.val_losses.append(val_loss)
            metrics[f"msft/{name}/val_loss"] = val_loss

            # Determine trend and adjust weight
            if state.best_val_loss == float("inf"):
                # First evaluation: just set baseline
                state.best_val_loss = val_loss
                trend = 0.0
            else:
                relative_change = (val_loss - state.best_val_loss) / max(abs(state.best_val_loss), 1e-8)

                if relative_change < -self.plateau_threshold:
                    # IMPROVING: val loss decreased meaningfully
                    state.best_val_loss = val_loss
                    state.consecutive_increases = 0
                    # Recover weight (up to original)
                    state.weight = min(state.weight * self.recovery_factor, state.original_weight)
                    trend = 1.0

                elif relative_change > self.plateau_threshold:
                    # OVERFITTING: val loss increased meaningfully
                    state.consecutive_increases += 1
                    # Decay weight (down to floor)
                    state.weight = max(state.weight * self.decay_factor, state.floor_weight)
                    state.total_decays += 1
                    trend = -1.0

                    if state.consecutive_increases == 1:
                        logger.info(
                            f"MSFT: '{name}' overfitting detected (val_loss={val_loss:.4f} > "
                            f"best={state.best_val_loss:.4f}), weight {state.weight/state.original_weight:.0%} of original"
                        )
                else:
                    # PLATEAU: val loss roughly unchanged
                    state.consecutive_increases = 0
                    trend = 0.0

            metrics[f"msft/{name}/weight"] = state.weight
            metrics[f"msft/{name}/trend"] = trend
            metrics[f"msft/{name}/weight_ratio"] = state.weight / state.original_weight

        model.train()
        return metrics

    def _compute_val_loss(
        self,
        model: nn.Module,
        batches: list[dict[str, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
        bf16: bool,
    ) -> float:
        """Compute average CE loss over evaluation batches."""
        total_loss = 0.0
        total_tokens = 0

        for batch in batches:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Shift for next-token prediction: logits[t] predicts input_ids[t+1]
            labels = shift_labels(batch["labels"].to(device))

            valid = (labels != IGNORE_INDEX).sum().item()
            if valid == 0:
                continue

            with torch.amp.autocast("cuda", dtype=dtype, enabled=bf16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    labels.view(-1),
                    reduction="sum",
                    ignore_index=IGNORE_INDEX,
                )
            total_loss += loss.item()
            total_tokens += valid

        return total_loss / max(total_tokens, 1)

    def get_current_weights(self) -> dict[str, float]:
        """Get current mixing weights for all sources."""
        return {name: state.weight for name, state in self.sources.items()}

    def get_mixing_probs(self) -> list[float]:
        """Get normalized probabilities for MixedDataset (sum to 1)."""
        weights = [state.weight for state in self.sources.values()]
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else [1.0 / len(weights)] * len(weights)

    def update_mixed_dataset(self, mixed_dataset) -> None:
        """Update a MixedDataset's sampling probabilities in-place.

        This allows dynamic weight adjustment without rebuilding the dataloader.
        """
        if hasattr(mixed_dataset, "probs"):
            new_probs = self.get_mixing_probs()
            mixed_dataset.probs = new_probs

    @property
    def summary_str(self) -> str:
        """One-line summary for logging."""
        parts = []
        for name, state in self.sources.items():
            ratio = state.weight / state.original_weight
            symbol = "↑" if state.consecutive_increases == 0 else f"↓{state.consecutive_increases}"
            parts.append(f"{name}={ratio:.0%}{symbol}")
        return " | ".join(parts)

    def detailed_summary(self) -> str:
        """Multi-line detailed summary."""
        lines = [f"Source Weights ({len(self.sources)} sources):"]
        for name, state in self.sources.items():
            ratio = state.weight / state.original_weight
            status = "improving" if state.consecutive_increases == 0 else f"overfitting×{state.consecutive_increases}"
            best = f"best={state.best_val_loss:.4f}" if state.best_val_loss < float("inf") else "no eval"
            lines.append(
                f"  {name:20s}: weight={state.weight:.4f} ({ratio:.0%} of original), "
                f"{status}, {best}, decays={state.total_decays}"
            )
        return "\n".join(lines)
