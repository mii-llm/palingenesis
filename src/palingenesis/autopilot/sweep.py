"""Learning rate sweep using the FULL optimized train() loop.

Each trial runs train(config) with max_steps=N and a different LR.
This ensures:
  - Same memory profile as the real run (won't OOM)
  - Same numerical path (Liger, AC, chunked loss all active)
  - Results transfer directly to the full training
  - torch.compile cache is shared across trials (same model architecture)

After each trial, we evaluate on validation data and check for collapse.
"""

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from palingenesis.autopilot.evaluate import evaluate
from palingenesis.autopilot.monitor import BehaviorMonitor, Signal
from palingenesis.config import Config
from palingenesis.train import train as full_train

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SweepResult:
    """Result of a single LR trial."""

    lr: float
    val_loss: float
    train_loss_final: float
    weight_drift: float
    had_issues: bool
    score: float  # lower = better
    steps_completed: int
    time_s: float


def lr_sweep(
    base_config: Config,
    tokenizer: AutoTokenizer,
    val_dataloader,
    lr_candidates: list[float] | None = None,
    steps_per_trial: int = 100,
    device: torch.device = torch.device("cuda"),
) -> list[SweepResult]:
    """Run LR sweep using the full optimized training loop.

    Each candidate LR runs train(config) with max_steps=steps_per_trial.
    All optimizations (Liger, AC, compile, chunked loss) are active.

    Args:
        base_config: Base training config (LR will be overridden per trial)
        tokenizer: Loaded tokenizer
        val_dataloader: Validation DataLoader for scoring
        lr_candidates: LR values to test
        steps_per_trial: Steps per trial
        device: GPU device

    Returns:
        List of SweepResult, sorted by score (best first)
    """
    if lr_candidates is None:
        lr_candidates = [5e-6, 1e-5, 2e-5, 5e-5, 1e-4]

    results = []
    sweep_dir = Path(base_config.train.output_dir) / "_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"LR Sweep: {len(lr_candidates)} candidates x {steps_per_trial} steps (full optimized loop)")

    # ONE monitor for the whole sweep, seeded with the BASE model's weight
    # norms. (A fresh monitor per trial just seeds itself on the trial model
    # and reports drift=0.0 — the forgetting penalty would never fire.)
    monitor = _seeded_drift_monitor(base_config)

    for i, lr in enumerate(lr_candidates):
        logger.info(f"  [{i+1}/{len(lr_candidates)}] LR={lr:.1e}")
        t0 = time.perf_counter()

        # Create a trial-specific config
        trial_config = _make_trial_config(base_config, lr, steps_per_trial, sweep_dir / f"lr_{lr:.0e}")

        # Run the FULL optimized training loop for this trial
        try:
            full_train(trial_config)
        except Exception as e:
            logger.warning(f"    Trial LR={lr:.1e} failed: {e}")
            results.append(
                SweepResult(
                    lr=lr,
                    val_loss=float("inf"),
                    train_loss_final=float("inf"),
                    weight_drift=0.0,
                    had_issues=True,
                    score=float("inf"),
                    steps_completed=0,
                    time_s=time.perf_counter() - t0,
                )
            )
            continue

        # Load the trained model from the trial's final checkpoint
        trial_model_path = Path(trial_config.train.output_dir) / "final"
        if not trial_model_path.exists():
            # No final saved -- check for step checkpoints
            trial_model_path = Path(trial_config.train.output_dir)

        # Evaluate on validation data
        try:
            trial_model = AutoModelForCausalLM.from_pretrained(
                trial_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=base_config.model.trust_remote_code,
                use_cache=False,
                low_cpu_mem_usage=True,
            ).to(device)

            val_metrics = evaluate(trial_model, val_dataloader, device, max_batches=30)
            val_loss = val_metrics["val_loss"]

            # Weight drift vs the base model (monitor seeded before the loop)
            drift, _ = monitor.check_weight_drift(trial_model) if monitor else (0.0, Signal.HEALTHY)

            del trial_model
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"    Eval failed for LR={lr:.1e}: {e}")
            val_loss = float("inf")
            drift = 0.0

        # Read train loss from the trial's log (last logged value)
        train_loss_final = _read_last_train_loss(trial_config.train.output_dir)

        # Score: val_loss + penalties
        had_issues = drift > 0.3 or val_loss == float("inf")
        score = val_loss
        if drift > 0.3:
            score += drift  # Penalty for forgetting
        if val_loss == float("inf"):
            score = float("inf")

        dt = time.perf_counter() - t0
        results.append(
            SweepResult(
                lr=lr,
                val_loss=val_loss,
                train_loss_final=train_loss_final,
                weight_drift=drift,
                had_issues=had_issues,
                score=score,
                steps_completed=steps_per_trial,
                time_s=dt,
            )
        )

        logger.info(f"    val_loss={val_loss:.4f} drift={drift:.3f} score={score:.4f} ({dt:.1f}s)")

        # Clean up trial checkpoints to save disk
        trial_output = Path(trial_config.train.output_dir)
        if trial_output.exists():
            shutil.rmtree(trial_output, ignore_errors=True)

    # Sort by score
    results.sort(key=lambda r: r.score)
    if results and results[0].score < float("inf"):
        logger.info(f"  Best LR: {results[0].lr:.1e} (val_loss={results[0].val_loss:.4f})")

    # Clean up sweep directory
    if sweep_dir.exists():
        shutil.rmtree(sweep_dir, ignore_errors=True)

    return results


def _seeded_drift_monitor(base_config: Config) -> BehaviorMonitor | None:
    """Build a BehaviorMonitor whose drift baseline is the BASE model.

    Loads the base model once on CPU just to record per-layer weight norms
    (the first check_weight_drift call seeds the baseline), then frees it.
    Returns None if the load fails — drift is then reported as 0.0.
    """
    monitor = BehaviorMonitor()
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_config.model.name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=base_config.model.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        monitor.check_weight_drift(base_model)  # first call seeds init norms
        del base_model
        return monitor
    except Exception as e:
        logger.warning(f"Could not seed drift monitor from base model ({e}); drift reported as 0")
        return None


def _make_trial_config(base: Config, lr: float, max_steps: int, output_dir: Path) -> Config:
    """Create a trial config: same as base but with different LR and short max_steps."""
    import copy

    # Shallow copy of dataclass sections
    trial = Config()
    trial.model = copy.copy(base.model)
    trial.data = copy.copy(base.data)
    trial.train = copy.copy(base.train)
    trial.parallel = copy.copy(base.parallel)
    trial.memory = copy.copy(base.memory)
    trial.plugins = copy.copy(base.plugins)
    trial.preprocess = copy.copy(base.preprocess)
    trial.logging = copy.copy(base.logging)

    # Override for this trial
    trial.train.learning_rate = lr
    trial.train.min_learning_rate = lr * 0.1
    trial.train.max_steps = max_steps
    trial.train.output_dir = str(output_dir)
    trial.train.save_steps = max_steps + 1  # Don't save intermediate checkpoints during sweep
    trial.train.logging_steps = max(max_steps // 10, 1)

    # Disable wandb/trackio for sweep trials (too noisy)
    trial.logging.use_wandb = False
    trial.logging.use_trackio = False
    trial.logging.run_name = f"sweep-lr{lr:.0e}"

    # Trials are scored by the shared evaluate() pass, not in-train eval
    trial.data.eval_dataset = ""

    return trial


def _read_last_train_loss(output_dir: str) -> float:
    """Try to read the last training loss from the output directory.

    Falls back to inf if no log is available.
    """
    # Our train() logs to stdout with format: step=N loss=X ...
    # We could also check wandb/trackio but they're disabled for sweeps
    # For now return inf -- the val_loss is the real metric anyway
    return float("inf")
