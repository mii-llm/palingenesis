"""Data ablation: test different configurations using the FULL optimized train loop.

Each ablation trial runs train(config) with max_steps=N and different data settings.
All optimizations are active (Liger, AC, compile, FSDP, etc.) so results are
representative of the real training.

After each trial, evaluates on a held-out validation set to score the configuration.
"""

import copy
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from palingenesis.autopilot.evaluate import evaluate
from palingenesis.config import Config
from palingenesis.train import train as full_train

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AblationResult:
    """Result of a single ablation trial."""

    name: str
    val_loss: float
    val_perplexity: float
    val_accuracy: float
    weight_drift: float
    steps: int
    time_s: float
    config_description: str


def run_ablations(
    base_config: Config,
    ablation_configs: list[dict],
    val_dataloader,
    steps_per_trial: int = 200,
    device: torch.device = torch.device("cuda"),
) -> list[AblationResult]:
    """Run data ablation experiments using the full optimized training loop.

    Each trial modifies the base config (e.g., data mix weights, epochs, LR)
    and runs train(config) for steps_per_trial steps. Results are scored by
    validation loss.

    Args:
        base_config: Base training config
        ablation_configs: List of dicts with "name" and "overrides" keys.
            Overrides are applied to the config before training.
        val_dataloader: Validation DataLoader for scoring
        steps_per_trial: Steps per ablation trial
        device: GPU device

    Returns:
        List of AblationResult, sorted by val_loss (best first)
    """
    results = []
    ablation_dir = Path(base_config.train.output_dir) / "_ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Data Ablation: {len(ablation_configs)} trials x {steps_per_trial} steps")

    # ONE monitor seeded with the base model's weight norms (a fresh monitor
    # per trial would always report drift=0.0 — see sweep.py)
    from palingenesis.autopilot.sweep import _seeded_drift_monitor

    monitor = _seeded_drift_monitor(base_config)

    for i, ablation in enumerate(ablation_configs):
        name = ablation.get("name", f"trial_{i}")
        overrides = ablation.get("overrides", {})

        logger.info(f"  [{i+1}/{len(ablation_configs)}] {name}")
        t0 = time.perf_counter()

        # Build trial config
        trial_config = _make_ablation_config(base_config, name, overrides, steps_per_trial, ablation_dir / f"trial_{i}")

        # Run full optimized training
        try:
            full_train(trial_config)
        except Exception as e:
            logger.warning(f"    Trial '{name}' failed: {e}")
            results.append(
                AblationResult(
                    name=name,
                    val_loss=float("inf"),
                    val_perplexity=float("inf"),
                    val_accuracy=0.0,
                    weight_drift=0.0,
                    steps=0,
                    time_s=time.perf_counter() - t0,
                    config_description=str(overrides),
                )
            )
            continue

        # Evaluate
        trial_model_path = Path(trial_config.train.output_dir) / "final"
        try:
            trial_model = AutoModelForCausalLM.from_pretrained(
                trial_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=base_config.model.trust_remote_code,
                use_cache=False,
                low_cpu_mem_usage=True,
            ).to(device)

            val_metrics = evaluate(trial_model, val_dataloader, device, max_batches=50)
            drift = monitor.check_weight_drift(trial_model)[0] if monitor else 0.0

            del trial_model
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"    Eval failed for '{name}': {e}")
            val_metrics = {"val_loss": float("inf"), "val_perplexity": float("inf"), "val_accuracy": 0.0}
            drift = 0.0

        dt = time.perf_counter() - t0
        results.append(
            AblationResult(
                name=name,
                val_loss=val_metrics["val_loss"],
                val_perplexity=val_metrics["val_perplexity"],
                val_accuracy=val_metrics["val_accuracy"],
                weight_drift=drift,
                steps=steps_per_trial,
                time_s=dt,
                config_description=str(overrides),
            )
        )

        logger.info(
            f"    val_loss={val_metrics['val_loss']:.4f} "
            f"ppl={val_metrics['val_perplexity']:.1f} "
            f"acc={val_metrics['val_accuracy']:.3f} "
            f"drift={drift:.3f} ({dt:.1f}s)"
        )

        # Clean up trial output
        trial_output = Path(trial_config.train.output_dir)
        if trial_output.exists():
            shutil.rmtree(trial_output, ignore_errors=True)

    # Sort by val_loss
    results.sort(key=lambda r: r.val_loss)
    if results and results[0].val_loss < float("inf"):
        logger.info(f"\n  Best: '{results[0].name}' (val_loss={results[0].val_loss:.4f})")

    # Clean up
    if ablation_dir.exists():
        shutil.rmtree(ablation_dir, ignore_errors=True)

    return results


def _make_ablation_config(
    base: Config,
    name: str,
    overrides: dict,
    max_steps: int,
    output_dir: Path,
) -> Config:
    """Create an ablation trial config from base + overrides."""
    trial = Config()
    trial.model = copy.copy(base.model)
    trial.data = copy.copy(base.data)
    trial.train = copy.copy(base.train)
    trial.parallel = copy.copy(base.parallel)
    trial.memory = copy.copy(base.memory)
    trial.plugins = copy.copy(base.plugins)
    trial.preprocess = copy.copy(base.preprocess)
    trial.logging = copy.copy(base.logging)

    # Apply overrides
    for key, value in overrides.items():
        parts = key.split(".")
        if len(parts) == 2:
            section_name, field_name = parts
            if hasattr(trial, section_name):
                section = getattr(trial, section_name)
                if hasattr(section, field_name):
                    setattr(section, field_name, value)

    # Trial-specific settings
    trial.train.max_steps = max_steps
    trial.train.output_dir = str(output_dir)
    trial.train.save_steps = max_steps + 1  # No intermediate saves
    trial.train.logging_steps = max(max_steps // 10, 1)
    trial.logging.use_wandb = False
    trial.logging.use_trackio = False
    trial.logging.run_name = f"ablation-{name}"

    # Trials are scored by the shared evaluate() pass, not in-train eval
    trial.data.eval_dataset = ""

    return trial


def generate_lr_ablations(base_lr: float) -> list[dict]:
    """Generate ablation configs for learning rate comparison."""
    return [
        {"name": f"lr={base_lr*0.5:.1e}", "overrides": {"train.learning_rate": base_lr * 0.5}},
        {"name": f"lr={base_lr:.1e} (base)", "overrides": {}},
        {"name": f"lr={base_lr*2:.1e}", "overrides": {"train.learning_rate": base_lr * 2}},
        {"name": f"lr={base_lr*5:.1e}", "overrides": {"train.learning_rate": base_lr * 5}},
    ]


def generate_data_ablations() -> list[dict]:
    """Generate ablation configs for common data strategies."""
    return [
        {"name": "baseline (1 epoch)", "overrides": {"train.epochs": 1}},
        {"name": "3 epochs (repetition)", "overrides": {"train.epochs": 3}},
        {"name": "5 epochs (heavy repetition)", "overrides": {"train.epochs": 5, "train.weight_decay": 0.3}},
        {"name": "with packing", "overrides": {"data.packing": True}},
        {"name": "with ECHO", "overrides": {"data.include_observations": True}},
        {"name": "with progressive turns", "overrides": {"data.turn_scaling": "progressive"}},
        {"name": "DEFT loss", "overrides": {"plugins.deft": True}},
        {"name": "CADFT loss", "overrides": {"plugins.cadft": True}},
        {"name": "SymNoise α=5", "overrides": {"plugins.sym_noise": True, "plugins.sym_noise_alpha": 5.0}},
        {"name": "SymNoise α=7", "overrides": {"plugins.sym_noise": True, "plugins.sym_noise_alpha": 7.0}},
    ]
