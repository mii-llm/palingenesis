"""Autopilot: autonomous training with profiling, LR sweep, and full training.

Resumable: saves progress after each phase. If interrupted, re-run the same
command and it skips completed phases automatically.

Output structure:
    output_dir/
        autopilot_state.json     -- progress state (which phases completed, best LR, etc.)
        autopilot_report.json    -- final detailed report
        final/                   -- best model in HF format (from train())
        step-*/                  -- training checkpoints (from train())
"""

import json
import logging
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from palingenesis.autopilot.adaptive_sweep import (
    TrialResult,
    adaptive_lr_sweep,
)
from palingenesis.autopilot.evaluate import evaluate
from palingenesis.autopilot.profile import auto_config, correct_lr_for_horizon, estimate_optimal_lr, profile_hardware
from palingenesis.autopilot.sweep import lr_sweep
from palingenesis.config import Config
from palingenesis.data import ChatDataset, _load_dataset_source, collate_fn
from palingenesis.distributed import cleanup_distributed, set_skip_cleanup

logger = logging.getLogger(__name__)

STATE_FILE = "autopilot_state.json"


def autopilot(
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    dataset: str = "HuggingFaceH4/ultrachat_200k",
    dataset_split: str = "train_sft",
    val_dataset: str | None = None,
    val_split: str = "test_sft",
    output_dir: str = "./autopilot-output",
    max_steps: int = 5000,
    seq_length: int = 4096,
    lr_sweep_steps: int = 100,
    do_ablation: bool = False,
    trust_remote_code: bool = True,
):
    """Run fully autonomous training. Resumable -- re-run to continue."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    state_path = output_path / STATE_FILE

    # Load or create state
    state = _load_state(state_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep process group alive across multiple train() calls
    set_skip_cleanup(True)

    logger.info("=" * 70)
    logger.info("AUTOPILOT: Autonomous Training Optimization")
    logger.info(f"  Output: {output_path}")
    if state.get("completed_phases"):
        logger.info(f"  Resuming: phases {state['completed_phases']} already done")
    logger.info("=" * 70)

    # Always need tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Profile + Auto-Configure
    # ══════════════════════════════════════════════════════════════════════
    if "profile" not in state.get("completed_phases", []):
        logger.info("\n[Phase 1] Profiling hardware...")
        t0 = time.perf_counter()

        hardware = profile_hardware()
        from transformers import AutoConfig

        model_cfg = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
        vocab_size = getattr(model_cfg, "vocab_size", 32000)
        hidden_size = getattr(model_cfg, "hidden_size", 4096)
        num_layers = getattr(model_cfg, "num_hidden_layers", 32)
        model_params_b = (num_layers * (12 * hidden_size**2) + vocab_size * hidden_size * 2) / 1e9

        recommended = auto_config(model_params_b, seq_length, vocab_size, hardware)

        # Estimate optimal LR from scaling laws (used as sweep center point)
        effective_batch_tokens = (
            recommended["train.per_device_batch_size"]
            * seq_length
            * recommended["train.gradient_accumulation_steps"]
            * hardware.get("num_gpus", 1)
        )
        estimated_lr = estimate_optimal_lr(model_params_b, effective_batch_tokens, max_steps)
        recommended["_estimated_lr"] = estimated_lr

        state["hardware"] = hardware
        state["recommended"] = recommended
        state["model_params_b"] = model_params_b
        state.setdefault("completed_phases", []).append("profile")
        _save_state(state, state_path)

        logger.info(f"  GPU: {hardware['gpu_name']} ({hardware['memory_gb']:.0f}GB)")
        logger.info(f"  Model: ~{model_params_b:.1f}B params")
        logger.info(
            f"  Auto-config: batch={recommended['train.per_device_batch_size']}, "
            f"grad_accum={recommended['train.gradient_accumulation_steps']}"
        )
        logger.info(f"  Phase 1 done ({time.perf_counter()-t0:.1f}s)")
    else:
        recommended = state["recommended"]
        logger.info("[Phase 1] Profile: cached, skipping")

    # Validation dataloader: shared by the LR sweep and the ablation phase.
    # Built lazily so cached (skipped) phases don't pay the loading cost.
    _val_dl_cache: list = []

    def _get_val_dl():
        if not _val_dl_cache:
            _val_dl_cache.append(
                _build_val_dataloader(
                    val_dataset or dataset,
                    val_split if val_dataset else dataset_split,
                    tokenizer,
                    seq_length,
                    recommended["train.per_device_batch_size"],
                )
            )
        return _val_dl_cache[0]

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Adaptive LR Sweep (coarse → refine → horizon-correct)
    # ══════════════════════════════════════════════════════════════════════
    if "lr_sweep" not in state.get("completed_phases", []):
        logger.info(f"\n[Phase 2] Adaptive LR Sweep ({lr_sweep_steps} steps/trial)...")
        t0 = time.perf_counter()

        # Build sweep base config (all optimizations ON)
        sweep_cfg = _build_config(
            model,
            dataset,
            dataset_split,
            seq_length,
            trust_remote_code,
            recommended,
            lr=2e-5,
            max_steps=lr_sweep_steps,
            output_dir=str(output_path / "_sweep"),
            val_dataset=val_dataset,
            val_split=val_split,
        )

        val_dl = _get_val_dl()

        # Try adaptive sweep (preferred); fall back to legacy if it fails
        model_params_b = state.get("model_params_b", 4.0)
        effective_batch_tokens = (
            recommended["train.per_device_batch_size"]
            * seq_length
            * recommended["train.gradient_accumulation_steps"]
            * state.get("hardware", {}).get("num_gpus", 1)
        )

        try:
            # Define trial runner that uses the full optimized train loop
            def _run_adaptive_trial(lr: float, trial_steps: int) -> TrialResult:
                """Run a single trial and return structured result."""
                import copy
                import shutil

                trial_cfg = Config()
                trial_cfg.model = copy.copy(sweep_cfg.model)
                trial_cfg.data = copy.copy(sweep_cfg.data)
                trial_cfg.train = copy.copy(sweep_cfg.train)
                trial_cfg.parallel = copy.copy(sweep_cfg.parallel)
                trial_cfg.memory = copy.copy(sweep_cfg.memory)
                trial_cfg.plugins = copy.copy(sweep_cfg.plugins)
                trial_cfg.preprocess = copy.copy(sweep_cfg.preprocess)
                trial_cfg.logging = copy.copy(sweep_cfg.logging)

                trial_cfg.train.learning_rate = lr
                trial_cfg.train.min_learning_rate = lr * 0.1
                trial_cfg.train.max_steps = trial_steps
                trial_dir = str(output_path / "_sweep" / f"lr_{lr:.0e}")
                trial_cfg.train.output_dir = trial_dir
                trial_cfg.train.save_steps = trial_steps + 1
                trial_cfg.train.logging_steps = max(trial_steps // 20, 1)
                trial_cfg.logging.use_wandb = False
                trial_cfg.logging.use_trackio = False
                # Trials are scored by the shared evaluate() pass below;
                # in-train eval would just double the cost of every trial.
                trial_cfg.data.eval_dataset = ""

                result = TrialResult(lr=lr, final_loss=float("inf"), initial_loss=0.0)
                try:
                    from palingenesis.train import train as full_train
                    full_train(trial_cfg)

                    # Extract loss curve from output (best-effort)
                    result.steps_completed = trial_steps
                    result.diverged = False

                    # Evaluate on validation data
                    trial_model_path = Path(trial_dir) / "final"
                    if not trial_model_path.exists():
                        trial_model_path = Path(trial_dir)

                    from transformers import AutoModelForCausalLM as AMLM
                    trial_model = AMLM.from_pretrained(
                        trial_model_path,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=trust_remote_code,
                        use_cache=False,
                        low_cpu_mem_usage=True,
                    ).to(device)

                    val_metrics = evaluate(trial_model, val_dl, device, max_batches=30)
                    result.final_loss = val_metrics["val_loss"]
                    result.initial_loss = val_metrics.get("initial_loss", result.final_loss * 1.5)

                    del trial_model
                    torch.cuda.empty_cache()

                except Exception as e:
                    logger.warning(f"Trial LR={lr:.1e} failed: {e}")
                    result.diverged = True
                    result.final_loss = float("inf")

                # Cleanup
                trial_path = Path(trial_dir)
                if trial_path.exists():
                    shutil.rmtree(trial_path, ignore_errors=True)

                return result

            best_lr, sweep_results = adaptive_lr_sweep(
                run_trial_fn=_run_adaptive_trial,
                model_params_b=model_params_b,
                effective_batch_tokens=effective_batch_tokens,
                steps_per_trial=lr_sweep_steps,
                full_training_steps=max_steps,
            )

        except Exception as e:
            # Fallback to legacy sweep if adaptive fails
            logger.warning(f"Adaptive sweep failed ({e}), falling back to legacy sweep")
            sweep_results_legacy = lr_sweep(
                base_config=sweep_cfg,
                tokenizer=tokenizer,
                val_dataloader=val_dl,
                steps_per_trial=lr_sweep_steps,
                device=device,
            )
            best_lr = (
                sweep_results_legacy[0].lr
                if (sweep_results_legacy and sweep_results_legacy[0].score < float("inf"))
                else 2e-5
            )
            best_lr = correct_lr_for_horizon(best_lr, lr_sweep_steps, max_steps)
            sweep_results = []

        state["best_lr"] = best_lr
        state["sweep_results"] = [
            {"lr": r.lr, "final_loss": r.final_loss, "score": r.score, "diverged": r.diverged}
            for r in sweep_results
        ] if sweep_results else []
        state.setdefault("completed_phases", []).append("lr_sweep")
        _save_state(state, state_path)

        logger.info(f"  Best LR: {best_lr:.1e}")
        logger.info(f"  Phase 2 done ({time.perf_counter()-t0:.1f}s)")
    else:
        best_lr = state["best_lr"]
        logger.info(f"[Phase 2] LR Sweep: cached (best_lr={best_lr:.1e}), skipping")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2.5 (optional): Data/config ablations at the chosen LR
    # ══════════════════════════════════════════════════════════════════════
    if do_ablation and "ablation" not in state.get("completed_phases", []):
        ablation_steps = lr_sweep_steps * 2
        logger.info(f"\n[Phase 2.5] Ablations (LR={best_lr:.1e}, {ablation_steps} steps/trial)...")
        t0 = time.perf_counter()

        from palingenesis.autopilot.ablate import generate_data_ablations, run_ablations

        ablation_cfg = _build_config(
            model,
            dataset,
            dataset_split,
            seq_length,
            trust_remote_code,
            recommended,
            lr=best_lr,
            max_steps=ablation_steps,
            output_dir=str(output_path / "_ablation"),
            val_dataset=val_dataset,
            val_split=val_split,
        )
        ablation_results = run_ablations(
            base_config=ablation_cfg,
            ablation_configs=generate_data_ablations(),
            val_dataloader=_get_val_dl(),
            steps_per_trial=ablation_steps,
            device=device,
        )
        state["ablation_results"] = [
            {
                "name": r.name,
                "val_loss": r.val_loss,
                "val_perplexity": r.val_perplexity,
                "val_accuracy": r.val_accuracy,
                "weight_drift": r.weight_drift,
                "config": r.config_description,
            }
            for r in ablation_results
        ]
        state.setdefault("completed_phases", []).append("ablation")
        _save_state(state, state_path)
        logger.info(f"  Phase 2.5 done ({time.perf_counter()-t0:.1f}s)")
    elif do_ablation:
        logger.info("[Phase 2.5] Ablations: cached, skipping")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Full Training (all optimizations, resumable via checkpoints)
    # ══════════════════════════════════════════════════════════════════════
    if "training" not in state.get("completed_phases", []):
        logger.info(f"\n[Phase 3] Full training (LR={best_lr:.1e}, {max_steps} steps)...")
        t0 = time.perf_counter()

        from palingenesis.train import train as full_train

        train_cfg = _build_config(
            model,
            dataset,
            dataset_split,
            seq_length,
            trust_remote_code,
            recommended,
            lr=best_lr,
            max_steps=max_steps,
            output_dir=str(output_path),
            val_dataset=val_dataset,
            val_split=val_split,
        )
        # Enable tracking for the real run (wandb only; trackio stays opt-in)
        train_cfg.logging.use_wandb = True
        train_cfg.logging.project = "palingenesis-autopilot"
        train_cfg.logging.run_name = f"autopilot-lr{best_lr:.0e}"
        # Enable checkpointing for resume
        train_cfg.train.save_steps = max(max_steps // 10, 100)
        train_cfg.train.resume_from = "auto"  # Resume if crashed mid-training

        full_train(train_cfg)

        state.setdefault("completed_phases", []).append("training")
        state["training_time_s"] = time.perf_counter() - t0
        _save_state(state, state_path)
        logger.info(f"  Phase 3 done ({state['training_time_s']:.1f}s)")
    else:
        logger.info("[Phase 3] Training: already completed, skipping")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 4: Report
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n[Phase 4] Final report")
    logger.info("=" * 70)
    logger.info(f"  Best LR: {best_lr:.1e}")
    logger.info(
        f"  Batch: {recommended['train.per_device_batch_size']}, "
        f"Accum: {recommended['train.gradient_accumulation_steps']}"
    )
    logger.info(f"  Output: {output_path}/final")
    logger.info("=" * 70)

    # Write final report
    report = {
        "best_lr": best_lr,
        "recommended_config": recommended,
        "sweep_results": state.get("sweep_results", []),
        "ablation_results": state.get("ablation_results", []),
        "hardware": state.get("hardware", {}),
        "model": model,
        "dataset": dataset,
        "max_steps": max_steps,
        "seq_length": seq_length,
    }
    with open(output_path / "autopilot_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Cleanup
    set_skip_cleanup(False)
    cleanup_distributed()

    return report


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_config(
    model: str,
    dataset: str,
    dataset_split: str,
    seq_length: int,
    trust_remote_code: bool,
    recommended: dict,
    lr: float,
    max_steps: int,
    output_dir: str,
    val_dataset: str | None = None,
    val_split: str | None = None,
) -> Config:
    """Build a fully-configured Config from autopilot parameters.

    Applies all research-backed optimizations by default:
    - DEFT loss (best overall SFT objective)
    - SymNoise (embedding regularization)
    - ECHO mode (learn from tool outputs)
    - AdaGC (per-tensor gradient clipping)
    - EMA (better generalization)
    - Selective AC (optimal memory/compute)
    - Chunked loss (memory efficient)
    - Gradient release (when possible)
    """
    cfg = Config()
    cfg.model.name_or_path = model
    cfg.model.trust_remote_code = trust_remote_code
    cfg.model.use_liger_kernel = True
    cfg.model.compile = True

    cfg.data.dataset = dataset
    cfg.data.dataset_split = dataset_split
    cfg.data.max_seq_length = seq_length
    cfg.data.streaming = True
    cfg.data.packing = True
    cfg.data.include_observations = True  # ECHO: world model from tool outputs
    cfg.data.turn_scaling = "progressive"
    # Eval set: prefer an explicit held-out dataset. Without one, fall back to
    # a fixed subset of the TRAIN split (works for any dataset, but numbers are
    # optimistic since eval samples may also appear in training).
    if val_dataset:
        cfg.data.eval_dataset = val_dataset
        cfg.data.eval_split = val_split or "test"
    else:
        cfg.data.eval_dataset = dataset
        cfg.data.eval_split = dataset_split
    cfg.data.eval_every = max(max_steps // 20, 10)  # ~20 eval points during training

    cfg.train.learning_rate = lr
    cfg.train.min_learning_rate = lr * 0.1
    cfg.train.max_steps = max_steps
    cfg.train.per_device_batch_size = recommended.get("train.per_device_batch_size", 1)
    cfg.train.gradient_accumulation_steps = recommended.get("train.gradient_accumulation_steps", 16)
    cfg.train.output_dir = output_dir
    cfg.train.gradient_checkpointing = "selective"
    cfg.train.bf16 = recommended.get("train.bf16", True)
    cfg.train.lr_scheduler = "power_decay"  # Better than cosine (arxiv:2602.06797)
    cfg.train.adagc = True  # Per-tensor adaptive gradient clipping
    cfg.train.ema = True  # EMA for better final checkpoint
    cfg.train.ema_every = 10
    cfg.train.adamc = True  # Corrected weight decay

    # Optimizer: use lion8bit if memory tight, otherwise adamw.
    # No LR adjustment for Lion: sweep trials and the final run both go
    # through this function with the same optimizer, so the swept LR is
    # already calibrated for whichever optimizer is selected.
    gpu_mem = recommended.get("_estimated_activation_budget_gb", 40)
    cfg.train.optimizer = "lion8bit" if gpu_mem < 10 else "adamw"

    cfg.parallel.fsdp = recommended.get("parallel.fsdp", False)
    cfg.parallel.context_parallel = recommended.get("parallel.context_parallel", False)

    cfg.memory.chunked_loss = recommended.get("memory.chunked_loss", True)
    cfg.memory.loss_num_chunks = recommended.get("memory.loss_num_chunks", 8)
    cfg.memory.float8_training = recommended.get("memory.float8_training", False)
    cfg.memory.selective_diff = True  # Always beneficial

    # Enable gradient release when GA=1 and compatible optimizer
    if cfg.train.gradient_accumulation_steps == 1 and cfg.train.optimizer != "muon":
        cfg.memory.gradient_release = True

    # Plugins: DEFT + SymNoise (research-backed optimal combo)
    cfg.plugins.deft = True
    cfg.plugins.sym_noise = True
    cfg.plugins.sym_noise_alpha = 5.0

    # Hyperball: norm-constrained optimization (zero cost, 20-30% speedup)
    # Only meaningful when not using gradient_release (which doesn't use the standard step)
    cfg.train.hyperball = True

    # Disable logging for sweep/ablation trials by default
    cfg.logging.use_wandb = False
    cfg.logging.use_trackio = False
    return cfg


def _build_val_dataloader(
    dataset_name: str,
    split: str,
    tokenizer,
    seq_length: int,
    batch_size: int,
):
    """Build a validation DataLoader.

    Uses _load_dataset_source so local jsonl/parquet files and prepared
    directories work, not just HF hub datasets.
    """
    from torch.utils.data import DataLoader

    try:
        raw = _load_dataset_source(dataset_name, split, streaming=True)
    except Exception:
        # Fallback: use training split
        raw = _load_dataset_source(dataset_name, "train", streaming=True)

    ds = ChatDataset(raw, tokenizer, seq_length, rank=0, world_size=1)
    pad_id = tokenizer.pad_token_id or 0
    return DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=2,
        pin_memory=True,
    )


def _load_state(path: Path) -> dict:
    """Load autopilot state from disk (for resume)."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_state(state: dict, path: Path):
    """Save autopilot state to disk (after each phase)."""
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    """CLI entry point for autopilot."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="Autopilot: autonomous training optimization")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_split", default="train_sft")
    parser.add_argument("--val_dataset", default=None)
    parser.add_argument("--val_split", default="test_sft")
    parser.add_argument("--output", default="./autopilot-output")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--lr_sweep_steps", type=int, default=100)
    parser.add_argument("--ablate", action="store_true")
    args = parser.parse_args()

    autopilot(
        model=args.model,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        val_dataset=args.val_dataset,
        val_split=args.val_split,
        output_dir=args.output,
        max_steps=args.max_steps,
        seq_length=args.seq_length,
        lr_sweep_steps=args.lr_sweep_steps,
        do_ablation=args.ablate,
    )


if __name__ == "__main__":
    main()
