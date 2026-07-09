"""Metrics: trackio + wandb, rank-aware. Only rank 0 logs."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from palingenesis.config import Config

logger = logging.getLogger(__name__)

_RUN_ID_FILE = "tracker_run_id.json"


class Tracker:
    """Dual backend tracker: wandb + trackio.

    Robustness properties (a metrics backend must never kill a training run):
      - init failures degrade to a warning, training continues
      - log() failures are caught and rate-limited-warned per backend
      - the wandb run id is persisted in output_dir; a run that RESUMES from a
        checkpoint (train.resume_from) continues the same wandb run, while a
        FRESH start in the same output_dir rotates to a new run id. Reusing the
        id on a fresh start would reattach to the old run at a higher history
        step, and wandb silently drops all rows with a lower step.
      - wandb.log is called WITHOUT an explicit step: train/global_step is
        embedded in the payload and define_metric pins it as the x-axis for
        all metrics. This keeps charts aligned across restarts and makes
        step-monotonicity data loss impossible.
    """

    def __init__(self, config: Config, is_main: bool):
        self._wandb = None
        self._trackio = None
        self._wandb_errors = 0
        self._trackio_errors = 0
        if not is_main:
            return

        name = config.logging.run_name or f"sft-{config.model.name_or_path.split('/')[-1]}"
        flat_config = _flatten(config)
        resuming = _will_resume(config)

        if config.logging.use_wandb:
            try:
                import wandb

                run_id = _load_or_create_run_id(config.train.output_dir, name, reuse=resuming)
                self._wandb = wandb.init(
                    project=config.logging.project,
                    name=name,
                    id=run_id,
                    config=flat_config,
                    resume="allow",
                    dir=config.train.output_dir,
                )
                # Use the training step as x-axis for ALL metrics so train/*,
                # eval/* and health/* line up on the same axis across resumes.
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step")
            except Exception as e:
                logger.warning(f"wandb init failed: {e}")
                self._wandb = None

        if config.logging.use_trackio:
            try:
                import trackio

                # trackio resumes by (project, name): same name + resume="allow"
                # continues the run after a crash/restart.
                self._trackio = trackio.init(
                    project=config.logging.project,
                    name=name,
                    config=flat_config,
                    resume="allow",
                )
            except Exception as e:
                logger.warning(f"trackio init failed: {e}")
                self._trackio = None

    def log(self, metrics: dict[str, Any], step: int | None = None):
        # Include the step as a metric so wandb's define_metric x-axis works
        if step is not None and "train/global_step" not in metrics:
            metrics = {**metrics, "train/global_step": step}

        if self._wandb:
            try:
                # No step= on purpose: wandb DROPS any row whose explicit step
                # is <= the run's history step (e.g. after a resume or a rerun
                # attached to an old run). The x-axis comes from
                # train/global_step in the payload via define_metric instead.
                self._wandb.log(metrics)
            except Exception as e:
                self._wandb_errors += 1
                if self._wandb_errors <= 3:
                    logger.warning(f"wandb log failed (continuing training): {e}")

        if self._trackio:
            try:
                import trackio

                trackio.log(metrics, step=step)
            except Exception as e:
                self._trackio_errors += 1
                if self._trackio_errors <= 3:
                    logger.warning(f"trackio log failed (continuing training): {e}")

    def finish(self):
        if self._wandb:
            try:
                self._wandb.finish()
            except Exception as e:
                logger.warning(f"wandb finish failed: {e}")
        if self._trackio:
            try:
                import trackio

                trackio.finish()
            except Exception as e:
                logger.warning(f"trackio finish failed: {e}")


def _will_resume(config: Config) -> bool:
    """True when this training will resume from a checkpoint.

    Mirrors train.py's resume logic: an explicit path always resumes;
    "auto" resumes only if a checkpoint actually exists in output_dir.
    """
    resume_from = config.train.resume_from
    if not resume_from:
        return False
    if resume_from == "auto":
        try:
            from palingenesis.checkpoint import find_latest_checkpoint

            return find_latest_checkpoint(config.train.output_dir) is not None
        except Exception:
            return False
    return True


def _load_or_create_run_id(output_dir: str, name: str, reuse: bool = True) -> str:
    """Persist a wandb run id next to the checkpoints.

    reuse=True (checkpoint resume): return the persisted id so logging
    continues in the SAME wandb run instead of scattering one training
    over several runs.

    reuse=False (fresh start): always mint a new id, even if one is
    persisted. Reattaching a fresh run to an old wandb run would resume
    it at the old history step and wandb would silently drop everything.
    """
    path = Path(output_dir) / _RUN_ID_FILE
    if reuse:
        try:
            if path.exists():
                data = json.loads(path.read_text())
                if data.get("id"):
                    return data["id"]
        except Exception:
            pass

    import secrets

    run_id = secrets.token_hex(8)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": run_id, "name": name}))
    except Exception as e:
        logger.warning(f"Could not persist tracker run id: {e}")
    return run_id


def _flatten(config: Config) -> dict:
    out = {}
    for section in ("model", "data", "train", "parallel", "memory", "plugins", "preprocess", "logging"):
        for k, v in asdict(getattr(config, section)).items():
            out[f"{section}/{k}"] = v
    return out


def setup_logging(rank: int):
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        format=f"[rank {rank}] %(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        force=True,
    )
    # Silence noisy third-party loggers. httpx/httpcore emit an INFO line for
    # EVERY HuggingFace Hub HTTP request, which floods the training log.
    for name in (
        "transformers",
        "datasets",
        "torch.distributed",
        "httpx",
        "httpcore",
        "urllib3",
        "huggingface_hub",
        "filelock",
        "fsspec",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
