"""Checkpointing: sharded DCP for FSDP, safetensors for single-GPU, auto-purge.

Design (aligned with torchtitan):
  - Intermediate checkpoints: sharded DCP (each rank saves its shard, zero extra memory)
  - Final export: HF-compatible safetensors (gathered to rank 0, usable by from_pretrained)
  - Auto-purge: keeps only the latest K checkpoints to avoid filling disk
  - Async-ready: uses dcp.save which supports async_save for non-blocking I/O

For FSDP: uses PyTorch Distributed Checkpoint (DCP) — each rank saves/loads its shard.
For single GPU: uses sharded safetensors via HF save_pretrained.
"""

import json
import logging
import re
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save

logger = logging.getLogger(__name__)

TRAINING_META_FILE = "training_meta.json"
MAX_SHARD_SIZE = "2GB"
DEFAULT_KEEP_LATEST_K = 5  # Auto-purge: keep only last 5 checkpoints


def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    step: int,
    output_dir: str,
    is_fsdp: bool = False,
    epoch: int = 0,
    micro_step: int = 0,
    keep_latest_k: int = DEFAULT_KEEP_LATEST_K,
):
    """Save a full training checkpoint, then purge old ones.

    Model is saved in sharded safetensors (2GB shards). On 8B models this
    produces ~8 files, each loadable independently for low-memory resume.

    Auto-purge: after saving, removes checkpoints older than the latest K.
    Set keep_latest_k=0 to disable purging (keep all checkpoints).
    """
    path = Path(output_dir) / f"step-{step}"

    if is_fsdp and dist.is_initialized() and dist.get_world_size() > 1:
        _save_fsdp(model, tokenizer, optimizer, scheduler, step, epoch, micro_step, path)
    else:
        _save_single(model, tokenizer, optimizer, scheduler, step, epoch, micro_step, path)

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(f"Checkpoint saved: step {step} -> {path}")
        # Auto-purge old checkpoints (only rank 0 manages filesystem)
        if keep_latest_k > 0:
            _purge_old_checkpoints(output_dir, keep_latest_k)


def _save_single(model, tokenizer, optimizer, scheduler, step, epoch, micro_step, path):
    """Single-GPU save: sharded safetensors for model, torch for optimizer."""
    path.mkdir(parents=True, exist_ok=True)

    # Model in sharded safetensors (HF format, loadable by from_pretrained)
    model.save_pretrained(
        path / "model",
        safe_serialization=True,
        max_shard_size=MAX_SHARD_SIZE,
    )
    tokenizer.save_pretrained(path / "model")

    # Optimizer: save state_dict (complex nested structure)
    # We split into per-group files if state is large
    _save_optimizer_sharded(optimizer, path / "optimizer")

    # Metadata
    _save_meta(scheduler, step, epoch, micro_step, path)

    # RNG states
    _save_rng(path)


def _save_fsdp(model, tokenizer, optimizer, scheduler, step, epoch, micro_step, path):
    """FSDP2 save via Distributed Checkpoint (sharded, no gathering).

    Each rank saves only its local shard — zero extra memory, scales to any model size.
    This is the torchtitan pattern: dcp.save() with sharded state dicts.
    """
    from torch.distributed.checkpoint import save as dcp_save
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    path.mkdir(parents=True, exist_ok=True)

    # Sharded state dicts: each rank holds only its shard. NO gathering.
    # full_state_dict=False means we save each rank's local shard directly.
    opts = StateDictOptions(full_state_dict=False)

    state = {
        "model": get_model_state_dict(model, options=opts),
        "optimizer": get_optimizer_state_dict(model, optimizer, options=opts),
    }

    # DCP save: each rank writes its shard to a separate file in the directory.
    # For 8 GPUs: creates 8 shard files per (model + optimizer).
    # Zero extra memory — each rank only serializes what it already has.
    dcp_save(state, checkpoint_id=str(path / "dcp"))

    # Only rank 0 saves non-distributed state (metadata, tokenizer, RNG)
    if dist.get_rank() == 0:
        tokenizer.save_pretrained(path / "tokenizer")
        _save_meta(scheduler, step, epoch, micro_step, path)
        _save_rng(path)

    dist.barrier()


def _save_optimizer_sharded(optimizer, path: Path):
    """Save optimizer state in chunks to avoid massive single file.

    Splits optimizer state by parameter groups. For 8B models with AdamW,
    full optimizer state is ~48GB in fp32. Sharding keeps each file manageable.
    """
    path.mkdir(parents=True, exist_ok=True)
    state_dict = optimizer.state_dict()

    # Save param_groups (small, JSON-compatible structure)
    with open(path / "param_groups.json", "w") as f:
        # param_groups contain non-tensor metadata
        groups_meta = []
        for g in state_dict["param_groups"]:
            groups_meta.append({k: v for k, v in g.items() if k != "params"})
            groups_meta[-1]["params"] = g["params"]  # list of param indices
        json.dump(groups_meta, f, default=str)

    # Save state tensors in shards (one file per N params)
    shard_size = 50  # params per shard file
    state = state_dict["state"]
    param_ids = sorted(state.keys())

    for shard_idx in range(0, len(param_ids), shard_size):
        shard_params = param_ids[shard_idx : shard_idx + shard_size]
        shard_tensors = {}
        shard_meta = {}
        for pid in shard_params:
            for key, val in state[pid].items():
                if isinstance(val, torch.Tensor):
                    # contiguous copy: safetensors rejects views/shared storage
                    shard_tensors[f"{pid}.{key}"] = val.detach().cpu().contiguous().clone()
                else:
                    # Anything non-tensor: step counters, but also nested
                    # structures (bitsandbytes 8-bit optimizers keep dicts
                    # that CONTAIN tensors). torch.save handles all of it;
                    # JSON does not — it crashed on Lion8bit state.
                    shard_meta[f"{pid}.{key}"] = val

        if shard_tensors:
            safetensors_save(shard_tensors, str(path / f"shard_{shard_idx:04d}.safetensors"))
        if shard_meta:
            torch.save(shard_meta, path / f"shard_{shard_idx:04d}_meta.pt")


def _save_meta(scheduler, step, epoch, micro_step, path):
    meta = {
        "step": step,
        "epoch": epoch,
        "micro_step": micro_step,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else {},
    }
    with open(path / TRAINING_META_FILE, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def _save_rng(path):
    rng = {
        "cpu_rng": torch.random.get_rng_state().to(torch.float32),
        "cuda_rng": torch.cuda.get_rng_state().to(torch.float32) if torch.cuda.is_available() else torch.zeros(1),
    }
    safetensors_save(rng, str(path / "rng_state.safetensors"))


# ══════════════════════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════════════════════


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_dir: str,
    is_fsdp: bool = False,
    device: torch.device | None = None,
) -> dict:
    """Load checkpoint for resume. Low-memory: loads shards one at a time.

    Returns metadata dict with step, epoch, micro_step.
    """
    path = Path(checkpoint_dir)
    if not path.exists():
        logger.warning(f"Checkpoint {path} not found, starting fresh.")
        return {"step": 0, "epoch": 0, "micro_step": 0}

    meta_path = path / TRAINING_META_FILE
    if not meta_path.exists():
        logger.warning(f"No {TRAINING_META_FILE} in {path}, starting fresh.")
        return {"step": 0, "epoch": 0, "micro_step": 0}

    with open(meta_path) as f:
        meta = json.load(f)

    if is_fsdp and dist.is_initialized() and dist.get_world_size() > 1:
        _load_fsdp(model, optimizer, path)
    else:
        _load_single(model, optimizer, path, device)

    # Scheduler
    if scheduler is not None and meta.get("scheduler_state"):
        scheduler.load_state_dict(meta["scheduler_state"])

    # RNG
    rng_path = path / "rng_state.safetensors"
    if rng_path.exists():
        rng = safetensors_load(str(rng_path))
        torch.random.set_rng_state(rng["cpu_rng"].to(torch.uint8))
        if device and device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda_rng"].to(torch.uint8))

    logger.info(f"Resumed from step={meta['step']}, epoch={meta['epoch']}")
    return meta


def _load_single(model, optimizer, path, device):
    """Load single-GPU checkpoint with low memory usage.

    Model weights are loaded shard-by-shard using safetensors memory mapping.
    Only one shard is in RAM at a time.
    """
    model_path = path / "model"
    if model_path.exists():
        # Load model shards one by one (low RAM)
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            # Sharded model: load each shard and apply to model
            with open(index_file) as f:
                index = json.load(f)
            # Get unique shard files
            shard_files = sorted(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = model_path / shard_file
                # Memory-mapped load: only the accessed tensors are actually read
                shard_dict = safetensors_load(str(shard_path), device=str(device) if device else "cpu")
                # Apply to model
                missing, unexpected = model.load_state_dict(shard_dict, strict=False)
                del shard_dict  # Free immediately
            logger.info(f"Loaded model from {len(shard_files)} shards")
        else:
            # Single file model
            sf_files = list(model_path.glob("*.safetensors"))
            if sf_files:
                state = safetensors_load(str(sf_files[0]), device=str(device) if device else "cpu")
                model.load_state_dict(state, strict=False)
                del state
                logger.info("Loaded model from single safetensors file")

    # Optimizer: load sharded
    optim_path = path / "optimizer"
    if optim_path.exists():
        _load_optimizer_sharded(optimizer, optim_path, device)
    elif (path / "optimizer.pt").exists():
        # Legacy: single file optimizer
        state = torch.load(path / "optimizer.pt", map_location=device or "cpu", weights_only=False)
        optimizer.load_state_dict(state)
        del state
        logger.info("Loaded optimizer (legacy single file)")


def _load_optimizer_sharded(optimizer, path: Path, device):
    """Load sharded optimizer state. One shard at a time for low memory."""
    # Load param_groups metadata
    groups_path = path / "param_groups.json"
    if not groups_path.exists():
        logger.warning("No param_groups.json found, skipping optimizer load")
        return

    with open(groups_path) as f:
        groups_meta = json.load(f)

    # Reconstruct state dict
    state = {}

    # Load shard files
    shard_files = sorted(path.glob("shard_*.safetensors"))
    for shard_file in shard_files:
        # Load tensors
        tensors = safetensors_load(str(shard_file), device=str(device) if device else "cpu")

        # Load corresponding meta (step counts, non-tensor state).
        # New format: torch.save (.pt) — preserves types exactly, including
        # nested structures with tensors (bitsandbytes 8-bit state).
        # Legacy format: .json (step counts only, pre-fix checkpoints).
        shard_meta = {}
        legacy_json = False
        meta_pt = shard_file.parent / (shard_file.stem + "_meta.pt")
        meta_json = shard_file.parent / (shard_file.stem + "_meta.json")
        if meta_pt.exists():
            shard_meta = torch.load(meta_pt, map_location=device or "cpu", weights_only=False)
        elif meta_json.exists():
            legacy_json = True
            with open(meta_json) as f:
                shard_meta = json.load(f)

        # Reconstruct per-param state from flat keys
        for key, tensor in tensors.items():
            pid_str, attr = key.rsplit(".", 1)
            pid = int(pid_str)
            if pid not in state:
                state[pid] = {}
            state[pid][attr] = tensor

        for key, val in shard_meta.items():
            pid_str, attr = key.rsplit(".", 1)
            pid = int(pid_str)
            if pid not in state:
                state[pid] = {}
            # Legacy JSON stored step as a plain number; torch optimizers
            # expect a tensor. The .pt path preserves the original type.
            if legacy_json and attr == "step":
                state[pid][attr] = torch.tensor(float(val))
            else:
                state[pid][attr] = val

        del tensors  # Free shard memory

    # Reconstruct full state_dict
    full_state = {"state": state, "param_groups": groups_meta}
    optimizer.load_state_dict(full_state)
    logger.info(f"Loaded optimizer from {len(shard_files)} shards")


def _load_fsdp(model, optimizer, path):
    """Load FSDP2 distributed checkpoint (sharded, each rank loads its shard).

    Mirrors the save: each rank loads only the shard it needs. No gathering.
    DCP handles the shard-to-rank mapping automatically based on the FSDP mesh.
    """
    from torch.distributed.checkpoint import load as dcp_load
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    dcp_path = path / "dcp"
    if not dcp_path.exists():
        logger.warning(f"DCP path {dcp_path} not found")
        return

    # Sharded load: each rank gets empty state dict containers,
    # DCP fills them with the correct shard for this rank.
    opts = StateDictOptions(full_state_dict=False)

    # Get empty state dict containers (shaped correctly for this rank's shard)
    model_state = get_model_state_dict(model, options=opts)
    optim_state = get_optimizer_state_dict(model, optimizer, options=opts)

    # Single DCP load call — loads both model and optimizer shards at once
    state = {"model": model_state, "optimizer": optim_state}
    dcp_load(state, checkpoint_id=str(dcp_path))

    # Apply loaded shards back to model and optimizer
    set_model_state_dict(model, model_state, options=opts)
    set_optimizer_state_dict(model, optimizer, optim_state, options=opts)

    logger.info("Loaded FSDP distributed checkpoint (sharded)")
    dist.barrier()


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SAVE + UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def save_final(model, tokenizer, output_dir: str, is_fsdp: bool = False):
    """Save final model in HF-compatible sharded safetensors format.

    For FSDP: gathers the full model state to rank 0, then saves in HF format.
    This requires ~2× model size in CPU RAM on rank 0 (gathered state + model).
    For a 4B model: ~16GB CPU RAM. For 35B: ~140GB. Plan accordingly.

    The final export is always HF-format (loadable by from_pretrained anywhere).
    Intermediate checkpoints use sharded DCP (fast, zero extra memory).
    """
    path = Path(output_dir) / "final"

    if is_fsdp and dist.is_initialized() and dist.get_world_size() > 1:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        dist.barrier()  # Ensure all ranks finished training before gathering

        # Gather full state to rank 0 (cpu_offload=True to avoid GPU OOM)
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state = get_model_state_dict(model, options=opts)

        if dist.get_rank() == 0:
            path.mkdir(parents=True, exist_ok=True)
            # Apply gathered state to a local model copy for save_pretrained
            model.load_state_dict(state, assign=True)
            model.save_pretrained(path, safe_serialization=True, max_shard_size=MAX_SHARD_SIZE)
            tokenizer.save_pretrained(path)
            logger.info(f"Final model saved (HF format) -> {path}")

        dist.barrier()
    else:
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path, safe_serialization=True, max_shard_size=MAX_SHARD_SIZE)
        tokenizer.save_pretrained(path)
        logger.info(f"Final model saved -> {path}")


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest VALID checkpoint directory by step number.

    A checkpoint is valid only if it contains either:
      - training_meta.json (our format)
      - .metadata (DCP format marker)
      - model.safetensors.index.json (HF sharded format)

    This prevents picking up half-written checkpoints from interrupted saves.
    (Aligned with torchtitan's validity check pattern.)
    """
    base = Path(output_dir)
    if not base.exists():
        return None

    valid_checkpoints = []
    for d in base.iterdir():
        if not d.is_dir() or not d.name.startswith("step-"):
            continue
        try:
            step_num = int(d.name.split("-")[1])
        except (ValueError, IndexError):
            continue

        # Check validity: at least one completion marker must exist
        has_meta = (d / TRAINING_META_FILE).exists()
        has_dcp = (d / "dcp" / ".metadata").exists()
        has_hf = (d / "model" / "model.safetensors.index.json").exists()

        if has_meta or has_dcp or has_hf:
            valid_checkpoints.append((step_num, str(d)))

    if not valid_checkpoints:
        return None

    valid_checkpoints.sort(key=lambda x: x[0], reverse=True)
    return valid_checkpoints[0][1]


def _purge_old_checkpoints(output_dir: str, keep_latest_k: int):
    """Remove old checkpoint directories, keeping only the latest K.

    Runs on rank 0 only. Deletes directories synchronously (simple, reliable).
    For production with very frequent checkpoints, consider moving to a
    background thread (like torchtitan's purge_thread).

    NOTE: Never purges 'best/' or 'final/' directories — only step-N checkpoints.
    """
    base = Path(output_dir)
    if not base.exists():
        return

    checkpoints = []
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith("step-"):
            match = re.search(r"step-(\d+)", d.name)
            if match:
                checkpoints.append((int(match.group(1)), d))

    if len(checkpoints) <= keep_latest_k:
        return

    # Sort by step number, delete oldest
    checkpoints.sort(key=lambda x: x[0])
    to_delete = checkpoints[:-keep_latest_k]

    for step_num, path in to_delete:
        try:
            shutil.rmtree(path)
            logger.info(f"Purged old checkpoint: step-{step_num}")
        except OSError as e:
            logger.warning(f"Failed to purge {path}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# BEST MODEL TRACKING
# ══════════════════════════════════════════════════════════════════════════════


class BestModelTracker:
    """Track the best model checkpoint by eval loss during training.

    SFT commonly overfits — the last checkpoint is often not the best one.
    This tracker saves a copy of the model whenever eval loss reaches a new
    minimum, so the user always has access to the best checkpoint.

    Output structure:
        output_dir/
        ├── step-100/     # periodic checkpoint (may be purged)
        ├── step-200/     # periodic checkpoint
        ├── best/         # lowest eval loss (NEVER purged)
        │   ├── model/
        │   ├── tokenizer/
        │   └── best_meta.json  (step, eval_loss)
        └── final/        # last step (always kept)

    Usage in training loop:
        tracker = BestModelTracker(output_dir)
        ...
        if eval_loss is not None:
            if tracker.update(eval_loss, step, model, tokenizer, is_fsdp):
                logger.info(f"New best model at step {step} (eval_loss={eval_loss:.4f})")
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.best_loss = float("inf")
        self.best_step = -1

    def update(
        self,
        eval_loss: float,
        step: int,
        model,
        tokenizer,
        is_fsdp: bool = False,
    ) -> bool:
        """Update best model if eval_loss is a new minimum.

        Returns True if a new best was saved, False otherwise.
        Only rank 0 performs the actual save. Other ranks participate
        in the FSDP gather if needed.
        """
        if eval_loss >= self.best_loss:
            return False

        self.best_loss = eval_loss
        self.best_step = step

        path = Path(self.output_dir) / "best"

        if is_fsdp and dist.is_initialized() and dist.get_world_size() > 1:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state = get_model_state_dict(model, options=opts)

            if dist.get_rank() == 0:
                # Remove old best, save new one
                if path.exists():
                    shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
                model.load_state_dict(state, assign=True)
                model.save_pretrained(path / "model", safe_serialization=True, max_shard_size=MAX_SHARD_SIZE)
                tokenizer.save_pretrained(path / "model")
                _save_best_meta(path, step, eval_loss)

            dist.barrier()
        else:
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(path / "model", safe_serialization=True, max_shard_size=MAX_SHARD_SIZE)
            tokenizer.save_pretrained(path / "model")
            _save_best_meta(path, step, eval_loss)

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f"Best model updated: step={step}, eval_loss={eval_loss:.4f} -> {path}")

        return True

    @property
    def has_best(self) -> bool:
        return self.best_step >= 0


def _save_best_meta(path: Path, step: int, eval_loss: float):
    """Save metadata for the best checkpoint."""
    meta = {"step": step, "eval_loss": round(eval_loss, 6)}
    with open(path / "best_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
