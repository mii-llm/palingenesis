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

# Prefixes that training wrappers add to parameter names: torch.compile's
# OptimizedModule, activation checkpointing, FSDP1. A saved model must use the
# architecture's own names, or from_pretrained initialises those weights randomly.
_WRAPPER_PREFIXES = ("_orig_mod.", "_checkpoint_wrapped_module.", "_fsdp_wrapped_module.")


def hf_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A state dict with the wrapper prefixes removed from every name."""
    clean = {}
    for name, tensor in state.items():
        for prefix in _WRAPPER_PREFIXES:
            name = name.replace(prefix, "")
        clean[name] = tensor
    return clean


def save_hf_model(model, tokenizer, path: Path, state: dict[str, torch.Tensor] | None = None,
                  source_layout: bool = False) -> None:
    """Save in Hugging Face format under the architecture's parameter names.

    `state` is an already gathered full state dict (FSDP); by default the model's own.
    The live model is never modified.

    source_layout: for a model loaded as a causal LM from a checkpoint of another
    architecture (Qwen3.5's checkpoints are multimodal; transformers loads their
    language model as Qwen3_5ForCausalLM), save in that checkpoint's layout: its
    config, the weights under its names, and the weights the causal LM does not have
    (the vision tower) copied from it. Then it loads wherever the original does (vLLM
    serves Qwen3.5 only through the multimodal architecture). Exports use it; the
    trainer's own resume checkpoints keep the model's names.
    """
    path.mkdir(parents=True, exist_ok=True)
    state = hf_state_dict(state if state is not None else model.state_dict())
    source = _source_checkpoint(model) if source_layout else None
    if source is None:
        model.save_pretrained(path, state_dict=state, safe_serialization=True, max_shard_size=MAX_SHARD_SIZE)
    else:
        _save_in_source_layout(model, state, path, *source)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)


def _source_checkpoint(model):
    """(config, name or path) of the checkpoint `model` was loaded from, when its architecture
    differs from the model's own (a causal-LM view of a multimodal checkpoint); else None."""
    from transformers import AutoConfig

    own = getattr(model, "config", None)
    name = getattr(own, "_name_or_path", "") or ""
    if not name:
        return None
    try:
        config = AutoConfig.from_pretrained(name)
    except Exception:                                     # noqa: BLE001 — no source to follow
        return None
    if config.model_type == own.model_type:
        return None
    return config, name


def _source_dir(name: str) -> Path:
    """The source checkpoint's directory (local, or its Hub snapshot: weights and json files)."""
    local = Path(name)
    if not local.is_dir():
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(name, allow_patterns=["*.safetensors", "*.json"]))
    return local


def _save_in_source_layout(model, state: dict, path: Path, config, name: str) -> None:
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors import safe_open
    from transformers.core_model_loading import revert_weight_conversion

    seen: set[int] = set()
    unique = {}
    for key, tensor in state.items():             # tied weights once, as save_pretrained (before renaming,
        if tensor.data_ptr() not in seen:          # which may build new tensors)
            seen.add(tensor.data_ptr())
            unique[key] = tensor
    weights = revert_weight_conversion(model, unique)                # the source checkpoint's names
    source_dir = _source_dir(name)
    files = sorted(source_dir.glob("*.safetensors"))
    source_keys = {}
    for file in files:
        with safe_open(str(file), framework="pt") as f:
            source_keys.update(dict.fromkeys(f.keys(), file))
    unmatched = sorted(k for k in weights if k not in source_keys)
    if unmatched:
        # Never fill a trained weight's slot from the source: that would silently export the
        # untrained weight. Every trained tensor must land on one of the source's names.
        raise RuntimeError(f"cannot save in the layout of {name}: trained weights without a counterpart there "
                           f"({unmatched[:5]}{' ...' if len(unmatched) > 5 else ''})")
    copied = 0
    for file in files:
        with safe_open(str(file), framework="pt") as f:
            for key in f.keys():
                if key not in weights:                   # a module the causal LM does not have (vision)
                    weights[key] = f.get_tensor(key)
                    copied += 1
    weights = {k: v.detach().contiguous().cpu() for k, v in weights.items()}
    split = split_torch_state_dict_into_shards(weights, max_shard_size=MAX_SHARD_SIZE)
    for filename, keys in split.filename_to_tensors.items():
        safetensors_save({k: weights[k] for k in keys}, str(path / filename), metadata={"format": "pt"})
    if split.is_sharded:
        index = {"metadata": {"total_size": sum(v.numel() * v.element_size() for v in weights.values())},
                 "weight_map": split.tensor_to_filename}
        (path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    config.save_pretrained(path)
    for extra in source_dir.glob("*processor*.json"):      # image/video processors: the architecture needs them
        shutil.copy(extra, path / extra.name)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(path)
    logger.info("Saved in the layout of %s (%s): %d trained tensors, %d copied from it",
                name, type(config).__name__, len(weights) - copied, copied)


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
    save_hf_model(model, tokenizer, path / "model")

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
    Only one shard is in RAM at a time. Every parameter of the model must be found
    in the checkpoint (a weight tied to another counts as found through it).
    """
    model_path = path / "model"
    if model_path.exists():
        # Model names without wrapper prefixes -> the model's own names
        own = {hf: name for name, hf in zip(model.state_dict(), hf_state_dict(dict.fromkeys(model.state_dict())))}
        shard_files = sorted(model_path.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No model weights (*.safetensors) in {model_path}")
        loaded: set[str] = set()
        for shard_file in shard_files:
            # Memory-mapped load: only the accessed tensors are actually read
            shard = hf_state_dict(safetensors_load(str(shard_file), device=str(device) if device else "cpu"))
            unknown = [k for k in shard if k not in own]
            if unknown:
                raise RuntimeError(f"Checkpoint {model_path} has weights this model does not: {unknown[:5]}")
            model.load_state_dict({own[k]: v for k, v in shard.items()}, strict=False)
            loaded.update(shard)
            del shard  # Free immediately
        tied = _tied_parameter_names(model)
        missing = [k for k in own if k not in loaded and k not in tied]
        if missing:
            raise RuntimeError(f"Checkpoint {model_path} lacks weights of this model: {missing[:5]}")
        logger.info(f"Loaded model from {len(shard_files)} file(s)")

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


def _tied_parameter_names(model) -> set[str]:
    """HF names of parameters that share storage with an earlier one (tied embeddings),
    which save_pretrained writes once."""
    seen: dict[int, str] = {}
    tied = set()
    for name, tensor in hf_state_dict(model.state_dict()).items():
        ptr = tensor.untyped_storage().data_ptr() if tensor.device.type != "meta" else id(tensor)
        if ptr in seen:
            tied.add(name)
        else:
            seen[ptr] = name
    return tied


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
    _save_gathered_or_local(model, tokenizer, path, is_fsdp)
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(f"Final model saved (HF format) -> {path}")


def _save_gathered_or_local(model, tokenizer, path: Path, is_fsdp: bool) -> None:
    """HF-format save; under FSDP the full state is gathered to rank 0 (CPU) and saved
    from there, leaving every rank's sharded model untouched."""
    if is_fsdp and dist.is_initialized() and dist.get_world_size() > 1:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        dist.barrier()  # every rank has finished its step before the gather
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state = get_model_state_dict(model, options=opts)
        if dist.get_rank() == 0:
            save_hf_model(model, tokenizer, path, state=state, source_layout=True)
        del state
        dist.barrier()
    else:
        save_hf_model(model, tokenizer, path, source_layout=True)


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest COMPLETE checkpoint directory by step number.

    training_meta.json is written after the model and optimizer state (and, under
    FSDP, after every rank's DCP shard), so it marks a checkpoint whose save
    finished; a directory without it is a save interrupted halfway and is skipped.
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

        if (d / TRAINING_META_FILE).exists():
            valid_checkpoints.append((step_num, str(d)))
        else:
            logger.warning(f"Skipping incomplete checkpoint {d} (no {TRAINING_META_FILE})")

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
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if is_main and path.exists():
            shutil.rmtree(path)
        _save_gathered_or_local(model, tokenizer, path / "model", is_fsdp)
        if is_main:
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
