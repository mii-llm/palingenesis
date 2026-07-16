"""
Main training loop — ultra-optimized for long-sequence agentic SFT.

Optimizations stacked:
  1. FSDP2 (fully_shard) — per-parameter sharding, communication overlap
  2. Context Parallel — sequence sharding across GPUs via Ring Attention
  3. Liger Kernel — fused RMSNorm, SwiGLU, RoPE, CrossEntropy (Triton)
  4. Chunked CE Loss — never materializes full [B, S, V] logit tensor
  5. Selective Activation Checkpointing — save SDPA + half matmuls, recompute rest
  6. torch.compile per-layer — graph optimization without full-model recompile
  7. bf16 mixed precision with fp32 gradient reduction
  8. Global valid-token normalization — correct loss scaling across DP ranks

Launch:
    # Single A100
    torchrun --standalone --nproc_per_node=1 -m palingenesis.train --config config.yaml

    # 8x A100
    torchrun --standalone --nproc_per_node=8 -m palingenesis.train --config config.yaml

    # Multi-node
    torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d \\
             --rdzv_endpoint=$MASTER:29500 -m palingenesis.train --config config.yaml
"""

import logging
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from palingenesis.checkpoint import (
    BestModelTracker,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_final,
)
from palingenesis.config import Config
from palingenesis.context_parallel import enable_context_parallel, shard_for_context_parallel
from palingenesis.data import IGNORE_INDEX, _load_dataset_source, build_dataloader
from palingenesis.distributed import apply_fsdp, build_mesh, cleanup_distributed, is_main, setup_distributed
from palingenesis.health import HealthMonitor
from palingenesis.kernels import apply_activation_checkpointing, apply_liger_kernel
from palingenesis.logging import Tracker, setup_logging
from palingenesis.loss import (
    cce_available,
    chunked_cross_entropy_loss,
    cross_entropy_loss,
    cut_cross_entropy_loss,
    shift_labels,
)
from palingenesis.optim import AdamCCorrection, build_optimizer, build_scheduler
from palingenesis.perf import (
    AdaGC,
    BaseModelMerge,
    CUDAPrefetcher,
    GCControl,
    ModelEMA,
    SpikeDetector,
)
from palingenesis.plugins import (
    SymNoiseHook,
    build_schedule_free_optimizer,
    cadft_loss,
    chunked_deft_loss,
    deft_loss,
    dft_loss,
    infosft_weighted_loss,
    pre_rl_loss,
)

logger = logging.getLogger(__name__)


# fp32 logits budget per loss chunk. 2GB → a 16×4096 batch of a 152K-vocab
# model runs 20 chunks instead of 64; a length-grouped 16×512 batch runs 3.
LOSS_CHUNK_TARGET_GB = 2.0


def _dynamic_num_chunks(batch_tokens: int, vocab_size: int, target_gb: float = LOSS_CHUNK_TARGET_GB) -> int:
    """Loss chunks for the ACTUAL batch, not the max_seq_length worst case.

    Each chunk materializes [tokens/chunks, vocab] fp32 logits; keep that under
    target_gb. Fewer chunks = fewer sequential lm_head fwd+bwd passes = faster.
    """
    logit_gb = batch_tokens * vocab_size * 4 / 1e9
    return max(1, min(64, math.ceil(logit_gb / target_gb)))


def _resolve_total_steps(config: Config, dataset_len: int | None, world_size: int) -> int:
    """Total optimizer steps for the LR schedule (warmup length + decay horizon).

    Priority:
      1. train.max_steps if set
      2. derived from epochs × dataset size (map-style datasets)
      3. 100k fallback (streaming without max_steps — warn loudly, because a
         short run would otherwise sit inside warmup forever and never reach
         peak LR, let alone decay)
    """
    if config.train.max_steps > 0:
        logger.info(f"LR schedule: total_steps={config.train.max_steps} (from train.max_steps)")
        return config.train.max_steps

    if dataset_len is not None:
        samples_per_step = (
            config.train.per_device_batch_size * world_size * config.train.gradient_accumulation_steps
        )
        steps_per_epoch = max(1, math.ceil(dataset_len / samples_per_step))
        total_steps = steps_per_epoch * config.train.epochs
        note = " (upper bound: packing reduces actual steps)" if config.data.packing else ""
        logger.info(
            f"LR schedule: total_steps={total_steps} derived from "
            f"{config.train.epochs} epoch(s) × {steps_per_epoch} steps/epoch "
            f"({dataset_len} samples / {samples_per_step} per step){note}"
        )
        return total_steps

    logger.warning(
        "train.max_steps is not set and the dataset is streaming (no length): "
        "assuming 100,000 steps for the LR schedule. With warmup_ratio="
        f"{config.train.warmup_ratio} that means {int(100_000 * config.train.warmup_ratio)} warmup steps — "
        "short runs will NEVER leave warmup or reach peak LR. Set train.max_steps explicitly."
    )
    return 100_000


def train(config: Config):
    # ── CUDA allocator: expandable segments ───────────────────────────────
    # Variable-length batches fragment the caching allocator (tens of GB
    # "reserved but unallocated" → spurious OOM despite free memory).
    # Must be set before the first CUDA allocation; user env wins if set.
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and "PYTORCH_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ── Distributed ───────────────────────────────────────────────────────
    rank, local_rank, world_size = setup_distributed()
    setup_logging(rank)
    device = torch.device(f"cuda:{local_rank}")

    logger.info(f"rank={rank} local_rank={local_rank} world_size={world_size}")

    # ── Config validation (raises on hard incompatibilities) ─────────────
    for warning in config.validate():
        logger.warning(f"Config warning: {warning}")

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(config.train.seed + rank)
    torch.cuda.manual_seed_all(config.train.seed + rank)
    torch.set_float32_matmul_precision(config.memory.float32_matmul_precision)

    # ── Liger Kernel (BEFORE model load) ──────────────────────────────────
    if config.model.use_liger_kernel:
        model_type = _infer_model_type(config.model.name_or_path)
        apply_liger_kernel(model_type)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Model ─────────────────────────────────────────────────────────────
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map[config.model.torch_dtype]

    logger.info(f"Loading model: {config.model.name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=model_dtype,
        attn_implementation=config.model.attn_implementation,
        trust_remote_code=config.model.trust_remote_code,
        low_cpu_mem_usage=True,  # Load shard-by-shard, no 2x RAM spike
    )
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # ── Float8 Training (H100+ speedup, before AC and FSDP) ─────────────
    if config.memory.float8_training:
        from palingenesis.float8 import apply_float8_training

        apply_float8_training(model)

    # ── Activation Checkpointing (before FSDP, before compile) ────────────
    apply_activation_checkpointing(model, mode=config.train.gradient_checkpointing)

    # ── Hybrid model freeze (DeltaNet/SSM layers frozen, only attention trains) ──
    if config.train.freeze_non_attention:
        _freeze_non_attention_layers(model)

    # ── Selective Differentiation (arxiv:2404.12406): eliminate activation memory for frozen layers ──
    if config.memory.selective_diff and config.train.freeze_non_attention:
        from palingenesis.memory import apply_selective_diff_v2

        apply_selective_diff_v2(model)

    # ── Device Mesh + FSDP2 ───────────────────────────────────────────────
    mesh = build_mesh(world_size, config.parallel)
    is_fsdp = config.parallel.fsdp and world_size > 1 and mesh is not None

    if is_fsdp:
        logger.info("Applying FSDP2 (fully_shard)")
        model = apply_fsdp(model, mesh, config.parallel, bf16=config.train.bf16)
    else:
        model = model.to(device)

    # ── Context Parallel ──────────────────────────────────────────────────
    cp_enabled = config.parallel.context_parallel and mesh is not None and "cp" in (mesh.mesh_dim_names or ())
    cp_mesh = mesh["cp"] if cp_enabled else None
    if cp_enabled:
        enable_context_parallel(cp_mesh, config.parallel.cp_rotate_method)

    # ── torch.compile (per-layer for efficiency) ──────────────────────────
    if config.model.compile:
        logger.info(
            f"Compiling model layers (backend={config.model.compile_backend}, mode={config.model.compile_mode})"
        )
        _compile_layers(model, backend=config.model.compile_backend, mode=config.model.compile_mode)

    # ── Dataset ───────────────────────────────────────────────────────────
    logger.info("Loading dataset(s)")
    if config.data.sources:
        logger.info(f"  Multi-source mode: {len(config.data.sources)} datasets")
        for src in config.data.sources:
            logger.info(
                f"    {src.get('dataset', '?')} (weight={src.get('weight', 1.0)}, mode={src.get('mode', 'sft')})"
            )
        dataloader = build_dataloader(
            config.data, tokenizer, config.data, rank, world_size, config.train.per_device_batch_size
        )
        dataset_len = None
    else:
        # Preprocess integration: when enabled, train on the prepared output
        # (produced by `pgs prepare --config <same config>`) instead of the raw dataset.
        dataset_id = config.data.dataset
        preserve_order = False
        if config.preprocess.enabled:
            from palingenesis.prepare import find_prepared_dataset, find_prepared_eval

            prepared = find_prepared_dataset(config.preprocess.output_dir)
            if prepared is None:
                raise FileNotFoundError(
                    f"preprocess.enabled=true but no prepared dataset found in "
                    f"'{config.preprocess.output_dir}'. Run first:\n"
                    f"  pgs prepare --config <this config>"
                )
            dataset_id = str(prepared)
            # Curriculum ordering is meaningful — don't shuffle it away
            preserve_order = config.preprocess.strategy == "curriculum"
            if preserve_order and config.data.length_group_buffer > 0:
                config.data.length_group_buffer = 0
                logger.info("  Curriculum strategy: length-grouped batching disabled to preserve ordering")
            logger.info(f"  Using prepared dataset: {prepared} (strategy={config.preprocess.strategy})")
            _log_prepared_meta(config.preprocess.output_dir)

            # Auto-use the held-out eval set written by `pgs prepare` (if the
            # user didn't point eval_dataset elsewhere). Disjoint from the
            # training pool → a trustworthy same-distribution eval/gap.
            if not config.data.eval_dataset:
                eval_file = find_prepared_eval(config.preprocess.output_dir)
                if eval_file is not None:
                    config.data.eval_dataset = str(eval_file)
                    logger.info(f"  Using held-out eval set: {eval_file}")

        logger.info(f"  Single dataset: {dataset_id}")
        dataset = _load_dataset_source(dataset_id, config.data.dataset_split, config.data.streaming)
        try:
            dataset_len = len(dataset)  # map-style (non-streaming) only
        except TypeError:
            dataset_len = None
        if not preserve_order:
            if config.data.streaming:
                dataset = dataset.shuffle(seed=config.train.seed, buffer_size=10_000)
            else:
                # Map-style datasets were previously NEVER shuffled — samples
                # arrived in file order every epoch (bad for optimization).
                dataset = dataset.shuffle(seed=config.train.seed)
        dataloader = build_dataloader(
            dataset, tokenizer, config.data, rank, world_size, config.train.per_device_batch_size
        )

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    total_steps = _resolve_total_steps(config, dataset_len, world_size)
    if config.plugins.schedule_free:
        warmup_steps = int(total_steps * config.train.warmup_ratio)
        optimizer = build_schedule_free_optimizer(
            model, config.train.learning_rate, config.train.weight_decay, warmup_steps
        )
        scheduler = None  # Schedule-Free doesn't need a scheduler
    else:
        optimizer = build_optimizer(
            model,
            config.train.learning_rate,
            config.train.weight_decay,
            config.train.llrd_decay,
            use_muon=(config.train.optimizer == "muon"),
            optimizer_name=config.train.optimizer,
        )
        scheduler = build_scheduler(
            optimizer,
            config.train.lr_scheduler,
            total_steps,
            config.train.warmup_ratio,
            config.train.min_learning_rate / config.train.learning_rate,
        )

    # ── AdamC: Corrected Weight Decay (prevents end-of-training gradient explosion) ──
    _adamc = (
        AdamCCorrection(optimizer, config.train.learning_rate) if config.train.adamc and scheduler is not None else None
    )

    # ── Hyperball: norm-constrained optimization for scale-invariant layers (arxiv:2606.16899) ──
    _hyperball = None
    if config.train.hyperball:
        from palingenesis.optim import HyperballWrapper

        constrained_params = []
        for name, p in model.named_parameters():
            if (
                p.requires_grad
                and p.ndim == 2
                and not any(k in name.lower() for k in ("embed", "norm", "bias", "lm_head"))
            ):
                constrained_params.append(p)
        if constrained_params:
            _hyperball = HyperballWrapper(optimizer, constrained_params)
            logger.info(f"Hyperball enabled: {len(constrained_params)} weight matrices norm-constrained")

    # ── MONA: curvature-aware acceleration for Muon/Lion (arxiv:2605.26842) ──
    _mona = None
    if config.train.mona:
        from palingenesis.optim import MONAAcceleration

        _mona = MONAAcceleration(model, beta_a=config.train.mona_beta_a, lite=config.train.mona_lite)
        logger.info(f"MONA acceleration enabled (beta_a={config.train.mona_beta_a}, lite={config.train.mona_lite})")

    # ── Gradient Release (FORGE, arxiv:2606.22932): optimizer step fused into backward ──
    _grad_release = None
    if config.memory.gradient_release:
        if config.train.gradient_accumulation_steps > 1:
            logger.warning(
                "gradient_release requires gradient_accumulation_steps=1, disabling. "
                "Hint: increase per_device_batch_size instead (freed memory allows it)."
            )
        elif config.train.ga_ramp_start > 0:
            logger.warning("gradient_release incompatible with ga_ramp (dynamic accumulation), disabling.")
        elif config.train.optimizer == "muon":
            logger.warning("gradient_release incompatible with Muon optimizer (needs full gradient), disabling.")
        else:
            from palingenesis.memory import GradientRelease

            _grad_release = GradientRelease(model, optimizer, adagc=None)
            _grad_release.enable()
            logger.info("Gradient release enabled: optimizer steps fused into backward pass")

    # ── Loss computation setup ──────────────────────────────────────────────
    use_chunked_loss = config.memory.chunked_loss
    use_cce = False  # Cut Cross-Entropy (Apple, ICLR 2025)
    lm_head = None

    # Determine which loss backend to use:
    # Priority: CCE (if available & no token-weighting plugins) > chunked CE > standard CE
    _needs_logits = (
        config.plugins.dft
        or config.plugins.cadft
        or config.plugins.deft
        or config.plugins.info_sft
        or config.plugins.pre_rl
    )

    # Special case: DEFT + chunked_loss = use chunked_deft_loss (memory-efficient DEFT)
    # (resolved below, after lm_head is found — it requires a usable lm_head)
    _use_chunked_deft = False

    try:
        _loss_vocab_size = model.config.vocab_size
    except AttributeError:
        _loss_vocab_size = 128256

    if use_chunked_loss:
        lm_head = _get_lm_head(model)
        _use_chunked_deft = config.plugins.deft and lm_head is not None
        if lm_head is None:
            logger.warning("Could not find lm_head, falling back to standard loss")
            use_chunked_loss = False
            _use_chunked_deft = False
        elif cce_available() and not _needs_logits:
            # CCE: zero-memory CE, replaces chunked when no plugin needs full logits
            use_cce = True
            use_chunked_loss = False
            logger.info("Using Cut Cross-Entropy (Apple): zero logit materialization")
        else:
            # Chunk count is computed PER BATCH from actual tokens (see
            # _dynamic_num_chunks). A static count sized for max_seq_length
            # ran 64 sequential tiny lm_head fwd+bwd passes even on short
            # length-grouped batches — pure loop overhead.
            worst_case = _dynamic_num_chunks(
                config.train.per_device_batch_size * config.data.max_seq_length, _loss_vocab_size
            )
            kind = "DEFT" if _use_chunked_deft else "CE"
            logger.info(
                f"Chunked {kind} loss: dynamic chunking (per-batch, ≤{LOSS_CHUNK_TARGET_GB:.0f}GB fp32 "
                f"logits per chunk; worst case {worst_case} chunks at seq {config.data.max_seq_length})"
            )

    # ── Tracker ───────────────────────────────────────────────────────────
    tracker = Tracker(config, is_main=is_main())

    # ── Health Monitor ────────────────────────────────────────────────────
    health = HealthMonitor(
        model,
        tier2_every=config.logging.health_tier2_every,
        tier3_every=config.logging.health_tier3_every,
        rl_readiness=config.logging.rl_readiness,
        rl_entropy_floor=config.logging.rl_entropy_floor,
    )

    # ── Best Model Tracker (saves best eval-loss checkpoint) ──────────────
    # Driven by either the single eval_dataset OR the multi-source composite score.
    _best_tracker = (
        BestModelTracker(config.train.output_dir)
        if (config.data.eval_dataset or config.data.eval_sources)
        else None
    )

    # ── Multi-source eval (per-capability losses + weighted composite) ─────
    # When eval_sources is set it REPLACES the single eval_dataset for eval +
    # best-model tracking (arxiv:2603.21606): each source is scored independently
    # (e.g. eval/lm/loss, eval/mcqa/loss) and a weighted composite drives
    # BestModelTracker — immune to the token-count domination that a single mixed
    # eval_dataset suffers. Every rank evaluates the same fixed set identically.
    _multi_evaluator = None
    if config.data.eval_sources:
        from palingenesis.multi_eval import MultiEvaluator

        _multi_evaluator = MultiEvaluator(
            config.data.eval_sources, tokenizer, config.data.max_seq_length, device
        )

    # ── Validation Set (optional) ─────────────────────────────────────────
    # Skipped when eval_sources is configured (the multi-evaluator takes over).
    eval_batches: list | None = None
    if config.data.eval_dataset and not config.data.eval_sources:
        logger.info(f"Loading eval dataset: {config.data.eval_dataset} (split={config.data.eval_split})")
        eval_ds = _load_dataset_source(config.data.eval_dataset, config.data.eval_split, streaming=True)
        from palingenesis.data import ChatDataset, _collate_fn

        # Same masking as TRAINING (include_observations, train_on_reasoning):
        # otherwise eval measures a different token set than the one being
        # optimized — e.g. with ECHO (observations trained, but excluded from
        # eval) assistant-only eval CE can rise while the actual training
        # objective improves, which reads as phantom "overfitting".
        # turn_scaling is irrelevant here: eval computes unweighted CE.
        eval_chat = ChatDataset(
            eval_ds,
            tokenizer,
            config.data.max_seq_length,
            config.data.messages_field,
            rank=0,
            world_size=1,
            include_observations=config.data.include_observations,
            train_on_reasoning=config.data.train_on_reasoning,
            last_turn_only=config.data.last_turn_only,
        )
        # Pre-collect fixed eval samples (no streaming randomness)
        eval_batches = []
        count = 0
        for sample in eval_chat:
            eval_batches.append(sample)
            count += 1
            if count >= config.data.eval_samples:
                break
        if eval_batches:
            # Collate into small batches (batch 4 keeps the full-logits eval
            # forward ~5GB instead of ~20GB at batch 16), length-sorted so
            # each batch pads only to its own max (eval loss is a token-
            # weighted sum — order-free). pad_to_multiple matches training so
            # eval reuses the same torch.compile shape buckets.
            eval_batches.sort(key=lambda s: s["input_ids"].size(0))
            pad_id = tokenizer.pad_token_id or 0
            eval_collated = []
            for i in range(0, len(eval_batches), 4):
                batch = eval_batches[i : i + 4]
                eval_collated.append(_collate_fn(batch, pad_id, pad_to_multiple=64))
            eval_batches = eval_collated
            logger.info(f"  Eval set: {count} samples, {len(eval_batches)} batches")
        else:
            eval_batches = None
            logger.warning("  Eval set empty, disabling validation")

    # ── Plugins ───────────────────────────────────────────────────────────
    if config.plugins.sym_noise:
        # Instance stays alive via the forward hook it registers on the model
        SymNoiseHook(model, alpha=config.plugins.sym_noise_alpha)

    # Pre-RL reference logits (stale snapshot for KL anchoring)
    _pre_rl_ref_logits: torch.Tensor | None = None

    # ── Resume from Checkpoint ────────────────────────────────────────────
    start_step = 0
    start_epoch = 0
    start_micro = 0

    resume_path = config.train.resume_from
    if resume_path == "auto":
        resume_path = find_latest_checkpoint(config.train.output_dir)
        if resume_path:
            logger.info(f"Auto-resume: found checkpoint at {resume_path}")
        else:
            logger.info("Auto-resume: no checkpoint found, starting fresh")

    if resume_path:
        meta = load_checkpoint(model, optimizer, scheduler, resume_path, is_fsdp, device)
        start_step = meta.get("step", 0)
        start_epoch = meta.get("epoch", 0)
        start_micro = meta.get("micro_step", 0)
        logger.info(f"Resumed: step={start_step}, epoch={start_epoch}")

    # ── Performance: GC Control ───────────────────────────────────────────
    gc_ctrl = GCControl(gc_every=100)
    gc_ctrl.disable_auto_gc()

    # ── Spike Detection ───────────────────────────────────────────────────
    _spike_detector = (
        SpikeDetector(
            z_threshold=config.train.spike_z_threshold,
            warmup=50,
        )
        if config.train.spike_detection
        else None
    )

    # ── AdaGC: Per-Tensor Adaptive Gradient Clipping (ICML 2026) ──────────
    _adagc = (
        AdaGC(
            model,
            lambda_rel=config.train.adagc_lambda,
            beta=config.train.adagc_beta,
            warmup_steps=int(total_steps * config.train.warmup_ratio),
            global_max_norm=config.train.max_grad_norm,
        )
        if config.train.adagc
        else None
    )

    # Connect AdaGC to gradient release (if both active)
    if _grad_release is not None and _adagc is not None:
        _grad_release.adagc = _adagc
        logger.info("AdaGC connected to gradient release (per-tensor clipping in backward)")

    # ── EMA: Exponential Moving Average of Weights ────────────────────────
    _ema = ModelEMA(model, decay=config.train.ema_decay) if config.train.ema else None

    # ── Base Model Merge: Periodic pull-back toward pretrained (SFA) ──────
    _base_merge = (
        BaseModelMerge(model, merge_ratio=config.train.base_merge_ratio, method=config.train.base_merge_method)
        if config.train.base_merge
        else None
    )

    # ── Training Loop ─────────────────────────────────────────────────────
    _effective_batch = config.train.per_device_batch_size * world_size * config.train.gradient_accumulation_steps
    _warmup_steps = int(total_steps * config.train.warmup_ratio)
    _steps_note = "" if config.train.max_steps > 0 else f" (~{config.train.epochs} epoch(s))"
    logger.info(
        f"Training started │ total_steps={total_steps}{_steps_note} │ "
        f"warmup={_warmup_steps} ({config.train.warmup_ratio:.0%}) │ "
        f"effective_batch={_effective_batch} "
        f"({config.train.per_device_batch_size}/gpu × {world_size} gpu × {config.train.gradient_accumulation_steps} ga)"
    )
    if start_step > 0:
        logger.info(f"Resuming from step {start_step} — {max(total_steps - start_step, 0)} steps remaining")
    model.train()

    global_step = start_step
    accum_loss = 0.0
    accum_tokens = 0
    accum_micro = 0
    accum_ce = 0.0  # unweighted CE (chunked-DEFT side metric; free)
    accum_gate = 0.0  # mean DEFT trust gate
    accum_ce_micro = 0
    tokens_total = 0  # cumulative trained tokens (this process)
    # Whether train/loss IS a cross-entropy (ppl and eval/gap only make sense
    # against a CE; DEFT/DFT/CADFT/InfoSFT values are differently scaled)
    _objective_is_ce = not (
        config.plugins.deft or config.plugins.dft or config.plugins.cadft
        or config.plugins.info_sft or config.plugins.pre_rl
    )
    grad_accum = config.train.gradient_accumulation_steps
    # GA ramp: start small, increase to target over training (arxiv:2602.14208)
    ga_ramp_start = config.train.ga_ramp_start
    ga_ramp_enabled = ga_ramp_start > 0 and ga_ramp_start < grad_accum
    t_start = time.perf_counter()
    t_step = time.perf_counter()

    for epoch in range(start_epoch, config.train.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.train.epochs}")

        # For resume: count micro-steps to skip already-processed ones
        skip_count = start_micro if epoch == start_epoch else 0
        if skip_count > 0:
            logger.info(f"  Skipping {skip_count} micro-steps for resume...")

        # Wrap dataloader with CUDA prefetcher for overlapped H2D transfer
        batch_iter = CUDAPrefetcher(dataloader, device)
        _accum_counter = 0  # tracks micro-steps within current accumulation window

        for micro_step, batch in enumerate(batch_iter):
            # Fast-skip: count steps without processing (O(1) per skip on streaming)
            if micro_step < skip_count:
                continue
            # batch is ALREADY on GPU (prefetched via CUDA stream)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            # Shift labels for next-token prediction: logits[t] predicts input_ids[t+1].
            # The data pipeline yields labels aligned with input_ids; without this
            # shift the model would be trained to copy the current token.
            # position_ids for packed sequences (document-aware masking)
            position_ids = batch.get("position_ids", None)
            labels = shift_labels(batch["labels"], position_ids)

            # ── Context Parallel: shard sequence ──────────────────────
            if cp_enabled:
                input_ids, attention_mask, labels = shard_for_context_parallel(
                    input_ids, attention_mask, labels, cp_mesh
                )

            # ── Gradient sync control for accumulation ────────────────
            # Dynamic GA: ramp from ga_ramp_start to grad_accum over training
            if ga_ramp_enabled:
                progress = global_step / max(total_steps, 1)
                current_ga = ga_ramp_start + int((grad_accum - ga_ramp_start) * progress)
                current_ga = max(ga_ramp_start, min(grad_accum, current_ga))
            else:
                current_ga = grad_accum
            _accum_counter += 1
            is_last_micro = _accum_counter >= current_ga
            if is_fsdp:
                model.set_requires_gradient_sync(is_last_micro)

            # ── Forward ───────────────────────────────────────────────
            # Count valid tokens for this micro-batch (local)
            local_valid = (labels != IGNORE_INDEX).sum()

            # All-reduce valid tokens across DP ranks for correct loss normalization.
            # Without this, each rank divides by its own local count, causing
            # gradient scale inconsistency when ranks have different valid counts.
            # (aligned with torchtitan: global_valid_tokens used as loss denominator)
            # Must happen on EVERY micro-step, not just the last one — otherwise
            # micro-batches within one accumulation window get inconsistent scaling.
            if is_fsdp:
                global_valid = local_valid.clone()
                dist.all_reduce(global_valid, op=dist.ReduceOp.SUM)
                global_valid_tokens = max(global_valid.item(), 1)
            else:
                global_valid_tokens = max(local_valid.item(), 1)

            # Gradient accumulation scaling is baked into the loss denominator:
            # each micro-loss becomes (sum CE) / (valid_tokens * GA), so summing
            # gradients over the window averages the micro-batches. Without the
            # GA factor the accumulated gradient (and effective LR) is GA× too
            # large. It must be in the denominator (not applied after the fact)
            # because the chunked paths run their backward per chunk internally.
            loss_denom = global_valid_tokens * current_ga

            # Build forward kwargs (includes position_ids for packed sequences)
            fwd_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if position_ids is not None:
                fwd_kwargs["position_ids"] = position_ids

            with torch.amp.autocast("cuda", dtype=model_dtype, enabled=config.train.bf16):
                if use_cce:
                    hidden = _get_hidden_states(model, input_ids, attention_mask, position_ids)
                    loss = cut_cross_entropy_loss(
                        hidden,
                        labels,
                        lm_head,
                        global_valid_tokens=loss_denom,
                    )
                elif _use_chunked_deft:
                    hidden = _get_hidden_states(model, input_ids, attention_mask, position_ids)
                    _deft_stats: dict = {}
                    loss = chunked_deft_loss(
                        hidden,
                        labels,
                        lm_head,
                        num_chunks=_dynamic_num_chunks(input_ids.numel(), _loss_vocab_size),
                        global_valid_tokens=loss_denom,
                        stats=_deft_stats,
                    )
                    if _deft_stats.get("valid"):
                        accum_ce += _deft_stats["ce_sum"] / _deft_stats["valid"]
                        accum_gate += _deft_stats["gate_sum"] / _deft_stats["valid"]
                        accum_ce_micro += 1
                elif use_chunked_loss:
                    hidden = _get_hidden_states(model, input_ids, attention_mask, position_ids)
                    loss = chunked_cross_entropy_loss(
                        hidden,
                        labels,
                        lm_head,
                        num_chunks=_dynamic_num_chunks(input_ids.numel(), _loss_vocab_size),
                        global_valid_tokens=loss_denom,
                    )
                elif config.plugins.cadft:
                    outputs = model(**fwd_kwargs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    loss = cadft_loss(logits, labels, beta=config.plugins.cadft_beta)
                    loss = loss / loss_denom
                elif config.plugins.deft:
                    outputs = model(**fwd_kwargs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    loss = deft_loss(logits, labels)
                    loss = loss / loss_denom
                elif config.plugins.dft:
                    outputs = model(**fwd_kwargs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    loss = dft_loss(logits, labels)
                    loss = loss / loss_denom
                elif config.plugins.info_sft:
                    outputs = model(**fwd_kwargs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    loss = infosft_weighted_loss(logits, labels, p_bar=config.plugins.info_sft_pbar)
                    loss = loss / loss_denom
                elif config.plugins.pre_rl:
                    # KL anchoring: reference logits from a STALE model snapshot.
                    # Without a stale snapshot, KL(current || current) = 0 always.
                    # We use _pre_rl_ref_logits cached from a previous step.
                    # Updated every eval_every steps (or first step if None).
                    outputs = model(**fwd_kwargs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    if _pre_rl_ref_logits is None or _pre_rl_ref_logits.shape != logits.shape:
                        # First step or shape changed: use current as baseline (KL=0 for warmup)
                        _pre_rl_ref_logits = logits.detach().clone()
                    ref_logits = _pre_rl_ref_logits
                    loss = pre_rl_loss(
                        logits,
                        labels,
                        ref_logits,
                        entropy_coeff=config.plugins.pre_rl_entropy_coeff,
                        kl_coeff=config.plugins.pre_rl_kl_coeff,
                    )
                    # pre_rl_loss normalizes internally; only GA scaling needed
                    loss = loss / current_ga
                else:
                    outputs = model(**fwd_kwargs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    loss = cross_entropy_loss(logits, labels, loss_denom)

            # ── RL-readiness: record output entropy (once per logged step) ──
            if (
                config.logging.rl_readiness
                and is_last_micro
                and (global_step + 1) % config.train.logging_steps == 0
            ):
                if use_cce or _use_chunked_deft or use_chunked_loss:
                    health.record_entropy_from_hidden(hidden.detach(), labels, lm_head)
                else:
                    health.record_logit_entropy(logits.detach(), labels)

            # ── Backward ──────────────────────────────────────────────
            # Undo the GA factor for logging: loss_val is the true per-token
            # micro-loss, while `loss` carries the 1/GA gradient scaling.
            loss_val = loss.detach().float().item() * current_ga
            loss.backward()

            # MONA: augment gradients with curvature-aware acceleration (before optimizer step)
            if _mona is not None:
                _mona.apply()

            accum_loss += loss_val
            accum_micro += 1
            _n_valid = local_valid.item()
            accum_tokens += _n_valid
            tokens_total += _n_valid

            # Tier 1: record per-microstep (zero overhead)
            health.record_microstep(loss_val, labels)

            # ── Optimizer Step ────────────────────────────────────────
            if is_last_micro:
                _accum_counter = 0  # reset for next accumulation window

                # With gradient_release: optimizer already stepped during backward.
                # Spike detection is handled by AdaGC at per-tensor level inside backward.
                if _grad_release is not None:
                    grad_norm = _grad_release.last_grad_norm
                    global_step += 1
                    if scheduler is not None:
                        scheduler.step()
                        if _adamc is not None:
                            _adamc.step()
                    if _ema is not None and global_step % config.train.ema_every == 0:
                        _ema.update()
                else:
                    # Standard mode: clip, detect spikes, step
                    if _adagc:
                        grad_norm = _adagc.clip(global_step)
                    elif config.train.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                    else:
                        grad_norm = _grad_norm(model)

                    # Spike Detection (ZClip-inspired): skip if grad is anomalous
                    gn_val = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                    spike_detected = _spike_detector.check(gn_val) if _spike_detector else False

                    if spike_detected:
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        logger.warning(
                            f"step={global_step} SPIKE SKIPPED (grad_norm={gn_val:.1f}, "
                            f"avg={_spike_detector.mean:.2f})"
                        )
                    else:
                        if _hyperball is not None:
                            _hyperball.step()  # optimizer.step() + norm projection
                        else:
                            optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                            if _adamc is not None:
                                _adamc.step()
                        # NOTE: zero_grad happens AFTER the health/logging block
                        # below. Zeroing here made health's gradient metrics
                        # (grad_cosine_sim, gw_ratio) permanently empty because
                        # they inspect p.grad after the step.
                        global_step += 1
                        if _ema is not None and global_step % config.train.ema_every == 0:
                            _ema.update()

                dt = time.perf_counter() - t_step

                # GC: run every 100 steps to prevent stalls
                gc_ctrl.step(global_step)

                # GNS: flush accumulated micro-batch data
                health.flush_gns()

                # Pre-RL: update stale reference every 10 steps
                # (provides non-zero KL gradient signal for anchoring)
                if config.plugins.pre_rl and global_step % 10 == 0:
                    _pre_rl_ref_logits = None  # will be refreshed on next forward

                # ── Log ───────────────────────────────────────────────
                if global_step % config.train.logging_steps == 0:
                    tok_s = accum_tokens / max(dt, 1e-6)
                    lr = scheduler.get_last_lr()[0] if scheduler is not None else config.train.learning_rate
                    gn = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                    # Average per-token loss over the accumulation window.
                    # NOTE: with DEFT/DFT-family objectives this is the gated
                    # loss, numerically much smaller than a CE. train/ce_loss
                    # (when available) is the comparable cross-entropy.
                    step_loss = accum_loss / max(accum_micro, 1)
                    ce_loss = accum_ce / accum_ce_micro if accum_ce_micro else None

                    metrics = {
                        "train/loss": step_loss,
                        "train/lr": lr,
                        "train/tokens_per_sec": tok_s,
                        "train/tokens_per_sec_global": tok_s * world_size,
                        "train/tokens_total": tokens_total,
                        "train/step_time_s": dt,
                        "train/grad_norm": gn,
                        "train/global_step": global_step,
                        "train/epoch": epoch + 1,
                    }
                    # ppl is only meaningful from a true cross-entropy
                    if ce_loss is not None:
                        metrics["train/ce_loss"] = ce_loss
                        metrics["train/ppl"] = math.exp(min(ce_loss, 20.0))
                        metrics["train/deft_gate"] = accum_gate / accum_ce_micro
                    elif _objective_is_ce:
                        metrics["train/ppl"] = math.exp(min(step_loss, 20.0))

                    if _spike_detector is not None:
                        metrics["train/spikes_skipped"] = _spike_detector.spikes_detected
                    if _adagc is not None:
                        metrics["train/adagc_clips"] = _adagc.total_clips

                    # Health diagnostics (tiered — tier2 every 10, tier3 every 100)
                    health_metrics = health.on_step(global_step, model)
                    if health_metrics:
                        metrics.update(health_metrics)

                    # ── Validation Loss ────────────────────────────────
                    if eval_batches and global_step % config.data.eval_every == 0:
                        eval_loss = _compute_eval_loss(model, eval_batches, device, model_dtype, config.train.bf16)
                        metrics["eval/loss"] = eval_loss
                        metrics["eval/ppl"] = math.exp(min(eval_loss, 20.0))
                        # Generalization gap: eval CE - train CE (rising gap =
                        # overfitting). Must compare CE to CE — against a DEFT
                        # value the gap would be dominated by unit mismatch.
                        if ce_loss is not None:
                            metrics["eval/gap"] = eval_loss - ce_loss
                        elif _objective_is_ce:
                            metrics["eval/gap"] = eval_loss - step_loss
                        # Best model tracking: save if eval loss is new minimum
                        if _best_tracker is not None:
                            _best_tracker.update(eval_loss, global_step, model, tokenizer, is_fsdp)

                    # ── Multi-source Validation (per-capability + composite) ──
                    elif _multi_evaluator is not None and global_step % config.data.eval_every == 0:
                        me = _multi_evaluator.evaluate(model, dtype=model_dtype)
                        # Guard: if every source loaded empty (e.g. missing files),
                        # score is a meaningless 0.0 — don't log it or (falsely) save
                        # it as the best checkpoint.
                        if me.per_source:
                            # Composite (weighted) score → the headline eval loss.
                            metrics["eval/loss"] = me.score
                            metrics["eval/ppl"] = math.exp(min(me.score, 20.0))
                            # Per-source losses: e.g. eval/lm/loss, eval/mcqa/loss.
                            for _name, _loss in me.per_source.items():
                                metrics[f"eval/{_name}/loss"] = _loss
                                metrics[f"eval/{_name}/ppl"] = math.exp(min(_loss, 20.0))
                            if me.regressions:
                                metrics["eval/regressions"] = len(me.regressions)
                            if ce_loss is not None:
                                metrics["eval/gap"] = me.score - ce_loss
                            elif _objective_is_ce:
                                metrics["eval/gap"] = me.score - step_loss
                            # Best model tracking on the composite score.
                            if _best_tracker is not None:
                                _best_tracker.update(me.score, global_step, model, tokenizer, is_fsdp)

                    tracker.log(metrics, step=global_step)
                    if is_main():
                        eval_str = f" eval={metrics.get('eval/loss', 0):.4f}" if "eval/loss" in metrics else ""
                        entropy_str = (
                            f" entropy={metrics['health/output_entropy']:.2f}"
                            if "health/output_entropy" in metrics
                            else ""
                        )
                        ce_str = f" ce={ce_loss:.4f}" if ce_loss is not None else ""
                        logger.info(
                            f"step={global_step} loss={step_loss:.4f}{ce_str} lr={lr:.2e} "
                            f"tok/s={tok_s:.0f} grad_norm={gn:.3f} dt={dt:.2f}s{entropy_str}{eval_str}"
                        )

                # Deferred gradient zeroing: AFTER health/logging (which inspect
                # p.grad for cosine-similarity / grad-weight-ratio metrics).
                # No-op under gradient_release (grads freed in backward) and
                # after spike-skip (already zeroed above).
                if _grad_release is None:
                    optimizer.zero_grad(set_to_none=True)

                accum_loss, accum_tokens, accum_micro = 0.0, 0, 0
                accum_ce, accum_gate, accum_ce_micro = 0.0, 0.0, 0
                t_step = time.perf_counter()

                # ── Checkpoint ────────────────────────────────────────
                if config.train.save_steps > 0 and global_step % config.train.save_steps == 0:
                    save_checkpoint(
                        model,
                        tokenizer,
                        optimizer,
                        scheduler,
                        global_step,
                        config.train.output_dir,
                        is_fsdp,
                        epoch=epoch,
                        micro_step=micro_step,
                    )

                # ── Base Model Merge (SFA: anti-forgetting) ───────────
                if _base_merge is not None and global_step % config.train.base_merge_every == 0:
                    _base_merge.merge_step()
                    if is_main():
                        logger.info(f"step={global_step} base_merge applied (ratio={config.train.base_merge_ratio})")

                if config.train.max_steps > 0 and global_step >= config.train.max_steps:
                    break

        if config.train.max_steps > 0 and global_step >= config.train.max_steps:
            break

    # ── Finish ────────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_start
    logger.info(f"Done. {global_step} steps in {total_time:.1f}s")

    gc_ctrl.cleanup()

    # Apply EMA weights before final save (if enabled)
    if _ema is not None:
        logger.info("Applying EMA weights for final save (better generalization)")
        _ema.apply_to_model()

    save_final(model, tokenizer, config.train.output_dir, is_fsdp)
    tracker.log(
        {"train/total_steps": global_step, "train/total_time_s": total_time, "train/tokens_total": tokens_total},
        step=global_step,
    )
    tracker.finish()
    cleanup_distributed()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _log_prepared_meta(output_dir: str):
    """Log provenance of the prepared dataset (model/strategy/counts) if available."""
    import json
    from pathlib import Path

    from palingenesis.prepare import PREPARED_META

    meta_path = Path(output_dir) / PREPARED_META
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text())
        logger.info(
            f"  Prepared with model={meta.get('model')} strategy={meta.get('strategy')} "
            f"samples={meta.get('num_samples')} source={meta.get('source_dataset')}"
        )
    except Exception:
        pass


def _infer_model_type(name: str) -> str | None:
    for family in ("llama", "mistral", "qwen", "gemma", "phi"):
        if family in name.lower():
            return family
    return None


def _compile_layers(model: torch.nn.Module, backend: str = "inductor", mode: str = "default"):
    """Compile each transformer layer individually (like torchtitan).

    Modes:
      - "default": standard inductor compilation
      - "reduce-overhead": minimizes CPU overhead (good for small batches)
      - "max-autotune": autotuning for each GEMM shape (5-15% faster, longer compile)
    """
    # Dynamo config for MoE compatibility (aligned with torchtitan):
    # capture_scalar_outputs: needed for data-dependent dynamic shapes in MoE dispatch
    # skip_fwd_side_effects: avoids breaking AC recompute under compile (RoPE cache etc.)
    # Flags vary across torch versions (e.g. skip_fwd_side_effects... doesn't
    # exist in 2.8) and dynamo's config module raises on unknown names — guard.
    for flag in ("capture_scalar_outputs", "skip_fwd_side_effects_in_bwd_under_checkpoint"):
        if hasattr(torch._dynamo.config, flag):
            setattr(torch._dynamo.config, flag, True)
        else:
            logger.debug(f"torch._dynamo.config.{flag} not available in torch {torch.__version__}; skipping")

    # All wrapped layers share ONE code object (the layer class / AC-wrapper
    # forward) and Dynamo's recompile cache is per code object. Hybrid models
    # (Qwen3.5: DeltaNet + full attention) legitimately need many entries:
    # {Long 2D linear-attn mask, Bool 4D causal mask, None for dense batches}
    # × {static, dynamic seq} × {train, eval batch size}. The default limit
    # of 8 gets exhausted mid-warmup and Dynamo silently falls back to EAGER
    # ("expected Long, actual Bool" recompile spam, then nothing).
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 24)

    # fullgraph=False: hybrid architectures route through custom linear-attention
    # ops (fla) where a graph break must degrade gracefully, not hard-error.
    compile_kwargs = {"backend": backend, "fullgraph": False}
    if mode != "default":
        compile_kwargs["mode"] = mode

    for attr_path in ("model.layers", "transformer.h", "transformer.layers"):
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            for i, layer in enumerate(obj):
                obj[i] = torch.compile(layer, **compile_kwargs)
            return
        except (AttributeError, TypeError):
            continue


def _get_lm_head(model: torch.nn.Module):
    """Find the lm_head linear layer."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    return None


def _get_hidden_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Get hidden states from the model backbone (before lm_head).

    Works for HF models that have model.model as the transformer backbone.
    """
    backbone = getattr(model, "model", None) or getattr(model, "transformer", None)
    if backbone is None:
        raise RuntimeError("Cannot find model backbone for chunked loss. Disable memory.chunked_loss.")

    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if position_ids is not None:
        kwargs["position_ids"] = position_ids

    outputs = backbone(**kwargs)
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _grad_norm(model: torch.nn.Module) -> float:
    # One device sync total. The previous per-param `.float().norm().item()`
    # loop did ~330 GPU syncs AND ~330 fp32 grad copies per call.
    norms = [p.grad.norm(2, dtype=torch.float32) for p in model.parameters() if p.grad is not None]
    if not norms:
        return 0.0
    return torch.stack(norms).norm(2).item()


def _chunked_ce_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_tokens: int = 2048,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Summed cross-entropy over flattened tokens, computed in token chunks.

    A single full-batch `cross_entropy(logits.view(-1, V).float(), ...)` upcasts
    the entire [B·S, V] logits to fp32 AND materializes an equally large internal
    log-softmax — ~20GB of transient fp32 at seq 4096 / 150K vocab, which OOMs
    eval even though the (chunked) training step fits. Upcasting only a slice at
    a time bounds the peak to chunk_tokens×V. Returns a scalar tensor (the sum);
    identical to the one-shot result up to fp accumulation order.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    n_tokens = flat_labels.size(0)

    total = flat_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, n_tokens, chunk_tokens):
        end = min(start + chunk_tokens, n_tokens)
        total = total + F.cross_entropy(
            flat_logits[start:end].float(),
            flat_labels[start:end],
            reduction="sum",
            ignore_index=ignore_index,
        )
    return total


@torch.no_grad()
def _compute_eval_loss(
    model: torch.nn.Module,
    eval_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    bf16: bool,
) -> float:
    """Compute average CE loss over pre-collected eval batches. No gradient."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in eval_batches:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = shift_labels(batch["labels"].to(device))

        valid = (labels != IGNORE_INDEX).sum().item()
        if valid == 0:
            continue

        with torch.amp.autocast("cuda", dtype=dtype, enabled=bf16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Chunked CE (fp32 outside autocast) — bounds the transient fp32
        # logits/log-softmax that would otherwise OOM at large seq×vocab.
        loss = _chunked_ce_sum(logits, labels)
        total_loss += loss.item()
        total_tokens += valid
        del outputs, logits

    model.train()
    return total_loss / max(total_tokens, 1)


def _freeze_non_attention_layers(model: torch.nn.Module):
    """Freeze all layers EXCEPT attention blocks + lm_head + norms.

    For hybrid models (Qwen3.5, Mamba-hybrid, etc.) where research shows
    adapting the recurrent/DeltaNet/SSM backbone is destructive (arxiv:2604.22127).
    Only the attention pathway (minority component) should be adapted.

    Detects attention layers by name patterns:
      - self_attn, attention, attn: attention projections (Q, K, V, O)
      - lm_head: output projection (always trainable)
      - norm: normalization layers (always trainable for adaptation)
      - embed: embeddings (trainable for new token learning)

    Everything else (DeltaNet, delta_net, ssm, recurrent, mlp in non-attention
    blocks) gets frozen.
    """
    attention_keywords = ("self_attn", "attention", "attn", "q_proj", "k_proj", "v_proj", "o_proj")
    always_train = ("lm_head", "norm", "embed", "layernorm")

    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        name_lower = name.lower()
        # Always keep trainable
        if any(k in name_lower for k in always_train):
            param.requires_grad = True
            trainable_count += param.numel()
        # Keep attention layers trainable
        elif any(k in name_lower for k in attention_keywords):
            param.requires_grad = True
            trainable_count += param.numel()
        # Freeze everything else (DeltaNet, SSM, recurrent, MLP in non-attn blocks)
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    total = frozen_count + trainable_count
    logger.info(
        f"Hybrid freeze: {trainable_count:,} trainable ({100*trainable_count/total:.1f}%), "
        f"{frozen_count:,} frozen ({100*frozen_count/total:.1f}%)"
    )


def main():
    config = Config.from_cli()
    train(config)


if __name__ == "__main__":
    main()
