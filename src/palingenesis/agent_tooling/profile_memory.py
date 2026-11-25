#!/usr/bin/env python3
"""GPU memory for a training config: exact where it can be, measured when asked.

Usage:
    pgs profile --config configs/qwen35_4b/a100_80gb.yaml --gpu 80
    pgs profile --config configs/qwen35_4b/a100_80gb.yaml --measure     # real step on this GPU

Static estimate (no GPU needed):
    - parameters: EXACT, by building the architecture on the meta device (GQA,
      linear attention, MoE experts, tied embeddings and freezing all counted)
    - optimizer states, gradients: exact per the optimizer's state layout and the
      weight dtype (torch AdamW keeps two states in the weight dtype, no master copy)
    - logits: the trainer's per-batch chunk size
    - activations: a ROUGH analytic estimate (architecture-dependent; use --measure)

--measure runs the real trainer (palingenesis.train.train) for two optimizer
steps on worst-case rows (every row rendered to max_seq_length), with the
configured batch size, gradient accumulation and evaluation (run once), and
reports torch's peak allocated memory: exactly what training allocates for the
largest possible batch. Single process, so FSDP sharding is not modelled.

--world_size N (or WORLD_SIZE) adds the memory of ONE rank of an N-process run with the config's
parallel.tp_degree / pp_degree / cp_degree: layer weights, gradients and optimizer states split over
tensor x pipeline ranks and sharded by FSDP2 over the data (x context) ranks; the embeddings and head
sharded over data x tensor ranks (palingenesis.tensor_parallel); activations divided by tp (sequence
parallelism) and held for the micro-batches a pipeline stage keeps in flight (1F1B: up to pp of them,
GPipe: all); one layer's gathered weights for FSDP's all-gather. Rough, like the activations above.
"""

import sys

import torch

from palingenesis.config import Config

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
# GiB throughout, like the measured peak: GPUs are sold by GiB ("80 GB" A100 = 80 GiB).
GB = 2**30


def _model_config(config: Config):
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(config.model.name_or_path, trust_remote_code=config.model.trust_remote_code)
    except Exception as exc:  # no silent defaults: an estimate for the wrong model is worse than none
        raise SystemExit(
            f"Cannot load the model config for {config.model.name_or_path!r} ({type(exc).__name__}: {exc}). "
            "Point model.name_or_path at a local model directory or make the Hub reachable."
        )


def _meta_model(config: Config, model_config):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    # multimodal: the whole architecture (vision tower and projector count, frozen or trained)
    auto = AutoModelForImageTextToText if config.multimodal.enabled else AutoModelForCausalLM
    with torch.device("meta"):
        model = auto.from_config(model_config, trust_remote_code=config.model.trust_remote_code)
    if config.multimodal.enabled:
        from palingenesis.multimodal import set_vision_trainable

        set_vision_trainable(model, config.multimodal.train_vision_tower, config.multimodal.train_projector)
    if config.train.freeze_non_attention:
        from palingenesis.train import _freeze_non_attention_layers

        _freeze_non_attention_layers(model)
    return model


def _optimizer_bytes(model, name: str, param_bytes: int, bf16_rounding: str = "stochastic") -> float:
    """Optimizer state bytes for the trainable parameters, per optimizer layout."""
    total = 0.0
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if name == "muon":  # Muon momentum on matrices, AdamW on the rest
            is_matrix = p.ndim >= 2 and "embed" not in pname.lower()
            total += (1 if is_matrix else 2) * n * param_bytes
        elif name in ("adamw8bit", "paged_adamw8bit"):
            total += 2 * n  # two 8-bit states (block stats negligible)
        elif name == "lion8bit":
            total += n  # one 8-bit momentum
        elif name == "adamw_bf16" and param_bytes == 2 and bf16_rounding == "kahan":
            total += 3 * n * param_bytes  # bf16 moments + the bf16 compensation buffer
        else:  # torch AdamW: exp_avg + exp_avg_sq, weight dtype
            total += 2 * n * param_bytes
    return total


def _text_dims(model_config):
    text = model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
    hidden = text.hidden_size
    layers = text.num_hidden_layers
    heads = getattr(text, "num_attention_heads", 1)
    kv_heads = getattr(text, "num_key_value_heads", None) or heads
    head_dim = getattr(text, "head_dim", None) or hidden // max(heads, 1)
    types = getattr(text, "layer_types", None) or ["full_attention"] * layers
    full_layers = sum(1 for t in types if t in ("full_attention", "attention", "hybrid"))
    return hidden, layers, text.vocab_size, kv_heads, head_dim, full_layers


def _vision_memory(config: Config, model_config, model, rows: int, seq: int, act_bytes: int) -> tuple[float, int]:
    """(GiB, parameters) of the vision side of a micro-batch, at worst: every token of every row an
    image token. The language model's share of those tokens is in the activations already (image
    tokens count against data.max_seq_length); this is the tower's own: the pixel patches (fp32,
    merge^2 patches per token) and its activations, kept for the backward only when the tower trains
    (a frozen tower holds one block's worth at a time; a trained projector alone keeps its input)."""
    from palingenesis.multimodal import vision_parameters

    vision = getattr(model_config, "vision_config", None)
    if not config.multimodal.enabled or vision is None:
        return 0.0, 0
    tower, projector = vision_parameters(model)
    merge = int(getattr(vision, "spatial_merge_size", 2))
    patches = rows * seq * merge**2
    patch_values = (
        int(getattr(vision, "in_channels", 3))
        * int(getattr(vision, "temporal_patch_size", 1))
        * int(getattr(vision, "patch_size", 14)) ** 2
    )
    per_block = patches * int(vision.hidden_size) * act_bytes * 10  # as for the language model's layers
    blocks = int(getattr(vision, "depth", None) or getattr(vision, "num_hidden_layers", 1))
    kept = blocks if config.multimodal.train_vision_tower else 1
    total = patches * patch_values * 4 + kept * per_block
    return total / GB, sum(p.numel() for p in tower.values()) + sum(p.numel() for p in projector.values())


def _distill_memory(config: Config, rows: int, seq: int, act_bytes: int) -> tuple[float, float]:
    """Off-policy distillation: (online teachers' weights, per-micro-batch teacher buffers) in GiB.

    Teachers: exact parameter counts from their configs (meta device) in their dtype, divided by the
    data-parallel ranks under placement shard (the world size from WORLD_SIZE, else 1). Buffers: an online
    teacher's final hidden states at every position plus one slice of its fp32 logits; a cached teacher's
    top-k ids and log-probs per position."""
    import os

    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    from palingenesis.distill.chunked import SLICE_ELEMENTS
    from palingenesis.distill.config import parse_teachers

    if not config.distill.enabled:
        return 0.0, 0.0
    world = int(os.environ.get("WORLD_SIZE", "1"))
    weights = buffers = 0.0
    for spec in parse_teachers(config.distill.teachers):
        if spec.cache:
            buffers += rows * seq * (spec.top_k * (4 + 2) + 5)
            continue
        cfg = AutoConfig.from_pretrained(spec.model, trust_remote_code=config.model.trust_remote_code)
        auto = AutoModelForImageTextToText if config.multimodal.enabled else AutoModelForCausalLM
        with torch.device("meta"):
            teacher = auto.from_config(cfg, trust_remote_code=config.model.trust_remote_code)
        n = sum(p.numel() for p in teacher.parameters())
        shards = world if spec.placement == "shard" or (spec.placement == "auto" and world > 1) else 1
        weights += n * (torch.finfo(_DTYPES[spec.dtype]).bits // 8) / shards
        buffers += rows * seq * _text_dims(cfg)[0] * act_bytes + SLICE_ELEMENTS * 4
    return weights / GB, buffers / GB


def _offloaded_per_layer(config: Config, ckpt: str, per_layer: float, layer_input: float) -> float:
    """Bytes activation offload moves to host memory per layer (rough, like the activations themselves)."""
    if ckpt == "none":
        return per_layer
    if ckpt == "full" or config.memory.activation_offload_boundaries_only:
        return layer_input
    from palingenesis.kernels import sac_routes_saved_tensors_hooks

    # selective: the layer input, plus what the policy keeps when this torch routes it through the offload hooks
    return per_layer * 0.4 if sac_routes_saved_tensors_hooks() else layer_input


def estimate_memory(config: Config, gpu_memory_gb: float = 80.0, world_size: int | None = None) -> dict:
    from palingenesis.train import _dynamic_num_chunks

    model_config = _model_config(config)
    model = _meta_model(config, model_config)
    param_bytes = torch.finfo(_DTYPES[config.model.torch_dtype]).bits // 8
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hidden, layers, vocab, kv_heads, head_dim, full_layers = _text_dims(model_config)

    params_gb = params * param_bytes / GB
    optimizer_gb = _optimizer_bytes(model, config.train.optimizer, param_bytes, config.train.bf16_rounding) / GB
    if config.train.hyperball:
        optimizer_gb += min(1.0, trainable * 4 / GB)  # one snapshot bucket (<= 1 GiB)
    grads_gb = trainable * param_bytes / GB
    if config.memory.gradient_release and config.train.gradient_accumulation_steps <= 1:
        grads_gb = 0.0
    reference_gb = params_gb if config.dpo.enabled else 0.0  # frozen reference policy

    # Sequences per micro-batch, and tokens whose activations are alive at once.
    rows = config.train.per_device_batch_size * (2 if config.dpo.enabled else 1)
    seq = config.data.max_seq_length
    live_tokens = rows * (min(config.memory.seco_chunk_size, seq) if config.memory.seco else seq)
    act_bytes = 2 if config.train.bf16 else param_bytes
    per_layer = live_tokens * hidden * act_bytes * 10  # rough: ~10 hidden-sized tensors kept per layer
    ckpt = config.train.gradient_checkpointing
    if ckpt == "none":
        activations_gb = layers * per_layer / GB
    elif ckpt == "selective":
        activations_gb = layers * per_layer * 0.4 / GB
    else:  # full: layer inputs kept + one layer recomputed
        activations_gb = (layers * live_tokens * hidden * act_bytes + per_layer) / GB
    offload_host_gb = 0.0
    if config.memory.activation_offload:
        # palingenesis.activation_offload: what a layer saves leaves the GPU, except in the last layer, the layers
        # whose copies are in flight (forward) or prefetched (backward), and the one computing
        offloaded = _offloaded_per_layer(config, ckpt, per_layer, live_tokens * hidden * act_bytes)
        m = config.memory
        resident = min(layers, 2 + max(m.activation_offload_in_flight, m.activation_offload_prefetch))
        activations_gb -= (layers - resident) * offloaded / GB
        offload_host_gb = (layers - 1) * offloaded / GB

    ce_tokens = live_tokens if config.memory.seco else rows * seq
    chunks = _dynamic_num_chunks(ce_tokens, vocab)
    logits_gb = 2 * ce_tokens / chunks * vocab * 4 / GB  # fp32 logits + their gradient, one chunk

    teacher_gb, teacher_buffers_gb = _distill_memory(config, rows, seq, act_bytes)
    if config.distill.enabled:
        logits_gb *= 2  # a slice of the student's logits and one of the teacher's (or its top-k support)

    kv_gb = 0.0
    if config.memory.seco:
        # K/V store + its gradient on the GPU (sdpa); only the gradient with offload;
        # the previous path (non-sdpa attention) also keeps a prefix copy.
        copies = 1 if config.memory.seco_kv_offload else (2 if config.model.attn_implementation == "sdpa" else 3)
        kv_gb = copies * rows * seq * full_layers * 2 * kv_heads * head_dim * act_bytes / GB

    vision_gb, vision_params = _vision_memory(config, model_config, model, rows, seq, act_bytes)

    exact_gb = params_gb + optimizer_gb + grads_gb + reference_gb + teacher_gb
    total_gb = (exact_gb + activations_gb + logits_gb + kv_gb + teacher_buffers_gb + vision_gb) * 1.10
    per_rank = _per_rank(
        config,
        model,
        world_size,
        weights_gb=params_gb + optimizer_gb + grads_gb,
        activations_gb=activations_gb,
        logits_gb=logits_gb,
        layers=layers,
    )
    if per_rank is not None:
        per_rank["gpu_memory_gb"] = gpu_memory_gb
        per_rank["fits"] = per_rank["total_gb"] <= gpu_memory_gb
    return {
        "per_rank": per_rank,
        "model": config.model.name_or_path,
        "total_params_B": params / 1e9,
        "trainable_ratio": trainable / max(params, 1),
        "weight_dtype": config.model.torch_dtype,
        "params_memory_gb": params_gb,
        "optimizer_memory_gb": optimizer_gb,
        "optimizer_label": config.train.optimizer + (" + hyperball" if config.train.hyperball else ""),
        "grad_memory_gb": grads_gb,
        "reference_memory_gb": reference_gb,
        "teacher_memory_gb": teacher_gb,
        "teacher_buffers_gb": teacher_buffers_gb,
        "vision_memory_gb": vision_gb,
        "vision_params_B": vision_params / 1e9,
        "activation_memory_gb": activations_gb,
        "activation_offload_host_gb": offload_host_gb,
        "ce_peak_gb": logits_gb,
        "kv_memory_gb": kv_gb,
        "total_estimated_gb": total_gb,
        "gpu_memory_gb": gpu_memory_gb,
        "fits": total_gb <= gpu_memory_gb,
        "headroom_gb": gpu_memory_gb - total_gb,
        "ac_mode": ckpt,
        "seq_len": seq,
        "batch_size": config.train.per_device_batch_size,
        "hidden_size": hidden,
        "num_layers": layers,
        "vocab_size": vocab,
    }


def _per_rank(config: Config, model, world_size, *, weights_gb, activations_gb, logits_gb, layers) -> dict | None:
    """One rank's share of an N-process run (see the module docstring); None for a single process without
    tensor / pipeline parallelism."""
    import os

    from palingenesis.parallel_dims import PP_SCHEDULES, ParallelDims, stages_per_rank

    p = config.parallel
    world = world_size or int(os.environ.get("WORLD_SIZE", "0")) or p.tp_degree * p.pp_degree * max(p.cp_degree, 1)
    if world <= 1:
        return None
    cp = (p.cp_degree or min(world // (p.tp_degree * p.pp_degree), 8)) if p.context_parallel else 1
    dims = ParallelDims(world, tp=p.tp_degree, pp=p.pp_degree, cp=cp)
    if dims.errors():
        return {"error": dims.errors()[0]}
    total = sum(q.numel() for q in model.parameters())
    root = sum(
        q.numel() for n, q in model.named_parameters() if "embed_tokens" in n or n.startswith("lm_head") or n == "model.norm.weight"
    )
    layer_share = (total - root) / max(total, 1)
    # layers split over tensor x pipeline ranks, then FSDP over data x context; the root over data x context x tensor
    layer_weights = weights_gb * layer_share / (dims.tp * dims.pp * dims.dp * dims.cp)
    root_weights = weights_gb * (1 - layer_share) / (dims.dp * dims.cp * dims.tp)
    gather = weights_gb * layer_share / 4 / max(layers, 1) / dims.tp * 2  # one layer's bf16 weights, gathered (x2: prefetch)
    rows = config.train.per_device_batch_size
    microbatches = p.pp_microbatches or rows
    in_flight = 1.0
    if dims.pp > 1:
        per = stages_per_rank(p.pp_schedule, p.pp_stages_per_rank)
        held = microbatches if PP_SCHEDULES[p.pp_schedule][0] in ("GPipe", "LoopedBFS") else min(microbatches, dims.pp * per)
        in_flight = held / microbatches  # activations of `held` micro-batches of rows / microbatches rows each
    acts = activations_gb / (dims.tp * dims.cp * dims.pp) * in_flight
    logits = logits_gb / (dims.tp * dims.cp) / (microbatches if dims.pp > 1 else 1)
    total_gb = (layer_weights + root_weights + gather + acts + logits) * 1.10
    return {
        "dims": dims.describe(),
        "weights_optimizer_grads_gb": layer_weights + root_weights,
        "fsdp_gather_gb": gather,
        "activations_gb": acts,
        "logits_gb": logits,
        "total_gb": total_gb,
    }


def _worst_case_rows(config: Config, tokenizer, count: int) -> list[dict]:
    """Rows that render to (just under) max_seq_length tokens: the largest batch
    the trainer can see. Random vocabulary text; preference rows under DPO."""
    import random

    rng = random.Random(0)
    target = config.data.max_seq_length

    def fill(prefix_messages, budget):
        n = budget
        while True:
            text = tokenizer.decode([rng.randrange(tokenizer.vocab_size) for _ in range(n)])
            messages = prefix_messages + [{"role": "assistant", "content": text}]
            length = len(tokenizer(tokenizer.apply_chat_template(messages, tokenize=False))["input_ids"])
            if length <= target:
                return messages[-1]
            n -= length - target + 8

    prompt = [{"role": "user", "content": "go"}]
    rows = []
    for _ in range(count):
        if config.dpo.enabled:
            rows.append(
                {"prompt": prompt, "chosen": [fill(prompt, target - 32)], "rejected": [fill(prompt, target - 32)]}
            )
        else:
            rows.append({"messages": prompt + [fill(prompt, target - 32)]})
    return rows


def measure_memory(config: Config, steps: int = 2) -> dict:
    """Peak GPU memory of the real trainer (palingenesis.train.train) running
    `steps` optimizer steps on worst-case rows: same code path as training, so
    everything the trainer allocates is counted. Single process (no FSDP)."""
    if not torch.cuda.is_available():
        raise SystemExit("--measure needs a GPU")
    import copy
    import json
    import tempfile

    from palingenesis.agent_tooling._pipeline import load_tokenizer
    from palingenesis.train import train

    tokenizer = load_tokenizer(config)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = copy.deepcopy(config)
        data_file = f"{tmp}/worst_case.jsonl"
        micro = cfg.train.gradient_accumulation_steps
        rows = _worst_case_rows(cfg, tokenizer, steps * micro * cfg.train.per_device_batch_size)
        with open(data_file, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        cfg.data.dataset, cfg.data.dataset_split, cfg.data.streaming = data_file, "train", False
        cfg.data.sources, cfg.data.pretrain_replay_dataset, cfg.data.pretokenize = [], "", False
        cfg.data.eval_every = steps  # the configured evaluation runs once, at the last step
        cfg.data.messages_field = "messages"
        cfg.data.num_workers = 0  # workers would split the few rows below one batch each
        cfg.preprocess.enabled = False
        cfg.distill.missing = "skip"  # a teacher cache holds the real rows, not these (online teachers still run)
        cfg.train.max_steps = steps
        cfg.train.save_steps, cfg.train.save_final, cfg.train.resume_from = 0, False, None
        cfg.train.output_dir = f"{tmp}/out"
        cfg.logging.use_wandb = cfg.logging.use_trackio = False
        # the health monitor's periodic memory report resets torch's peak counter
        cfg.logging.health_tier2_every = cfg.logging.health_tier3_every = 10**9
        torch.cuda.reset_peak_memory_stats()
        result = train(cfg)
    # GiB: GPUs are sold by GiB ("80 GB" A100 = 80 GiB), so the numbers match the card
    return {
        "measured_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
        # of the last optimizer step: the first ones carry compilation and allocator warm-up
        "tokens_per_sec": result.metrics.get("train/tokens_per_sec"),
        "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "rows": config.train.per_device_batch_size * (2 if config.dpo.enabled else 1),
        "micro_batches": config.train.gradient_accumulation_steps,
        "evaluated": bool(config.data.eval_sources or config.data.eval_dataset),
        "seq_len": config.data.max_seq_length,
    }


def print_report(est: dict):
    print("=" * 70)
    print("MEMORY PROFILE ESTIMATE")
    print("=" * 70)
    print(
        f"  Model: {est['model']} ({est['total_params_B']:.2f}B params, {est['weight_dtype']} weights, "
        f"{est['trainable_ratio']:.0%} trainable)"
    )
    print(f"  Sequence length: {est['seq_len']:,} | Batch size: {est['batch_size']}")
    print(f"  Hidden: {est['hidden_size']} | Layers: {est['num_layers']} | Vocab: {est['vocab_size']:,}")
    print()
    print("  Exact (single GPU, before any FSDP sharding):")
    print(f"    Model parameters:     {est['params_memory_gb']:6.1f} GiB")
    print(f"    Optimizer states:     {est['optimizer_memory_gb']:6.1f} GiB  [{est['optimizer_label']}]")
    print(f"    Gradients:            {est['grad_memory_gb']:6.1f} GiB")
    if est["reference_memory_gb"]:
        print(f"    DPO reference model:  {est['reference_memory_gb']:6.1f} GiB")
    if est.get("teacher_memory_gb"):
        print(f"    Online teachers:      {est['teacher_memory_gb']:6.1f} GiB  [frozen, distill.teachers]")
    print("  Estimated:")
    print(f"    Logits (one chunk):   {est['ce_peak_gb']:6.1f} GiB")
    if est["kv_memory_gb"]:
        print(f"    SeCO K/V cache:       {est['kv_memory_gb']:6.1f} GiB")
    if est.get("teacher_buffers_gb"):
        print(f"    Teacher targets:      {est['teacher_buffers_gb']:6.1f} GiB  [per micro-batch]")
    if est.get("vision_memory_gb"):
        print(
            f"    Vision tower (rough): {est['vision_memory_gb']:6.1f} GiB  [pixels + activations if every token "
            f"were an image; its {est['vision_params_B'] * 1e3:.0f}M parameters are in the weights above]"
        )
    print(f"    Activations (rough):  {est['activation_memory_gb']:6.1f} GiB  [checkpointing: {est['ac_mode']}]")
    if est.get("activation_offload_host_gb"):
        print(
            f"    Offloaded activations: {est['activation_offload_host_gb']:5.1f} GiB  [host memory, pinned; "
            "memory.activation_offload]"
        )
    print(f"    {'─' * 50}")
    print(
        f"    Total (+10% overhead): {est['total_estimated_gb']:5.1f} GiB of {est['gpu_memory_gb']:.0f} GiB "
        f"({est['headroom_gb']:+.1f} GiB)"
    )
    rank = est.get("per_rank")
    if rank and "error" in rank:
        print(f"  Per rank: {rank['error']}")
    elif rank:
        print(f"  Per rank ({rank['dims']}):")
        print(f"    Weights, optimizer, gradients: {rank['weights_optimizer_grads_gb']:6.1f} GiB")
        print(f"    FSDP gathered layer:           {rank['fsdp_gather_gb']:6.1f} GiB")
        print(f"    Activations (rough):           {rank['activations_gb']:6.1f} GiB")
        print(f"    Logits (one chunk):            {rank['logits_gb']:6.1f} GiB")
        print(f"    Total (+10%):                  {rank['total_gb']:6.1f} GiB of {rank['gpu_memory_gb']:.0f} GiB")
    print()
    verdict = "should fit" if est["fits"] else "likely does NOT fit"
    print(
        f"  {'✓' if est['fits'] else '✗'} Estimate: {verdict}. Activations are approximate; "
        "`--measure` runs a real step for the exact peak."
    )
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--gpu_memory_gb",
        "--gpu",
        type=float,
        default=80.0,
        dest="gpu_memory_gb",
        help="GPU memory in GiB, as GPUs are rated (default: 80)",
    )
    parser.add_argument("--measure", action="store_true", help="run two real optimizer steps and report the peak")
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="processes of the run: adds one rank's share under parallel.tp_degree / pp_degree / FSDP",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    est = estimate_memory(config, args.gpu_memory_gb, world_size=args.world_size)
    print_report(est)
    if args.measure:
        m = measure_memory(config)
        print(
            f"  Measured peak: {m['measured_peak_gb']:.1f} GiB of {m['gpu_total_gb']:.0f} GiB "
            f"(real trainer: 2 optimizer steps x {m['micro_batches']} micro-batches of {m['rows']} x "
            f"{m['seq_len']} tokens{', plus one evaluation' if m['evaluated'] else ''})\n"
        )
        sys.exit(0 if m["measured_peak_gb"] < m["gpu_total_gb"] else 1)
    fits = est["per_rank"]["fits"] if est.get("per_rank") and "fits" in est["per_rank"] else est["fits"]
    sys.exit(0 if fits else 1)


if __name__ == "__main__":
    main()
