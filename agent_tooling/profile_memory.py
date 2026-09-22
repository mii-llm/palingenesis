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

--measure runs two optimizer steps with the trainer's own components (model,
activation checkpointing, freezing, optimizer, Hyperball, chunked loss / SeCO /
DPO) on random tokens at the configured batch size and max_seq_length, and
reports torch's peak allocated memory. Single GPU; FSDP sharding is not modelled.
"""

import sys

import torch

import agent_tooling._path_setup  # noqa: F401
from palingenesis.config import Config

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
GB = 1e9


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
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=config.model.trust_remote_code)
    if config.train.freeze_non_attention:
        from palingenesis.train import _freeze_non_attention_layers

        _freeze_non_attention_layers(model)
    return model


def _optimizer_bytes(model, name: str, param_bytes: int) -> float:
    """Optimizer state bytes for the trainable parameters, per optimizer layout."""
    total = 0.0
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if name == "muon":                                   # Muon momentum on matrices, AdamW on the rest
            is_matrix = p.ndim >= 2 and "embed" not in pname.lower()
            total += (1 if is_matrix else 2) * n * param_bytes
        elif name in ("adamw8bit", "paged_adamw8bit"):
            total += 2 * n                                    # two 8-bit states (block stats negligible)
        elif name == "lion8bit":
            total += n                                        # one 8-bit momentum
        else:                                                 # torch AdamW: exp_avg + exp_avg_sq, weight dtype
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


def estimate_memory(config: Config, gpu_memory_gb: float = 80.0) -> dict:
    from palingenesis.train import _dynamic_num_chunks

    model_config = _model_config(config)
    model = _meta_model(config, model_config)
    param_bytes = torch.finfo(_DTYPES[config.model.torch_dtype]).bits // 8
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hidden, layers, vocab, kv_heads, head_dim, full_layers = _text_dims(model_config)

    params_gb = params * param_bytes / GB
    optimizer_gb = _optimizer_bytes(model, config.train.optimizer, param_bytes) / GB
    if config.train.hyperball:
        optimizer_gb += min(1.0, trainable * 4 / GB)          # one snapshot bucket (<= 1 GiB)
    grads_gb = trainable * param_bytes / GB
    if config.memory.gradient_release and config.train.gradient_accumulation_steps <= 1:
        grads_gb = 0.0
    reference_gb = params_gb if config.dpo.enabled else 0.0  # frozen reference policy

    # Sequences per micro-batch, and tokens whose activations are alive at once.
    rows = config.train.per_device_batch_size * (2 if config.dpo.enabled else 1)
    seq = config.data.max_seq_length
    live_tokens = rows * (min(config.memory.seco_chunk_size, seq) if config.memory.seco else seq)
    act_bytes = 2 if config.train.bf16 else param_bytes
    per_layer = live_tokens * hidden * act_bytes * 10         # rough: ~10 hidden-sized tensors kept per layer
    ckpt = config.train.gradient_checkpointing
    if ckpt == "none":
        activations_gb = layers * per_layer / GB
    elif ckpt == "selective":
        activations_gb = layers * per_layer * 0.4 / GB
    else:                                                     # full: layer inputs kept + one layer recomputed
        activations_gb = (layers * live_tokens * hidden * act_bytes + per_layer) / GB

    ce_tokens = live_tokens if config.memory.seco else rows * seq
    chunks = _dynamic_num_chunks(ce_tokens, vocab)
    logits_gb = 2 * ce_tokens / chunks * vocab * 4 / GB       # fp32 logits + their gradient, one chunk

    kv_gb = 0.0
    if config.memory.seco:                                    # K/V leaf + its gradient + one prefix copy
        kv_gb = 3 * rows * seq * full_layers * 2 * kv_heads * head_dim * act_bytes / GB

    exact_gb = params_gb + optimizer_gb + grads_gb + reference_gb
    total_gb = (exact_gb + activations_gb + logits_gb + kv_gb) * 1.10
    return {
        "model": config.model.name_or_path,
        "total_params_B": params / 1e9,
        "trainable_ratio": trainable / max(params, 1),
        "weight_dtype": config.model.torch_dtype,
        "params_memory_gb": params_gb,
        "optimizer_memory_gb": optimizer_gb,
        "optimizer_label": config.train.optimizer + (" + hyperball" if config.train.hyperball else ""),
        "grad_memory_gb": grads_gb,
        "reference_memory_gb": reference_gb,
        "activation_memory_gb": activations_gb,
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


def measure_memory(config: Config, steps: int = 2) -> dict:
    """Peak GPU memory of real optimizer steps built from the trainer's components."""
    if not torch.cuda.is_available():
        raise SystemExit("--measure needs a GPU")
    from transformers import AutoModelForCausalLM

    from palingenesis.kernels import apply_activation_checkpointing, apply_liger_kernel
    from palingenesis.loss import chunked_cross_entropy_loss, shift_labels
    from palingenesis.optim import Hyperball, build_optimizer, is_hyperball_param
    from palingenesis.train import (
        _dynamic_num_chunks,
        _freeze_non_attention_layers,
        _get_hidden_states,
        _get_lm_head,
        _infer_model_type,
    )

    device = torch.device("cuda")
    dtype = _DTYPES[config.model.torch_dtype]
    torch.cuda.reset_peak_memory_stats()
    if config.model.use_liger_kernel:
        apply_liger_kernel(_infer_model_type(config.model.name_or_path))

    def load(path):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation=config.model.attn_implementation,
                                                 trust_remote_code=config.model.trust_remote_code)
        m.config.use_cache = False
        return m

    model = load(config.model.name_or_path)
    apply_activation_checkpointing(model, mode=config.train.gradient_checkpointing)
    if config.train.freeze_non_attention:
        _freeze_non_attention_layers(model)
    model = model.to(device).train()
    ref = None
    if config.dpo.enabled:
        ref = load(config.dpo.reference_model or config.model.name_or_path).to(device).eval().requires_grad_(False)
    optimizer = build_optimizer(model, config.train.learning_rate, config.train.weight_decay, config.train.llrd_decay,
                                use_muon=config.train.optimizer == "muon", optimizer_name=config.train.optimizer)
    stepper = optimizer
    if config.train.hyperball:
        constrained = [p for n, p in model.named_parameters() if p.requires_grad and is_hyperball_param(n, p)]
        stepper = Hyperball(optimizer, constrained, config.train.hyperball_lr)
    head = _get_lm_head(model)
    vocab = head.weight.shape[0]
    rows = config.train.per_device_batch_size * (2 if config.dpo.enabled else 1)
    ids = torch.randint(0, vocab, (rows, config.data.max_seq_length), device=device)
    labels = ids.clone()                                     # every token scored: the worst case
    mask = torch.ones_like(ids)
    for _ in range(steps):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config.train.bf16):
            if config.memory.seco:
                from palingenesis.seco import chunkwise_forward_backward

                chunkwise_forward_backward(model, ids, labels, chunk_size=config.memory.seco_chunk_size,
                                           budget=config.memory.spaco_budget or None,
                                           num_loss_chunks=lambda n: _dynamic_num_chunks(n, vocab))
            elif config.dpo.enabled:
                from palingenesis.dpo import preference_step, reference_logps

                chunks = _dynamic_num_chunks(ids.numel(), vocab)
                ref_lp = reference_logps(ref, _get_hidden_states, _get_lm_head, ids, mask, labels, chunks)
                hidden = _get_hidden_states(model, ids, mask, None)
                preference_step(hidden, labels, head, ref_lp, rows // 2, beta=config.dpo.beta,
                                sft_weight=config.dpo.sft_weight, num_chunks=chunks).loss.backward()
            else:
                hidden = _get_hidden_states(model, ids, mask, None)
                shifted = shift_labels(labels)
                chunked_cross_entropy_loss(hidden, shifted, head, num_chunks=_dynamic_num_chunks(ids.numel(), vocab),
                                           global_valid_tokens=float((shifted != -100).sum())).backward()
        stepper.step()
        optimizer.zero_grad(set_to_none=True)
    # GiB here: GPUs are sold by GiB ("80 GB" A100 = 80 GiB), so the numbers match the card
    return {"measured_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_total_gb": torch.cuda.get_device_properties(device).total_memory / 2**30,
            "rows": rows, "seq_len": config.data.max_seq_length}


def print_report(est: dict):
    print("=" * 70)
    print("MEMORY PROFILE ESTIMATE")
    print("=" * 70)
    print(f"  Model: {est['model']} ({est['total_params_B']:.2f}B params, {est['weight_dtype']} weights, "
          f"{est['trainable_ratio']:.0%} trainable)")
    print(f"  Sequence length: {est['seq_len']:,} | Batch size: {est['batch_size']}")
    print(f"  Hidden: {est['hidden_size']} | Layers: {est['num_layers']} | Vocab: {est['vocab_size']:,}")
    print()
    print("  Exact (single GPU, before any FSDP sharding):")
    print(f"    Model parameters:     {est['params_memory_gb']:6.1f} GB")
    print(f"    Optimizer states:     {est['optimizer_memory_gb']:6.1f} GB  [{est['optimizer_label']}]")
    print(f"    Gradients:            {est['grad_memory_gb']:6.1f} GB")
    if est["reference_memory_gb"]:
        print(f"    DPO reference model:  {est['reference_memory_gb']:6.1f} GB")
    print("  Estimated:")
    print(f"    Logits (one chunk):   {est['ce_peak_gb']:6.1f} GB")
    if est["kv_memory_gb"]:
        print(f"    SeCO K/V cache:       {est['kv_memory_gb']:6.1f} GB")
    print(f"    Activations (rough):  {est['activation_memory_gb']:6.1f} GB  [checkpointing: {est['ac_mode']}]")
    print(f"    {'─' * 50}")
    print(f"    Total (+10% overhead): {est['total_estimated_gb']:5.1f} GB of {est['gpu_memory_gb']:.0f} GB "
          f"({est['headroom_gb']:+.1f} GB)")
    print()
    verdict = "should fit" if est["fits"] else "likely does NOT fit"
    print(f"  {'✓' if est['fits'] else '✗'} Estimate: {verdict}. Activations are approximate; "
          "`--measure` runs a real step for the exact peak.")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu_memory_gb", "--gpu", type=float, default=80.0, dest="gpu_memory_gb",
                        help="GPU memory in GB (default: 80)")
    parser.add_argument("--measure", action="store_true", help="run two real optimizer steps and report the peak")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    est = estimate_memory(config, args.gpu_memory_gb)
    print_report(est)
    if args.measure:
        m = measure_memory(config)
        print(f"  Measured peak: {m['measured_peak_gb']:.1f} GiB of {m['gpu_total_gb']:.0f} GiB "
              f"({m['rows']} x {m['seq_len']} tokens, 2 optimizer steps)\n")
        sys.exit(0 if m["measured_peak_gb"] < m["gpu_total_gb"] else 1)
    sys.exit(0 if est["fits"] else 1)


if __name__ == "__main__":
    main()
