#!/usr/bin/env python3
"""Memory profiler — estimate peak GPU memory before training.

Provides a breakdown of where memory goes and whether your config will
fit on your hardware. No training needed — just estimates.

Usage:
    python -m agent_tooling.profile_memory --config configs/long_context.yaml --gpu_memory_gb 80

Reports:
    - Model parameters memory
    - Optimizer states memory (AdamW = 2x params in fp32)
    - Activation memory estimate (with/without checkpointing)
    - Gradient memory
    - CE loss peak (with/without chunked loss)
    - Total estimate and whether it fits
"""

import sys

import agent_tooling._path_setup  # noqa: F401

from palingenesis.config import Config


def estimate_memory(config: Config, gpu_memory_gb: float = 80.0) -> dict:
    """Estimate peak memory usage from config (no GPU needed)."""

    # Try to get model config for sizing
    try:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(
            config.model.name_or_path,
            trust_remote_code=config.model.trust_remote_code,
        )
        hidden_size = getattr(model_config, "hidden_size", 4096)
        num_layers = getattr(model_config, "num_hidden_layers", 32)
        num_heads = getattr(model_config, "num_attention_heads", 32)
        intermediate_size = getattr(model_config, "intermediate_size", hidden_size * 4)
        vocab_size = getattr(model_config, "vocab_size", 32000)
    except Exception:
        # Defaults for ~8B model
        hidden_size = 4096
        num_layers = 32
        num_heads = 32
        intermediate_size = 14336
        vocab_size = 128256

    # Config values
    seq_len = config.data.max_seq_length
    batch_size = config.train.per_device_batch_size
    dtype_bytes = 2  # bf16
    world_size = 1  # For FSDP estimate we divide by this later

    # ── Model Parameters ──────────────────────────────────────────────
    # Attention: Q, K, V, O projections per layer
    attn_params_per_layer = 4 * hidden_size * hidden_size
    # FFN: gate + up + down
    ffn_params_per_layer = 3 * hidden_size * intermediate_size
    # Norms
    norm_params_per_layer = 2 * hidden_size
    layer_params = attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer
    total_params = num_layers * layer_params + vocab_size * hidden_size * 2  # embed + lm_head

    params_memory_gb = total_params * dtype_bytes / 1e9

    # ── Optimizer States ─────────────────────────────────────────────
    # Depends on optimizer choice
    optimizer_name = config.train.optimizer
    trainable_params = total_params
    if config.train.freeze_non_attention:
        # ~25% of params trainable (attention + norms + lm_head)
        trainable_params = int(total_params * 0.30)

    optimizer_bytes_per_param = {
        "adamw": 12,  # fp32 copy + m + v = 4+4+4
        "muon": 4,  # momentum only (no v)
        "adamw8bit": 6,  # 8-bit m + v + fp32 correction
        "paged_adamw8bit": 6,  # same as adamw8bit (pages to CPU on OOM)
        "lion8bit": 2,  # 8-bit momentum only
    }.get(optimizer_name, 12)

    optimizer_memory_gb = trainable_params * optimizer_bytes_per_param / 1e9
    optimizer_label = f"{optimizer_name} ({optimizer_bytes_per_param} bytes/param)"

    # ── Gradients ─────────────────────────────────────────────────────
    grad_memory_gb = trainable_params * dtype_bytes / 1e9

    # Gradient release (FORGE): eliminates gradient buffer entirely
    gradient_release_active = getattr(config.memory, "gradient_release", False)
    if gradient_release_active and config.train.gradient_accumulation_steps <= 1:
        grad_memory_gb = 0.0  # gradients freed per-param during backward
        grad_note = "(gradient_release: freed per-param, ~0 GB)"
    else:
        grad_note = ""

    # ── Activations ───────────────────────────────────────────────────
    # Per layer: input activations + attention scores
    # Without checkpointing: all layers store activations
    # With checkpointing: only ~2 layers stored (fwd + recompute)

    # Rough formula: per layer activation = batch * seq * hidden * bytes * factor
    # Factor depends on what's stored: q/k/v/attn_output/ffn_input/ffn_output
    activation_per_layer = batch_size * seq_len * hidden_size * dtype_bytes * 10  # ~10 tensors retained
    # Attention scores: batch * num_heads * seq * seq * dtype (the quadratic part)
    # With flash attention this is O(seq) not O(seq^2), so we use O(seq) estimate
    attn_activation_per_layer = batch_size * num_heads * seq_len * 64 * dtype_bytes  # flash: O(seq * block_size)

    if config.train.gradient_checkpointing == "none":
        activation_memory_gb = num_layers * (activation_per_layer + attn_activation_per_layer) / 1e9
        ac_label = "NONE (all activations stored)"
    elif config.train.gradient_checkpointing == "selective":
        # Selective: ~40% savings (save SDPA + half matmuls)
        activation_memory_gb = num_layers * (activation_per_layer + attn_activation_per_layer) * 0.4 / 1e9
        ac_label = "SELECTIVE (~60% reduction)"
    else:  # full
        # Full: only store input per layer + recompute
        activation_memory_gb = 2 * (activation_per_layer + attn_activation_per_layer) / 1e9
        ac_label = "FULL (~90% reduction, 33% slower)"

    # Selective differentiation: frozen layers don't store activations
    selective_diff_active = getattr(config.memory, "selective_diff", False)
    if selective_diff_active and config.train.freeze_non_attention:
        # Only ~25-30% of layers (attention) store activations
        trainable_layer_fraction = 0.30
        activation_memory_gb *= trainable_layer_fraction
        ac_label += f" + SELECTIVE_DIFF (only {trainable_layer_fraction:.0%} layers save activations)"

    # ── CE Loss Peak ──────────────────────────────────────────────────
    if config.memory.chunked_loss:
        # Only materializes [B, S/N, V] at a time
        chunk_size = seq_len // config.memory.loss_num_chunks
        ce_peak_gb = batch_size * chunk_size * vocab_size * 4 / 1e9  # float32 logits
        ce_label = f"CHUNKED ({config.memory.loss_num_chunks} chunks)"
    else:
        ce_peak_gb = batch_size * seq_len * vocab_size * 4 / 1e9
        ce_label = "FULL (materializes all logits)"

    # ── FSDP Sharding ─────────────────────────────────────────────────
    fsdp_note = ""
    if config.parallel.fsdp:
        fsdp_note = "(will be ÷ N_GPUs with FSDP)"
        # Don't divide here — this shows single-GPU peak for comparison

    # ── Context Parallel ──────────────────────────────────────────────
    cp_note = ""
    if config.parallel.context_parallel:
        cp_note = "(activation memory ÷ CP_degree)"

    # ── Total ─────────────────────────────────────────────────────────
    total_gb = params_memory_gb + optimizer_memory_gb + grad_memory_gb + activation_memory_gb + ce_peak_gb
    # Add 10% overhead for fragmentation, temp buffers, CUDA context
    total_with_overhead = total_gb * 1.10
    fits = total_with_overhead <= gpu_memory_gb

    return {
        "model": config.model.name_or_path,
        "total_params": total_params,
        "total_params_B": total_params / 1e9,
        "params_memory_gb": params_memory_gb,
        "optimizer_memory_gb": optimizer_memory_gb,
        "optimizer_label": optimizer_label,
        "grad_memory_gb": grad_memory_gb,
        "grad_note": grad_note,
        "activation_memory_gb": activation_memory_gb,
        "ce_peak_gb": ce_peak_gb,
        "total_estimated_gb": total_with_overhead,
        "gpu_memory_gb": gpu_memory_gb,
        "fits": fits,
        "headroom_gb": gpu_memory_gb - total_with_overhead,
        "ac_mode": ac_label,
        "ce_mode": ce_label,
        "fsdp_note": fsdp_note,
        "cp_note": cp_note,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "trainable_ratio": trainable_params / max(total_params, 1),
    }


def print_report(est: dict):
    print("=" * 70)
    print("MEMORY PROFILE ESTIMATE")
    print("=" * 70)
    print(f"  Model: {est['model']} ({est['total_params_B']:.1f}B params)")
    print(f"  Sequence length: {est['seq_len']:,} | Batch size: {est['batch_size']}")
    print(f"  Hidden: {est['hidden_size']} | Layers: {est['num_layers']} | Vocab: {est['vocab_size']:,}")
    print()
    print("  Memory Breakdown (single GPU, pre-FSDP):")
    print(f"    Model parameters:     {est['params_memory_gb']:6.1f} GB  {est['fsdp_note']}")
    print(f"    Optimizer states:     {est['optimizer_memory_gb']:6.1f} GB  [{est.get('optimizer_label', 'adamw')}]")
    print(f"    Gradients:            {est['grad_memory_gb']:6.1f} GB  {est.get('grad_note', '')} {est['fsdp_note']}")
    print(f"    Activations:          {est['activation_memory_gb']:6.1f} GB  [{est['ac_mode']}] {est['cp_note']}")
    print(f"    CE loss peak:         {est['ce_peak_gb']:6.1f} GB  [{est['ce_mode']}]")
    print(f"    {'─' * 50}")
    print(f"    Total (+10% overhead): {est['total_estimated_gb']:5.1f} GB")
    print(f"    GPU available:         {est['gpu_memory_gb']:5.1f} GB")
    print(f"    Headroom:              {est['headroom_gb']:+5.1f} GB")
    print()

    if est["fits"]:
        print(f"  ✓ Should FIT on {est['gpu_memory_gb']:.0f}GB GPU")
    else:
        over_by = -est["headroom_gb"]
        print(f"  ✗ Will NOT fit on {est['gpu_memory_gb']:.0f}GB GPU (over by {over_by:.1f} GB)")
        print()
        print("  Suggestions to reduce memory (in order of impact):")
        optimizer_name = est.get("optimizer_label", "adamw")
        if "adamw" in optimizer_name and "8bit" not in optimizer_name:
            savings = est["optimizer_memory_gb"] * 0.5
            print(f"    → Switch to muon optimizer: saves ~{savings:.0f} GB (no v buffer)")
            if savings < over_by:
                savings8 = est["optimizer_memory_gb"] * 0.83
                print(f"    → Switch to lion8bit: saves ~{savings8:.0f} GB (1 byte/param momentum)")
        elif "muon" in optimizer_name:
            savings = est["optimizer_memory_gb"] * 0.75
            print(f"    → Switch to lion8bit: saves ~{savings:.0f} GB more")
        if "NONE" in est["ac_mode"]:
            print("    → Enable gradient checkpointing: train.gradient_checkpointing=selective")
        if "FULL" in est["ce_mode"] and est["ce_peak_gb"] > 1.0:
            print(f"    → Enable chunked loss: saves ~{est['ce_peak_gb']*0.9:.1f} GB")
        if not est.get("freeze_note"):
            train_ratio = est.get("trainable_ratio", 1.0)
            if train_ratio > 0.5:
                savings = est["optimizer_memory_gb"] * 0.7
                print(f"    → Enable freeze_non_attention: saves ~{savings:.0f} GB (if hybrid model)")
        print("    → Reduce batch_size or max_seq_length")
        print("    → Use FSDP (multi-GPU) or cpu_offload=true")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu_memory_gb", type=float, default=80.0, help="GPU memory in GB (default: 80 for A100)")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    est = estimate_memory(config, args.gpu_memory_gb)
    print_report(est)
    sys.exit(0 if est["fits"] else 1)


if __name__ == "__main__":
    main()
