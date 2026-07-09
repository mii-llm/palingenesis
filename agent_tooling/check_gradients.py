#!/usr/bin/env python3
"""Gradient health check — per-layer norm analysis.

Runs a single forward-backward pass on a small batch and reports
per-layer gradient statistics to diagnose:
    - Vanishing gradients (norms < 1e-7)
    - Exploding gradients (norms > 100 before clipping)
    - Dead layers (zero gradient)
    - Imbalanced norms across depth (gradient should flow smoothly)

Usage:
    python -m agent_tooling.check_gradients --config configs/single_gpu.yaml

Requires 1 GPU. Loads model, runs 1 forward-backward, reports norms.
"""

import sys

import agent_tooling._path_setup  # noqa: F401

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from palingenesis.config import Config
from palingenesis.data import ChatDataset, collate_fn, IGNORE_INDEX


def check_gradients(config: Config) -> dict:
    """Run one step and collect per-layer gradient norms."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map[config.model.torch_dtype]

    print(f"Loading model: {config.model.name_or_path} ({model_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=model_dtype,
        trust_remote_code=config.model.trust_remote_code,
        use_cache=False,
    ).to(device)
    model.train()

    # Get one batch
    dataset = load_dataset(
        config.data.dataset,
        split=config.data.dataset_split,
        streaming=config.data.streaming,
    )
    chat_ds = ChatDataset(
        dataset,
        tokenizer,
        config.data.max_seq_length,
        config.data.messages_field,
        rank=0,
        world_size=1,
        include_observations=config.data.include_observations,
        turn_scaling=config.data.turn_scaling,
        train_on_reasoning=getattr(config.data, "train_on_reasoning", True),
    )

    samples = []
    for s in chat_ds:
        samples.append(s)
        if len(samples) >= 2:
            break

    if not samples:
        return {"issues": ["CRITICAL: No samples processed from dataset."]}

    pad_id = tokenizer.pad_token_id or 0
    batch = collate_fn(samples, pad_id)
    batch = {k: v.to(device) for k, v in batch.items()}

    # Forward-backward
    print("Running forward-backward pass...")
    with torch.amp.autocast("cuda", dtype=model_dtype, enabled=config.train.bf16):
        outputs = model(**batch)
        loss = outputs.loss

    loss.backward()

    # Collect per-layer gradient norms
    layer_norms = {}
    total_params = 0
    grad_params = 0
    zero_grad_params = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        total_params += param.numel()

        if param.grad is not None:
            grad_params += param.numel()
            norm = param.grad.float().norm(2).item()
            layer_norms[name] = {
                "norm": norm,
                "shape": list(param.shape),
                "numel": param.numel(),
                "max_abs": param.grad.float().abs().max().item(),
                "has_nan": torch.isnan(param.grad).any().item(),
                "has_inf": torch.isinf(param.grad).any().item(),
            }
            if norm == 0:
                zero_grad_params += param.numel()
        else:
            layer_norms[name] = {"norm": 0.0, "shape": list(param.shape), "numel": param.numel(), "dead": True}
            zero_grad_params += param.numel()

    # Analysis
    issues = []
    norms = [v["norm"] for v in layer_norms.values() if v["norm"] > 0]
    has_nan = any(v.get("has_nan", False) for v in layer_norms.values())
    has_inf = any(v.get("has_inf", False) for v in layer_norms.values())
    dead_layers = [k for k, v in layer_norms.items() if v.get("dead") or v["norm"] == 0]
    tiny_grad = [k for k, v in layer_norms.items() if 0 < v["norm"] < 1e-7]
    huge_grad = [k for k, v in layer_norms.items() if v["norm"] > 100]

    if has_nan:
        issues.append("CRITICAL: NaN in gradients — model numerics broken.")
    if has_inf:
        issues.append("CRITICAL: Inf in gradients — loss explosion.")
    if len(dead_layers) > len(layer_norms) * 0.3:
        issues.append(
            f"WARNING: {len(dead_layers)} layers have zero gradient ({100*len(dead_layers)/len(layer_norms):.0f}%)."
        )
    if tiny_grad:
        issues.append(f"WARNING: {len(tiny_grad)} layers have vanishing gradients (<1e-7).")
    if huge_grad:
        issues.append(f"WARNING: {len(huge_grad)} layers have large gradients (>100).")

    # Check gradient flow across depth (for transformer layers)
    layer_pattern_norms = []
    for name, info in layer_norms.items():
        if "layers." in name and ".self_attn." in name and "q_proj" in name:
            layer_pattern_norms.append(info["norm"])

    if len(layer_pattern_norms) > 4:
        ratio = max(layer_pattern_norms) / max(min(layer_pattern_norms), 1e-30)
        if ratio > 1000:
            issues.append(
                f"WARNING: Gradient norm ratio across layers is {ratio:.0f}x — "
                "possible vanishing/exploding gradient flow."
            )

    if not issues:
        issues.append("OK: Gradient health looks good.")

    return {
        "loss": loss.item(),
        "total_grad_norm": sum(n**2 for n in norms) ** 0.5 if norms else 0,
        "total_params": total_params,
        "params_with_grad": grad_params,
        "zero_grad_params": zero_grad_params,
        "num_layers": len(layer_norms),
        "min_norm": min(norms) if norms else 0,
        "max_norm": max(norms) if norms else 0,
        "median_norm": sorted(norms)[len(norms) // 2] if norms else 0,
        "dead_layers": dead_layers[:10],
        "tiny_grad_layers": tiny_grad[:5],
        "huge_grad_layers": huge_grad[:5],
        "issues": issues,
        # Per-layer detail (top 20 by norm)
        "top_layers": sorted(
            [(k, v["norm"]) for k, v in layer_norms.items() if v["norm"] > 0],
            key=lambda x: x[1],
            reverse=True,
        )[:20],
    }


def print_report(result: dict):
    print("=" * 70)
    print("GRADIENT HEALTH REPORT")
    print("=" * 70)
    print(f"  Loss value: {result['loss']:.4f}")
    print(f"  Total grad norm: {result['total_grad_norm']:.4f}")
    print(f"  Parameters: {result['total_params']:,} total, {result['params_with_grad']:,} with grad")
    print(f"  Zero-grad params: {result['zero_grad_params']:,}")
    print(
        f"  Layer norms: min={result['min_norm']:.2e} median={result['median_norm']:.2e} max={result['max_norm']:.2e}"
    )
    print()

    if result["top_layers"]:
        print("  Top 10 layers by gradient norm:")
        for name, norm in result["top_layers"][:10]:
            short_name = name.replace("model.", "").replace("self_attn.", "attn.")
            print(f"    {norm:.4e}  {short_name}")
        print()

    if result["dead_layers"]:
        print(f"  Dead layers (zero grad): {result['dead_layers'][:5]}")
    if result["tiny_grad_layers"]:
        print(f"  Vanishing layers (<1e-7): {result['tiny_grad_layers']}")
    if result["huge_grad_layers"]:
        print(f"  Exploding layers (>100): {result['huge_grad_layers']}")

    print()
    for issue in result["issues"]:
        prefix = "  ✓" if issue.startswith("OK") else "  ⚠"
        print(f"{prefix} {issue}")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    result = check_gradients(config)
    print_report(result)
    sys.exit(0 if all("OK" in i for i in result["issues"]) else 1)


if __name__ == "__main__":
    main()
