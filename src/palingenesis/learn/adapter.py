"""Scope adapters: MLP-only LoRA through PEFT, the frozen ancestor stack, and exact offline stacking.

A scope's adapter learns only that scope's notes. It is trained on top of its ancestors' live adapters,
which are held frozen as forward hooks on the base linears (FrozenDelta): not PEFT adapters, so nothing
swaps active adapters or flips requires_grad, and PEFT's disable_adapter() gives exactly the "parent
stack" P = base + ancestors.

Stacking: linear LoRAs add, and sum_i s_i B_i A_i = B_cat A_cat with A_cat the rows of every A_i and
B_cat the columns of every s_i B_i. A chain (org + team + user) is therefore served as ONE adapter of
rank r_o + r_t + r_u, exactly, with lora_alpha = rank (PEFT scale 1). Padding with zero rank (up to an
engine's accepted rank) keeps it exact.

PEFT is the optional `learn` extra: pip install 'palingenesis[learn]'.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
from pathlib import Path

import torch
from torch import Tensor, nn

PEFT_PREFIX = "base_model.model."
ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"


def require_peft():
    try:
        import peft
    except ImportError:
        raise ImportError(
            "pgs learn trains LoRA adapters with PEFT, which is not installed: pip install 'palingenesis[learn]'"
        ) from None
    return peft


def target_modules(model: nn.Module, regex: str) -> list[str]:
    """Names of the nn.Linear modules matching the anchored target regex (MLP projections of the decoder)."""
    pattern = re.compile(regex)
    names = [
        name for name, module in model.named_modules() if isinstance(module, nn.Linear) and pattern.fullmatch(name)
    ]
    if not names:
        raise ValueError(f"no linear module matches adapter.target_regex {regex!r}")
    return names


def attach_lora(model: nn.Module, rank: int, alpha: float, regex: str):
    """`model` wrapped by PEFT with one trainable LoRA on the regex's modules (dropout 0, no bias)."""
    peft = require_peft()
    target_modules(model, regex)  # fail early with a readable error
    config = peft.LoraConfig(
        r=rank, lora_alpha=alpha, target_modules=regex, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"
    )
    return peft.get_peft_model(model, config)


def lora_state(peft_model) -> dict[str, Tensor]:
    """A detached copy of the trainable LoRA parameters (a "previous model" snapshot)."""
    return {n: p.detach().clone() for n, p in peft_model.named_parameters() if "lora_" in n}


@contextlib.contextmanager
def swapped_lora(peft_model, state: dict[str, Tensor]):
    """Run with the LoRA parameters temporarily set to `state` (copied in and back, no autograd, no
    requires_grad change). Used for the "previous" and "self" teacher arms."""
    params = dict(peft_model.named_parameters())
    saved = {}
    with torch.no_grad():
        for name, value in state.items():
            saved[name] = params[name].detach().clone()
            params[name].copy_(value)
    try:
        yield
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                params[name].copy_(value)


# ---------------------------------------------------------------------------- reading and stacking


def read_adapter(path: str | Path) -> tuple[dict[str, tuple[Tensor, Tensor]], dict]:
    """A PEFT LoRA directory as {module name: (A [r, in], B [out, r] already scaled)} and its config.

    The scale alpha / r (alpha / sqrt(r) with rsLoRA) is folded into B, so delta W = B @ A.
    """
    from safetensors.torch import load_file

    path = Path(path)
    config = json.loads((path / ADAPTER_CONFIG).read_text())
    if config.get("use_dora"):
        raise ValueError(f"{path}: DoRA adapters are not linear and cannot be stacked")
    tensors = load_file(str(path / ADAPTER_WEIGHTS))
    out: dict[str, tuple[Tensor, Tensor]] = {}
    for key, a in tensors.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module = key[: -len(".lora_A.weight")]
        module = module[len(PEFT_PREFIX) :] if module.startswith(PEFT_PREFIX) else module
        b = tensors[key.replace(".lora_A.weight", ".lora_B.weight")]
        rank = a.shape[0]
        alpha = config.get("alpha_pattern", {}).get(module, config["lora_alpha"])
        scale = alpha / math.sqrt(rank) if config.get("use_rslora") else alpha / rank
        out[module] = (a.float(), b.float() * scale)
    if not out:
        raise ValueError(f"{path}: no LoRA weights found")
    return out, config


def stack(
    adapters: list[dict[str, tuple[Tensor, Tensor]]], pad_to: int | None = None
) -> dict[str, tuple[Tensor, Tensor]]:
    """One adapter equal to the sum of `adapters` (scale folded in): A and B concatenated along the rank.

    A module missing from one adapter gets zero rows there, so every module has the same total rank;
    `pad_to` adds zero rank up to that size. Both keep the sum exact.
    """
    modules = sorted({m for a in adapters for m in a})
    out = {}
    for module in modules:
        shapes = next(a[module] for a in adapters if module in a)
        in_f, out_f = shapes[0].shape[1], shapes[1].shape[0]
        a_parts, b_parts = [], []
        for adapter in adapters:
            rank = next(iter(adapter.values()))[0].shape[0]
            a, b = adapter.get(module, (torch.zeros(rank, in_f), torch.zeros(out_f, rank)))
            a_parts.append(a)
            b_parts.append(b)
        a, b = torch.cat(a_parts, 0), torch.cat(b_parts, 1)
        if pad_to is not None:
            if pad_to < a.shape[0]:
                raise ValueError(f"pad_to {pad_to} is below the stacked rank {a.shape[0]}")
            a = torch.cat([a, a.new_zeros(pad_to - a.shape[0], in_f)], 0)
            b = torch.cat([b, b.new_zeros(out_f, pad_to - b.shape[1])], 1)
        out[module] = (a, b)
    return out


def write_adapter(tensors: dict[str, tuple[Tensor, Tensor]], out: str | Path, base_model: str) -> dict:
    """Save scale-folded LoRA tensors as a PEFT adapter (lora_alpha = r, so PEFT's scale is 1)."""
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ranks = {a.shape[0] for a, _ in tensors.values()}
    if len(ranks) != 1:
        raise ValueError(f"every module must have the same rank, got {sorted(ranks)}")
    rank = ranks.pop()
    weights = {}
    for module, (a, b) in tensors.items():
        weights[f"{PEFT_PREFIX}{module}.lora_A.weight"] = a.float().contiguous().cpu()
        weights[f"{PEFT_PREFIX}{module}.lora_B.weight"] = b.float().contiguous().cpu()
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        # Suffixes, as serving engines expect; the weights name the exact modules.
        "target_modules": sorted({m.rsplit(".", 1)[-1] for m in tensors}),
        "use_rslora": False,
        "use_dora": False,
        "modules_to_save": None,
    }
    save_file(weights, str(out / ADAPTER_WEIGHTS))
    (out / ADAPTER_CONFIG).write_text(json.dumps(config, indent=2) + "\n")
    return config


def stack_dirs(paths: list[str | Path], out: str | Path, base_model: str, pad_to: int | None = None) -> dict:
    """Stack PEFT adapter directories (a scope chain, ancestors first) into one PEFT adapter at `out`."""
    return write_adapter(stack([read_adapter(p)[0] for p in paths], pad_to=pad_to), out, base_model)


def stack_composite(paths: list[str | Path], out: str | Path, base_model: str, pad_to: int, manifest: dict) -> None:
    """stack_dirs into `out` plus its manifest.json, built in a private temporary directory renamed into place:
    a reader (the vLLM resolver plugin, another replica on the same store) sees no composite or a complete one.
    manifest.json marks a complete composite."""
    import os
    import threading

    from palingenesis.learn.store import remove_tree

    out = Path(out)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    remove_tree(tmp)
    stack_dirs(paths, tmp, base_model, pad_to=pad_to)
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if out.exists() and not (out / "manifest.json").exists():
        remove_tree(out)  # the remains of a killed stack
    try:
        os.rename(tmp, out)
    except OSError:
        remove_tree(tmp)  # another writer finished first: stacking is deterministic, its copy is the same
        if not (out / "manifest.json").exists():
            raise


def rename_modules(src: str | Path, out: str | Path, renames: dict[str, str]) -> None:
    """Copy a PEFT adapter with module names rewritten (the first matching prefix of `renames` after PEFT's own
    prefix): engines that name a checkpoint's modules differently (SGLang drops `language_model.`)."""
    from safetensors.torch import load_file, save_file

    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    weights = {}
    for key, value in load_file(str(src / ADAPTER_WEIGHTS)).items():
        body = key[len(PEFT_PREFIX) :] if key.startswith(PEFT_PREFIX) else key
        for old, new in renames.items():
            if body.startswith(old):
                body = new + body[len(old) :]
                break
        weights[PEFT_PREFIX + body] = value
    save_file(weights, str(out / ADAPTER_WEIGHTS))
    (out / ADAPTER_CONFIG).write_text((src / ADAPTER_CONFIG).read_text())


class FrozenDelta:
    """Adds a frozen low-rank delta (stacked ancestors) to target linears through forward hooks.

    Exact in any base dtype: the delta is computed in fp32 from the layer's input and added to its output,
    as an unmerged LoRA is. remove() takes the hooks off.
    """

    def __init__(self, model: nn.Module, tensors: dict[str, tuple[Tensor, Tensor]]):
        modules = dict(model.named_modules())
        missing = sorted(set(tensors) - set(modules))
        if missing:
            raise ValueError(f"the ancestor adapters name modules this model does not have: {missing[:3]}")
        self.handles = []
        self.tensors = {}
        for name, (a, b) in tensors.items():
            module = modules[name]
            device = module.weight.device
            pair = (a.to(device, torch.float32), b.to(device, torch.float32))
            self.tensors[name] = pair
            self.handles.append(module.register_forward_hook(self._hook(pair)))

    @staticmethod
    def _hook(pair):
        a, b = pair

        def hook(module, inputs, output):
            x = inputs[0]
            return output + ((x.to(a.dtype) @ a.t()) @ b.t()).to(output.dtype)

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
