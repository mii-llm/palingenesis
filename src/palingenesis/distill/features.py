"""Capturing module outputs (hidden states, attention projections, ...) for feature matching.

`ModuleCapture` registers a forward hook on each requested module; the hook stores the module's
output only while capturing is switched on (the forward of the current micro-batch), so the extra
forward passes activation checkpointing replays during backward are ignored. The student's outputs
keep their graph; the teacher's are taken under no_grad.
"""

from __future__ import annotations

import contextlib
import re

import torch
from torch import Tensor, nn

_WRAPPERS = ("_orig_mod.", "_checkpoint_wrapped_module.", "_fsdp_wrapped_module.")


def decoder_layers(model: nn.Module) -> list[nn.Module]:
    """The model's repeated decoder layers (as FSDP and activation checkpointing find them)."""
    from palingenesis.distributed import _find_transformer_layers

    return _find_transformer_layers(model)


def _clean(name: str) -> str:
    for w in _WRAPPERS:
        name = name.replace(w, "")
    return name


def resolve_module(model: nn.Module, where: int | str) -> nn.Module:
    """A decoder layer by index (negative from the end), or a module by path: the path is matched
    against the end of the module names (ignoring wrapper prefixes), so `layers.3.mlp` and
    `model.layers.3.mlp` both work. Raises when nothing, or more than one module, matches."""
    if isinstance(where, int) or (isinstance(where, str) and re.fullmatch(r"-?\d+", where)):
        layers = decoder_layers(model)
        i = int(where)
        if not -len(layers) <= i < len(layers):
            raise ValueError(f"layer index {i} out of range: the model has {len(layers)} decoder layers")
        return layers[i]
    target = str(where).strip(".")
    matches = [
        (n, m) for n, m in model.named_modules() if n and (_clean(n) == target or _clean(n).endswith("." + target))
    ]
    unique = {id(m): (n, m) for n, m in matches}
    if not unique:
        raise ValueError(f"no module named like {target!r} in {type(model).__name__}")
    if len(unique) > 1:
        names = ", ".join(_clean(n) for n, _ in list(unique.values())[:4])
        raise ValueError(f"{target!r} matches several modules ({names}): give a longer path")
    return next(iter(unique.values()))[1]


def _output_tensor(output) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, Tensor):
        return hidden
    raise TypeError(f"cannot read a tensor from a module output of type {type(output).__name__}")


class ModuleCapture:
    """Outputs of named modules of one model, recorded during `capturing()`."""

    def __init__(self, model: nn.Module, targets: dict[str, int | str]):
        self.outputs: dict[str, Tensor] = {}
        self._active = False
        self._handles = []
        for key, where in targets.items():
            module = resolve_module(model, where)
            self._handles.append(module.register_forward_hook(self._hook(key)))

    def _hook(self, key: str):
        def hook(module, args, output):
            if self._active:
                self.outputs[key] = _output_tensor(output)

        return hook

    @contextlib.contextmanager
    def capturing(self):
        self.outputs = {}
        self._active = True
        try:
            yield self.outputs
        finally:
            self._active = False

    def take(self) -> dict[str, Tensor]:
        out, self.outputs = self.outputs, {}
        return out

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def model_hidden_size(model: nn.Module) -> int:
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    return int(cfg.hidden_size)


@torch.no_grad()
def probe_output_dim(model: nn.Module, where: int | str, device) -> int:
    """The last dimension of a module's output, from a forward on 4 tokens (for projections of
    modules whose width is not the hidden size)."""
    capture = ModuleCapture(model, {"probe": where})
    try:
        ids = torch.zeros((1, 4), dtype=torch.long, device=device)
        with capture.capturing() as out:
            model(input_ids=ids)
        return int(out["probe"].shape[-1])
    finally:
        capture.remove()
