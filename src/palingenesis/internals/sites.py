"""Where the residual stream is: a model's decoder layers, final norm and output head.

Residual index r, used everywhere in palingenesis.internals:

    r = 0        the input of decoder layer 0 (the embeddings, after any embedding scale)
    r = i + 1    the output of decoder layer i, before the next layer (and before the final norm)
    r = n        the output of the last layer, BEFORE the final norm

This matches `output_hidden_states` for r < n. For r = n it does not: transformers 5 returns the final norm's
output as the last entry of `hidden_states` (verified on Llama, Qwen3, Qwen3.5 and Gemma 3), so a logit lens built
on `hidden_states` normalises that entry twice. Here every r is a raw residual and the lenses apply the final norm
themselves.

Hybrid models (Qwen3.5 / Qwen3-Next: Gated DeltaNet linear-attention layers next to full-attention layers) have one
residual stream like any decoder: the token mixer differs per layer, the residual additions do not. `layer_types`
records which mixer wrote each layer's update.

The decoder layers are found by structure, not by class name, so the same code reads plain HF models, multimodal
wrappers (Qwen3_5ForConditionalGeneration: model.language_model.layers), PEFT models and the library's own wrappers
(activation-checkpoint wrappers, layers compiled in place, FSDP2 fully_shard, which keeps the module tree).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch.nn as nn

from palingenesis.logits import output_head

_LIST_NAMES = ("layers", "h", "blocks", "block")
_NORM_NAMES = ("norm", "final_layernorm", "ln_f", "final_layer_norm", "final_norm")
_SKIP = re.compile(r"(^|\.)(visual|vision_tower|vision_model|audio_tower|mtp)(\.|$)")


def text_config(model: nn.Module):
    config = model.config
    return config.get_text_config() if hasattr(config, "get_text_config") else config


@dataclass
class Sites:
    """The modules the internals tools hook. `layers[i]` is decoder layer i; `norm` the final norm (None for a model
    without one); `head` the output head with the model's logit transform (palingenesis.logits.output_head)."""

    layers: nn.ModuleList
    norm: nn.Module | None
    head: nn.Module | None
    path: str
    layer_types: list[str] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def n_residuals(self) -> int:
        """Residual indices are 0..n_layers inclusive."""
        return len(self.layers) + 1

    def mixer(self, r: int) -> str:
        """The token mixer that wrote the last update into residual r ("embedding" for r = 0)."""
        if r == 0:
            return "embedding"
        return self.layer_types[r - 1] if self.layer_types else "attention"


def find_sites(model: nn.Module) -> Sites:
    """Locate the decoder layers (the ModuleList of length num_hidden_layers with the shortest path, outside any
    vision / audio tower or multi-token-prediction head), the final norm next to them and the output head."""
    n = getattr(text_config(model), "num_hidden_layers", None)
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or _SKIP.search(name):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in _LIST_NAMES and len(module) > 0 and (n is None or len(module) == n):
            candidates.append((name.count("."), name, module))
    if not candidates:
        raise RuntimeError(
            "internals: cannot find the decoder layers (a ModuleList named layers / h / blocks"
            + (f" of length num_hidden_layers={n}" if n is not None else "")
            + ")"
        )
    _, path, layers = min(candidates, key=lambda c: (c[0], c[1]))
    parent = model.get_submodule(path.rsplit(".", 1)[0]) if "." in path else model
    norm = next((getattr(parent, a) for a in _NORM_NAMES if isinstance(getattr(parent, a, None), nn.Module)), None)
    types = list(getattr(text_config(model), "layer_types", None) or [])
    return Sites(layers=layers, norm=norm, head=output_head(model), path=path, layer_types=types[: len(layers)])


def parse_layers(spec: str | int | list | tuple | None, n_residuals: int) -> list[int]:
    """Residual indices from a spec: None / "all" (every r), an int, "17-19" (inclusive), "0,4,8", "every:4"
    (0, 4, 8, ... and always the last), negative indices from the end (-1 = the last layer's output)."""
    last = n_residuals - 1
    if spec is None or spec == "all":
        return list(range(n_residuals))
    if isinstance(spec, int):
        items: list = [spec]
    elif isinstance(spec, (list, tuple)):
        items = list(spec)
    else:
        spec = str(spec).strip()
        if spec.startswith("every:"):
            step = int(spec.split(":", 1)[1])
            if step < 1:
                raise ValueError(f"layers {spec!r}: the step must be >= 1")
            return sorted(set(range(0, n_residuals, step)) | {last})
        items = [s.strip() for s in spec.split(",") if s.strip()]
    out: set[int] = set()
    for item in items:
        span = re.fullmatch(r"(\d+)-(\d+)", item) if isinstance(item, str) else None
        if span:
            a, b = (_resolve(int(x), n_residuals) for x in span.groups())
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(_resolve(int(item), n_residuals))
    return sorted(out)


def _resolve(r: int, n_residuals: int) -> int:
    if r < 0:
        r += n_residuals
    if not 0 <= r < n_residuals:
        raise ValueError(f"residual index out of range: valid 0..{n_residuals - 1} (or negative from the end)")
    return r
