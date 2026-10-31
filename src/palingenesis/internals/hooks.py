"""Forward hooks on the residual stream: read it at chosen positions, or rewrite it.

Reading (`Recorder`) gathers the selected positions INSIDE the hook, so a batch's full [B, T, d] residual is never
kept: memory is the selected rows times the number of layers read. With `stop=True` the forward is cut right after the
deepest layer read (a probe at layer 12 of 24 costs half a forward).

Writing (`edit_residual` and the helpers built on it: `steer`, `ablate`, `patch`) replaces a layer's output, so every
later layer, the final norm and the head see the edit, for every kind of token mixer (attention or Gated DeltaNet:
the recurrent state of later layers is computed from the edited residual).

Every handle is removed in a `finally`, also when the forward raises; `active_hooks(model)` counts what is left, which
the tests assert is zero.

Decoder layers return a tensor in transformers 5 and a tuple (hidden, ...) in older versions or other code; both are
handled. Residual r = 0 is read and written through a forward PRE-hook on layer 0 (its input), so embedding scales
(Gemma) are included.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn

from palingenesis.internals.sites import Sites, find_sites


class StopForward(Exception):
    """Raised by a Recorder with stop=True once the deepest requested residual is read."""


@dataclass
class Selection:
    """Which positions of a [B, T, d] residual become output rows, and how they are pooled.

    `index` holds flat positions b * T + t; `segment` the output row each selected position belongs to; `pool`
    "mean" averages the positions of a segment (in float32), "none" keeps one output row per position (then
    segment = arange(K)). Built by palingenesis.internals.positions from the chat template's masks."""

    index: torch.Tensor  # [K] long
    segment: torch.Tensor  # [K] long
    n_out: int
    pool: str = "none"

    @classmethod
    def last(cls, lengths: list[int], width: int) -> Selection:
        """The last real token of each right-padded row."""
        idx = torch.tensor([b * width + n - 1 for b, n in enumerate(lengths)], dtype=torch.long)
        return cls(idx, torch.arange(len(lengths)), len(lengths), "none")

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        flat = h.reshape(-1, h.shape[-1])
        index = self.index.to(flat.device)
        picked = flat.index_select(0, index)
        if self.pool == "none":
            return picked
        segment = self.segment.to(flat.device)
        out = torch.zeros(self.n_out, flat.shape[-1], dtype=torch.float32, device=flat.device)
        out.index_add_(0, segment, picked.float())
        counts = torch.bincount(segment, minlength=self.n_out).clamp_min(1).to(out.dtype)
        return out / counts[:, None]


def _hidden(output):
    h = output if torch.is_tensor(output) else output[0]
    if h.ndim != 3:  # e.g. code that takes output[0] of a tensor output gets batch row 0, silently
        raise RuntimeError(f"internals: expected a [batch, tokens, hidden] residual, got shape {tuple(h.shape)}")
    return h


def _replace(output, h: torch.Tensor):
    if torch.is_tensor(output):
        return h
    return (h, *output[1:]) if isinstance(output, tuple) else type(output)((h, *output[1:]))


def _register(
    sites: Sites, r: int, fn: Callable[[torch.Tensor], torch.Tensor | None], edit: bool = False
) -> torch.utils.hooks.RemovableHandle:
    """Hook residual r: fn(h) is called with the [B, T, d] residual; a returned tensor replaces it. Edits are
    prepended, so every reader of the same residual (a Recorder) sees the edited value whatever the order the
    contexts were entered in."""
    if r == 0:

        def pre(module, args, kwargs):
            if args:
                new = fn(_hidden(args[0]))
                return None if new is None else ((new, *args[1:]), kwargs)
            new = fn(_hidden(kwargs["hidden_states"]))
            return None if new is None else (args, {**kwargs, "hidden_states": new})

        pre._palingenesis_internals = True
        return sites.layers[0].register_forward_pre_hook(pre, with_kwargs=True, prepend=edit)

    def post(module, args, output):
        new = fn(_hidden(output))
        return None if new is None else _replace(output, new)

    post._palingenesis_internals = True
    return sites.layers[r - 1].register_forward_hook(post, prepend=edit)


def active_hooks(model: nn.Module) -> int:
    """Hooks of this module still registered on the decoder layers (0 after every context exits). transformers' own
    output-capturing hooks (installed once output_hidden_states is used) are not counted."""
    sites = find_sites(model)
    return sum(
        getattr(fn, "_palingenesis_internals", False)
        for m in sites.layers
        for fn in (*m._forward_hooks.values(), *m._forward_pre_hooks.values())
    )


class Recorder:
    """Records residuals `layers` at `select`'s positions on every forward while active.

        with Recorder(model, [0, 12, 24]) as rec:
            rec.select = Selection.last(lengths, width)
            run_backbone(model, input_ids, attention_mask)   # palingenesis.internals.capture.run_backbone
            acts = rec.take()                                # {r: [n_out, d]}

    `dtype` casts what is kept (default: the residual's dtype; mean pooling is float32); `device` moves it (e.g. "cpu"
    to free accelerator memory between batches); `stop` cuts the forward after the deepest layer (StopForward, caught
    by run_backbone)."""

    def __init__(
        self,
        model: nn.Module,
        layers: list[int],
        select: Selection | None = None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        stop: bool = False,
        sites: Sites | None = None,
    ):
        self.sites = sites or find_sites(model)
        self.layers = sorted(set(layers))
        if not self.layers:
            raise ValueError("Recorder: no layers to record")
        self.select, self.dtype, self.device, self.stop = select, dtype, device, stop
        self.acts: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def _record(self, r: int, h: torch.Tensor) -> None:
        v = self.select.apply(h) if self.select is not None else h
        v = v.detach()
        if self.dtype is not None:
            v = v.to(self.dtype)
        if self.device is not None:
            v = v.to(self.device)
        self.acts[r] = v
        if self.stop and r == self.layers[-1]:
            raise StopForward

    def take(self) -> dict[int, torch.Tensor]:
        """The residuals of the last forward; clears them."""
        missing = [r for r in self.layers if r not in self.acts]
        if missing:
            raise RuntimeError(f"Recorder: residuals {missing} were not reached by the forward")
        acts, self.acts = self.acts, {}
        return acts

    def __enter__(self) -> Recorder:
        try:
            for r in self.layers:
                self._handles.append(_register(self.sites, r, lambda h, r=r: self._record(r, h)))
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, *exc) -> None:
        self.close()


@contextlib.contextmanager
def edit_residual(
    model: nn.Module, edits: dict[int, Callable[[torch.Tensor], torch.Tensor]], sites: Sites | None = None
) -> Iterator[None]:
    """While active, residual r is replaced by edits[r](h) on every forward (h: [B, T, d], the same dtype back)."""
    sites = sites or find_sites(model)
    handles = []
    try:
        for r, fn in edits.items():
            handles.append(_register(sites, r, fn, edit=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


STEER_POSITIONS = ("all", "last_and_new")


@contextlib.contextmanager
def steer(
    model: nn.Module,
    vectors: dict[int, torch.Tensor],
    positions: str = "all",
    sites: Sites | None = None,
) -> Iterator[None]:
    """Add vectors[r] ([d], raw residual units) to residual r.

    positions: "all" adds at every position of every forward (ActAdd / CAA / repeng apply steering to all positions;
    use it for generation). "last_and_new" adds, on the first forward (the prompt prefill), only at the last position,
    and at every position of later forwards (the generated tokens): the prompt's own representation stays untouched.
    It assumes the prompt is left-padded or unpadded, as in generate()."""
    if positions not in STEER_POSITIONS:
        raise ValueError(f"steer: positions must be one of {STEER_POSITIONS}, not {positions!r}")
    calls = {r: 0 for r in vectors}

    def make(r: int, v: torch.Tensor):
        def fn(h: torch.Tensor) -> torch.Tensor:
            add = v.to(device=h.device, dtype=h.dtype)
            first = calls[r] == 0
            calls[r] += 1
            if positions == "last_and_new" and first:
                h = h.clone()
                h[:, -1] = h[:, -1] + add
                return h
            return h + add

        return fn

    with edit_residual(model, {r: make(r, v) for r, v in vectors.items()}, sites):
        yield


@contextlib.contextmanager
def ablate(
    model: nn.Module, direction: torch.Tensor, layers: list[int] | None = None, sites: Sites | None = None
) -> Iterator[None]:
    """Directional ablation (Arditi et al. 2024): h <- h - (h . u) u with u the unit direction, at residuals `layers`
    (default: every residual, 0..n), every position. Removes the direction from what later layers read."""
    sites = sites or find_sites(model)
    layers = list(range(sites.n_residuals)) if layers is None else layers
    u = direction.float() / direction.float().norm().clamp_min(1e-12)

    def fn(h: torch.Tensor) -> torch.Tensor:
        uu = u.to(h.device)
        proj = (h.float() @ uu)[..., None] * uu
        return (h.float() - proj).to(h.dtype)

    with edit_residual(model, {r: fn for r in layers}, sites):
        yield


@contextlib.contextmanager
def patch(
    model: nn.Module, r: int, positions: torch.Tensor, values: torch.Tensor, sites: Sites | None = None
) -> Iterator[None]:
    """Activation patching: residual r at positions[b, k] (absolute positions of a right-padded batch) is replaced by
    values[b, k] ([B, K, d]) on every forward while active."""
    if positions.shape != values.shape[:2]:
        raise ValueError(f"patch: positions {tuple(positions.shape)} vs values {tuple(values.shape)}")

    def fn(h: torch.Tensor) -> torch.Tensor:
        h = h.clone()
        rows = torch.arange(h.shape[0], device=h.device)[:, None].expand_as(positions)
        h[rows, positions.to(h.device)] = values.to(device=h.device, dtype=h.dtype)
        return h

    with edit_residual(model, {r: fn}, sites):
        yield
