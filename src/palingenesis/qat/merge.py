"""Merging ternary models: the result is ternary again (alpha * code per group), verified after writing.

A plain average of k ternary tensors is not ternary: with codes c_i in {-1, 0, 1} and scales alpha_i, the mean
(1/k) sum_i alpha_i c_i takes up to 3^k values per group (2k + 1 even with equal scales). Every method here ends with
a projection back onto the ternary set and differs in what it projects:

    requantize   mean of the source tensors, then the quantiser (scale scheme + rounding) again. On latent exports
                 (`export: latent`) this is "merge the full-precision latent weights, then re-ternarise"; on folded
                 exports it re-ternarises the mean of the ternary weights.
    vote         in code space, a weighted plurality per element: with s+, s0, s- the weight behind each code, the
                 sign with the larger weight is kept if its weight exceeds s0 (and reaches `min_share`), else the
                 code is 0. Equal weight for +1 and -1 gives 0; `tie: sign` keeps a sign that only ties with zero
                 (two models: what one of them set and the other left at zero).
    ties         sign election (TIES-Merging, arXiv:2306.01708): per element the sign with the larger total mass
                 sum_i a_i |t_i| wins, the entries of that sign are averaged (the others dropped), and the result is
                 re-ternarised. t_i is the weight itself, or W_i - base with `base` (then base + lam * merged). With
                 ternary sources the zero codes play the role of TIES' trimmed entries; `density` < 1 also trims the
                 smallest |t_i| of each tensor.
    consensus    relative to a shared ternary start (`base`): a code keeps its base value unless sources changed it; a
                 change is taken when the sources that changed the code agree on the new value (weighted plurality; a
                 tie, or fewer than `min_agree` of the weight behind the change, keeps the base code). A change made
                 by one expert only survives whole instead of being diluted by the experts that did not make it.

`refit` (default) sets each group's alpha to the least-squares fit of the weighted mean of the sources given the
merged codes, alpha = <M, c> / <c, c>; without it alpha is the weighted mean of the sources' alphas.

Tensors that are not ternary (norms, biases, anything outside `ternary_config.json`) are averaged with the same
weights. Sources are folded or latent exports of palingenesis.qat:TernaryQAT (their `ternary_config.json` names the
ternary tensors and group sizes), or any checkpoints together with `layout` / `group_size`.

    pgs merge ternary runs/a/final runs/b/final runs/c/final --method vote -o merged/
    pgs merge ternary runs/a/final runs/b/final --method consensus --base runs/start/final -o merged/
    pgs merge ternary --check merged/
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch import Tensor

from palingenesis.merge import Checkpoint, MergeError, MergeReport, _checkpoints, _combine, _relabel
from palingenesis.qat.ternary import TernarySpec, decode_ternary, group_view

TERNARY_CONFIG = "ternary_config.json"
METHODS = ("requantize", "vote", "ties", "consensus")


# ─────────────────────────────────────────────────────────────── layout


def read_layout(model: str | Path) -> dict | None:
    """The `ternary_config.json` of an export, or None."""
    path = Path(model) / TERNARY_CONFIG
    return json.loads(path.read_text()) if path.exists() else None


def guess_layout(ck: Checkpoint, group_size: int, include: str = r".*", exclude: str = r"embed|lm_head") -> dict:
    """Every 2-D float tensor whose name matches `include` and not `exclude` (one scale per row where the columns
    are not a multiple of group_size)."""
    inc, exc = re.compile(include), re.compile(exclude) if exclude else None
    tensors = {}
    for name in ck.keys():
        shape = ck.shape(name)
        if len(shape) != 2 or not inc.search(name) or (exc is not None and exc.search(name)):
            continue
        tensors[name] = group_size if group_size <= 0 or shape[1] % group_size == 0 else -1
    return {"tensors": tensors, "quantizer": TernarySpec(group_size).describe()}


def _spec_from(layout: dict, group_size: int, scale: str | None) -> TernarySpec:
    q = dict(layout.get("quantizer") or {})
    return TernarySpec(
        group_size, scale or q.get("scale", "absmean"), scale_args=None if scale else q.get("scale_args")
    )


# ─────────────────────────────────────────────────────────────── per-tensor merges


def _codes_and_alpha(w: Tensor, spec: TernarySpec) -> tuple[Tensor, Tensor]:
    """(codes, alpha) in group layout: read back exactly from a ternary tensor, else from the quantiser (a latent)."""
    decoded = decode_ternary(w, spec.group_size)
    if decoded is not None:
        return decoded[0].float(), decoded[1]
    g = group_view(w.float(), spec.group_size)
    alpha, delta = spec.scales(g)
    return spec.codes(g, alpha, delta), alpha


def _finish(codes: Tensor, mean: Tensor, alphas: Tensor, refit: bool) -> Tensor:
    """alpha * codes in group layout; alpha refit to `mean` (the weighted mean of the sources) or `alphas`."""
    if refit:
        support = codes.abs().sum(-1, keepdim=True)
        alpha = ((mean * codes).sum(-1, keepdim=True) / support.clamp(min=1)).clamp(min=0)
        alpha = torch.where(support > 0, alpha, alphas)
    else:
        alpha = alphas
    return alpha * codes


def merge_tensor(
    method: str,
    sources: list[Tensor],
    weights: list[float],
    spec: TernarySpec,
    *,
    base: Tensor | None = None,
    tie: str = "zero",
    min_share: float = 0.0,
    density: float = 1.0,
    lam: float = 1.0,
    min_agree: float = 0.0,
    refit: bool = True,
    stats: dict | None = None,
) -> Tensor:
    """One ternary tensor from the same tensor of several models (float32, the sources' shape)."""
    if method not in METHODS:
        raise MergeError(f"method must be one of {METHODS}, not {method!r}")
    shape = sources[0].shape
    a = torch.tensor(weights, dtype=torch.float32)
    a = a / a.sum()
    if method == "requantize":
        mean = sum(w * x.float() for w, x in zip(a.tolist(), sources))
        return spec.quantize(mean)
    if method == "ties":
        start = base.float() if base is not None else torch.zeros(shape)
        task = torch.stack([x.float() - start for x in sources])
        if density < 1.0:
            k = max(1, int(density * task[0].numel()))
            cut = task.abs().flatten(1).kthvalue(task[0].numel() - k + 1, dim=1).values
            task = task * (task.abs() >= cut.view(-1, *[1] * len(shape)))
        mass = (a.view(-1, *[1] * len(shape)) * task).sum(0)
        elected = torch.sign(mass)
        agree = (torch.sign(task) == elected) & (task != 0)
        w_agree = a.view(-1, *[1] * len(shape)) * agree
        merged = (w_agree * task).sum(0) / w_agree.sum(0).clamp(min=1e-12)
        if stats is not None:
            stats["conflict"] = stats.get("conflict", 0) + int(((task > 0).any(0) & (task < 0).any(0)).sum())
            stats["total"] = stats.get("total", 0) + task[0].numel()
        return spec.quantize(start + lam * merged)
    decoded = [_codes_and_alpha(x, spec) for x in sources]
    codes = torch.stack([c for c, _ in decoded])  # [k, rows, groups, g]
    alphas = sum(w * al for w, (_, al) in zip(a.tolist(), decoded))
    mean = sum(w * c * al for w, (c, al) in zip(a.tolist(), decoded))
    aw = a.view(-1, 1, 1, 1)
    if stats is not None:
        stats["conflict"] = stats.get("conflict", 0) + int(((codes > 0).any(0) & (codes < 0).any(0)).sum())
        stats["support_split"] = stats.get("support_split", 0) + int(((codes != 0).any(0) & (codes == 0).any(0)).sum())
        stats["total"] = stats.get("total", 0) + codes[0].numel()
    if method == "vote":
        if tie not in ("zero", "sign"):
            raise MergeError(f"tie must be 'zero' or 'sign', not {tie!r}")
        plus, minus, zero = ((aw * (codes == value)).sum(0) for value in (1.0, -1.0, 0.0))
        top = torch.maximum(plus, minus)
        beats_zero = top >= zero - 1e-9 if tie == "sign" else top > zero + 1e-9
        out = torch.sign(plus - minus) * (beats_zero & (top >= min_share))
    else:  # consensus
        if base is None:
            raise MergeError("consensus needs a base (the ternary model the sources started from)")
        base_codes, _ = _codes_and_alpha(base, spec)
        changed = codes != base_codes
        votes = torch.stack([(aw * (changed & (codes == value))).sum(0) for value in (-1.0, 0.0, 1.0)])
        top, idx = votes.max(0)
        second = votes.sort(0, descending=True).values[1]
        behind = (aw * changed).sum(0)
        take = (top > second) & (top > 0) & (behind >= min_agree)
        out = torch.where(take, idx.float() - 1.0, base_codes)
        if stats is not None:
            stats["changed"] = stats.get("changed", 0) + int((out != base_codes).sum())
    if stats is not None:
        stats["zeros"] = stats.get("zeros", 0) + int((out == 0).sum())
    return _finish(out, mean, alphas, refit).reshape(shape)


# ─────────────────────────────────────────────────────────────── checkpoints


def merge_ternary(
    models: list[str | Path],
    out: str | Path,
    *,
    method: str = "vote",
    weights: list[float] | None = None,
    base: str | Path | None = None,
    layout: dict | None = None,
    group_size: int | None = None,
    scale: str | None = None,
    tie: str = "zero",
    min_share: float = 0.0,
    density: float = 1.0,
    lam: float = 1.0,
    min_agree: float = 0.0,
    refit: bool = True,
    verify: bool = True,
    template: str | Path | None = None,
    missing: str = "error",
    dtype=None,
    overwrite: bool = False,
    extra: str = "error",
) -> MergeReport:
    """Merge checkpoints into a ternary checkpoint (see the module docstring). `layout`: a ternary_config dict
    (default: the first model's `ternary_config.json`, else every 2-D tensor outside the embedding and the head with
    `group_size`). `scale`: the scale scheme of re-ternarisation (default: the layout's). Raises MergeError if the
    written tensors are not ternary (`verify`)."""
    if method not in METHODS:
        raise MergeError(f"method must be one of {METHODS}, not {method!r}")
    if len(models) < 2:
        raise MergeError("merging needs at least two models")
    cks = _checkpoints(models)
    weights = list(weights) if weights is not None else [1.0] * len(cks)
    if len(weights) != len(cks) or min(weights) < 0 or sum(weights) <= 0:
        raise MergeError("weights: one non-negative weight per model, not all zero")
    total = sum(weights)
    weights = [w / total for w in weights]
    b = Checkpoint(base) if base is not None else None
    if method == "consensus" and b is None:
        raise MergeError("consensus needs --base (the ternary model the sources started from)")
    tpl = Checkpoint(template) if template is not None else cks[0]
    if layout is None:
        layout = read_layout(cks[0].dir)
    if layout is None:
        if group_size is None:
            raise MergeError(f"{cks[0].source} has no {TERNARY_CONFIG}: pass group_size (and a layout to be exact)")
        layout = guess_layout(cks[0], group_size)
    tensors = {k: (group_size if group_size is not None else g) for k, g in layout["tensors"].items()}
    stats: dict = {}

    def fn(name, w0, ws):
        if name not in tensors:
            return sum(w * x for w, x in zip(weights, ws))
        spec = _spec_from(layout, tensors[name], scale)
        return merge_tensor(
            method,
            ws,
            weights,
            spec,
            base=w0,
            tie=tie,
            min_share=min_share,
            density=density,
            lam=lam,
            min_agree=min_agree,
            refit=refit,
            stats=stats,
        )

    report = _combine(
        f"ternary_{method}",
        list(zip(cks, weights)),
        out,
        base=b,
        template=tpl,
        missing=missing,
        dtype=dtype,
        overwrite=overwrite,
        extra=extra,
        per_tensor=fn,
    )
    out_layout = {**layout, "tensors": tensors, "export": "folded"}
    (Path(report.out) / TERNARY_CONFIG).write_text(json.dumps(out_layout, indent=2))
    n = max(stats.get("total", 0), 1)
    summary = {
        "ternary_tensors": len(tensors),
        **{f"{k}_share": round(v / n, 6) for k, v in stats.items() if k != "total"},
    }
    if verify:
        check = check_ternary(report.out)
        summary["ternary_verified"] = check["ok"]
        if not check["ok"]:
            raise MergeError(f"merged tensors are not ternary: {check['failed'][:3]}")
    return _relabel(report, f"ternary_{method}", summary)


def check_ternary(model: str | Path, layout: dict | None = None, rtol: float = 1e-3) -> dict:
    """Read every tensor the layout calls ternary and test it: {"ok", "checked", "failed": [names], "zero_share",
    "bits_per_weight"} (codes at log2(3) bits plus 16 bits per scale, over the ternary tensors)."""
    ck = Checkpoint(model)
    layout = layout or read_layout(ck.dir)
    if layout is None:
        raise MergeError(f"{ck.source} has no {TERNARY_CONFIG}: pass a layout")
    failed, zeros, total, scales, checked = [], 0, 0, 0, 0
    for name, g in layout["tensors"].items():
        if name not in ck:
            continue
        decoded = decode_ternary(ck.tensor(name), g, rtol)
        checked += 1
        if decoded is None:
            failed.append(name)
            continue
        codes, alpha = decoded
        zeros += int((codes == 0).sum())
        total += codes.numel()
        scales += alpha.numel()
    return {
        "ok": not failed and checked > 0,
        "checked": checked,
        "failed": failed,
        "zero_share": zeros / max(total, 1),
        "bits_per_weight": (1.584962500721156 * total + 16 * scales) / max(total, 1),
    }
