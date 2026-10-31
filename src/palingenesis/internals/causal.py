"""Causal checks: is a probed direction USED by the model, or only decodable from it?

A probe that reads a property says nothing about whether the model's behaviour depends on it (Belinkov 2022;
Ravichander et al. 2021; Elazar et al. 2021). Three interventions answer that, all on the residual stream through the
hooks of palingenesis.internals.hooks, measured on an exact decision log-odds (palingenesis.margins: logit
differences, not probabilities, which saturate; Zhang & Nanda 2023, Heimersheim & Nanda 2024):

    direction_effect   add alpha * (mu+ - mu-) at the probe's layer (every position, prompt included: in hybrid models
                       a decision is carried by the recurrent state from the prefill; steering only the generated tokens
                       had no effect in Yap 2026), for a grid of alpha, and fit the slope of the decision margin; the
                       same for `n_random` random directions of the same norm (the null: any large push moves margins
                       a little). Then ablate the direction at every residual (Arditi et al. 2024) and compare the
                       margin change with ablating random directions. Steering uses the diff-of-means vector, not the
                       classifier weights (Marks & Tegmark 2023; ITI; Im & Li 2025).
    patching_sweep     activation patching between twin prompts (palingenesis.margins twins: pos / neg sides that differ
                       minimally, symmetric token replacement rather than Gaussian noise, Zhang & Nanda 2023): the
                       residual at (layer, position-from-the-end) of one side replaces the other's, one layer at a time.
                       "denoise" (pos into neg) tests sufficiency, "noise" (neg into pos) necessity (Heimersheim &
                       Nanda 2024). Reported as the normalised effect (patched - base) / (source - base).
    generate           text with a steering vector active, to read what a dose does.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from palingenesis.internals.hooks import Recorder, Selection, ablate, patch, steer
from palingenesis.internals.positions import Encoder
from palingenesis.internals.probes import Probe
from palingenesis.internals.sites import find_sites, parse_layers


@torch.no_grad()
def decision_margins(
    model: nn.Module,
    tok,
    ids: Sequence[Sequence[int]],
    decision: Sequence[int],
    *,
    batch_size: int = 16,
) -> torch.Tensor:
    """[N] exact log-odds of the decision token set at the last position of each prompt (any active hooks apply)."""
    from palingenesis.margins import MarginReader, log_odds

    reader = MarginReader(model, tok, device=next(model.parameters()).device)
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = torch.empty(len(ids), dtype=torch.float64)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        logits = reader.logits_at_last([list(ids[i]) for i in chunk])
        out[chunk] = log_odds(logits, list(decision)).double().cpu()
    return out


def _slope(alphas: Sequence[float], values: Sequence[float]) -> float:
    a = torch.tensor(alphas, dtype=torch.float64)
    v = torch.tensor(values, dtype=torch.float64)
    a = a - a.mean()
    return float((a * (v - v.mean())).sum() / (a * a).sum().clamp_min(1e-12))


def direction_effect(
    model: nn.Module,
    tok,
    rows: Sequence[dict],
    probe: Probe,
    decision: Sequence[int],
    *,
    alphas: Sequence[float] = (-2.0, -1.0, 0.0, 1.0, 2.0),
    n_random: int = 8,
    ablation: bool = True,
    chat_template_kwargs: dict | None = None,
    batch_size: int = 16,
    seed: int = 0,
) -> dict:
    """Dose-response of the decision margin along the probe's diff-of-means direction vs random directions of the
    same norm, plus directional ablation (see the module docstring). Rows are rendered at their decision position
    (last_prompt); alpha is in units of |mu+ - mu-| (alpha = 1 moves the negative-class mean onto the positive one)."""
    if len(alphas) < 2:
        raise ValueError("direction_effect: at least two alphas")
    kwargs = chat_template_kwargs if chat_template_kwargs is not None else probe.meta.get("chat_template_kwargs", {})
    encoder = Encoder(tok, "last_prompt", kwargs)
    ids = [encoder.prompt_ids(r) for r in rows]
    sites = find_sites(model)
    r = probe.layer
    if not 0 <= r < sites.n_residuals:
        raise ValueError(f"probe layer {r} out of range for this model (0..{sites.n_layers})")
    vec = probe.mean_diff.float()

    def curve(v: torch.Tensor) -> list[float]:
        values = []
        for a in alphas:
            with steer(model, {r: a * v}, sites=sites):
                values.append(float(decision_margins(model, tok, ids, decision, batch_size=batch_size).mean()))
        return values

    base = decision_margins(model, tok, ids, decision, batch_size=batch_size)
    g = torch.Generator().manual_seed(seed)
    randoms = []
    for _ in range(n_random):
        d = torch.randn(vec.shape, generator=g)
        randoms.append(d / d.norm() * vec.norm())
    margins = curve(vec)
    random_slopes = [_slope(alphas, curve(d)) for d in randoms]
    slope = _slope(alphas, margins)
    rs = torch.tensor(random_slopes, dtype=torch.float64)
    out = {
        "layer": r,
        "n": len(ids),
        "alphas": list(alphas),
        "base_margin": float(base.mean()),
        "margins": margins,
        "slope": slope,
        "random_slopes": random_slopes,
        "z": float((slope - rs.mean()) / rs.std().clamp_min(1e-12)) if n_random > 1 else None,
        "mean_diff_norm": float(vec.norm()),
        "resid_norm": probe.resid_norm,
    }
    if ablation:
        with ablate(model, vec, sites=sites):
            abl = decision_margins(model, tok, ids, decision, batch_size=batch_size)
        rand_abl = []
        for d in randoms:
            with ablate(model, d, sites=sites):
                rand_abl.append(
                    float((decision_margins(model, tok, ids, decision, batch_size=batch_size) - base).mean())
                )
        out["ablation"] = {"delta": float((abl - base).mean()), "random_deltas": rand_abl}
    return out


@torch.no_grad()
def patching_sweep(
    model: nn.Module,
    tok,
    twins: Sequence[dict],
    decision: Sequence[int],
    *,
    layers: str | list[int] | None = "all",
    offsets: Sequence[int] = (-1,),
    mode: str = "denoise",
    chat_template_kwargs: dict | None = None,
    batch_size: int = 8,
) -> dict:
    """Activation patching between the two sides of each twin (see the module docstring). Positions are offsets from
    each side's last prompt token (-1 = the decision position); sides of different lengths are aligned from the end.
    Returns per layer and offset the mean normalised effect, and the mean margins of both sides."""
    if mode not in ("denoise", "noise"):
        raise ValueError("mode must be denoise (pos into neg) or noise (neg into pos)")
    if any(o >= 0 for o in offsets):
        raise ValueError("offsets count from the end: -1 is the last prompt token")
    sites = find_sites(model)
    rs = parse_layers(layers, sites.n_residuals)
    encoder = Encoder(tok, "last_prompt", chat_template_kwargs)
    src_key, dst_key = ("pos", "neg") if mode == "denoise" else ("neg", "pos")
    src_ids = [encoder.prompt_ids(t[src_key]) for t in twins]
    dst_ids = [encoder.prompt_ids(t[dst_key]) for t in twins]
    shortest = min(len(x) for x in src_ids + dst_ids)
    if -min(offsets) > shortest:
        raise ValueError(f"offset {min(offsets)} reaches before the start of a {shortest}-token prompt")
    src_m = decision_margins(model, tok, src_ids, decision, batch_size=batch_size)
    dst_m = decision_margins(model, tok, dst_ids, decision, batch_size=batch_size)
    device = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    k = len(offsets)
    effect = {r: torch.zeros(k, dtype=torch.float64) for r in rs}
    from palingenesis.internals.capture import run_backbone
    from palingenesis.margins import MarginReader, log_odds

    reader = MarginReader(model, tok, device=device)
    for start in range(0, len(twins), batch_size):
        idx = list(range(start, min(start + batch_size, len(twins))))
        src = [src_ids[i] for i in idx]
        width = max(len(x) for x in src)
        ids = torch.full((len(idx), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for b, x in enumerate(src):
            ids[b, : len(x)] = torch.tensor(x)
            mask[b, : len(x)] = 1
        src_pos = torch.tensor([[len(x) + o for o in offsets] for x in src])
        flat = (torch.arange(len(idx))[:, None] * width + src_pos).reshape(-1)
        select = Selection(flat, torch.arange(len(flat)), len(flat), "none")
        with Recorder(model, rs, select, sites=sites, stop=True) as rec:
            run_backbone(model, ids.to(device), mask.to(device))
            values = rec.take()
        dst = [dst_ids[i] for i in idx]
        dst_pos = torch.tensor([[len(x) + o for o in offsets] for x in dst])
        for r in rs:
            v = values[r].reshape(len(idx), k, -1)
            for j in range(k):
                with patch(model, r, dst_pos[:, j : j + 1], v[:, j : j + 1], sites=sites):
                    m = log_odds(reader.logits_at_last(dst), list(decision)).double().cpu()
                denom = src_m[idx] - dst_m[idx]
                norm = torch.where(denom.abs() > 1e-6, (m - dst_m[idx]) / denom, torch.zeros_like(m))
                effect[r][j] += norm.sum()
    n = len(twins)
    return {
        "mode": mode,
        "offsets": list(offsets),
        "n": n,
        "source_margin": float(src_m.mean()),
        "base_margin": float(dst_m.mean()),
        "layers": {r: [round(float(e / n), 4) for e in effect[r]] for r in rs},
        "mixers": {r: sites.mixer(r) for r in rs},
    }


@torch.no_grad()
def generate(
    model: nn.Module,
    tok,
    messages: list[dict],
    *,
    vectors: dict[int, torch.Tensor] | None = None,
    positions: str = "all",
    chat_template_kwargs: dict | None = None,
    max_new_tokens: int = 128,
    **gen_kwargs,
) -> str:
    """The model's reply to `messages` (greedy unless gen_kwargs say otherwise) with steering vectors active."""
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **(chat_template_kwargs or {}))
    enc = tok(text, add_special_tokens=False, return_tensors="pt").to(next(model.parameters()).device)
    gen_kwargs.setdefault("do_sample", False)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if vectors:
        with steer(model, vectors, positions):
            out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=pad, **gen_kwargs)
    else:
        out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=pad, **gen_kwargs)
    return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
