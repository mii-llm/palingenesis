"""Logit lens and tuned lens: what each residual would predict if the model stopped there.

    logit lens (nostalgebraist 2020)   logits_r = head(norm(h_r))
    tuned lens (Belrose et al. 2023)   logits_r = head(norm(h_r + A_r h_r + b_r)), one affine translator per layer
                                       (A, b zero-initialised: it starts as the logit lens), trained to minimise
                                       KL(final distribution || lens distribution) at the same position

`norm` is the model's own final norm module (Qwen3.5 / Gemma compute x̂ (1 + w): never re-implemented) and `head` the
output head with the model's logit transform (Gemma's softcap, Cohere / Granite scales: palingenesis.logits). Every
residual is raw (see palingenesis.internals.sites), so the last one is normalised exactly once. `check_readout`
asserts that head(norm(h_n)) reproduces the model's own logits before any lens number is reported (a wrong norm, a
missing transform or the wrong submodule would otherwise give plausible-looking garbage).

Per layer, averaged over the selected positions (`lens_report`):

    kl          KL(final || lens) in nats: how far the layer's prediction is from the model's
    top1        share of positions where the lens's top token is the final top token
    ce          cross-entropy of the actual next token (teacher forcing; positions inside the text only)
    margin      the exact log-odds of a decision token set (palingenesis.margins.log_odds), e.g. "<tool_call>" at the
                decision position, or Italian vs English first tokens: at which depth the decision forms, and whether
                a training stage moved it there or only at the end

With tied embeddings, early logit-lens layers mostly decode the input token (Paulo et al. 2024 on Mamba); the tuned
lens fixes that. Logits are built for at most `chunk` rows at a time, never [layers, tokens, vocab].
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from palingenesis.internals.capture import _batches, model_fingerprint, run_backbone
from palingenesis.internals.hooks import Recorder
from palingenesis.internals.positions import Encoder, collate
from palingenesis.internals.sites import Sites, find_sites, parse_layers, text_config


def readout(sites: Sites, h: torch.Tensor) -> torch.Tensor:
    """Logits the model would produce from residual h: head(norm(h)), float32."""
    if sites.head is None:
        raise RuntimeError("internals lens: the model has no output head")
    if sites.norm is not None:
        h = sites.norm(h)
    return sites.head(h).float()


@torch.no_grad()
def check_readout(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> float:
    """Raise unless head(norm(residual n)) equals the model's logits; returns the relative max difference."""
    sites = find_sites(model)
    with Recorder(model, [sites.n_layers], sites=sites) as rec:
        reference = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        h = rec.take()[sites.n_layers]
    ours = readout(sites, h)
    diff = float((ours - reference).abs().max() / reference.abs().max().clamp_min(1e-30))
    dtype = next(model.parameters()).dtype
    if diff > (1e-4 if dtype in (torch.float32, torch.float64) else 2e-2):
        raise RuntimeError(
            f"internals lens: head(norm(last residual)) differs from the model's logits (relative {diff:.1e}); "
            "the final norm or the output head was not found correctly for this architecture"
        )
    return diff


class TunedLens(nn.Module):
    """One affine translator per residual r < n (none for r = n, which is the model's own readout)."""

    def __init__(self, n_layers: int, hidden_size: int):
        super().__init__()
        self.n_layers, self.hidden_size = n_layers, hidden_size
        self.translators = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(n_layers))
        for t in self.translators:  # identity at init: weight decay pulls toward the logit lens, not toward zero
            nn.init.zeros_(t.weight)
            nn.init.zeros_(t.bias)
        self.meta: dict = {}

    def forward(self, h: torch.Tensor, r: int) -> torch.Tensor:
        if r >= self.n_layers:
            return h
        t = self.translators[r]
        return h + t(h.to(t.weight.dtype)).to(h.dtype)

    def save(self, path: str | Path) -> Path:
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {"n_layers": self.n_layers, "hidden_size": self.hidden_size, "meta": self.meta}
        state = {k: v.detach().float().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(state, str(path), metadata={"palingenesis_tuned_lens": json.dumps(header)})
        return path

    @classmethod
    def load(cls, path: str | Path) -> TunedLens:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            header = json.loads((f.metadata() or {})["palingenesis_tuned_lens"])
            state = {k: f.get_tensor(k) for k in f.keys()}
        lens = cls(header["n_layers"], header["hidden_size"])
        lens.load_state_dict(state)
        lens.meta = header.get("meta", {})
        return lens


def _kl(final_logp: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """KL(final || lens) per row."""
    logq = torch.log_softmax(logits, -1)
    return (final_logp.exp() * (final_logp - logq)).sum(-1)


@torch.no_grad()
def lens_report(
    model: nn.Module,
    tok,
    rows: Sequence[dict],
    *,
    position: str = "assistant_all",
    layers: str | list[int] | None = "all",
    decision: Sequence[int] | None = None,
    tuned: TunedLens | None = None,
    chat_template_kwargs: dict | None = None,
    batch_size: int = 8,
    max_batch_tokens: int = 16384,
    max_tokens: int | None = None,
    chunk: int = 64,
) -> dict:
    """Per-layer lens metrics (see the module docstring), averaged over every selected position of every row."""
    sites = find_sites(model)
    n = sites.n_layers
    rs = sorted(set(parse_layers(layers, sites.n_residuals)) | {n})
    encoder = Encoder(tok, position, chat_template_kwargs, max_tokens)
    items = [(i, e) for i, row in enumerate(rows) if (e := encoder.encode(row)) is not None]
    if not items:
        raise ValueError(f"internals lens: no row has anything at position {position!r} ({encoder.dropped})")
    if any(e.pool == "mean" for _, e in items):
        raise ValueError("internals lens: a pooled position (*_mean) has no single next token; use *_all or *_last")
    device = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    sums = {r: {"kl": 0.0, "top1": 0.0, "ce": 0.0, "margin": 0.0} for r in rs}
    count, ce_count = 0, 0
    was_training = model.training
    model.eval()
    first = True
    try:
        with Recorder(model, rs, sites=sites) as rec:
            for batch_items in _batches(items, batch_size, max_batch_tokens):
                batch = collate([e for _, e in batch_items], pad)
                ids, mask = batch.input_ids.to(device), batch.attention_mask.to(device)
                if first:
                    check_readout(model, ids[:1, : int(mask[0].sum())])
                    first = False
                rec.select = batch.select
                run_backbone(model, ids, mask)
                acts = rec.take()
                lengths = mask.sum(1).tolist()
                targets = torch.tensor(
                    [int(batch.input_ids[b, p + 1]) if p + 1 < lengths[b] else -100 for b, p in batch.owners]
                )
                for start in range(0, len(batch.owners), chunk):
                    sl = slice(start, start + chunk)
                    final_logits = readout(sites, acts[n][sl])
                    final_logp = torch.log_softmax(final_logits, -1)
                    final_top = final_logits.argmax(-1)
                    tgt = targets[sl].to(final_logits.device)
                    has = tgt != -100
                    for r in rs:
                        h = acts[r][sl]
                        logits = final_logits if r == n else readout(sites, tuned(h, r) if tuned is not None else h)
                        s = sums[r]
                        s["kl"] += float(_kl(final_logp, logits).sum())
                        s["top1"] += float((logits.argmax(-1) == final_top).sum())
                        if has.any():
                            s["ce"] += float(torch.nn.functional.cross_entropy(logits[has], tgt[has], reduction="sum"))
                        if decision:
                            from palingenesis.margins import log_odds

                            s["margin"] += float(log_odds(logits, list(decision)).sum())
                    count += len(final_top)
                    ce_count += int(has.sum())
    finally:
        model.train(was_training)
    out = {"position": position, "tuned": tuned is not None, "positions": count, "layers": {}}
    for r in rs:
        s = sums[r]
        entry = {"mixer": sites.mixer(r), "kl": s["kl"] / count, "top1": s["top1"] / count}
        entry["ce"] = s["ce"] / ce_count if ce_count else None
        if decision:
            entry["margin"] = s["margin"] / count
        out["layers"][r] = {
            k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v) for k, v in entry.items()
        }
    return out


def fit_tuned_lens(
    model: nn.Module,
    tok,
    rows: Sequence[dict],
    *,
    position: str = "assistant_all",
    steps: int = 250,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 8,
    max_batch_tokens: int = 16384,
    max_rows_per_step: int = 2048,
    chat_template_kwargs: dict | None = None,
    max_tokens: int | None = None,
    seed: int = 0,
) -> TunedLens:
    """Train a tuned lens on the residuals of `rows` at `position` (default: every assistant token, the
    post-training distribution). AdamW (the Adam setting of Belrose et al.'s code: lr 1e-3; weight decay 1e-3, toward the identity since the
    translator is residual), gradient clipping at 1,
    one backward per layer so only one layer's [rows, vocab] logits are alive; the model stays frozen (no_grad
    capture; norm and head parameters are not given gradients). `max_rows_per_step` positions are sampled per step."""
    sites = find_sites(model)
    n = sites.n_layers
    device = next(model.parameters()).device
    lens = TunedLens(n, int(text_config(model).hidden_size)).to(device)
    lens.meta = {"model": model_fingerprint(model, tok), "position": position, "steps": steps, "lr": lr}
    encoder = Encoder(tok, position, chat_template_kwargs, max_tokens)
    items = [(i, e) for i, row in enumerate(rows) if (e := encoder.encode(row)) is not None]
    if not items:
        raise ValueError(f"tuned lens: no row has anything at position {position!r} ({encoder.dropped})")
    opt = torch.optim.AdamW(lens.parameters(), lr=lr, weight_decay=weight_decay)  # decay toward the identity
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    g = torch.Generator().manual_seed(seed)
    frozen = [(p, p.requires_grad) for p in model.parameters()]
    was_training = model.training
    model.eval()
    try:
        for p, _ in frozen:
            p.requires_grad_(False)
        batches = list(_batches(items, batch_size, max_batch_tokens))
        with Recorder(model, list(range(n + 1)), sites=sites) as rec:
            for step in range(steps):
                batch_items = batches[int(torch.randint(len(batches), (1,), generator=g))]
                batch = collate([e for _, e in batch_items], pad)
                rec.select = batch.select
                with torch.no_grad():
                    run_backbone(model, batch.input_ids.to(device), batch.attention_mask.to(device))
                acts = rec.take()
                k = len(batch.owners)
                pick = torch.randperm(k, generator=g)[:max_rows_per_step].to(device)
                with torch.no_grad():
                    final_logp = torch.log_softmax(readout(sites, acts[n][pick]), -1)
                opt.zero_grad(set_to_none=True)
                for r in range(n):
                    h = acts[r][pick].float()
                    loss = _kl(final_logp, readout(sites, lens(h, r).to(acts[r].dtype))).mean()
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
                opt.step()
    finally:
        for p, flag in frozen:
            p.requires_grad_(flag)
        model.train(was_training)
    return lens
