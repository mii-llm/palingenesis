"""Use a saved probe: score rows, gate them, label a dataset, or reward RL rollouts.

    scorer = ProbeScorer("probes/known/best.safetensors")            # loads the model recorded in the probe
    p = scorer.probabilities([{"messages": [...]}, ...])             # [N] calibrated P(label)
    flags = scorer.gate(rows, at="fpr_0.05")                         # thresholds chosen on the probe's validation split

The scorer renders each row at the probe's position with the probe's chat template kwargs, reads only the probe's
residual and stops the forward there (a probe at layer 18 of 24 costs 3/4 of a prefill). It refuses a model of another
shape and warns when the weights differ from the ones the probe was fitted on: a probe moved to another checkpoint is a
hypothesis to check (`pgs internals compare` reports transfer), not a given; RL can move representations enough to
break a frozen probe (Taufeeque et al. 2026).

Three ways to bring a probe into training, from safest to riskiest:

1. LABELS (`score_rows`, `pgs internals score`): a column written into the RL / DPO dataset before training, from a
   frozen checkpoint, for a prompt-only probe (position last_prompt). The policy cannot influence a label computed
   before it runs; relabel with the current checkpoint between stages instead of sampling k answers per prompt.
2. GATE (`ProbeScorer.gate`): at deployment, abstain or route when the probe fires; the threshold is a calibrated
   operating point (FPR on the validation split; re-check it after every training stage and per domain: Kadavath et
   al. 2022 found calibration does not transfer).
3. REWARD (`probe_reward`): the probe scores rollouts during RL. The scoring model is a FROZEN copy, never the
   policy, and no gradient flows through it; still, optimising against a detector teaches the policy to evade it
   (Bailey et al. 2024; Cundy & Gleave 2025: GRPO learned to evade a lie detector unless its TPR was very high and
   the KL to the reference strong). Keep a held-out monitor probe that is never a reward, prefer probes on the prompt
   (which the policy cannot change) or off-policy uses (labels, DPO pairs), and watch the gap between the frozen and a
   re-fitted probe on the policy's outputs.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from palingenesis.internals.capture import _batches, model_fingerprint, run_backbone
from palingenesis.internals.hooks import Recorder
from palingenesis.internals.positions import Encoder, collate
from palingenesis.internals.probes import Probe
from palingenesis.internals.sites import find_sites

logger = logging.getLogger(__name__)


class ProbeMismatch(ValueError):
    """The probe cannot be applied to this model (different hidden size or depth)."""


def check_compatible(probe: Probe, model: nn.Module, tok=None) -> dict:
    """Raise ProbeMismatch on a shape mismatch; warn when the weights or chat template differ. Returns the model's
    fingerprint."""
    want = probe.meta.get("model", {})
    have = model_fingerprint(model, tok)
    for key in ("hidden_size", "num_layers"):
        if want.get(key) and want[key] != have[key]:
            raise ProbeMismatch(f"probe fitted on {key}={want[key]}, this model has {have[key]}")
    if probe.weight.shape[0] != have["hidden_size"]:
        raise ProbeMismatch(
            f"probe weight has {probe.weight.shape[0]} features, the model's hidden size is {have['hidden_size']}"
        )
    if want.get("weights_sha") and want["weights_sha"] != have["weights_sha"]:
        logger.warning(
            "internals: probe fitted on %s, applied to other weights (%s): check transfer before trusting it",
            want.get("name_or_path") or want["weights_sha"],
            have.get("name_or_path") or have["weights_sha"],
        )
    if (
        want.get("chat_template_sha")
        and have["chat_template_sha"]
        and want["chat_template_sha"] != have["chat_template_sha"]
    ):
        logger.warning("internals: the tokenizer's chat template differs from the one the probe was fitted with")
    return have


class ProbeScorer:
    """Apply a saved probe (path or Probe) with a model (path, module, or None = the path recorded in the probe)."""

    def __init__(
        self,
        probe: Probe | str | Path,
        model: nn.Module | str | None = None,
        tok=None,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        batch_size: int = 16,
        max_batch_tokens: int = 32768,
        max_tokens: int | None = None,
    ):
        self.probe = probe if isinstance(probe, Probe) else Probe.load(probe)
        if model is None or isinstance(model, (str, Path)):
            from transformers import AutoModelForCausalLM, AutoTokenizer

            path = str(model or self.probe.meta.get("model", {}).get("name_or_path") or "")
            if not path:
                raise ValueError("ProbeScorer: the probe records no model path; pass model=")
            device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
            model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, low_cpu_mem_usage=True).to(device)
            tok = tok or AutoTokenizer.from_pretrained(path)
        if tok is None:
            raise ValueError("ProbeScorer: pass tok= with a model module")
        self.model, self.tok = model.eval(), tok
        self.sites = find_sites(model)
        check_compatible(self.probe, model, tok)
        self.encoder = Encoder(tok, self.probe.position, self.probe.meta.get("chat_template_kwargs", {}), max_tokens)
        self.batch_size, self.max_batch_tokens = batch_size, max_batch_tokens
        self._lock = threading.Lock()  # rewards run in threads; one forward at a time per scorer

    @torch.no_grad()
    def logits(self, rows: Sequence[dict]) -> torch.Tensor:
        """[N] probe logits, one per row; NaN for a row with nothing at the probe's position (or too long)."""
        if self.probe.position.endswith("_all"):
            raise ValueError(
                "ProbeScorer: a *_all probe scores tokens, not rows; fit it at *_mean / *_last / last_prompt"
            )
        out = torch.full((len(rows),), float("nan"))
        items = [(i, e) for i, row in enumerate(rows) if (e := self.encoder.encode(row)) is not None]
        if not items:
            return out
        device = next(self.model.parameters()).device
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else (self.tok.eos_token_id or 0)
        with self._lock, Recorder(self.model, [self.probe.layer], stop=True, sites=self.sites) as rec:
            for batch_items in _batches(items, self.batch_size, self.max_batch_tokens):
                batch = collate([e for _, e in batch_items], pad)
                rec.select = batch.select
                run_backbone(self.model, batch.input_ids.to(device), batch.attention_mask.to(device))
                z = self.probe.logit(rec.take()[self.probe.layer]).cpu()
                for (b, _), v in zip(batch.owners, z):
                    out[batch_items[b][0]] = v
        return out

    def probabilities(self, rows: Sequence[dict]) -> torch.Tensor:
        a, b = self.probe.calibration
        return torch.sigmoid(a * self.logits(rows) + b)

    def gate(self, rows: Sequence[dict], at: str = "fpr_0.05") -> list[bool | None]:
        """True where the probe fires at the named operating point (None where a row could not be scored)."""
        if at not in self.probe.thresholds:
            raise KeyError(f"probe has no threshold {at!r} (have {sorted(self.probe.thresholds)})")
        z = self.logits(rows)
        return [None if torch.isnan(v) else bool(v > self.probe.thresholds[at]) for v in z]


# -------------------------------------------------------------------------------------------- dataset labels


def score_rows(scorer: ProbeScorer, rows: list[dict], column: str, *, gate_at: str | None = None) -> list[dict]:
    """Copies of `rows` with the calibrated probability in `column` (None if unscorable) and, with gate_at, a boolean
    `<column>_flag`."""
    if gate_at is not None and gate_at not in scorer.probe.thresholds:
        raise KeyError(f"probe has no threshold {gate_at!r} (have {sorted(scorer.probe.thresholds)})")
    z = scorer.logits(rows)
    a, b = scorer.probe.calibration
    p = torch.sigmoid(a * z + b)
    out = []
    for row, zi, pi in zip(rows, z.tolist(), p.tolist()):
        new = dict(row)
        new[column] = None if pi != pi else round(pi, 6)
        if gate_at is not None:
            new[f"{column}_flag"] = None if zi != zi else bool(zi > scorer.probe.thresholds[gate_at])
        out.append(new)
    return out


# -------------------------------------------------------------------------------------------- RL reward

_SCORERS: dict[tuple, ProbeScorer] = {}
_SCORERS_LOCK = threading.Lock()


def _scorer(probe: str, model: str | None, device: str | None) -> ProbeScorer:
    key = (str(probe), model, device)
    with _SCORERS_LOCK:
        if key not in _SCORERS:
            _SCORERS[key] = ProbeScorer(probe, model, device=device)
        return _SCORERS[key]


def probe_reward(
    prompts: list | None = None,
    messages: list | None = None,
    *,
    probe: str,
    model: str | None = None,
    device: str | None = "cpu",
    invert: bool = False,
    **columns,
) -> list[float | None]:
    """Batched RL reward: the calibrated probability of a saved probe, read by a FROZEN scoring model (default: the
    model recorded in the probe, on CPU so it never competes with the policy and vLLM for GPU memory).

    A last_prompt probe scores the prompt (the policy cannot move it: use it to condition another reward, e.g. pay for
    abstaining only where the probe says the model does not know); any other position scores the whole conversation
    including the rollout. `invert` returns 1 - p. Read the warnings in this module's docstring before rewarding a
    rollout with a probe.

        rewards:
          known: {fn: palingenesis.internals.score:probe_reward, weight: 0.0,
                  args: {probe: probes/known/best.safetensors}}     # weight 0: logged as rewards/known, a monitor
    """
    scorer = _scorer(probe, model, device)
    if scorer.probe.position == "last_prompt":
        if prompts is None:
            raise ValueError("probe_reward: a last_prompt probe needs the prompts")
        rows = [{"messages": list(p)} for p in prompts]
    else:
        if messages is None:
            raise ValueError("probe_reward: a span probe needs the conversations (messages)")
        rows = [{"messages": list(m)} for m in messages]
    p = scorer.probabilities(rows)
    if invert:
        p = 1 - p
    return [None if v != v else float(v) for v in p.tolist()]


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
