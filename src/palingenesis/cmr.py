"""Common-mode rejection (CMR): train a skill without moving the model's overall propensity for a decision.

Measured failure this addresses: every agentic RL / DPO / interpolation run moved tool use along one axis, more needed
calls with more unneeded calls, or fewer of both. Distance-to-start anchors (KL, L2) have zero gradient at the start and
react only after the drift. CMR acts on the first-order effect from step 1.

Twins (x+, x-) per family: the same request where x+ needs a decision (e.g. a tool call) and x- does not (the tool set
swapped for irrelevant tools, a required argument removed). The decision margin at the first assistant token is

    m(x) = log p(d | x) - log(1 - p(d | x)),   d = the decision token (e.g. "<tool_call>")

Common mode: m(x+) + m(x-), the overall propensity. Differential mode: m(x+) - m(x-), the discrimination. Each refresh
estimates the common-mode direction u_f = mean over pairs of grad[m(x+) + m(x-)] for every family f, orthonormalised
across families (Gram-Schmidt; a family whose residual is below `min_residual` of its norm is dropped).

After every optimizer step the parameter UPDATE (not the gradient: adaptive optimizers rescale a projected gradient back
off the subspace, and momentum carries the old common mode) is corrected so its component along each u_f is zero
("hold") or a pre-registered amount ("target"):

    s_f = u_f . theta (before the step, float64 accumulation) ; c_f = u_f . theta_new - s_f ; theta -= (c_f - delta_f) u_f

To first order the common margins are unchanged by the step, while the differential margins (and everything else) move
freely. The parameters corrected are the ones the optimizer updates (the fp32 masters under MasterWeights).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import random
from dataclasses import dataclass, field

import torch

_nullcontext = contextlib.nullcontext

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CMRConfig:
    enabled: bool = False
    # {family: path to JSONL of {"pos": {"messages", "tools"?}, "neg": {"messages", "tools"?}}}
    families: dict = field(default_factory=dict)
    decision_token: str = "<tool_call>"  # must be a single token of the student's tokenizer
    # per-family override, e.g. {"cont": "<|im_end|>"} for the end-vs-continue decision after a prefix of calls
    decision_tokens: dict = field(default_factory=dict)
    # families held on the pos margin only (u = grad m+, the neg side is monitored but free): for twins that are not
    # minimal pairs, where holding m+ + m- would let m+ and m- drift in opposite directions
    pos_only: list = field(default_factory=list)
    pairs_per_refresh: int = 16  # twins per family used to estimate u_f
    micro_pairs: int = 4  # twins per forward / backward inside a refresh (activation memory)
    refresh_every: int = 1  # optimizer steps between direction refreshes
    mode: str = "hold"  # "hold": common-mode change 0; "target": steer the mean common margin toward `target`
    target: dict = field(default_factory=dict)  # mode "target": {family: mean common margin to reach}
    gain: float = 0.1  # mode "target": fraction of the remaining gap closed per step (along u_f, first order)
    min_residual: float = 0.05  # Gram-Schmidt: drop a family whose direction is (nearly) inside the others'
    chat_template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})
    seed: int = 0

    def validate(self) -> list[str]:
        errors = []
        if self.enabled and not self.families:
            errors.append("cmr.families: at least one twin family is required when cmr.enabled")
        if self.mode not in ("hold", "target"):
            errors.append(f"cmr.mode must be 'hold' or 'target', not {self.mode!r}")
        if self.mode == "target" and set(self.target) - set(self.families):
            errors.append("cmr.target names families that are not in cmr.families")
        if set(self.pos_only) - set(self.families):
            errors.append("cmr.pos_only names families that are not in cmr.families")
        if set(self.decision_tokens) - set(self.families):
            errors.append("cmr.decision_tokens names families that are not in cmr.families")
        if self.chat_template_kwargs.get("enable_thinking"):
            errors.append("cmr: thinking twins are not supported (the first assistant token is inside the think block)")
        if self.pairs_per_refresh < 1 or self.refresh_every < 1:
            errors.append("cmr.pairs_per_refresh and cmr.refresh_every must be >= 1")
        return errors


def load_twins(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if not (r.get("pos", {}).get("messages") and r.get("neg", {}).get("messages")):
                    raise ValueError(f"{path}: twin rows need pos.messages and neg.messages")
                rows.append(r)
    if not rows:
        raise ValueError(f"{path}: no twins")
    return rows


def decision_token_id(tok, token: str) -> int:
    ids = tok.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"cmr.decision_token {token!r} is {len(ids)} tokens for this tokenizer; it must be one")
    return ids[0]


class CommonModeRejection:
    """See the module docstring. `model` computes the margins (its parameters receive the margin gradients); `params`
    are the tensors the optimizer updates and that get corrected: the model's own parameters, or the fp32 masters whose
    .grad the model's backward fills (MasterWeights)."""

    def __init__(self, config: CMRConfig, model, tok, params: list[torch.Tensor], device: str | torch.device = "cuda"):
        errors = config.validate()
        if errors:
            raise ValueError("; ".join(errors))
        self.config, self.model, self.tok, self.params, self.device = config, model, tok, params, device
        self.token_id = decision_token_id(tok, config.decision_token)
        self.token_ids = {
            f: decision_token_id(tok, config.decision_tokens.get(f, config.decision_token)) for f in config.families
        }
        self.twins = {f: load_twins(p) for f, p in config.families.items()}
        self.rng = random.Random(config.seed)
        self.dirs: dict[str, list[torch.Tensor]] = {}  # family -> unit direction (per-parameter fp32 chunks)
        self.norms: dict[str, float] = {}
        self.common: dict[str, float] = {}  # mean common margin of the last refresh batch
        self.before: dict[str, float] = {}
        self.stats: dict[str, float] = {}
        self.steps = 0

    # ------------------------------------------------------------------ margins

    def _prompt_ids(self, side: dict) -> list[int]:
        """The prompt up to the decision position. With "prefix_calls" ([{name, arguments}]), an assistant turn holding
        those calls is rendered and cut right after its last "</tool_call>": the next token decides between ending the
        turn and making another call."""
        kw = dict(tools=side.get("tools") or None, tokenize=False, **self.config.chat_template_kwargs)
        calls = side.get("prefix_calls")
        if not calls:
            text = self.tok.apply_chat_template(side["messages"], add_generation_prompt=True, **kw)
        else:
            turn = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments") or {}}}
                    for c in calls
                ],
            }
            full = self.tok.apply_chat_template(list(side["messages"]) + [turn], add_generation_prompt=False, **kw)
            cut = full.rfind("</tool_call>")
            if cut < 0:
                raise ValueError("cmr: the chat template rendered no </tool_call> for prefix_calls")
            text = full[: cut + len("</tool_call>")]
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def margins(self, sides: list[dict], token_id: int | None = None) -> torch.Tensor:
        """Decision margin at the first assistant token for each prompt. One forward with RIGHT padding, read at each
        row's last real position: causal attention and recurrent (Gated DeltaNet) layers never see the padding there,
        which left padding could not guarantee for linear-attention state."""
        ids = [self._prompt_ids(s) for s in sides]
        width = max(len(x) for x in ids)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        input_ids = torch.full((len(ids), width), pad, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        from palingenesis.logits import final_hidden_states, output_head

        hidden = final_hidden_states(self.model, input_ids.to(self.device), mask.to(self.device))  # [B, T, H]
        last = torch.tensor([len(x) - 1 for x in ids], device=hidden.device)
        h = hidden[torch.arange(len(ids), device=hidden.device), last]  # head only on the B last real positions
        logits = output_head(self.model)(h).float()
        tid = self.token_id if token_id is None else token_id
        # exact log-odds log p(d) - log(1 - p(d)) = logit_d - logsumexp(other logits): no clamp, so a saturated side
        # (p near 0 or 1) keeps its gradient
        others = logits.clone()
        others[:, tid] = float("-inf")
        return logits[:, tid] - torch.logsumexp(others, dim=-1)

    @torch.no_grad()
    def evaluate(self, rows: list[dict], family: str | None = None) -> dict[str, float]:
        """Mean common / differential margin and decision rates on twins (e.g. held-out dev twins)."""
        was = self.model.training
        self.model.eval()
        tid = self.token_ids.get(family) if family else None
        mp, mn = self.margins([r["pos"] for r in rows], tid), self.margins([r["neg"] for r in rows], tid)
        self.model.train(was)
        return {
            "common": float((mp + mn).mean()),
            "differential": float((mp - mn).mean()),
            "p_pos": float(torch.sigmoid(mp).mean()),
            "p_neg": float(torch.sigmoid(mn).mean()),
        }

    # ------------------------------------------------------------------ directions

    def _grads(self) -> list[torch.Tensor]:
        return [
            p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)
            for p in self.params
        ]

    def _zero(self) -> None:
        for p in self.params:
            p.grad = None
        for p in self.model.parameters():
            p.grad = None

    def refresh(self) -> None:
        """Estimate u_f for every family, orthonormalise. Backward in micro-batches of `micro_pairs` (exact: the mean is
        linear) under bf16 autocast on CUDA, so activation memory stays bounded."""
        self._zero()
        self.dirs, self.norms = {}, {}  # free the previous directions before building the new ones
        raw: dict[str, list[torch.Tensor]] = {}
        micro = max(1, self.config.micro_pairs)
        cast = torch.autocast("cuda", dtype=torch.bfloat16) if str(self.device).startswith("cuda") else _nullcontext()
        for f, rows in self.twins.items():
            batch = self.rng.sample(rows, min(self.config.pairs_per_refresh, len(rows)))
            total = 0.0
            for i in range(0, len(batch), micro):
                chunk = batch[i : i + micro]
                with cast:
                    tid = self.token_ids[f]
                    mp = self.margins([r["pos"] for r in chunk], tid)
                    mn = None if f in self.config.pos_only else self.margins([r["neg"] for r in chunk], tid)
                part = (mp if mn is None else mp + mn).sum() / len(batch)
                part.backward()
                total += float(part.detach())
            raw[f] = self._grads()
            self.common[f] = total
            self._zero()
        for f, g in raw.items():
            norm0 = math.sqrt(sum(float(torch.sum(x * x, dtype=torch.float64)) for x in g))
            for d in self.dirs.values():  # Gram-Schmidt against the kept families
                dot = sum(float(torch.sum(x * y, dtype=torch.float64)) for x, y in zip(g, d))
                g = [x - dot * y for x, y in zip(g, d)]
            norm = math.sqrt(sum(float(torch.sum(x * x, dtype=torch.float64)) for x in g))
            if norm0 == 0 or norm < self.config.min_residual * norm0:
                logger.warning("cmr: family %s dropped this refresh (residual %.3g of %.3g)", f, norm, norm0)
                continue
            self.dirs[f] = [x / norm for x in g]
            self.norms[f] = norm0
            self.stats[f"cmr/u_norm/{f}"] = norm0
            self.stats[f"cmr/common/{f}"] = self.common[f]
        del raw

    # ------------------------------------------------------------------ the correction

    @torch.no_grad()
    def _dot(self, d: list[torch.Tensor]) -> float:
        return sum(float(torch.sum(x * p.detach().float(), dtype=torch.float64)) for x, p in zip(d, self.params))

    def pre_step(self) -> None:
        """Before optimizer.step(): refresh the directions when due and record u_f . theta."""
        if self.steps % self.config.refresh_every == 0 or not self.dirs:
            self.refresh()
        self.before = {f: self._dot(d) for f, d in self.dirs.items()}

    @torch.no_grad()
    def post_step(self) -> dict[str, float]:
        """After optimizer.step(): remove (or set) the update's component along each u_f. Returns metrics."""
        self.steps += 1
        out = dict(self.stats)
        for f, d in self.dirs.items():
            c = self._dot(d) - self.before[f]  # component of this step's update along u_f
            delta = 0.0
            if self.config.mode == "target" and f in self.config.target:
                # first order: d(common) = norm_f * (u_f . d theta); close `gain` of the gap per step
                delta = self.config.gain * (self.config.target[f] - self.common[f]) / self.norms[f]
            corr = c - delta
            for x, p in zip(d, self.params):
                p.sub_(corr * x.to(p.dtype))
            out[f"cmr/component/{f}"] = c
            out[f"cmr/correction/{f}"] = corr
            # what is left along u_f after the correction (should be ~delta; fp32 rounding of tiny per-element steps)
            out[f"cmr/residual/{f}"] = self._dot(d) - self.before[f] - delta
        return out
