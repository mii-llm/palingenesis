"""Decision margins: the log-odds of one decision at an exact token position, read from a causal LM.

A *side* is one prompt: {"messages": [...], "tools"?: [...]} renders the conversation with the generation prompt, so the
decision is the first assistant token (e.g. "<tool_call>" = call a tool now). With "prefix_calls" ([{name, arguments}])
an assistant turn holding those calls (after the text "prefix_content", default none) is rendered and cut right after
its last "</tool_call>": the next token decides between ending the turn ("<|im_end|>") and calling again.

The margin of a decision set D (usually one token) is the exact log-odds

    m(x) = log p(D | x) - log(1 - p(D | x)) = logsumexp(logits[D]) - logsumexp(logits[not D])

computed without clamping, so a saturated side keeps its gradient. One forward per batch with RIGHT padding, read at each
row's last real position: causal attention and recurrent layers (Gated DeltaNet) never see the padding there, which
left padding cannot guarantee for linear-attention state. The output head runs only at those B positions (no
full-vocabulary logits over the sequence).

Used by common-mode rejection (palingenesis.cmr) and margin surgery (palingenesis.surgery).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import torch


def load_twins(path: str) -> list[dict]:
    """JSONL rows {"pos": side, "neg": side, ...}; both sides need messages."""
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
    """The id of a decision token, which must be exactly one token of this tokenizer."""
    ids = tok.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"decision token {token!r} is {len(ids)} tokens for this tokenizer; it must be one")
    return ids[0]


def decision_ids(tok, tokens: str | Sequence[str]) -> list[int]:
    """Ids of a decision token or decision set (each element a single token; duplicates refused)."""
    tokens = [tokens] if isinstance(tokens, str) else list(tokens)
    ids = [decision_token_id(tok, t) for t in tokens]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"decision set {tokens!r} must be non-empty with distinct tokens")
    return ids


class MarginReader:
    """Reads decision margins of `model` (gradients flow unless called under no_grad)."""

    def __init__(self, model, tok, chat_template_kwargs: dict | None = None, device: str | torch.device | None = None):
        self.model, self.tok = model, tok
        self.chat_template_kwargs = dict(
            chat_template_kwargs if chat_template_kwargs is not None else {"enable_thinking": False}
        )
        if device is None:
            device = next(model.parameters()).device
        self.device = device

    def prompt_ids(self, side: dict) -> list[int]:
        """Token ids of the prompt up to the decision position (see the module docstring)."""
        kw = dict(tools=side.get("tools") or None, tokenize=False, **self.chat_template_kwargs)
        calls = side.get("prefix_calls")
        if not calls:
            text = self.tok.apply_chat_template(side["messages"], add_generation_prompt=True, **kw)
        else:
            turn = {
                "role": "assistant",
                "content": side.get("prefix_content") or "",
                "tool_calls": [
                    {"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments") or {}}}
                    for c in calls
                ],
            }
            full = self.tok.apply_chat_template(list(side["messages"]) + [turn], add_generation_prompt=False, **kw)
            cut = full.rfind("</tool_call>")
            if cut < 0:
                raise ValueError("margins: the chat template rendered no </tool_call> for prefix_calls")
            text = full[: cut + len("</tool_call>")]
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def logits_at_decision(self, sides: list[dict]) -> torch.Tensor:
        """[B, V] float32 logits at each side's decision position."""
        ids = [self.prompt_ids(s) for s in sides]
        return self.logits_at_last(ids)

    def logits_at_last(self, ids: list[list[int]]) -> torch.Tensor:
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
        h = hidden[torch.arange(len(ids), device=hidden.device), last]
        return output_head(self.model)(h).float()

    def margins(self, sides: list[dict], token_ids: int | Sequence[int]) -> torch.Tensor:
        """[B] exact log-odds of the decision set `token_ids` at each side's decision position."""
        return log_odds(self.logits_at_decision(sides), token_ids)


def log_odds(logits: torch.Tensor, token_ids: int | Sequence[int]) -> torch.Tensor:
    """logsumexp(logits[D]) - logsumexp(logits[not D]) per row, exact (no clamp)."""
    ids = [token_ids] if isinstance(token_ids, int) else list(token_ids)
    inside = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    inside[ids] = True
    pos = logits.masked_fill(~inside, float("-inf"))
    neg = logits.masked_fill(inside, float("-inf"))
    return torch.logsumexp(pos, dim=-1) - torch.logsumexp(neg, dim=-1)
