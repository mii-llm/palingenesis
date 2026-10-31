"""Token positions to read, from the chat template: the same renders and masks training uses.

A position spec is `last_prompt` or `<span>_<reduce>`:

    last_prompt        the last token of the prompt rendered with the generation prompt, i.e. the position whose
                       next-token distribution is the first answer token (Kadavath et al. 2022 read P(IK) here;
                       Kossen et al. 2024's "token before generation"). The prompt is the row's messages without a
                       trailing assistant turn, rendered exactly as palingenesis.margins renders a decision side
                       (tools, `prefix_calls` for the decision after a tool call, chat_template_kwargs).
    assistant_<r>      the tokens training puts loss on (palingenesis.data.ChatDataset's assistant mask: text, tool
                       calls, a reasoning block if the turn has one, the end-of-turn token)
    answer_<r>         the same without the reasoning block (the mask with train_on_reasoning: false)
    reasoning_<r>      the reasoning block only (assistant minus answer)

with <r> one of mean (one row: the span's mean, in float32), first, last (one row each) or all (one row per token).

Rows are JSON objects with `messages` (and optionally `tools`, `prefix_calls`, `prefix_content`,
`chat_template_kwargs`, which override the defaults key by key, as in ChatDataset). Batches are right-padded: causal
attention and recurrent layers (Gated DeltaNet) never see the padding at a real position, which left padding cannot
guarantee for a linear-attention state (see palingenesis.margins).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from palingenesis.internals.hooks import Selection

SPANS = ("assistant", "answer", "reasoning")
REDUCERS = ("mean", "first", "last", "all")
POSITIONS = ("last_prompt", *(f"{s}_{r}" for s in SPANS for r in REDUCERS))


def check_position(position: str) -> None:
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {', '.join(POSITIONS)}; not {position!r}")


@dataclass
class Encoded:
    """One row rendered: token ids and the positions read (one output row each, or one pooled row)."""

    ids: list[int]
    picks: list[int]
    pool: str  # mean | none


class Encoder:
    """Renders rows for one position spec. `encode` returns None for a row that has nothing at that position (no
    assistant span, an empty reasoning block) or is longer than max_tokens; `dropped` counts them by reason."""

    def __init__(
        self,
        tok,
        position: str = "last_prompt",
        chat_template_kwargs: dict | None = None,
        max_tokens: int | None = None,
    ):
        check_position(position)
        self.tok, self.position, self.max_tokens = tok, position, max_tokens
        self.chat_template_kwargs = dict(chat_template_kwargs if chat_template_kwargs is not None else {})
        self.dropped: dict[str, int] = {}
        self._datasets: dict = {}

    def _drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def _kwargs(self, row: dict) -> dict:
        own = row.get("chat_template_kwargs") or {}
        return {**self.chat_template_kwargs, **own}

    def encode(self, row: dict) -> Encoded | None:
        if not row.get("messages"):
            raise ValueError("internals: every row needs `messages`")
        enc = self._last_prompt(row) if self.position == "last_prompt" else self._span(row)
        if enc is None:
            return None
        if self.max_tokens is not None and len(enc.ids) > self.max_tokens:
            self._drop("too_long")
            return None
        return enc

    def prompt_ids(self, row: dict) -> list[int]:
        """Token ids of the row's prompt up to the generation position."""
        from palingenesis.margins import MarginReader

        messages = list(row["messages"])
        while messages and messages[-1].get("role") == "assistant" and not row.get("prefix_calls"):
            messages.pop()
        if not messages:
            raise ValueError("internals: a row has only assistant messages")
        side = {k: row[k] for k in ("tools", "prefix_calls", "prefix_content") if row.get(k) is not None}
        reader = MarginReader(None, self.tok, chat_template_kwargs=self._kwargs(row), device="cpu")
        return reader.prompt_ids({**side, "messages": messages})

    def _last_prompt(self, row: dict) -> Encoded:
        ids = self.prompt_ids(row)
        return Encoded(ids, [len(ids) - 1], "none")

    def _mask(self, row: dict, reasoning: bool) -> tuple[list[int], torch.Tensor] | None:
        from palingenesis.data import IGNORE_INDEX, ChatDataset

        kwargs = self._kwargs(row)
        key = (reasoning, repr(sorted(kwargs.items())))
        ds = self._datasets.get(key)
        if ds is None:
            ds = ChatDataset(
                None,
                self.tok,
                max_seq_length=self.max_tokens or 1 << 30,
                train_on_reasoning=reasoning,
                chat_template_kwargs=kwargs,
            )
            self._datasets[key] = ds
        out = ds._process({k: v for k, v in row.items() if k != "chat_template_kwargs"})
        if out is None:
            return None
        return out["input_ids"].tolist(), out["labels"] != IGNORE_INDEX

    def _span(self, row: dict) -> Encoded | None:
        span, reduce = self.position.rsplit("_", 1)
        full = self._mask(row, reasoning=True)
        if full is None:
            self._drop("no_assistant_span")
            return None
        ids, mask = full
        if span != "assistant":
            answer = self._mask(row, reasoning=False)
            if answer is None or answer[0] != ids:
                self._drop("answer_render_differs")
                return None
            mask = answer[1] if span == "answer" else mask & ~answer[1]
        picks = torch.nonzero(mask).flatten().tolist()
        if not picks:
            self._drop(f"empty_{span}_span")
            return None
        if reduce == "first":
            return Encoded(ids, picks[:1], "none")
        if reduce == "last":
            return Encoded(ids, picks[-1:], "none")
        return Encoded(ids, picks, "mean" if reduce == "mean" else "none")


@dataclass
class Batch:
    """Right-padded inputs and the Selection mapping them to output rows. `owners[j]` is (item index in the batch,
    token position) of output row j."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    select: Selection
    owners: list[tuple[int, int]]


def collate(items: list[Encoded], pad_id: int) -> Batch:
    width = max(len(e.ids) for e in items)
    input_ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    index, segment, owners = [], [], []
    pooled = items[0].pool == "mean"
    if any((e.pool == "mean") != pooled for e in items):
        raise ValueError("collate: mixed pooling in one batch")
    for b, e in enumerate(items):
        input_ids[b, : len(e.ids)] = torch.tensor(e.ids)
        mask[b, : len(e.ids)] = 1
        for p in e.picks:
            index.append(b * width + p)
            segment.append(b if pooled else len(owners))
            if not pooled:
                owners.append((b, p))
        if pooled:
            owners.append((b, -1))
    select = Selection(
        torch.tensor(index, dtype=torch.long),
        torch.tensor(segment, dtype=torch.long),
        len(owners),
        "mean" if pooled else "none",
    )
    return Batch(input_ids, mask, select, owners)
