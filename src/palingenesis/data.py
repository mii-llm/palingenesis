"""Data pipeline: multi-dataset mixing, chat masking, pretraining, packing, collation.

Supports three data modes:
  1. SFT (default): chat template masking, only assistant tokens get loss
  2. Pretraining/CPT: loss on ALL tokens (continued pretraining)
  3. Mixed: multiple datasets with weighted sampling, each with its own mode

Config examples:

  # Single SFT dataset (simple mode, backward compatible)
  data:
    dataset: HuggingFaceH4/ultrachat_200k
    dataset_split: train_sft
    messages_field: messages

  # Multiple datasets with mixing weights
  data:
    sources:
      - dataset: your-org/agentic-traces
        split: train
        weight: 0.80
        mode: sft
        messages_field: messages
      - dataset: your-org/general-instruct
        split: train
        weight: 0.15
        mode: sft
        messages_field: messages
      - dataset: your-org/pretraining-data
        split: train
        weight: 0.05
        mode: pretrain
        text_field: text
"""

import bisect
import functools
import json
import logging
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerBase

from palingenesis.config import DataConfig
from palingenesis.validate_data import THINK_TAGS

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def _load_dataset_source(dataset_id: str, split: str, streaming: bool):
    from datasets import load_dataset

    path = Path(dataset_id)

    # Prepared-output directory (from `pgs prepare`): resolve to the data file
    if path.is_dir():
        from palingenesis.prepare import find_prepared_dataset

        prepared = find_prepared_dataset(path)
        if prepared is not None:
            path = prepared

    if path.exists() and path.suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path), split="train", streaming=streaming)

    if path.exists() and path.suffix in {".jsonl", ".json"}:
        return load_dataset("json", data_files=str(path), split="train", streaming=streaming)

    return load_dataset(dataset_id, split=split, streaming=streaming)


def _shard_streaming_dataset(dataset, rank: int, world_size: int):
    """Shard a dataset across processes (rank) and dataloader workers.

    Two dataset shapes need different handling:

    * **Streaming** (``datasets.IterableDataset``): calling ``.shard(num_shards=N)``
      with ``N`` greater than the dataset's own ``num_shards`` (e.g. a single-file
      JSONL has 1 shard) creates empty sub-shards that crash on ``.features``
      (``IndexError: list index out of range``). So we shard across processes with
      ``split_dataset_by_node`` (which degrades to a strided skip when there are
      fewer shards than nodes) and let HF's own DataLoader integration handle the
      per-worker split — it assigns shards to workers when possible and otherwise
      stops the surplus workers, which is exactly what we want.
    * **Map-style** (``datasets.Dataset``): contiguous ``.shard()`` is safe and
      cheap for both rank and worker, so we keep the original behaviour.
    """
    worker_info = torch.utils.data.get_worker_info()

    try:
        from datasets import IterableDataset as _HFIterableDataset

        is_streaming = isinstance(dataset, _HFIterableDataset)
    except Exception:
        is_streaming = False

    if is_streaming:
        # Per-worker sharding is handled automatically by HF when this streaming
        # dataset is iterated inside a worker, so we only shard across processes.
        if world_size > 1:
            try:
                from datasets.distributed import split_dataset_by_node

                dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
            except Exception:
                # Best-effort fallback; never over-shard (that is what crashes).
                num_shards = getattr(dataset, "num_shards", None) or getattr(dataset, "n_shards", None)
                if hasattr(dataset, "shard") and (num_shards is None or num_shards >= world_size):
                    dataset = dataset.shard(num_shards=world_size, index=rank)
        return dataset

    # Map-style dataset: contiguous shard across both rank and worker.
    shard_index = rank
    shard_count = world_size
    if worker_info is not None and worker_info.num_workers > 1:
        shard_index = shard_index * worker_info.num_workers + worker_info.id
        shard_count *= worker_info.num_workers

    if shard_count <= 1:
        return dataset

    if hasattr(dataset, "shard"):
        return dataset.shard(num_shards=shard_count, index=shard_index)

    dataset = dataset.skip(shard_index)
    return dataset.take_every(shard_count)


def _shard_then_shuffle(dataset, rank: int, world_size: int, shuffle_buffer: int, shuffle_seed: int):
    """Per-worker shard, THEN buffer-shuffle. The order is load-bearing:
    `shuffle().shard()` on a streaming dataset leaves every worker except the
    first with an empty shard list (datasets 5.x), killing the DataLoader.
    Shard-first is also the semantically right order — each worker streams its
    own files and shuffles locally within its buffer."""
    dataset = _shard_streaming_dataset(dataset, rank, world_size)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
    return dataset


# ══════════════════════════════════════════════════════════════════════════════
# CHAT-TEMPLATE TURN MARKERS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TurnMarkers:
    """Where a chat template opens and closes an assistant turn.

    header:    text written before every assistant turn whatever it contains, e.g.
               "<|im_start|>assistant\\n". What follows it is what the model generates.
    end:       the marker that closes an assistant turn, e.g. "<|im_end|>".
    turn_open: the special token that opens every turn, e.g. "<|im_start|>" ("" if the
               header has none); an assistant turn never extends past the next one.
    """

    header: str
    end: str
    turn_open: str


# Probe strings: plain ASCII words that no template rewrites and no tokenizer splits oddly.
_PROBE_USER, _PROBE_ANSWER, _PROBE_USER2, _PROBE_ANSWER2, _PROBE_REASONING = (
    "PgsProbeUserOne", "PgsProbeAnswerOne", "PgsProbeUserTwo", "PgsProbeAnswerTwo", "PgsProbeReasoningOne"
)


def renders_reasoning(render) -> bool:
    """Whether a chat template puts an assistant turn's reasoning field into the text."""
    try:
        text = render([{"role": "user", "content": _PROBE_USER},
                       {"role": "assistant", "content": _PROBE_ANSWER, "reasoning_content": _PROBE_REASONING,
                        "reasoning": _PROBE_REASONING}])
    except Exception:
        return False
    return _PROBE_REASONING in text


def detect_think_tags(render) -> tuple[str, str] | None:
    """The delimiters a chat template puts around an assistant turn's reasoning, read off a
    probe render: the text between the reasoning and the answer is the closing tag, and the
    opening tag is the closing one without its slash ("</think>", "[/THINK]", "◁/think▷",
    "</seed:think>"). None when the template does not render reasoning or its delimiters
    do not follow that pattern (the caller then uses the configured tags)."""
    try:
        text = render([{"role": "user", "content": _PROBE_USER},
                       {"role": "assistant", "content": _PROBE_ANSWER, "reasoning_content": _PROBE_REASONING,
                        "reasoning": _PROBE_REASONING}])
    except Exception:
        return None
    i = text.find(_PROBE_REASONING)
    j = text.find(_PROBE_ANSWER, i + len(_PROBE_REASONING)) if i != -1 else -1
    if j == -1:
        return None
    close = text[i + len(_PROBE_REASONING): j].strip()
    if "/" not in close or any(c.isspace() for c in close):
        return None
    open_ = close.replace("/", "", 1)
    return (open_, close) if text[:i].rstrip().endswith(open_) else None


def derive_turn_markers(render, tokenizer) -> TurnMarkers | None:
    """Derive the assistant-turn markers of a chat template by rendering probe turns.

    The header is the longest common prefix of the text the template puts before an
    assistant turn in every situation: the generation prompt, a final turn, a turn in
    the history, and a turn with reasoning. Templates differ exactly there (Qwen3.x open
    the generation prompt with "<think>\\n" and drop it from history), so the common
    prefix is the part that never belongs to the model's output. The end marker is the
    first special token after a final turn's text. Returns None when the template
    cannot be probed this way; callers then locate turns by their text.
    """
    user = {"role": "user", "content": _PROBE_USER}
    answer = {"role": "assistant", "content": _PROBE_ANSWER}
    try:
        base = render([user])
        gen = render([user], add_generation_prompt=True)
        final = render([user, answer])
        history = render([user, answer, {"role": "user", "content": _PROBE_USER2},
                          {"role": "assistant", "content": _PROBE_ANSWER2}])
    except Exception:
        return None
    try:
        with_reasoning = render([user, {**answer, "reasoning_content": _PROBE_REASONING,
                                        "reasoning": _PROBE_REASONING}])
    except Exception:
        with_reasoning = None

    heads = [gen]
    for text, probe in ((final, _PROBE_ANSWER), (history, _PROBE_ANSWER), (with_reasoning, _PROBE_REASONING)):
        if text is None:
            continue
        i = text.find(probe, len(base))
        if not text.startswith(base) or i == -1:
            if probe == _PROBE_REASONING:
                continue  # templates that do not render reasoning
            return None
        heads.append(text[:i])
    if not gen.startswith(base):
        return None
    header = os.path.commonprefix([h[len(base):] for h in heads])
    if not header.strip():
        return None

    special_ids = special_token_ids(tokenizer)

    def first_special(text: str) -> tuple[int, int] | None:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        for tid, (o0, o1) in zip(enc["input_ids"], enc["offset_mapping"]):
            if tid in special_ids and o1 > o0:
                return o0, o1
        return None

    after = final[final.index(_PROBE_ANSWER) + len(_PROBE_ANSWER):]
    found = first_special(after)
    if found is None or after[: found[0]].strip():
        return None
    end = after[found[0]: found[1]]
    after_history = history[history.index(_PROBE_ANSWER) + len(_PROBE_ANSWER):]
    if not after_history.lstrip().startswith(end):
        return None
    opener = first_special(header)
    turn_open = header[opener[0]: opener[1]] if opener is not None and not header[: opener[0]].strip() else ""
    return TurnMarkers(header=header, end=end, turn_open=turn_open)


def special_token_ids(tokenizer) -> set[int]:
    """Ids of every special token. `all_special_ids` lists only the named ones in
    transformers 5 (Qwen3.5 leaves out <|im_start|>); added tokens flagged special
    complete it."""
    ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    ids.update(i for i, tok in added.items() if getattr(tok, "special", False))
    return ids


class _TokenSpans:
    """Character → token lookups over a fast tokenizer's offset mapping."""

    def __init__(self, offsets: list[tuple[int, int]]):
        self.offsets = offsets
        self._starts = [o0 for o0, _ in offsets]
        self._ends = [o1 for _, o1 in offsets]

    def overlapping(self, c0: int, c1: int) -> list[int]:
        """Tokens overlapping the characters [c0, c1). Overlap, not containment: a
        byte-level BPE token that merges a leading space into the first character of the
        span (o0 == c0 - 1) belongs to it."""
        if c1 <= c0:
            return []
        lo = bisect.bisect_right(self._ends, c0)
        hi = bisect.bisect_left(self._starts, c1)
        return [ti for ti in range(lo, hi) if self._ends[ti] > self._starts[ti]]

    def first_at(self, c: int) -> int:
        """First non-empty token starting at or after character c."""
        ti = bisect.bisect_left(self._starts, c)
        while ti < len(self.offsets) and self._ends[ti] <= self._starts[ti]:
            ti += 1
        return ti


@functools.lru_cache(maxsize=16)
def _think_block(tags: tuple[str, str]) -> re.Pattern:
    """A reasoning block and the whitespace after it; an unclosed block runs to the end."""
    return re.compile(rf"{re.escape(tags[0])}.*?(?:{re.escape(tags[1])}\s*|$)", re.DOTALL)


def _subtract_intervals(span: tuple[int, int], excluded: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pieces, pos = [], span[0]
    for e0, e1 in sorted(excluded):
        if e0 > pos:
            pieces.append((pos, min(e0, span[1])))
        pos = max(pos, e1)
    if pos < span[1]:
        pieces.append((pos, span[1]))
    return [(a, b) for a, b in pieces if b > a]


# Tool-output regions that templates embed inside other turns (Qwen wraps a tool
# message in <tool_response> inside a user turn); trained under include_observations.
_ECHO_MARKERS = (
    ("<tool_response>", "</tool_response>"),
    ("<|tool▁output|>", "<|tool▁output▁end|>"),
    ("<observation>", "</observation>"),
    ("[Tool Output]", "[/Tool Output]"),
    ("```output\n", "```"),
)


def _echo_text_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for start_marker, end_marker in _ECHO_MARKERS:
        pos = 0
        while (start := text.find(start_marker, pos)) != -1:
            c0 = start + len(start_marker)
            c1 = text.find(end_marker, c0)
            c1 = len(text) if c1 == -1 else c1
            spans.append((c0, c1))
            pos = c1 + len(end_marker)
    return spans


# ══════════════════════════════════════════════════════════════════════════════
# CORE DATASETS
# ══════════════════════════════════════════════════════════════════════════════


class ChatDataset(IterableDataset):
    """SFT dataset: chat template masking, only assistant tokens get loss.

    When include_observations=True (ECHO mode), tool/observation role tokens
    also receive loss. This trains the model to predict tool outputs, teaching
    it a world model of tool behavior (arxiv:2605.24517, ICML 2026).
    """

    def __init__(
        self,
        dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        messages_field: str = "messages",
        rank: int = 0,
        world_size: int = 1,
        include_observations: bool = False,
        turn_scaling: str = "uniform",
        train_on_reasoning: bool = True,
        last_turn_only: bool = False,
        shuffle_buffer: int = 0,
        shuffle_seed: int = 0,
        tools_field: str = "tools",
        think_tags: tuple[str, str] | list[str] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.messages_field = messages_field
        # Per-row tool definitions, passed to the chat template as `tools=` so the
        # rendered system prompt declares them exactly as at inference time.
        self.tools_field = tools_field
        self.rank = rank
        self.world_size = world_size
        self.include_observations = include_observations
        self.turn_scaling = turn_scaling
        self.train_on_reasoning = train_on_reasoning
        # Delimiters of reasoning baked into the data's assistant content. None: the chat
        # template's own (detect_think_tags), else <think></think>. The masking always uses
        # the template's, so data baked with one model's tags trains another model's format.
        self.think_tags = tuple(think_tags) if think_tags else None
        # Template kwargs for every row (e.g. {"enable_thinking": true}); a row's own
        # `chat_template_kwargs` field overrides them key by key, so thinking and
        # non-thinking rows can be mixed in one dataset.
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        # Streaming shuffle, applied per worker AFTER sharding (see _shard_then_shuffle).
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_seed = shuffle_seed
        # last_turn_only: mask every assistant turn except the final one. Selects which
        # turns get loss (training) / are scored (eval). Use for eval-format SFT where
        # earlier assistant turns are a FIXED few-shot prefix (e.g. n-shot MCQA
        # exemplars) that must not receive loss.
        self.last_turn_only = last_turn_only
        # Per-row chat-template kwargs (e.g. {"enable_thinking": false}), read from the
        # example's `chat_template_kwargs` field and applied to EVERY render of that row,
        # including the fallback paths. Rows without the field render with none.
        self._template_kwargs: dict[str, Any] = {}
        # Turn markers of the chat template, derived once per set of template kwargs
        # (see _turn_markers).
        self._markers_cache: dict[str, TurnMarkers | None] = {}
        self._has_generation_span: bool | None = None
        # Per-iteration counts (kept, truncated, dropped_too_long, dropped_no_target),
        # logged when the stream ends so silently skipped rows are visible.
        self.stats: Counter = Counter()

    def _renders_reasoning(self) -> bool:
        """Whether the chat template (with this row's kwargs) renders a turn's reasoning field;
        cached per kwargs."""
        cache = self.__dict__.setdefault("_reasoning_cache", {})
        key = json.dumps({k: v for k, v in self._template_kwargs.items() if k != "tools"}, sort_keys=True,
                         default=str)
        if key not in cache:
            cache[key] = renders_reasoning(lambda m, **kw: self._render_chat(m, tokenize=False, **kw))
        return cache[key]

    def _render_tags(self) -> tuple[str, str]:
        """Reasoning delimiters in the rendered text: the template's, else the configured ones."""
        cache = self.__dict__.setdefault("_tags_cache", {})
        key = json.dumps({k: v for k, v in self._template_kwargs.items() if k != "tools"}, sort_keys=True,
                         default=str)
        if key not in cache:
            detected = detect_think_tags(lambda m, **kw: self._render_chat(m, tokenize=False, **kw))
            cache[key] = detected or self.think_tags or THINK_TAGS
        return cache[key]

    def _data_tags(self) -> tuple[str, str]:
        """Reasoning delimiters baked into the data's content."""
        return self.think_tags or self._render_tags()

    def _render_chat(self, messages: list[dict], **kwargs):
        """apply_chat_template with the current row's template kwargs. Explicit keyword
        arguments win, so a call site can still force e.g. add_generation_prompt."""
        return self.tokenizer.apply_chat_template(messages, **{**self._template_kwargs, **kwargs})

    def __iter__(self):
        dataset = _shard_then_shuffle(self.dataset, self.rank, self.world_size,
                                      self.shuffle_buffer, self.shuffle_seed)
        self.stats.clear()
        for example in dataset:
            too_long = self.stats["dropped_too_long"]
            result = self._process(example)
            if result is not None:
                self.stats["kept"] += 1
                yield result
            elif self.stats["dropped_too_long"] == too_long:
                self.stats["dropped_no_target"] += 1
        self._log_stats()

    def _log_stats(self) -> None:
        dropped = self.stats["dropped_too_long"] + self.stats["dropped_no_target"]
        if not dropped and not self.stats["truncated"]:
            return
        worker = torch.utils.data.get_worker_info()
        where = f" (rank {self.rank}, worker {worker.id})" if worker is not None else f" (rank {self.rank})"
        logger.info(
            f"Chat data{where}: {self.stats['kept']} conversations kept, {self.stats['truncated']} cut to "
            f"fit max_seq_length={self.max_seq_length}, {self.stats['dropped_too_long']} dropped (no trained "
            f"assistant turn fits), {self.stats['dropped_no_target']} dropped (nothing to train on)"
        )

    @staticmethod
    def _keep_last_segment(mask: torch.Tensor) -> torch.Tensor:
        """Zero all True runs except the last contiguous one.

        The template marks EVERY assistant turn's content as True; for last-turn-only
        training we keep just the final contiguous run (the real answer) and mask the
        earlier runs (e.g. fixed few-shot exemplar answers)."""
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            return mask
        # Gaps > 1 between consecutive True indices separate turns.
        breaks = (idx[1:] - idx[:-1] > 1).nonzero(as_tuple=False).flatten()
        if breaks.numel() == 0:
            return mask  # single assistant span already
        last_run_start = int(idx[int(breaks[-1]) + 1].item())
        new_mask = torch.zeros_like(mask)
        new_mask[last_run_start:] = mask[last_run_start:]
        return new_mask

    def _template_has_generation_span(self) -> bool:
        """Whether the chat template marks assistant output with {% generation %}, the
        only case in which transformers can return an assistant mask."""
        if self._has_generation_span is None:
            template = getattr(self.tokenizer, "chat_template", None)
            if isinstance(template, dict):
                template = template.get("default") or next(iter(template.values()), "")
            self._has_generation_span = bool(
                isinstance(template, str) and re.search(r"\{%-?\s*generation\s*-?%\}", template)
            )
        return self._has_generation_span

    def _process(self, example: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        from palingenesis.validate_data import (
            is_trained_message,
            normalize_messages,
            normalize_tools,
            restore_baked_think,
        )

        kwargs = example.get("chat_template_kwargs") or {}
        if isinstance(kwargs, str):  # JSON-encoded in some dataset exports
            kwargs = json.loads(kwargs) if kwargs.strip() else {}
        self._template_kwargs = {**self.chat_template_kwargs, **kwargs}
        tools = normalize_tools(example.get(self.tools_field))
        if tools is not None and "tools" not in self._template_kwargs:
            self._template_kwargs["tools"] = tools
        messages = example.get(self.messages_field)
        if not messages:
            # Try alternative field names (conversations, chat, dialogue, etc.)
            for alt in ("conversations", "conversation", "chat", "dialogue", "turns"):
                messages = example.get(alt)
                if messages:
                    break
        if not messages:
            return None

        # Role normalization: handle non-standard formats (ShareGPT, Alpaca, OpenAI
        # tool calls with JSON-string arguments, conversations stored as JSON strings)
        normalized = normalize_messages(example, self.messages_field, think_tags=self._data_tags())
        if normalized:
            messages = normalized
            if not self._renders_reasoning():        # baked <think> blocks stay in the content
                messages = restore_baked_think(messages)
        elif not isinstance(messages, list):
            return None
        # If normalization returns None, use raw messages (may still work with some templates)

        # Smart truncation: if conversation exceeds max_seq_length, truncate at
        # the last complete turn boundary that fits AND contains an assistant turn.
        # This preserves training signal (partial conversations with no assistant = useless).
        messages = self._smart_truncate(messages)
        if not messages:
            return None

        # The template's {% generation %} mask cannot tell turns apart, so per-message
        # training flags (`"loss": false`) need the turn-aware masker.
        per_turn_flags = any(
            m.get("role") == "assistant" and not is_trained_message(m) for m in messages
        )
        if per_turn_flags or self.turn_scaling != "uniform" or not self._template_has_generation_span():
            return self._fallback(messages)

        try:
            templated = self._render_chat(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                return_dict=True,
                truncation=True,
                max_length=self.max_seq_length,
            )
            input_ids = torch.tensor(templated["input_ids"], dtype=torch.long)
            attn_mask = torch.tensor(
                templated.get("attention_mask", [1] * len(templated["input_ids"])), dtype=torch.long
            )
            mask_key = "assistant_masks" if "assistant_masks" in templated else "assistant_tokens_mask"
            assistant_mask = torch.tensor(templated[mask_key], dtype=torch.bool)
        except (TypeError, KeyError, ValueError):
            return self._fallback(messages)

        # Check if the mask is all zeros (Qwen3.5 doesn't support return_assistant_tokens_mask)
        if assistant_mask.sum() == 0:
            return self._fallback(messages)

        # Last-turn-only: drop every assistant span but the final contiguous one.
        if self.last_turn_only:
            assistant_mask = self._keep_last_segment(assistant_mask)
            if assistant_mask.sum() == 0:
                return self._fallback(messages)

        labels = input_ids.clone()
        labels[~assistant_mask] = IGNORE_INDEX
        labels[attn_mask == 0] = IGNORE_INDEX

        # train_on_reasoning=False: the template's generation span includes any <think>
        # block, so the mask above trains it. Strip those spans back out to match the
        # documented behavior (loss only on the post-</think> answer) — same semantics
        # the fallback path applies. Only affects already-trained tokens, so it respects
        # last_turn_only and never touches user/system regions.
        if not self.train_on_reasoning:
            full_text = self._render_chat(messages, tokenize=False, add_generation_prompt=False)
            labels = self._strip_reasoning_labels(input_ids, labels, full_text)

        # ECHO: if include_observations, also unmask tool_response regions within user messages
        # Some models (Qwen3.5) wrap tool outputs as <tool_response>...</tool_response> inside user turns
        if self.include_observations:
            labels = self._apply_echo_from_text(input_ids, labels, messages)

        if (labels != IGNORE_INDEX).sum() == 0:
            return None
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

    def _split_reasoning(self, msg: dict) -> tuple[str | None, str]:
        """Return (reasoning_raw, answer_raw): strings expected to appear verbatim in the
        rendered text. reasoning_raw is None when the turn carries no reasoning. Handles
        the `reasoning` field (current convention), `reasoning_content` (older), and
        `<think>...</think>` embedded in content."""
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        for key in ("reasoning", "reasoning_content"):
            rc = msg.get(key)
            if isinstance(rc, str) and rc.strip():
                return rc.strip(), content.strip()
        from palingenesis.validate_data import split_leading_think

        baked, rest = split_leading_think(content, self._data_tags())   # only a LEADING block is reasoning
        if baked is not None:
            return (baked.strip() or None), rest.strip()
        return None, content.strip()

    def _strip_reasoning_labels(self, input_ids: torch.Tensor, labels: torch.Tensor, full_text: str) -> torch.Tensor:
        """Set labels to IGNORE across every `<think>...</think>` span (+ trailing
        whitespace) in the render. Used on the fast path when train_on_reasoning=False so
        reasoning doesn't receive loss even though the template's generation span encloses
        it. Only flips tokens that are currently trained, so it respects last_turn_only and
        leaves user/system regions untouched.

        Needs a fast tokenizer (offset mapping) and offsets that align with input_ids; on
        any mismatch it returns labels unchanged (reasoning stays trained — safe no-op).
        """
        tags = self._render_tags()
        if not getattr(self.tokenizer, "is_fast", False) or tags[1] not in full_text:
            return labels
        try:
            enc = self.tokenizer(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_seq_length,
            )
        except Exception:
            return labels
        offsets = enc.get("offset_mapping")
        if not offsets or enc["input_ids"] != input_ids.tolist():
            return labels
        for m in re.finditer(rf"{re.escape(tags[0])}.*?{re.escape(tags[1])}", full_text, flags=re.DOTALL):
            c0, c1 = m.start(), m.end()
            # Reasoning opens a turn: the token before it is the untrained header. A block
            # preceded by trained text is part of an answer (literal tags), and stays trained.
            first = next((i for i, (a, b) in enumerate(offsets) if b > c0), None)
            if first is not None and first > 0 and labels[first - 1] != IGNORE_INDEX:
                continue
            while c1 < len(full_text) and full_text[c1] in " \t\r\n":
                c1 += 1
            for ti, (o0, o1) in enumerate(offsets):
                if o1 > o0 and o0 < c1 and o1 > c0 and int(labels[ti]) != IGNORE_INDEX:
                    labels[ti] = IGNORE_INDEX
        return labels

    def _fallback(self, messages: list[dict]) -> dict[str, torch.Tensor] | None:
        """Mask assistant tokens when the template has no `{% generation %}` span.

        Fast tokenizers use the robust offset-based masker (`_fallback_offsets`), which
        makes NO prefix-consistency assumption and therefore handles templates that
        rewrite history -- e.g. Qwen3.x dropping <think> from past assistant turns, or
        MiniMax-M2 interleaved thinking. Slow tokenizers (no offset mapping) use the
        legacy progressive-tokenization masker, which is correct for the prefix-consistent
        templates they ship.
        """
        if getattr(self.tokenizer, "is_fast", False):
            try:
                # None here means "nothing to train on" (e.g. every turn flagged
                # loss: false), a verdict the progressive masker must not overturn.
                return self._fallback_offsets(messages)
            except Exception as e:
                logger.debug(f"offset masker failed ({e!r}); using the progressive masker")
        return self._fallback_progressive(messages)

    def _turn_markers(self) -> "TurnMarkers | None":
        """The template's assistant-turn markers for the current row's template kwargs
        (cached: they depend on the template and e.g. enable_thinking, not on the row)."""
        kwargs = {k: v for k, v in self._template_kwargs.items() if k != "tools"}
        key = json.dumps(kwargs, sort_keys=True, default=str)
        if key not in self._markers_cache:
            def render(messages, **kw):
                return self.tokenizer.apply_chat_template(messages, tokenize=False, **{**kwargs, **kw})

            self._markers_cache[key] = derive_turn_markers(render, self.tokenizer)
        return self._markers_cache[key]

    def _fallback_offsets(self, messages: list[dict]) -> dict[str, torch.Tensor] | None:
        """Template-agnostic masking on the final render, via offset mapping.

        Renders the conversation once and tokenizes it with offsets. Each assistant turn
        is located by the template's own assistant header (see `derive_turn_markers`):
        everything the template writes after the header, up to and including the
        end-of-turn marker, is what the model generates at inference, so all of it is
        trained -- text, tool calls, the closing `</think>` and the end-of-turn token. A
        `<think>` block is trained only when the turn carries reasoning and
        train_on_reasoning is set: an empty block that the template inserts for a turn
        without reasoning is scaffolding, not model output.

        Only the final render is used (never render(messages[:i]) as a prefix of
        render(messages)), so templates that rewrite history (Qwen3.x drop reasoning
        from earlier turns) are handled. Requires a fast tokenizer (raises otherwise).
        Returns None when nothing in the conversation is trained.
        """
        from palingenesis.validate_data import is_trained_message

        full = self._render_chat(messages, tokenize=False, add_generation_prompt=False)
        enc = self._encode(full)
        ids = enc["input_ids"][: self.max_seq_length]
        offsets = (enc.get("offset_mapping") or [])[: self.max_seq_length]
        if not ids:
            return None
        if not offsets:
            raise ValueError("tokenizer returned no offset mapping")
        input_ids = torch.tensor(ids, dtype=torch.long)
        attn_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        spans = _TokenSpans(offsets)

        markers = self._turn_markers()
        if markers is None:
            located = self._locate_turns_by_content(full, messages, spans, input_ids)
        else:
            located = self._locate_turns_by_markers(full, messages, spans, markers)
        turns, echo = located  # turns: [(token indices, message)] per assistant message, in order

        def train(idx: list[int]) -> None:
            if idx:
                t = torch.tensor(idx, dtype=torch.long)
                labels[t] = input_ids[t]

        for tset in echo:  # tool/observation ECHO turns: always trained, never gated by last_turn_only
            train(tset)

        trained_turns = [(tset, i) for i, (tset, msg) in enumerate(turns) if is_trained_message(msg)]
        if self.last_turn_only and trained_turns:
            trained_turns = trained_turns[-1:]
        for tset, _ in trained_turns:
            train(tset)

        labels[attn_mask == 0] = IGNORE_INDEX

        if self.include_observations:
            for c0, c1 in _echo_text_spans(full):
                train(spans.overlapping(c0, c1))

        if (labels != IGNORE_INDEX).sum() == 0:
            return None

        result = {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

        n_trained = len(trained_turns)
        if self.turn_scaling != "uniform" and n_trained > 1:
            loss_weights = torch.zeros_like(input_ids, dtype=torch.float32)
            loss_weights[labels != IGNORE_INDEX] = 1.0
            for rank, (tset, _) in enumerate(trained_turns):
                if self.turn_scaling == "progressive":
                    w = ((rank + 1) / n_trained) ** 0.5
                else:  # last_heavy
                    w = 2.0 if rank == n_trained - 1 else 1.0
                if tset:
                    loss_weights[torch.tensor(tset, dtype=torch.long)] = w
            loss_weights[labels == IGNORE_INDEX] = 0.0
            valid = loss_weights > 0
            if valid.any():
                loss_weights[valid] /= loss_weights[valid].mean()
            result["loss_weights"] = loss_weights

        return result

    def _locate_turns_by_markers(self, full: str, messages: list[dict], spans: "_TokenSpans",
                                 markers: "TurnMarkers"):
        """Token indices of every assistant turn (header excluded, end-of-turn included),
        and of the ECHO-trained tool/observation contents."""
        echo_roles = {"tool", "observation", "ipython", "function"} if self.include_observations else set()
        turns: list[tuple[list[int], dict]] = []
        echo: list[list[int]] = []
        cursor = 0
        n = len(full)

        def next_turn(start: int) -> int:
            """Where the next turn opens (bounds an assistant turn and a content search)."""
            nxt = [p for p in (full.find(markers.turn_open, start) if markers.turn_open else -1,
                               full.find(markers.header, start)) if p != -1]
            return min(nxt) if nxt else n

        for msg in messages:
            role = msg.get("role")
            if role in echo_roles:
                _, answer = self._split_reasoning(msg)
                stop = full.find(markers.header, cursor)
                p = full.find(answer, cursor, n if stop == -1 else stop) if answer else -1
                if p != -1:
                    echo.append(spans.overlapping(p, p + len(answer)))
                continue
            if role != "assistant":
                continue

            h = full.find(markers.header, cursor)
            if h == -1:
                break
            start = h + len(markers.header)
            limit = next_turn(start)
            reasoning, _ = self._split_reasoning(msg)
            e = full.find(markers.end, start, limit) if markers.end else -1
            end = e + len(markers.end) if e != -1 else limit
            cursor = end

            train_think = self.train_on_reasoning and reasoning is not None
            pieces = [(start, end)]
            if not train_think:
                # Only a block that OPENS the turn is reasoning; <think> tags later in the answer
                # are text the model writes, and trained as such.
                lead = _think_block(self._render_tags()).match(
                    full, start + (len(full[start:end]) - len(full[start:end].lstrip())), end)
                excluded = [lead.span()] if lead else []
                if reasoning and not self.train_on_reasoning:
                    # Reasoning rendered outside <think> tags (other templates' channels).
                    p = full.find(reasoning, start, end)
                    if p != -1:
                        excluded.append((p, p + len(reasoning)))
                pieces = _subtract_intervals((start, end), excluded)
            tset = sorted({ti for c0, c1 in pieces for ti in spans.overlapping(c0, c1)})
            turns.append((tset, msg))
        return turns, echo

    def _locate_turns_by_content(self, full: str, messages: list[dict], spans: "_TokenSpans",
                                 input_ids: torch.Tensor):
        """Legacy locator for templates whose turn markers cannot be derived: finds each
        turn's reasoning/answer TEXT in the render, plus one end-of-turn special token.
        Text the template renders from other fields (tool calls) is not found."""
        special_ids = special_token_ids(self.tokenizer)
        n_tok = len(input_ids)
        echo_roles = {"tool", "observation", "ipython", "function"} if self.include_observations else set()
        turns: list[tuple[list[int], dict]] = []
        echo: list[list[int]] = []
        cursor = 0

        def is_ws(ti: int) -> bool:
            o0, o1 = spans.offsets[ti]
            return o1 > o0 and full[o0:o1].strip() == ""

        for msg in messages:
            role = msg.get("role")
            reasoning_raw, answer_raw = self._split_reasoning(msg)

            # Advance the cursor past this turn's reasoning + answer text, for EVERY role,
            # so later searches never match backwards into an earlier turn.
            r0 = r1 = -1
            if reasoning_raw:
                p = full.find(reasoning_raw, cursor)
                if p != -1:
                    r0, r1 = p, p + len(reasoning_raw)
                    cursor = r1
            a0 = a1 = -1
            if answer_raw:
                p = full.find(answer_raw, cursor)
                if p != -1:
                    a0, a1 = p, p + len(answer_raw)
                    cursor = a1

            if role != "assistant" and role not in echo_roles:
                continue

            tset: set[int] = set()
            if role == "assistant" and self.train_on_reasoning and reasoning_raw and r0 != -1:
                think_open = full.rfind(self._render_tags()[0], 0, r0)
                tset.update(spans.overlapping(think_open if think_open != -1 else r0, r1))
                if a0 != -1:
                    tset.update(spans.overlapping(r1, a0))
            if a0 != -1:
                tset.update(spans.overlapping(a0, a1))

            # Terminator: skip whitespace, include ONE end-of-turn special token.
            anchor = a1 if a1 != -1 else r1
            if anchor != -1 and role == "assistant":
                ti = spans.first_at(anchor)
                while ti < n_tok and is_ws(ti):
                    tset.add(ti)
                    ti += 1
                if ti < n_tok and int(input_ids[ti]) in special_ids:
                    tset.add(ti)

            if role == "assistant":
                turns.append((sorted(tset), msg))
            else:
                echo.append(sorted(tset))
        return turns, echo

    def _fallback_progressive(self, messages: list[dict]) -> dict[str, torch.Tensor] | None:
        """Legacy fallback masking: progressive tokenization to find exact turn boundaries.

        Used for slow tokenizers (no offset mapping). Assumes the template is
        prefix-consistent: render(messages[:i+1]) is a token-prefix of render(messages).
        This holds for the templates slow tokenizers ship, but NOT for history-rewriting
        templates (Qwen3.x) -- those require a fast tokenizer + `_fallback_offsets`.

        Strategy for precise boundaries:
        1. Tokenize full conversation to get input_ids
        2. For each turn i, tokenize messages[:i] and messages[:i+1]
        3. The tokens unique to messages[:i+1] belong to turn i
        4. For assistant turns: additionally exclude the header/role tokens
           by tokenizing the header prefix separately

        This eliminates the ~1-3 token boundary imprecision of the naive approach.
        """
        from palingenesis.validate_data import is_trained_message

        # Tokenize the full conversation
        full_text = self._render_chat(messages, tokenize=False, add_generation_prompt=False)
        tokens = self.tokenizer(full_text, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attn_mask = tokens["attention_mask"].squeeze(0)
        seq_len = len(input_ids)
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Roles that get loss
        train_roles = {"assistant"}
        if self.include_observations:
            train_roles.update({"tool", "observation", "ipython", "function"})

        # Track turn boundaries for progressive scaling
        assistant_turn_idx = 0
        total_assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
        turn_boundaries: list[tuple[int, int, int]] = []  # (start, end, turn_idx)

        # Tokenize progressively: messages[:0], messages[:1], messages[:2], ...
        # The token count at each prefix gives us exact turn boundaries.
        prev_len = 0
        for i, msg in enumerate(messages):
            try:
                # Tokenize prefix including this turn
                prefix_text = self._render_chat(
                    messages[: i + 1], tokenize=False, add_generation_prompt=False
                )
                prefix_ids = self.tokenizer(prefix_text, truncation=True, max_length=self.max_seq_length)["input_ids"]
                curr_len = len(prefix_ids)
            except Exception:
                # Some prefixes (e.g. system-only prefixes) are not renderable
                # with all chat templates. Skip them and keep scanning.
                prev_len = max(prev_len, 0)
                continue

            if msg.get("role") in train_roles and is_trained_message(msg):
                # This turn gets loss. But we want to exclude the header/role tokens
                # (e.g., "<|start_header_id|>assistant<|end_header_id|>\n\n")
                # Strategy: tokenize messages[:i] + a stub that produces the header
                # then everything after that header is the actual content.

                # Find where the content starts by tokenizing the prefix WITHOUT
                # this turn's content (just the header/role marker).
                # The header is: everything between prev_len and the start of content.
                content = msg.get("content", "") or ""
                has_reasoning = bool(msg.get("reasoning_content"))
                if (content or has_reasoning) and msg.get("role") == "assistant":
                    # Tokenize a stub version of this turn to find where trained
                    # content starts. What the stub strips decides what gets loss:
                    #   train_on_reasoning=True  → strip content AND reasoning, so
                    #     the boundary lands BEFORE the <think> block and reasoning
                    #     tokens are trained (required for reasoning distillation).
                    #   train_on_reasoning=False → keep reasoning in the stub, so
                    #     the boundary lands AFTER </think> and only the final
                    #     response is trained.
                    stub_msg = {**msg, "content": ""}
                    if self.train_on_reasoning:
                        stub_msg.pop("reasoning_content", None)
                    stub_messages = messages[:i] + [stub_msg]
                    try:
                        stub_text = self._render_chat(
                            stub_messages, tokenize=False, add_generation_prompt=False
                        )
                        stub_ids = self.tokenizer(stub_text, truncation=True, max_length=self.max_seq_length)["input_ids"]
                        # The stub renders the turn-CLOSING tokens (e.g. <|im_end|>)
                        # right after the header, so len(stub_ids) overshoots the
                        # content start by the closing-tag length. The exact
                        # boundary is where the stub and the real render diverge:
                        # their longest common token prefix.
                        limit = min(len(stub_ids), len(prefix_ids))
                        content_start = 0
                        while content_start < limit and stub_ids[content_start] == prefix_ids[content_start]:
                            content_start += 1
                    except Exception:
                        # If stub fails (some templates need non-empty content), use prev_len
                        content_start = prev_len
                else:
                    # For tool/observation roles, unmask the entire turn (header is short/irrelevant)
                    content_start = prev_len

                # Clamp to sequence length
                s = min(content_start, seq_len)
                e = min(curr_len, seq_len)

                if s < e:
                    labels[s:e] = input_ids[s:e]

                if msg.get("role") == "assistant":
                    turn_boundaries.append((s, e, assistant_turn_idx))
                    assistant_turn_idx += 1

            prev_len = curr_len

        # Last-turn-only: re-mask every assistant span except the final one. Runs on
        # the assistant turn_boundaries, so ECHO tool/observation spans are untouched.
        if self.last_turn_only and total_assistant_turns > 1:
            last_idx = total_assistant_turns - 1
            for s, e, idx in turn_boundaries:
                if idx != last_idx:
                    labels[s:e] = IGNORE_INDEX

        # ECHO: Also unmask <tool_response> regions inside user messages
        if self.include_observations:
            labels = self._apply_echo_from_text(input_ids, labels, messages)

        if (labels != IGNORE_INDEX).sum() == 0:
            return None

        result = {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

        # Apply per-turn loss scaling if configured
        if self.turn_scaling != "uniform" and total_assistant_turns > 1 and turn_boundaries:
            loss_weights = torch.ones_like(input_ids, dtype=torch.float32)
            loss_weights[labels == IGNORE_INDEX] = 0.0

            if self.turn_scaling == "progressive":
                for s, e, idx in turn_boundaries:
                    w = ((idx + 1) / total_assistant_turns) ** 0.5
                    loss_weights[s:e] = w
            elif self.turn_scaling == "last_heavy":
                for s, e, idx in turn_boundaries:
                    if idx == total_assistant_turns - 1:
                        loss_weights[s:e] = 2.0
                    else:
                        loss_weights[s:e] = 1.0

            valid_mask = loss_weights > 0
            if valid_mask.any():
                mean_w = loss_weights[valid_mask].mean()
                loss_weights[valid_mask] /= mean_w

            result["loss_weights"] = loss_weights

        return result

    def _apply_echo_from_text(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        messages: list[dict],
    ) -> torch.Tensor:
        """Apply ECHO observation loss by detecting tool output regions in tokenized text.

        Handles multiple patterns for tool outputs:
          1. Explicit role: "tool", "observation", "ipython", "function" → already handled in fallback
          2. Qwen-style: <tool_response>...</tool_response> inside user messages
          3. Generic markers: [Tool Output], [Observation], ```output inside user messages
          4. Content field: messages with tool_call_id or name field (tool responses)

        For pattern 2 (Qwen3.5), the chat template converts:
          {"role": "tool", "content": "4"} → user message with <tool_response>4</tool_response>

        We find these regions in the tokenized output and unmask them for loss.
        """
        # Decode full text to find tool_response markers
        full_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)

        # Markers that indicate tool output content (model-agnostic)
        TOOL_START_MARKERS = ["<tool_response>", "<|tool▁output|>", "<observation>", "[Tool Output]", "```output\n"]
        TOOL_END_MARKERS = ["</tool_response>", "<|tool▁output▁end|>", "</observation>", "[/Tool Output]", "```"]

        for start_marker, end_marker in zip(TOOL_START_MARKERS, TOOL_END_MARKERS):
            # Find all occurrences of this marker pair
            search_pos = 0
            while True:
                start_idx = full_text.find(start_marker, search_pos)
                if start_idx == -1:
                    break
                # Find the content start (after the opening tag)
                content_start = start_idx + len(start_marker)
                end_idx = full_text.find(end_marker, content_start)
                if end_idx == -1:
                    # No closing tag — take until end of text
                    end_idx = len(full_text)

                # Convert character positions to token positions
                # Tokenize the prefix up to content_start and end_idx
                prefix_to_start = full_text[:content_start]
                prefix_to_end = full_text[:end_idx]

                tok_start = len(self.tokenizer.encode(prefix_to_start, add_special_tokens=False))
                tok_end = len(self.tokenizer.encode(prefix_to_end, add_special_tokens=False))

                # Clamp to valid range
                tok_start = min(tok_start, len(input_ids))
                tok_end = min(tok_end, len(input_ids))

                # Unmask these tokens (give them loss)
                if tok_start < tok_end:
                    labels[tok_start:tok_end] = input_ids[tok_start:tok_end]

                search_pos = end_idx + len(end_marker)

        return labels

    def _encode(self, text: str):
        """Tokenize a render with offsets, remembering the last one: truncation and
        masking both need the full conversation's encoding."""
        cached = self._last_encoding
        if cached is not None and cached[0] == text:
            return cached[1]
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        self._last_encoding = (text, enc)
        return enc

    _last_encoding: tuple[str, Any] | None = None

    def _smart_truncate(self, messages: list[dict]) -> list[dict] | None:
        """Cut an over-long conversation after the last trained assistant turn that fits.

        Keeps whole turns (a conversation cut mid-turn trains on a fragment and loses the
        end-of-turn token) and ends on an assistant turn (trailing user/tool turns would
        cost compute without loss). Returns None when not even the first trained
        assistant turn fits.

        Cheap for the common case: a token is at least one UTF-8 byte, so a conversation
        whose serialized messages and tools, plus a margin for template text, fit in
        max_seq_length bytes fits in max_seq_length tokens. Otherwise the full render is
        tokenized once; if it is too long, the cut is searched on renders alone (token
        counts estimated from the full render's characters per token) and only the
        candidates next to the limit are tokenized.
        """
        from palingenesis.validate_data import is_trained_message

        payload = len(json.dumps(messages, ensure_ascii=False, default=str).encode())
        tools = self._template_kwargs.get("tools")
        if tools:
            payload += len(json.dumps(tools, ensure_ascii=False, default=str).encode())
        if payload + 64 * len(messages) + 2048 <= self.max_seq_length:
            return messages

        def render(k: int) -> str:
            return self._render_chat(messages[:k], tokenize=False, add_generation_prompt=False)

        try:
            full = render(len(messages))
            n_full = len(self._encode(full)["input_ids"])
        except Exception:
            return messages  # the masker reports templates that cannot render this row
        if n_full <= self.max_seq_length:
            return messages

        chars_per_token = len(full) / max(n_full, 1)

        def fits(k: int) -> bool:
            try:
                return len(self.tokenizer(render(k), add_special_tokens=False)["input_ids"]) <= self.max_seq_length
            except Exception:
                return False

        ends = [k for k in range(1, len(messages) + 1)
                if messages[k - 1].get("role") == "assistant" and is_trained_message(messages[k - 1])]
        # Largest end whose ESTIMATED length fits (renders only; they grow with the turns).
        lo, hi, guess = 0, len(ends) - 1, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                estimate = len(render(ends[mid])) / chars_per_token
            except Exception:
                estimate = float("inf")
            if estimate <= self.max_seq_length:
                guess, lo = mid, mid + 1
            else:
                hi = mid - 1
        # Settle the estimate with exact counts: step down while too long, up while the
        # next one still fits.
        i = max(guess, 0)
        while i >= 0 and not fits(ends[i]):
            i -= 1
        while 0 <= i < len(ends) - 1 and fits(ends[i + 1]):
            i += 1
        if i < 0:
            self.stats["dropped_too_long"] += 1
            return None
        self.stats["truncated"] += 1
        return messages[: ends[i]]


class PretrainDataset(IterableDataset):
    """Pretraining/CPT dataset: loss on ALL tokens (no masking)."""

    def __init__(
        self,
        dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        text_field: str = "text",
        rank: int = 0,
        world_size: int = 1,
        shuffle_buffer: int = 0,
        shuffle_seed: int = 0,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_field = text_field
        self.rank = rank
        self.world_size = world_size
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_seed = shuffle_seed

    def __iter__(self):
        dataset = _shard_then_shuffle(self.dataset, self.rank, self.world_size,
                                      self.shuffle_buffer, self.shuffle_seed)
        for example in dataset:
            result = self._process(example)
            if result is not None:
                yield result

    def _process(self, example: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        text = example.get(self.text_field, "")
        if not text:
            return None
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attn_mask = tokens["attention_mask"].squeeze(0)
        # Pretraining: loss on ALL tokens (standard causal LM)
        labels = input_ids.clone()
        labels[attn_mask == 0] = IGNORE_INDEX
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-DATASET MIXING
# ══════════════════════════════════════════════════════════════════════════════


class MixedDataset(IterableDataset):
    """Weighted interleaving of multiple datasets.

    Samples from each source with probability proportional to its weight.
    Stops when any source is exhausted (epoch boundary).

    Each source can be either SFT (chat masking) or pretrain (all-token loss).
    """

    def __init__(
        self,
        sources: list[IterableDataset],
        weights: list[float],
        seed: int = 42,
        names: list[str] | None = None,
    ):
        assert len(sources) == len(weights)
        assert all(w >= 0 for w in weights)
        total = sum(weights)
        self.sources = sources
        self.probs = [w / total for w in weights]
        self.seed = seed
        self.names = names or [f"source[{i}]" for i in range(len(sources))]

    def __iter__(self):
        rng = random.Random(self.seed)
        iterators = [iter(s) for s in self.sources]
        indices = list(range(len(self.sources)))
        active = list(indices)
        yielded = [0] * len(self.sources)

        while active:
            # When every source is still active this is identical to sampling over
            # `indices` with `self.probs` (determinism preserved for the healthy case).
            probs = [self.probs[i] for i in active]
            idx = rng.choices(active, weights=probs, k=1)[0]
            try:
                item = next(iterators[idx])
                yielded[idx] += 1
                yield item
            except StopIteration:
                if yielded[idx] == 0:
                    # A source that produced NOTHING is a misconfiguration (wrong
                    # field/format/split, empty file), not an epoch boundary. Drop
                    # it loudly and keep training on the rest instead of silently
                    # ending the epoch (which, with packing, yields 0 steps).
                    logger.warning(
                        "Data source '%s' yielded 0 usable examples — dropping it from the "
                        "mix. Check its messages/text field, format and split.",
                        self.names[idx],
                    )
                    active.remove(idx)
                    continue
                break  # A non-empty source exhausted → epoch boundary.


# ══════════════════════════════════════════════════════════════════════════════
# PACKING
# ══════════════════════════════════════════════════════════════════════════════


class PackedDataset(IterableDataset):
    """Packs whole conversations into blocks of at most max_len tokens.

    Documents are never split: a conversation cut across two blocks would train its
    tail without the prompt it answers. Blocks carry `position_ids` that restart at 0
    for every document, which is what keeps documents apart in the forward pass (see
    palingenesis.packing).

    sort_buffer > 0: first-fit-decreasing bin packing over a buffer of that many
    documents (longest first, each into the first open block with room). Blocks that
    are not full stay open across buffers, so the fill rate stays high without a
    large buffer; at most `max_open` blocks are kept, the fullest emitted first.
    sort_buffer == 0: next-fit in arrival order (a block is emitted when the next
    document does not fit).

    Output per block (trailing blocks may be shorter than max_len; the collator pads):
      - input_ids, labels (and loss_weights, when present): the documents concatenated
      - attention_mask: all ones
      - position_ids: restart at 0 at each document
    """

    def __init__(self, base: IterableDataset, max_len: int, eos_id: int = 0, sort_buffer: int = 256,
                 max_open: int = 64):
        self.base = base
        self.max_len = max_len
        self.eos_id = eos_id  # unused; kept for call-site compatibility
        self.sort_buffer = sort_buffer
        self.max_open = max(1, max_open)

    def __iter__(self):
        if self.sort_buffer > 0:
            yield from self._bin_packing()
        else:
            yield from self._next_fit()

    def _doc(self, ex: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Oversized documents are cut to max_len (the chat pipeline already keeps whole
        # turns within max_seq_length; this only guards other sources).
        keys = ["input_ids", "labels"] + (["loss_weights"] if "loss_weights" in ex else [])
        return {k: ex[k][: self.max_len] for k in keys}

    @staticmethod
    def _block(docs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        block = {k: torch.cat([d[k] for d in docs]) for k in docs[0]}
        n = block["input_ids"].numel()
        block["attention_mask"] = torch.ones(n, dtype=torch.long)
        block["position_ids"] = torch.cat([torch.arange(d["input_ids"].numel()) for d in docs])
        return block

    def _next_fit(self):
        docs: list[dict[str, torch.Tensor]] = []
        used = 0
        for ex in self.base:
            doc = self._doc(ex)
            n = doc["input_ids"].numel()
            if docs and used + n > self.max_len:
                yield self._block(docs)
                docs, used = [], 0
            docs.append(doc)
            used += n
        if docs:
            yield self._block(docs)

    def _bin_packing(self):
        open_bins: list[tuple[int, list[dict[str, torch.Tensor]]]] = []  # (tokens used, docs)
        buffer: list[dict[str, torch.Tensor]] = []

        def flush():
            buffer.sort(key=lambda d: d["input_ids"].numel(), reverse=True)
            for doc in buffer:
                n = doc["input_ids"].numel()
                for i, (used, docs) in enumerate(open_bins):
                    if used + n <= self.max_len:
                        docs.append(doc)
                        open_bins[i] = (used + n, docs)
                        break
                else:
                    open_bins.append((n, [doc]))
            buffer.clear()
            full = [docs for used, docs in open_bins if used == self.max_len]
            open_bins[:] = [(used, docs) for used, docs in open_bins if used < self.max_len]
            open_bins.sort(key=lambda b: b[0], reverse=True)
            while len(open_bins) > self.max_open:
                full.append(open_bins.pop(0)[1])
            return full

        for ex in self.base:
            buffer.append(self._doc(ex))
            if len(buffer) >= self.sort_buffer:
                for docs in flush():
                    yield self._block(docs)
        for docs in flush():
            yield self._block(docs)
        for _, docs in open_bins:
            yield self._block(docs)


# ══════════════════════════════════════════════════════════════════════════════
# COLLATION + BUILDER
# ══════════════════════════════════════════════════════════════════════════════


class LengthGroupedDataset(IterableDataset):
    """Reorder a sample stream so samples in the same batch have similar lengths.

    Without packing, `_collate_fn` pads every batch to its longest sample. With
    randomly-shuffled chat data (a few 4K-token samples among many short ones),
    that means MOST of the forward/backward FLOPs are spent on pad tokens.

    This buffers `buffer_size` samples, sorts them by length, cuts the sorted
    buffer into consecutive groups of `batch_size`, shuffles the group ORDER
    (so short/long batches interleave randomly), and yields group by group.
    The downstream DataLoader (same batch_size, aligned stream) reassembles
    exactly those groups, so per-batch padding drops to the within-group
    length spread — typically near zero after sorting.

    Sample-level randomness comes from the upstream shuffle; within a buffer
    only ORDER is affected, never which samples are seen.
    """

    def __init__(self, dataset: IterableDataset, batch_size: int, buffer_size: int = 512, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        # Align buffer to batch_size so group boundaries match DataLoader batches
        self.buffer_size = max(buffer_size - buffer_size % batch_size, batch_size)
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        buf: list[dict[str, torch.Tensor]] = []
        for sample in self.dataset:
            buf.append(sample)
            if len(buf) >= self.buffer_size:
                yield from self._drain(buf, rng)
                buf = []
        if buf:
            yield from self._drain(buf, rng)

    def _drain(self, buf: list[dict[str, torch.Tensor]], rng: random.Random):
        buf.sort(key=lambda s: s["input_ids"].size(0))
        groups = [buf[i : i + self.batch_size] for i in range(0, len(buf), self.batch_size)]
        # A partial group only exists in the final buffer; keep it LAST so it
        # can't shift the batch alignment of full groups (drop_last eats it).
        full = [g for g in groups if len(g) == self.batch_size]
        partial = [g for g in groups if len(g) < self.batch_size]
        rng.shuffle(full)
        for g in full + partial:
            yield from g


def _collate_fn(
    batch: list[dict[str, torch.Tensor]], pad_id: int, pad_to_multiple: int = 1
) -> dict[str, torch.Tensor]:
    """Pad to longest in batch (rounded up to pad_to_multiple).

    pad_to_multiple > 1 keeps shapes tensor-core aligned and drastically cuts
    the number of distinct shapes torch.compile sees (fewer recompiles).

    Packed rows (with position_ids): the padding gets positions 0, 1, 2, ..., i.e. it
    is one more document, attending only to itself, instead of one document per pad
    token. loss_weights (per-turn scaling) are padded with 0.
    """
    max_len = max(x["input_ids"].size(0) for x in batch)
    if pad_to_multiple > 1:
        max_len = ((max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    has_positions = "position_ids" in batch[0]
    has_weights = any("loss_weights" in x for x in batch)
    out: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
    if has_positions:
        out["position_ids"] = []
    if has_weights:
        out["loss_weights"] = []

    for item in batch:
        n = item["input_ids"].size(0)
        pad_len = max_len - n
        out["input_ids"].append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        out["attention_mask"].append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        out["labels"].append(torch.cat([item["labels"], torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)]))
        if has_positions:
            out["position_ids"].append(torch.cat([item["position_ids"], torch.arange(pad_len, dtype=torch.long)]))
        if has_weights:
            weights = item.get("loss_weights")
            if weights is None:
                weights = (item["labels"] != IGNORE_INDEX).float()
            out["loss_weights"].append(torch.cat([weights.float(), torch.zeros(pad_len)]))

    return {k: torch.stack(v) for k, v in out.items()}


# Keep backward-compatible name
collate_fn = _collate_fn


def build_dataset(
    dataset_or_config,
    tokenizer: PreTrainedTokenizerBase,
    config: DataConfig,
    rank: int,
    world_size: int,
    batch_size: int,
    streaming_shuffle_buffer: int = 0,
) -> IterableDataset:
    """Assemble the final training IterableDataset (everything the DataLoader wraps).

    Handles three cases:
    1. Pre-built dataset object (backward compat)
    2. Single dataset from config (config.dataset)
    3. Multiple datasets from config (config.sources)

    Also handles:
    4. Pretraining replay: auto-mixes generic data to prevent forgetting AND improve target task
       (arxiv:2603.04964, Stanford/Liang 2026)

    The returned stream yields per-sequence dicts (input_ids/attention_mask/labels,
    plus position_ids when packed) — i.e. tokenization, masking, mixing and packing
    are already applied.
    """
    # Determine the final IterableDataset
    if config.sources:
        # Multi-dataset mode: build each source and mix
        source_datasets = []
        weights = []
        names = []
        for src in config.sources:
            raw = _load_dataset_source(src["dataset"], src.get("split", "train"), streaming=config.streaming)
            # Shuffle happens inside the dataset AFTER per-worker sharding —
            # shuffle-then-shard crashes streaming workers (see _shard_then_shuffle).
            # Map-style datasets can be shuffled eagerly and use a different
            # shuffle() signature (no buffer_size).
            if not config.streaming:
                raw = raw.shuffle(seed=config.seed)
            shuffle_buffer = 10_000 if config.streaming else 0
            mode = src.get("mode", "sft")
            if mode == "sft":
                ds = ChatDataset(
                    raw,
                    tokenizer,
                    config.max_seq_length,
                    messages_field=src.get("messages_field", "messages"),
                    rank=rank,
                    world_size=world_size,
                    include_observations=config.include_observations,
                    turn_scaling=config.turn_scaling,
                    train_on_reasoning=getattr(config, "train_on_reasoning", True),
                    last_turn_only=src.get("last_turn_only", getattr(config, "last_turn_only", False)),
                    shuffle_buffer=shuffle_buffer,
                    shuffle_seed=config.seed,
                    tools_field=src.get("tools_field", config.tools_field),
                    think_tags=src.get("think_tags", getattr(config, "think_tags", None)),
                    chat_template_kwargs={**(getattr(config, "chat_template_kwargs", None) or {}),
                                          **(src.get("chat_template_kwargs") or {})},
                )
            elif mode == "pretrain":
                ds = PretrainDataset(
                    raw,
                    tokenizer,
                    config.max_seq_length,
                    text_field=src.get("text_field", "text"),
                    rank=rank,
                    world_size=world_size,
                    shuffle_buffer=shuffle_buffer,
                    shuffle_seed=config.seed,
                )
            else:
                raise ValueError(f"Unknown data mode: {mode}. Use 'sft' or 'pretrain'.")

            source_datasets.append(ds)
            weights.append(src.get("weight", 1.0))
            names.append(str(src.get("name", src.get("dataset", f"source[{len(names)}]"))))

        final_ds: IterableDataset = MixedDataset(source_datasets, weights, seed=config.seed, names=names)
    elif hasattr(dataset_or_config, "__iter__") and not isinstance(dataset_or_config, DataConfig):
        # Pre-loaded HF dataset object passed directly. The caller decides
        # whether to shuffle (streaming_shuffle_buffer > 0) — e.g. curriculum-
        # ordered prepared data must NOT be shuffled.
        raw = dataset_or_config
        final_ds = ChatDataset(
            raw,
            tokenizer,
            config.max_seq_length,
            config.messages_field,
            rank,
            world_size,
            include_observations=config.include_observations,
            turn_scaling=config.turn_scaling,
            train_on_reasoning=getattr(config, "train_on_reasoning", True),
            last_turn_only=getattr(config, "last_turn_only", False),
            shuffle_buffer=streaming_shuffle_buffer,
            shuffle_seed=config.seed,
            tools_field=config.tools_field,
            think_tags=getattr(config, "think_tags", None),
            chat_template_kwargs=getattr(config, "chat_template_kwargs", None),
        )
    else:
        # Single dataset from config
        raw = _load_dataset_source(config.dataset, config.dataset_split, config.streaming)
        final_ds = ChatDataset(
            raw,
            tokenizer,
            config.max_seq_length,
            config.messages_field,
            rank,
            world_size,
            include_observations=config.include_observations,
            turn_scaling=config.turn_scaling,
            train_on_reasoning=getattr(config, "train_on_reasoning", True),
            last_turn_only=getattr(config, "last_turn_only", False),
            shuffle_buffer=10_000 if config.streaming else 0,
            shuffle_seed=config.seed,
            tools_field=config.tools_field,
            think_tags=getattr(config, "think_tags", None),
            chat_template_kwargs=getattr(config, "chat_template_kwargs", None),
        )

    # ── Pretraining Replay (arxiv:2603.04964) ─────────────────────────────────
    # Surprising finding: mixing generic pretraining data during SFT improves
    # the TARGET task (not just prevents forgetting). The replay data acts as
    # implicit regularization that keeps the model in a good optimization basin.
    if config.pretrain_replay_dataset:
        replay_raw = _load_dataset_source(config.pretrain_replay_dataset, "train", streaming=True)
        replay_ds = PretrainDataset(
            replay_raw,
            tokenizer,
            config.max_seq_length,
            text_field="text",
            rank=rank,
            world_size=world_size,
            shuffle_buffer=10_000,
            shuffle_seed=config.seed,
        )
        # Mix: (1-w) * target_data + w * replay_data
        w = config.pretrain_replay_weight
        final_ds = MixedDataset(
            [final_ds, replay_ds],
            [1.0 - w, w],
            seed=config.seed,
            names=["target_mix", f"replay:{config.pretrain_replay_dataset} (text_field='text')"],
        )

    # Optional packing
    if config.packing:
        final_ds = PackedDataset(final_ds, config.max_seq_length, tokenizer.eos_token_id or 0, sort_buffer=256)
    elif batch_size > 1 and getattr(config, "length_group_buffer", 512) > 0:
        # No packing → pad-to-longest batches. Group similar lengths so the
        # padding (= wasted FLOPs) collapses to the within-group spread.
        final_ds = LengthGroupedDataset(
            final_ds, batch_size, buffer_size=config.length_group_buffer, seed=config.seed
        )
        logger.info(
            f"Length-grouped batching: buffer={config.length_group_buffer} "
            f"(cuts pad-token compute; set data.length_group_buffer: 0 to disable)"
        )

    return final_ds


def _dataloader_from_dataset(
    final_ds: IterableDataset, tokenizer: PreTrainedTokenizerBase, num_workers: int, batch_size: int
) -> DataLoader:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return DataLoader(
        final_ds,
        batch_size=batch_size,
        collate_fn=lambda b: collate_fn(b, pad_id, pad_to_multiple=64),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def build_dataloader(
    dataset_or_config,
    tokenizer: PreTrainedTokenizerBase,
    config: DataConfig,
    rank: int,
    world_size: int,
    batch_size: int,
    streaming_shuffle_buffer: int = 0,
) -> DataLoader:
    """Build the complete data pipeline (assemble the dataset, then wrap in a DataLoader)."""
    final_ds = build_dataset(
        dataset_or_config,
        tokenizer,
        config,
        rank,
        world_size,
        batch_size,
        streaming_shuffle_buffer=streaming_shuffle_buffer,
    )
    return _dataloader_from_dataset(final_ds, tokenizer, config.num_workers, batch_size)


# ══════════════════════════════════════════════════════════════════════════════
# PRE-TOKENIZED CACHE
# ══════════════════════════════════════════════════════════════════════════════
# Materialize the fully-assembled (tokenized → masked → mixed → packed) training
# stream to disk once, then on later runs load the tensors directly — skipping all
# per-step tokenization AND making the exact step count a cheap read. A fingerprint
# over every input that affects the tokens invalidates a stale cache automatically.
# Incompatible with dynamic-weight training (MSFT) and seq-len curriculum, which
# can't be baked into a static stream — those are rejected in Config.validate().

PRETOK_DATA = "train.parquet"
PRETOK_META = "pretokenized_meta.json"


def pretokenize_fingerprint(config, tokenizer) -> str:
    """Stable SHA-256 over everything that changes the materialized token stream."""
    import hashlib
    import json as _json

    d = config.data

    def _src_sig(src: dict) -> dict:
        p = Path(src.get("dataset", ""))
        stat = None
        try:
            if p.exists():
                st = p.stat()
                stat = [st.st_size, int(st.st_mtime)]
        except OSError:
            stat = None
        return {
            "dataset": src.get("dataset", ""),
            "split": src.get("split", "train"),
            "weight": src.get("weight", 1.0),
            "mode": src.get("mode", "sft"),
            "messages_field": src.get("messages_field", "messages"),
            "text_field": src.get("text_field", "text"),
            "last_turn_only": src.get("last_turn_only", getattr(d, "last_turn_only", False)),
            "think_tags": src.get("think_tags"),
            "chat_template_kwargs": src.get("chat_template_kwargs"),
            "stat": stat,
        }

    if d.sources:
        sources_sig = [_src_sig(s) for s in d.sources]
    else:
        sources_sig = [
            _src_sig({"dataset": d.dataset, "split": d.dataset_split, "messages_field": d.messages_field})
        ]

    payload = {
        "version": 1,
        "tokenizer": getattr(tokenizer, "name_or_path", ""),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "eos_token_id": tokenizer.eos_token_id,
        "max_seq_length": d.max_seq_length,
        "packing": d.packing,
        "length_group_buffer": getattr(d, "length_group_buffer", 0),
        "train_on_reasoning": getattr(d, "train_on_reasoning", True),
        "think_tags": getattr(d, "think_tags", None),
        "chat_template_kwargs": getattr(d, "chat_template_kwargs", None),
        "turn_scaling": getattr(d, "turn_scaling", "uniform"),
        "include_observations": getattr(d, "include_observations", False),
        "seed": d.seed,
        "replay": [d.pretrain_replay_dataset, d.pretrain_replay_weight],
        "sources": sources_sig,
    }
    blob = _json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def pretokenized_cache_valid(cache_dir, fingerprint: str) -> tuple[bool, str]:
    """(is_valid, reason). Valid iff the cache files exist and the fingerprint matches."""
    import json as _json

    meta = Path(cache_dir) / PRETOK_META
    data = Path(cache_dir) / PRETOK_DATA
    if not meta.exists() or not data.exists():
        return False, "no cache found"
    try:
        m = _json.loads(meta.read_text())
    except Exception:
        return False, "unreadable cache metadata"
    if m.get("fingerprint") != fingerprint:
        return False, "config/data/tokenizer changed since cache was built"
    return True, "valid"


def materialize_pretokenized(final_ds_factory, cache_dir, fingerprint: str, config, tokenizer) -> int:
    """Iterate the assembled (masked/mixed/packed) stream once and write tensors to parquet.

    Writes ``{cache_dir}/train.parquet`` (columns: input_ids, attention_mask, labels,
    and position_ids when packed) plus ``pretokenized_meta.json`` with the fingerprint
    and exact sequence count. Returns the number of sequences written.
    """
    import json as _json

    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (PRETOK_DATA + ".tmp")
    meta_path = out / PRETOK_META

    final_ds = final_ds_factory()
    writer: "pq.ParquetWriter | None" = None
    count = 0
    has_pos: bool | None = None
    buf: dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": [], "position_ids": []}

    def _flush():
        nonlocal writer
        if not buf["input_ids"]:
            return
        cols = {
            "input_ids": buf["input_ids"],
            "attention_mask": buf["attention_mask"],
            "labels": buf["labels"],
        }
        if has_pos:
            cols["position_ids"] = buf["position_ids"]
        table = pa.table(cols)
        if writer is None:
            writer = pq.ParquetWriter(str(tmp), table.schema)
        writer.write_table(table)
        for v in buf.values():
            v.clear()

    try:
        for ex in final_ds:
            if has_pos is None:
                has_pos = "position_ids" in ex
            buf["input_ids"].append(ex["input_ids"].tolist())
            buf["attention_mask"].append(ex["attention_mask"].tolist())
            buf["labels"].append(ex["labels"].tolist())
            if has_pos:
                buf["position_ids"].append(ex["position_ids"].tolist())
            count += 1
            if len(buf["input_ids"]) >= 1000:
                _flush()
        _flush()
    finally:
        if writer is not None:
            writer.close()

    if count == 0:
        raise ValueError(
            "Pre-tokenization produced 0 sequences — the data pipeline yielded nothing. "
            "Check the source paths/fields and masking settings."
        )

    tmp.replace(out / PRETOK_DATA)
    meta = {
        "fingerprint": fingerprint,
        "num_sequences": count,
        "max_seq_length": config.data.max_seq_length,
        "packing": config.data.packing,
        "has_position_ids": bool(has_pos),
    }
    meta_path.write_text(_json.dumps(meta, indent=2))
    return count


class PretokenizedDataset(IterableDataset):
    """Yields already-tokenized/packed tensors from a cached parquet, sharded by rank/worker.

    No tokenization, masking, mixing or packing — the cached rows are the final
    training sequences produced by an earlier ``materialize_pretokenized`` pass.
    """

    def __init__(self, dataset, rank: int = 0, world_size: int = 1):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        dataset = _shard_streaming_dataset(self.dataset, self.rank, self.world_size)
        for ex in dataset:
            out = {
                "input_ids": torch.as_tensor(ex["input_ids"], dtype=torch.long),
                "attention_mask": torch.as_tensor(ex["attention_mask"], dtype=torch.long),
                "labels": torch.as_tensor(ex["labels"], dtype=torch.long),
            }
            pos = ex.get("position_ids")
            if pos is not None:
                out["position_ids"] = torch.as_tensor(pos, dtype=torch.long)
            yield out


def build_pretokenized_dataloader(cache_dir, tokenizer, config: DataConfig, rank, world_size, batch_size) -> DataLoader:
    """Load the cached pre-tokenized parquet and wrap it in a DataLoader (no re-tokenization).

    Packed caches are all-max-length (no padding), so length grouping is a no-op there.
    For non-packed caches we re-apply length-grouped batching (honoring
    ``length_group_buffer``) so the cache doesn't lose the pad-token throughput win.
    """
    from datasets import load_dataset

    data_file = Path(cache_dir) / PRETOK_DATA
    ds = load_dataset("parquet", data_files=str(data_file), split="train", streaming=False)
    final_ds: IterableDataset = PretokenizedDataset(ds, rank=rank, world_size=world_size)

    if not config.packing and batch_size > 1 and getattr(config, "length_group_buffer", 512) > 0:
        final_ds = LengthGroupedDataset(
            final_ds, batch_size, buffer_size=config.length_group_buffer, seed=config.seed
        )

    return _dataloader_from_dataset(final_ds, tokenizer, config.num_workers, batch_size)
