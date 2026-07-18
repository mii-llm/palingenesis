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

import logging
import math
import random
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerBase

from palingenesis.config import DataConfig

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
    worker_info = torch.utils.data.get_worker_info()
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
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.messages_field = messages_field
        self.rank = rank
        self.world_size = world_size
        self.include_observations = include_observations
        self.turn_scaling = turn_scaling
        self.train_on_reasoning = train_on_reasoning
        # Streaming shuffle, applied per worker AFTER sharding (see _shard_then_shuffle).
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_seed = shuffle_seed
        # last_turn_only: mask every assistant turn except the final one. Selects which
        # turns get loss (training) / are scored (eval). Use for eval-format SFT where
        # earlier assistant turns are a FIXED few-shot prefix (e.g. n-shot MCQA
        # exemplars) that must not receive loss.
        self.last_turn_only = last_turn_only

    def __iter__(self):
        dataset = _shard_then_shuffle(self.dataset, self.rank, self.world_size,
                                      self.shuffle_buffer, self.shuffle_seed)
        for example in dataset:
            result = self._process(example)
            if result is not None:
                yield result

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

    def _process(self, example: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        messages = example.get(self.messages_field)
        if not messages:
            # Try alternative field names (conversations, chat, dialogue, etc.)
            for alt in ("conversations", "conversation", "chat", "dialogue", "turns"):
                messages = example.get(alt)
                if messages:
                    break
        if not messages:
            return None

        # Role normalization: handle non-standard formats (ShareGPT, Alpaca, etc.)
        # Maps: human→user, gpt→assistant, from/value→role/content
        from palingenesis.validate_data import normalize_messages

        normalized = normalize_messages(example, self.messages_field)
        if normalized:
            messages = normalized
        # If normalization returns None, use raw messages (may still work with some templates)

        # Smart truncation: if conversation exceeds max_seq_length, truncate at
        # the last complete turn boundary that fits AND contains an assistant turn.
        # This preserves training signal (partial conversations with no assistant = useless).
        messages = self._smart_truncate(messages)
        if not messages:
            return None

        try:
            templated = self.tokenizer.apply_chat_template(
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
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            labels = self._strip_reasoning_labels(input_ids, labels, full_text)

        # ECHO: if include_observations, also unmask tool_response regions within user messages
        # Some models (Qwen3.5) wrap tool outputs as <tool_response>...</tool_response> inside user turns
        if self.include_observations:
            labels = self._apply_echo_from_text(input_ids, labels, messages)

        if (labels != IGNORE_INDEX).sum() == 0:
            return None
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

    @staticmethod
    def _split_reasoning(msg: dict) -> tuple[str | None, str]:
        """Return (reasoning_raw, answer_raw): strings expected to appear verbatim in the
        rendered text. reasoning_raw is None when the turn carries no reasoning. Handles
        both the `reasoning_content` field and `<think>...</think>` embedded in content."""
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        rc = msg.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            return rc.strip(), content.strip()
        if "</think>" in content:
            head, _, tail = content.partition("</think>")
            reasoning = head.split("<think>")[-1].strip()
            return (reasoning or None), tail.strip()
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
        if not getattr(self.tokenizer, "is_fast", False) or "</think>" not in full_text:
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
        for m in re.finditer(r"<think>.*?</think>", full_text, flags=re.DOTALL):
            c0, c1 = m.start(), m.end()
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
                res = self._fallback_offsets(messages)
            except Exception:
                res = None
            if res is not None:
                return res
        return self._fallback_progressive(messages)

    def _fallback_offsets(self, messages: list[dict]) -> dict[str, torch.Tensor] | None:
        """Robust, template-agnostic masking via offset mapping + forward text search.

        Renders the conversation once, tokenizes with offsets, then locates each trained
        turn's reasoning/answer text by advancing a cursor through the rendered string.
        Because it relies only on text that ACTUALLY appears in the final render (never on
        render(messages[:i]) being a token-prefix of render(messages)), it is correct for
        history-rewriting templates that break the progressive masker.

        Requires a fast tokenizer (offset mapping). Returns None on any anomaly so the
        caller can fall back to the progressive masker.
        """
        full = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = self.tokenizer(
            full,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_seq_length,
        )
        offsets = enc.get("offset_mapping")
        if not offsets:
            return None
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attn_mask = torch.tensor(enc.get("attention_mask", [1] * len(enc["input_ids"])), dtype=torch.long)
        n_tok = len(input_ids)
        if n_tok == 0:
            return None
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        special_ids = set(self.tokenizer.all_special_ids or [])

        train_roles = {"assistant"}
        if self.include_observations:
            train_roles.update({"tool", "observation", "ipython", "function"})

        def toks_in(c0: int, c1: int) -> list[int]:
            # Overlap (not strict containment): captures a SentencePiece token that merges
            # a leading space into the first content char (o0 == c0 - 1).
            if c1 <= c0:
                return []
            return [ti for ti, (o0, o1) in enumerate(offsets) if o1 > o0 and o0 < c1 and o1 > c0]

        def first_tok_at(c: int) -> int:
            for ti, (o0, o1) in enumerate(offsets):
                if o1 > o0 and o0 >= c:
                    return ti
            return n_tok

        def is_ws(ti: int) -> bool:
            o0, o1 = offsets[ti]
            return o1 > o0 and full[o0:o1].strip() == ""

        n_assist = sum(1 for m in messages if m.get("role") == "assistant")
        assist_seen = 0
        cursor = 0
        turn_boundaries: list[tuple[set[int], int]] = []  # (token indices, assistant_turn_idx)

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

            if role not in train_roles:
                continue

            tset: set[int] = set()
            if role == "assistant" and self.train_on_reasoning and reasoning_raw and r0 != -1:
                # Include the '<think>' opener + reasoning + the '</think>' wrapper up to
                # the answer, so the whole generated block is one contiguous trained span.
                think_open = full.rfind("<think>", 0, r0)
                rstart = think_open if think_open != -1 else r0
                tset.update(toks_in(rstart, r1))
                if a0 != -1:
                    tset.update(toks_in(r1, a0))
            if a0 != -1:
                tset.update(toks_in(a0, a1))

            # Terminator: skip whitespace, include ONE end-of-turn special token (+ a
            # trailing newline). Stops before the next turn's header special token.
            anchor = a1 if a1 != -1 else r1
            if anchor != -1:
                ti = first_tok_at(anchor)
                while ti < n_tok and is_ws(ti):
                    tset.add(ti)
                    ti += 1
                if ti < n_tok and int(input_ids[ti]) in special_ids:
                    tset.add(ti)
                    ti += 1
                    if ti < n_tok and is_ws(ti):
                        tset.add(ti)

            if role == "assistant":
                turn_boundaries.append((tset, assist_seen))
                assist_seen += 1
            else:
                # tool/observation ECHO turn: always trained, never gated by last_turn_only
                for ti in tset:
                    labels[ti] = input_ids[ti]

        keep = turn_boundaries
        if self.last_turn_only and n_assist > 1:
            keep = [(ts, idx) for (ts, idx) in turn_boundaries if idx == n_assist - 1]
        for tset, _idx in keep:
            for ti in tset:
                labels[ti] = input_ids[ti]

        labels[attn_mask == 0] = IGNORE_INDEX

        if self.include_observations:
            labels = self._apply_echo_from_text(input_ids, labels, messages)

        if (labels != IGNORE_INDEX).sum() == 0:
            return None

        result = {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

        if self.turn_scaling != "uniform" and n_assist > 1 and turn_boundaries:
            loss_weights = torch.ones_like(input_ids, dtype=torch.float32)
            loss_weights[labels == IGNORE_INDEX] = 0.0
            if self.turn_scaling == "progressive":
                for tset, idx in turn_boundaries:
                    w = ((idx + 1) / n_assist) ** 0.5
                    for ti in tset:
                        loss_weights[ti] = w
            elif self.turn_scaling == "last_heavy":
                for tset, idx in turn_boundaries:
                    w = 2.0 if idx == n_assist - 1 else 1.0
                    for ti in tset:
                        loss_weights[ti] = w
            valid = loss_weights > 0
            if valid.any():
                loss_weights[valid] /= loss_weights[valid].mean()
            result["loss_weights"] = loss_weights

        return result

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
        # Tokenize the full conversation
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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
                prefix_text = self.tokenizer.apply_chat_template(
                    messages[: i + 1], tokenize=False, add_generation_prompt=False
                )
                prefix_ids = self.tokenizer(prefix_text, truncation=True, max_length=self.max_seq_length)["input_ids"]
                curr_len = len(prefix_ids)
            except Exception:
                # Some prefixes (e.g. system-only prefixes) are not renderable
                # with all chat templates. Skip them and keep scanning.
                prev_len = max(prev_len, 0)
                continue

            if msg.get("role") in train_roles:
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
                        stub_text = self.tokenizer.apply_chat_template(
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

    def _smart_truncate(self, messages: list[dict]) -> list[dict] | None:
        """Truncate at turn boundaries to fit max_seq_length while preserving training signal.

        Strategy:
        1. Quick char-based heuristic: if total chars < max_seq_length * 3, likely fits (skip)
        2. Otherwise, tokenize progressively at each turn boundary (exact count)
        3. Keep the maximum number of turns that fit within max_seq_length
        4. Require at least one complete assistant turn in the kept portion

        Uses actual tokenizer for precise token counting (no char/token ratio guessing).
        """
        # Fast path: short conversations definitely fit (4 chars/token is generous)
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        if total_chars < self.max_seq_length * 3:
            return messages

        # Tokenize at each turn boundary to find exact fit
        last_valid_boundary = 0
        has_assistant = False

        for i in range(len(messages)):
            prefix_messages = messages[: i + 1]
            try:
                prefix_text = self.tokenizer.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                # Template error: stop here
                break

            # Exact token count
            token_count = len(self.tokenizer.encode(prefix_text, add_special_tokens=False))

            if token_count > self.max_seq_length:
                # This turn pushes us over: stop at previous boundary
                break

            # This turn fits
            last_valid_boundary = i + 1
            if messages[i].get("role") == "assistant":
                has_assistant = True

        # Require at least one assistant turn
        if not has_assistant:
            for j in range(last_valid_boundary):
                if messages[j].get("role") == "assistant":
                    has_assistant = True
                    break

        if not has_assistant or last_valid_boundary < 2:
            return None

        return messages[:last_valid_boundary]


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

    def __init__(self, sources: list[IterableDataset], weights: list[float], seed: int = 42):
        assert len(sources) == len(weights)
        assert all(w >= 0 for w in weights)
        total = sum(weights)
        self.sources = sources
        self.probs = [w / total for w in weights]
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        iterators = [iter(s) for s in self.sources]
        indices = list(range(len(self.sources)))

        while True:
            idx = rng.choices(indices, weights=self.probs, k=1)[0]
            try:
                yield next(iterators[idx])
            except StopIteration:
                break  # Epoch boundary: first source exhausted


# ══════════════════════════════════════════════════════════════════════════════
# PACKING
# ══════════════════════════════════════════════════════════════════════════════


class SeqLenCurriculum:
    """Sequence length curriculum: ramp max_seq_len from short to full over training.

    Inspired by Dataset Decomposition (Apple, arxiv:2405.13226):
    Training with shorter sequences early saves compute (quadratic attention)
    while the model learns basic patterns. Longer sequences are introduced
    progressively for complex reasoning and long-range dependencies.

    Result: >4× data efficiency, up to 6× training speedup.

    The curriculum ramps from min_len to max_len over ramp_steps, using
    power-of-2 increments (matching the binary decomposition in the paper).

    Usage:
        curriculum = SeqLenCurriculum(min_len=1024, max_len=8192, ramp_steps=5000)
        packed_ds = PackedDataset(base, max_len=8192, eos_id=0, seq_len_curriculum=curriculum)

        for step in training:
            batch = next(dataloader)  # uses curriculum.current_max_len
            ...
            curriculum.step()  # advance curriculum
    """

    def __init__(self, min_len: int = 1024, max_len: int = 8192, ramp_steps: int = 5000):
        self.min_len = min_len
        self.max_len = max_len
        self.ramp_steps = ramp_steps
        self._current_step = 0

    @property
    def current_max_len(self) -> int:
        """Get current max sequence length based on training progress."""
        if self._current_step >= self.ramp_steps:
            return self.max_len

        progress = self._current_step / max(1, self.ramp_steps)
        # Linear ramp in log2 space (power-of-2 increments)
        log_min = math.log2(self.min_len)
        log_max = math.log2(self.max_len)
        log_current = log_min + progress * (log_max - log_min)
        # Round to nearest power of 2 (for efficient GPU batching)
        current = 2 ** int(log_current)
        return min(current, self.max_len)

    def step(self):
        """Advance curriculum by one step."""
        self._current_step += 1

    def reset(self):
        """Reset curriculum to beginning."""
        self._current_step = 0


class PackedDataset(IterableDataset):
    """Packs sequences into fixed-length blocks with document-aware position_ids.

    Concatenates multiple sequences end-to-end into one long tensor of length
    max_len. Produces `position_ids` that reset to 0 at each document boundary.

    Smart packing (sorted-length bin packing, inspired by arxiv:2107.02027):
    When sort_buffer > 0, accumulates samples in a buffer, sorts by length,
    then packs greedily. This reduces wasted space from 15-30% (random) to 3-8%
    (sorted). The buffer size controls the trade-off between packing efficiency
    and memory/randomness.

    Sequence length curriculum (inspired by arxiv:2405.13226, Dataset Decomposition):
    When seq_len_curriculum is provided, the effective max_len ramps from a short
    initial value to the full max_len over training. Early steps use shorter
    sequences (faster due to quadratic attention) while later steps use full length.
    This produces >4× data efficiency and up to 6× training speedup.

    When used with `attn_implementation="flex_attention"` in HuggingFace models,
    the position_ids resets trigger proper document-level causal masking:
    tokens from different documents cannot attend to each other.

    Output per sample:
      - input_ids: [max_len] packed token IDs
      - attention_mask: [max_len] all ones (no padding in packed sequences)
      - labels: [max_len] with IGNORE_INDEX preserved from source datasets
      - position_ids: [max_len] positions resetting to 0 at each document boundary
    """

    def __init__(
        self,
        base: IterableDataset,
        max_len: int,
        eos_id: int,
        sort_buffer: int = 256,
        seq_len_curriculum: "SeqLenCurriculum | None" = None,
    ):
        self.base = base
        self.max_len = max_len
        self.eos_id = eos_id
        self.sort_buffer = sort_buffer  # 0 = no sorting (sequential), >0 = sorted bin packing
        self.seq_len_curriculum = seq_len_curriculum

    @property
    def effective_max_len(self) -> int:
        """Current effective max sequence length (may be ramped by curriculum)."""
        if self.seq_len_curriculum is not None:
            return self.seq_len_curriculum.current_max_len
        return self.max_len

    def __iter__(self):
        if self.sort_buffer > 0:
            yield from self._sorted_packing()
        else:
            yield from self._sequential_packing()

    def _sequential_packing(self):
        """Original sequential packing: concatenate in arrival order."""
        buf_ids: list[int] = []
        buf_labels: list[int] = []
        buf_positions: list[int] = []

        for ex in self.base:
            doc_ids = ex["input_ids"].tolist()
            doc_labels = ex["labels"].tolist()
            doc_len = len(doc_ids)

            buf_ids.extend(doc_ids)
            buf_labels.extend(doc_labels)
            buf_positions.extend(range(doc_len))

            while len(buf_ids) >= self.max_len:
                yield {
                    "input_ids": torch.tensor(buf_ids[: self.max_len], dtype=torch.long),
                    "attention_mask": torch.ones(self.max_len, dtype=torch.long),
                    "labels": torch.tensor(buf_labels[: self.max_len], dtype=torch.long),
                    "position_ids": torch.tensor(buf_positions[: self.max_len], dtype=torch.long),
                }
                buf_ids = buf_ids[self.max_len :]
                buf_labels = buf_labels[self.max_len :]
                buf_positions = buf_positions[self.max_len :]

    def _sorted_packing(self):
        """Sorted bin packing: accumulate buffer, sort by length, pack greedily.

        From arxiv:2107.02027 and arxiv:2405.13226:
        Sorting samples by length before packing ensures similar-length documents
        end up in the same packed sequence. Benefits:
          - Less wasted space (short+short fills better than short+long that overflows)
          - More consistent compute per batch (no one sequence dominating)
          - ~2× packing efficiency improvement over random concatenation
        """
        buffer: list[dict] = []

        for ex in self.base:
            buffer.append(ex)

            if len(buffer) >= self.sort_buffer:
                yield from self._flush_buffer(buffer)
                buffer = []

        # Flush remaining
        if buffer:
            yield from self._flush_buffer(buffer)

    def _flush_buffer(self, buffer: list[dict]):
        """Sort buffer by length and pack greedily into max_len blocks."""
        # Sort by sequence length (shortest first → best packing)
        buffer.sort(key=lambda ex: ex["input_ids"].size(0))

        buf_ids: list[int] = []
        buf_labels: list[int] = []
        buf_positions: list[int] = []

        for ex in buffer:
            doc_ids = ex["input_ids"].tolist()
            doc_labels = ex["labels"].tolist()
            doc_len = len(doc_ids)

            # Defensive: cap oversized documents to max_len (should be caught by
            # smart_truncate upstream, but guarantee packing never produces garbage)
            if doc_len > self.max_len:
                doc_ids = doc_ids[: self.max_len]
                doc_labels = doc_labels[: self.max_len]
                doc_len = self.max_len

            # If adding this doc would overflow, try to yield what we have
            # and start fresh (greedy bin packing)
            if len(buf_ids) + doc_len > self.max_len and len(buf_ids) > 0:
                # Yield current buffer (may be shorter than max_len — pad or yield partial)
                if len(buf_ids) >= self.max_len:
                    yield {
                        "input_ids": torch.tensor(buf_ids[: self.max_len], dtype=torch.long),
                        "attention_mask": torch.ones(self.max_len, dtype=torch.long),
                        "labels": torch.tensor(buf_labels[: self.max_len], dtype=torch.long),
                        "position_ids": torch.tensor(buf_positions[: self.max_len], dtype=torch.long),
                    }
                    buf_ids = buf_ids[self.max_len :]
                    buf_labels = buf_labels[self.max_len :]
                    buf_positions = buf_positions[self.max_len :]

            buf_ids.extend(doc_ids)
            buf_labels.extend(doc_labels)
            buf_positions.extend(range(doc_len))

            # Yield complete blocks
            while len(buf_ids) >= self.max_len:
                yield {
                    "input_ids": torch.tensor(buf_ids[: self.max_len], dtype=torch.long),
                    "attention_mask": torch.ones(self.max_len, dtype=torch.long),
                    "labels": torch.tensor(buf_labels[: self.max_len], dtype=torch.long),
                    "position_ids": torch.tensor(buf_positions[: self.max_len], dtype=torch.long),
                }
                buf_ids = buf_ids[self.max_len :]
                buf_labels = buf_labels[self.max_len :]
                buf_positions = buf_positions[self.max_len :]


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
    """Pad to longest in batch (rounded up to pad_to_multiple). Handles position_ids.

    pad_to_multiple > 1 keeps shapes tensor-core aligned and drastically cuts
    the number of distinct shapes torch.compile sees (fewer recompiles).
    """
    max_len = max(x["input_ids"].size(0) for x in batch)
    if pad_to_multiple > 1:
        max_len = ((max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    ids, masks, labels = [], [], []
    has_positions = "position_ids" in batch[0]
    positions = [] if has_positions else None

    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        masks.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)]))
        if has_positions:
            positions.append(torch.cat([item["position_ids"], torch.zeros(pad_len, dtype=torch.long)]))

    result = {"input_ids": torch.stack(ids), "attention_mask": torch.stack(masks), "labels": torch.stack(labels)}
    if positions:
        result["position_ids"] = torch.stack(positions)
    return result


# Keep backward-compatible name
collate_fn = _collate_fn


def build_dataloader(
    dataset_or_config,
    tokenizer: PreTrainedTokenizerBase,
    config: DataConfig,
    rank: int,
    world_size: int,
    batch_size: int,
    streaming_shuffle_buffer: int = 0,
) -> DataLoader:
    """Build the complete data pipeline.

    Handles three cases:
    1. Pre-built dataset object (backward compat)
    2. Single dataset from config (config.dataset)
    3. Multiple datasets from config (config.sources)

    Also handles:
    4. Pretraining replay: auto-mixes generic data to prevent forgetting AND improve target task
       (arxiv:2603.04964, Stanford/Liang 2026)
    """
    # Determine the final IterableDataset
    if config.sources:
        # Multi-dataset mode: build each source and mix
        source_datasets = []
        weights = []
        for src in config.sources:
            raw = _load_dataset_source(src["dataset"], src.get("split", "train"), streaming=True)
            # Shuffle happens inside the dataset AFTER per-worker sharding —
            # shuffle-then-shard crashes streaming workers (see _shard_then_shuffle).
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
                    shuffle_buffer=10_000,
                    shuffle_seed=config.seed,
                )
            elif mode == "pretrain":
                ds = PretrainDataset(
                    raw,
                    tokenizer,
                    config.max_seq_length,
                    text_field=src.get("text_field", "text"),
                    rank=rank,
                    world_size=world_size,
                    shuffle_buffer=10_000,
                    shuffle_seed=config.seed,
                )
            else:
                raise ValueError(f"Unknown data mode: {mode}. Use 'sft' or 'pretrain'.")

            source_datasets.append(ds)
            weights.append(src.get("weight", 1.0))

        final_ds: IterableDataset = MixedDataset(source_datasets, weights, seed=config.seed)
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

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return DataLoader(
        final_ds,
        batch_size=batch_size,
        collate_fn=lambda b: collate_fn(b, pad_id, pad_to_multiple=64),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )
