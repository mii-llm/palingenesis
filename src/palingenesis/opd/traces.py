"""On-policy distillation on recorded agent traces, one assistant turn at a time.

A trace (system, user, assistant, tool, assistant, tool, ...) gives one training
context per assistant turn: everything before it, tool outputs included. The
student regenerates the turn from there and the teacher scores what it wrote. No
environment or tool execution is needed (the tools' outputs are the recorded
ones), and one trace yields as many on-policy samples as it has turns.

Every context is rendered with the student's chat template exactly as it would
be at inference, and the student generates from those exact token ids. Within
one agentic loop the templates keep the earlier turns' reasoning (Qwen3.5 and
MiniMax-M2.x after the last user query, GLM-5.x everywhere by default), so turn
k's context is a prefix of turn k+1's: the trace is one TRUNK with the turns
hanging off it. A template that rewrites history (reasoning dropped once a new
user message arrives) only moves where a turn's context leaves the trunk: each
branch attaches at its longest common token prefix with the trunk and carries
the rest of its context itself. Either way, every model reads the shared context
once (palingenesis.seco_tree), not once per turn:

  student rollouts   vLLM with prefix caching: the trunk is prefilled once
  teacher            seco_tree.tree_hidden_states: trunk once, then each branch
  student training   seco_tree.tree_forward_backward: exact gradients through
                     the trunk, activation memory of one chunk

The trunk also holds the recorded turns themselves. The teacher's hidden states
there come from the pass it makes anyway, so `loss.trace_kd_weight` adds
distillation on those recorded turns at the cost of the output-head projections
only (off-policy, next to the on-policy branches).
"""

import bisect
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from palingenesis.validate_data import THINK_TAGS, normalize_messages, normalize_tools, restore_baked_think

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ planning


@dataclass
class BranchPlan:
    turn: int  # index of the assistant message the student regenerates
    context: list[int]  # the full context the student generates from
    attach: int  # trunk position the branch continues from
    prefix: list[int]  # branch tokens before the completion (the context's tail, or its last token)


@dataclass
class TracePlan:
    trunk: list[int]
    branches: list[BranchPlan]
    kd_spans: list[tuple[int, int]] = field(default_factory=list)  # trunk token ranges of recorded turns

    def kd_positions(self) -> list[int]:
        """Trunk positions whose hidden state predicts a recorded-turn token (p predicts trunk[p + 1])."""
        return [p for a, b in self.kd_spans for p in range(a - 1, b - 1)]


def is_trainable_turn(message: dict) -> bool:
    return (
        message.get("role") == "assistant"
        and message.get("loss", True) is not False
        and bool(message.get("content") or message.get("reasoning_content") or message.get("tool_calls"))
    )


class TracePlanner:
    """Renders a trace's contexts with the student's chat template and lays them out as a tree."""

    def __init__(
        self,
        tok,
        chat_template_kwargs: dict,
        stop_ids: tuple[int, ...],
        max_context: int,
        branches_per_trace: int,
        recorded_kd: bool,
    ):
        self.tok = tok
        self.kwargs = chat_template_kwargs
        self.stop_ids = set(stop_ids)
        self.max_context = max_context
        self.branches_per_trace = branches_per_trace
        self.recorded_kd = recorded_kd
        probe = [{"role": "user", "content": "x"}]
        without = tok.apply_chat_template(probe, tokenize=False, **chat_template_kwargs)
        with_prompt = tok.apply_chat_template(probe, add_generation_prompt=True, tokenize=False, **chat_template_kwargs)
        # The text that opens an assistant turn at generation (e.g. "<|im_start|>assistant\n<think>\n"),
        # and its part that every rendered assistant turn starts with, reasoning kept or not (the
        # header, "<|im_start|>assistant\n"): templates drop earlier turns' reasoning once a new user
        # query arrives, and those turns then render without the opening's think block.
        self.turn_opening = with_prompt[len(without) :] if with_prompt.startswith(without) else ""
        history = tok.apply_chat_template(
            probe + [{"role": "assistant", "content": "y"}, {"role": "user", "content": "z"}],
            tokenize=False,
            **chat_template_kwargs,
        )
        rewritten = history[len(without) :] if history.startswith(without) else ""
        self.turn_header = os.path.commonprefix([self.turn_opening, rewritten]) or self.turn_opening
        self.bos = tok.bos_token_id
        from palingenesis.data import detect_think_tags, renders_reasoning

        def probe_render(m, **kw):
            return tok.apply_chat_template(m, tokenize=False, **{**chat_template_kwargs, **kw})

        # a template that ignores the reasoning field keeps baked <think> blocks in the content
        self.renders_reasoning = renders_reasoning(probe_render)
        self.think_tags = detect_think_tags(probe_render) or THINK_TAGS

    def render(self, messages: list[dict], tools: list[dict] | None, turn: int) -> str:
        return self.tok.apply_chat_template(
            messages[:turn], tools=tools or None, add_generation_prompt=True, tokenize=False, **self.kwargs
        )

    def _encode(self, text: str) -> tuple[list[int], list[int]]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, ends = list(enc["input_ids"]), [e for _, e in enc["offset_mapping"]]
        if self.bos is not None and (not ids or ids[0] != self.bos):
            ids, ends = [self.bos] + ids, [0] + ends
        return ids, ends

    def plan(self, messages: list[dict], tools: list[dict] | None, rng: random.Random) -> TracePlan | None:
        if not self.renders_reasoning:
            messages = restore_baked_think(messages)
        turns = [k for k, m in enumerate(messages) if is_trainable_turn(m)]
        if not turns:
            return None
        texts: dict[int, str] = {}

        def text(k: int) -> str:
            if k not in texts:
                texts[k] = self.render(messages, tools, k)
            return texts[k]

        # Contexts grow with the turn index: binary-search the last turn whose context may fit,
        # in characters (a cheap prefilter: 8 per token is far above any tokenizer's average),
        # then check the chosen trunk in tokens.
        lo, hi = 0, len(turns)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(text(turns[mid])) <= self.max_context * 8:
                lo = mid + 1
            else:
                hi = mid
        pool = turns[:lo]
        while pool:
            count = len(pool) if self.branches_per_trace <= 0 else min(self.branches_per_trace, len(pool))
            chosen = sorted(rng.sample(pool, count))
            # The trunk: the longest context that fits (every branch attaches at its common prefix with
            # it), so it also holds as many recorded turns as can be distilled, the chosen ones included.
            candidates = [k for k in pool if k >= chosen[-1]] if self.recorded_kd else [chosen[-1]]
            trunk = None
            lo_c, hi_c = 0, len(candidates)  # contexts grow with the turn: binary search
            while lo_c < hi_c:
                mid = (lo_c + hi_c) // 2
                ids, offsets = self._encode(text(candidates[mid]))
                if len(ids) <= self.max_context:
                    trunk_text, trunk, ends, trunk_turn = text(candidates[mid]), ids, offsets, candidates[mid]
                    lo_c = mid + 1
                else:
                    hi_c = mid
            if trunk is not None:
                break
            pool = [k for k in pool if k < chosen[-1]]  # the longest one does not fit: drop it and retry
        else:
            return None
        branches = []
        for k in chosen:
            s = text(k)
            common = len(os.path.commonprefix([s, trunk_text]))
            t = bisect.bisect_right(ends, common)  # trunk tokens entirely inside the common prefix
            cut = ends[t - 1] if t else 0
            tail = self.tok.encode(s[cut:], add_special_tokens=False) if cut < len(s) else []
            if t == 0 and not tail:
                continue
            context = trunk[:t] + tail
            if len(context) > self.max_context:
                continue
            if tail:
                branches.append(BranchPlan(k, context, t, tail))
            else:  # re-feed the last context token: its output predicts
                branches.append(BranchPlan(k, context, t - 1, [trunk[t - 1]]))  # the first completion token
        if not branches:
            return None
        plan = TracePlan(trunk, branches)
        if self.recorded_kd:
            plan.kd_spans = self._recorded_turns(messages, trunk_turn, trunk_text, trunk, ends)
        return plan

    def _recorded_turns(
        self, messages: list[dict], last_turn: int, trunk_text: str, trunk: list[int], ends: list[int]
    ) -> list[tuple[int, int]]:
        """Token ranges [a, b) of the recorded assistant turns inside the trunk, each through its
        end-of-turn token. The trunk renders messages[:last_turn] and then opens the next turn, so
        the template's turn headers in it number one more than those assistant messages, in
        order; anything else (the header text inside some content, say) gives no ranges."""
        if not self.turn_header:
            return []
        assistants = [m for m in messages[:last_turn] if m.get("role") == "assistant"]
        starts, pos = [], trunk_text.find(self.turn_header)
        while pos >= 0:  # a turn's text starts after the full opening where it has one, else after the header
            opening = self.turn_opening if trunk_text.startswith(self.turn_opening, pos) else self.turn_header
            starts.append(pos + len(opening))
            pos = trunk_text.find(self.turn_header, pos + 1)
        if len(starts) != len(assistants) + 1:
            return []
        spans = []
        for message, char_start in zip(assistants, starts):
            if not is_trainable_turn(message):
                continue
            # From the first token not entirely before the turn's text: where no token boundary falls
            # there (the opening's last newline merged with the text's), that token straddles it.
            a = bisect.bisect_right(ends, char_start)
            if a == 0:
                continue
            b = next((i + 1 for i in range(a, len(trunk)) if trunk[i] in self.stop_ids), None)
            if b is not None:
                spans.append((a, b))
        return spans


# -------------------------------------------------------------------- data


def load_trace_rows(
    path: str, messages_field: str = "messages", tools_field: str = "tools", think_tags: tuple[str, str] | None = None
) -> list[dict]:
    """Agent traces from JSONL or parquet: rows with messages (list or JSON string) and
    optional tools; the other columns (topic, id, ...) are kept. `think_tags` delimit
    reasoning baked into assistant content (default <think></think>)."""
    if path.endswith(".parquet"):
        import pandas as pd

        records = pd.read_parquet(path).to_dict("records")
    else:
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
    rows, skipped = [], 0
    for record in records:
        messages = normalize_messages(record, messages_field, think_tags=think_tags)
        tools = record.get(tools_field)
        if isinstance(tools, str):
            tools = json.loads(tools) if tools.strip() else None
        tools = normalize_tools(tools) if tools else None
        if not messages or not any(is_trainable_turn(m) for m in messages):
            skipped += 1
            continue
        rows.append(
            {
                **{k: v for k, v in record.items() if k not in (messages_field, tools_field)},
                "messages": messages,
                "tools": tools,
            }
        )
    if skipped:
        logger.warning("%s: skipped %d rows without a trainable assistant turn", path, skipped)
    if not rows:
        raise ValueError(f"{path}: no usable agent traces")
    return rows


@dataclass
class TraceSample:
    """One trace's rollouts, ready for the teacher and the student."""

    plan: TracePlan
    completions: list[list[int]]  # per branch (cleaned: through the end-of-turn token)
    behaviour_lp: list[list[float]]
    finish: list[str]
    teacher: str
    meta: dict[str, Any]
    teacher_hidden: list[Tensor] | None = None  # per branch [len(completion), H]
    teacher_kd_hidden: Tensor | None = None  # [len(plan.kd_positions()), H]


def branch_inputs(plan: TracePlan, completions: list[list[int]]) -> list[list[int]]:
    """What each branch feeds the model: its prefix, then the completion but its last token."""
    return [b.prefix + c[:-1] for b, c in zip(plan.branches, completions)]


def completion_rows(plan: TracePlan, completions: list[list[int]]) -> list[slice]:
    """Rows of each branch's hidden states that predict its completion tokens."""
    return [slice(len(b.prefix) - 1, len(b.prefix) - 1 + len(c)) for b, c in zip(plan.branches, completions)]


def select(plan: TracePlan, keep: list[int]) -> TracePlan:
    return TracePlan(plan.trunk, [plan.branches[i] for i in keep], plan.kd_spans)


def topic_teacher(row: dict, topic_field: str, topic_teachers: dict[str, list], default: str) -> str:
    """The teacher of a row's topic (`topic_teachers`: teacher -> topics), else `default`."""
    if topic_field:
        topic = row.get(topic_field)
        for teacher, topics in topic_teachers.items():
            if topic in topics:
                return teacher
    return default


def hidden_rows(hidden: Tensor, rows: slice) -> Tensor:
    return hidden[0, rows] if hidden.dim() == 3 else hidden[rows]


def stack_positions(positions: list[int], device) -> Tensor:
    return torch.tensor(positions, dtype=torch.long, device=device)
