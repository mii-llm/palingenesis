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

Traces recorded from the student itself (its own episodes in the environment)
put its own states into training, the failure states included: repetition,
rumination, turns that never end. `OnsetScan` finds where a recorded turn starts
to repeat earlier assistant text (a form-agnostic measure on token ids: no
phrase, call or format is named) and adds branches there: the student
regenerates from the token where it began to repeat itself, and the teacher
scores what it writes. Distillation on recorded turns can stop a few tokens
into each such run: deeper in, a teacher reading the repeated text agrees with
it, and those tokens would only dilute the loss.

Two more knobs serve such traces, both off by default:

  FailedTurns     turns whose tool call failed (an observation matching the
                  source's error patterns, a per-message error field, a call to
                  an unknown tool or with arguments that are not an object) get
                  extra loss weight, the first one of a failed episode its own,
                  and may get extra regenerated turns (SAGE-OPD 2606.19659 gives
                  deterministic failures full weight; STRIDE 2609.14636 finds the
                  teacher's endorsement drops at the first wrong action). Weights
                  are renormalized over the batch, so the loss keeps its scale.
  TeacherCutoff   stops supervising a trace where the teacher stops endorsing
                  the recorded student turns: the per-turn mean of the teacher's
                  log-prob of the recorded tokens, summed from the first turn
                  (STRIDE's statistic) or averaged over the last few turns, falls
                  below a threshold. Turns after that one, recorded or
                  regenerated, are dropped or down-weighted: there the teacher
                  reads a history it would never have written.
"""

import bisect
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field, replace
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
    onset: bool = False  # starts inside a recorded turn, where it began to repeat itself (OnsetScan)
    weight: float = 1.0  # loss weight of its tokens (FailedTurns, TeacherCutoff)
    error: bool = False  # regenerates a turn whose recorded tool call failed (FailedTurns)


@dataclass
class TracePlan:
    trunk: list[int]
    branches: list[BranchPlan]
    # trunk token ranges of the recorded turns distilled (OnsetScan.kd_tokens cuts the repetitive runs out of them)
    kd_spans: list[tuple[int, int]] = field(default_factory=list)
    # (message index, a, b): every recorded assistant turn inside the trunk, filled when FailedTurns or
    # TeacherCutoff need them; turn_weights: message index -> loss weight (absent = 1)
    turns: list[tuple[int, int, int]] = field(default_factory=list)
    turn_weights: dict[int, float] = field(default_factory=dict)

    def kd_positions(self) -> list[int]:
        """Trunk positions whose hidden state predicts a recorded-turn token (p predicts trunk[p + 1])."""
        return [p for a, b in self.kd_spans for p in range(a - 1, b - 1)]

    def kd_weights(self) -> list[float]:
        """The loss weight of each kd_positions() entry: the weight of the turn its predicted token belongs to."""
        if not self.turn_weights:
            return [1.0] * len(self.kd_positions())
        starts = [a for _, a, _ in self.turns]
        out = []
        for a, b in self.kd_spans:
            i = bisect.bisect_right(starts, a) - 1
            k, _, end = self.turns[i] if i >= 0 else (None, 0, 0)
            w = self.turn_weights.get(k, 1.0) if k is not None and a < end else 1.0
            out.extend([w] * (b - a))
        return out


@dataclass(frozen=True)
class OnsetScan:
    """Where recorded assistant turns start to repeat themselves, and what the planner does there.

    A token is a repeat when the `ngram` tokens ending at it already occurred in assistant text before (this turn's
    or an earlier turn's; tool outputs and prompts do not count, so quoting an observation is not a repeat). N-grams
    written by more than `common` of the earlier assistant turns are the turns' shared scaffolding (call markup,
    habitual openers) and never count (with the default .5, a turn copying the one turn before it is caught from the
    third turn on). A run starts where at least `threshold` of the last `window` tokens
    are repeats, at the first repeated n-gram of that window, and ends where the share falls below it. This measures
    the student's own text and names no form: exact loops, sub-word runs, counting enumerations, near-paraphrases that
    keep reusing their phrases, and turns that copy earlier turns all qualify; legitimate copies trigger it too, which
    costs only compute (the teacher agrees with a good copy, so its KL there is small).

    branches: regenerated turns per trace starting at the earliest runs (0 = none): the student continues from the
    token before the run, the teacher scores its continuation. kd_tokens: recorded-turn distillation keeps this many
    tokens of each run and drops the rest (-1 = keep all).
    """

    branches: int = 0
    kd_tokens: int = -1
    ngram: int = 6
    window: int = 32
    threshold: float = 0.75
    common: float = 0.5

    @property
    def active(self) -> bool:
        return self.branches > 0 or self.kd_tokens >= 0


def repetition_runs(
    ids: list[int],
    spans: list[tuple[int, int]],
    ngram: int = 6,
    window: int = 32,
    threshold: float = 0.75,
    common: float = 0.5,
) -> list[tuple[int, int]]:
    """Token ranges [start, end) inside `spans` (the assistant turns, in order) where the text repeats earlier
    assistant text; see OnsetScan. A run never crosses a turn boundary; a turn that repeats an earlier one starts a
    run at (or near) its first token."""
    turns_with: dict[tuple[int, ...], int] = {}  # n-gram -> assistant turns it occurred in so far
    last_turn: dict[tuple[int, ...], int] = {}
    runs: list[tuple[int, int]] = []
    need = threshold * window
    for turn, (a, b) in enumerate(spans):
        hits: list[bool] = []
        for t in range(a, b):
            if t - a + 1 < ngram:
                hits.append(False)
                continue
            g = tuple(ids[t - ngram + 1 : t + 1])
            here = last_turn.get(g) == turn
            earlier = turns_with.get(g, 0) - here  # earlier turns that wrote it
            hits.append((here or earlier > 0) and earlier <= common * turn)
            if not here:
                last_turn[g] = turn
                turns_with[g] = earlier + 1
        count, start = 0, None
        for i, hit in enumerate(hits):
            count += hit
            if i >= window:
                count -= hits[i - window]
            if start is None and count >= need:
                first = next(j for j in range(max(0, i - window + 1), i + 1) if hits[j])
                start = a + max(first - ngram + 1, 0)
                if runs and runs[-1][1] > start:  # the window reaches back into the previous run of this turn
                    start = runs[-1][1]
            elif start is not None and count < need:
                runs.append((start, a + i))
                start = None
        if start is not None:
            runs.append((start, b))
    return runs


def cut_runs(spans: list[tuple[int, int]], runs: list[tuple[int, int]], keep: int) -> list[tuple[int, int]]:
    """`spans` without the tokens of each run past its first `keep`."""
    out = []
    for a, b in spans:
        pieces = [(a, b)]
        for s, e in runs:
            lo, hi = s + keep, e
            if lo >= hi:
                continue
            pieces = [p for x, y in pieces for p in ((x, min(y, lo)), (max(x, hi), y)) if p[0] < p[1]]
        out.extend(pieces)
    return out


@dataclass(frozen=True)
class FailedTurns:
    """Which recorded assistant turns made a tool call that failed, and what the planner does with them.

    Read from the data, never from the rendered text, so any harness and template qualify. A turn failed when
    - an observation answering it (the messages after it up to the next assistant turn whose role is in `roles`,
      tool results by default) matches one of `patterns` (regular expressions, re.search), e.g. "^Error:",
      "command not found", "Traceback \\(most recent";
    - it or one of those observations carries a truthy `field` (a per-message error record), or
    - with `invalid_calls`, it calls a tool the trace does not declare, or with arguments that are not a JSON object.

    weight: loss weight of a failed turn's tokens, regenerated and recorded. first_weight: the first failed turn's,
    in episodes that failed (`success_field` falsy or <= 0; every episode when empty): the first wrong action,
    approximated by the first failed call. branches: extra regenerated turns per trace at failed turns, the first
    one first. Weights are relative: the trainer normalizes the loss by the weighted token count.
    """

    patterns: tuple[str, ...] = ()
    roles: tuple[str, ...] = ("tool",)
    field: str = ""
    invalid_calls: bool = False
    weight: float = 1.0
    first_weight: float = 1.0
    success_field: str = ""
    branches: int = 0

    @property
    def detects(self) -> bool:
        return bool(self.patterns or self.field or self.invalid_calls)

    @property
    def active(self) -> bool:
        return self.detects and (self.weight != 1.0 or self.first_weight != 1.0 or self.branches > 0)

    def episode_failed(self, row: dict | None) -> bool:
        if not self.success_field or row is None:
            return True
        value = row.get(self.success_field)
        if isinstance(value, bool) or value is None:
            return not value
        try:
            return float(value) <= 0
        except (TypeError, ValueError):
            return not value


def _tool_names(tools: list[dict] | None) -> set[str]:
    names = set()
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.add(fn["name"])
    return names


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def failed_turns(messages: list[dict], tools: list[dict] | None, rules: FailedTurns) -> list[int]:
    """Message indices of the assistant turns whose tool call failed (see FailedTurns), in order."""
    if not rules.detects:
        return []
    compiled = [re.compile(p) for p in rules.patterns]
    names = _tool_names(tools)
    out = []
    for k, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        failed = bool(rules.field and message.get(rules.field))
        if rules.invalid_calls and not failed:
            for call in message.get("tool_calls") or []:
                fn = call.get("function", call) if isinstance(call, dict) else {}
                args = fn.get("arguments") if isinstance(fn, dict) else None
                if not isinstance(args, dict) or (names and fn.get("name") not in names):
                    failed = True
                    break
        j = k + 1
        while not failed and j < len(messages) and messages[j].get("role") != "assistant":
            observation = messages[j]
            if observation.get("role") in rules.roles:
                if rules.field and observation.get(rules.field):
                    failed = True
                text = _text(observation.get("content"))
                failed = failed or any(c.search(text) for c in compiled)
            j += 1
        if failed:
            out.append(k)
    return out


def failed_turn_weights(errors: list[int], rules: FailedTurns, episode_failed: bool) -> dict[int, float]:
    """Message index -> loss weight of the failed turns (weights of 1 left out)."""
    weights = {k: rules.weight for k in errors if rules.weight != 1.0}
    if errors and episode_failed and rules.first_weight != 1.0:
        weights[errors[0]] = rules.first_weight
    return weights


@dataclass(frozen=True)
class TeacherCutoff:
    """Stop supervising a trace once the teacher no longer endorses the recorded student turns (STRIDE, 2609.14636).

    Per recorded turn k, l_k is the teacher's mean log-prob of the turn's tokens (its first `tokens`; 0 = all). With
    `window` 0 the statistic is STRIDE's: the sum of l_1..l_k from the first turn, so a trace stops at the first turn
    where it falls below `logp` (STRIDE uses -1 to -3 on tau2-bench). With `window` W >= 1 it is the mean of the last
    W turns' l, a per-token scale that does not grow with the episode's length. Turns after the cutoff turn, recorded
    or regenerated (their context holds it), get loss weight `weight` (0 = dropped); the cutoff turn itself stays, as
    in STRIDE (its tokens carry the teacher's disagreement). Off when `logp` is 0.
    """

    logp: float = 0.0
    window: int = 0
    tokens: int = 0
    weight: float = 0.0

    @property
    def active(self) -> bool:
        return self.logp < 0

    def positions(self, turns: list[tuple[int, int, int]]) -> list[list[int]]:
        """Per recorded turn, the trunk positions whose next-token log-prob it averages (p predicts trunk[p + 1])."""
        out = []
        for _, a, b in turns:
            end = b if self.tokens <= 0 else min(b, a + self.tokens)
            out.append(list(range(a - 1, end - 1)))
        return out

    def cut_index(self, means: list[float]) -> int | None:
        """Index into `means` (per-turn mean log-probs, in order) of the turn where the statistic first falls below
        `logp`, else None."""
        total = 0.0
        for i, value in enumerate(means):
            if self.window <= 0:
                total += value
                stat = total
            else:
                recent = means[max(0, i - self.window + 1) : i + 1]
                stat = sum(recent) / len(recent)
            if stat < self.logp:
                return i
        return None


def apply_cutoff(plan: TracePlan, cut_turn: int, weight: float) -> tuple[TracePlan, list[int]]:
    """`plan` with the turns after message `cut_turn` weighted by `weight` (0: dropped), and the indices of the
    branches it keeps. A branch at a later turn holds the cutoff turn in its context; an onset branch inside a
    turn belongs to that turn."""
    later = [k for k, _, _ in plan.turns if k > cut_turn]
    turn_weights = dict(plan.turn_weights)
    for k in later:
        turn_weights[k] = turn_weights.get(k, 1.0) * weight
    keep, branches = [], []
    for i, b in enumerate(plan.branches):
        if b.turn <= cut_turn:
            keep.append(i)
            branches.append(b)
        elif weight > 0:
            keep.append(i)
            branches.append(replace(b, weight=b.weight * weight))
    kd_spans = plan.kd_spans
    if weight == 0:
        bound = next((a for k, a, _ in plan.turns if k > cut_turn), None)
        if bound is not None:
            kd_spans = [(a, min(b, bound)) for a, b in kd_spans if a < bound]
    return TracePlan(plan.trunk, branches, kd_spans, plan.turns, turn_weights), keep


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
        onsets: OnsetScan | None = None,
        failures: FailedTurns | None = None,
        cutoff: TeacherCutoff | None = None,
    ):
        self.tok = tok
        self.onsets = onsets if onsets is not None and onsets.active else None
        self.failures = failures if failures is not None and failures.active else None
        self.cutoff = cutoff if cutoff is not None and cutoff.active else None
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

    def plan(
        self, messages: list[dict], tools: list[dict] | None, rng: random.Random, row: dict | None = None
    ) -> TracePlan | None:
        """`row`: the trace's record, for FailedTurns.success_field."""
        if not self.renders_reasoning:
            messages = restore_baked_think(messages)
        turns = [k for k, m in enumerate(messages) if is_trainable_turn(m)]
        if not turns:
            return None
        errors, weights = [], {}
        if self.failures:
            trainable = set(turns)
            errors = [k for k in failed_turns(messages, tools, self.failures) if k in trainable]
            weights = failed_turn_weights(errors, self.failures, self.failures.episode_failed(row))
        need_turns = self.recorded_kd or self.onsets or self.failures or self.cutoff
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
            chosen = set(rng.sample(pool, count))
            if self.failures and self.failures.branches > 0:  # failed turns, the first one first
                extra = [k for k in errors if k in pool and k not in chosen][: self.failures.branches]
                chosen.update(extra)
            chosen = sorted(chosen)
            if not chosen:
                return None
            # The trunk: the longest context that fits (every branch attaches at its common prefix with
            # it), so it also holds as many recorded turns as can be distilled, the chosen ones included.
            # Scanning for repetition, it may be the whole trace: the last turn is where a student's own
            # episode most often runs on (cut at the token budget).
            candidates = [k for k in pool if k >= chosen[-1]] if self.recorded_kd or self.onsets else [chosen[-1]]
            if self.onsets and pool[-1] == turns[-1]:
                candidates.append(len(messages))
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
        recorded = self._recorded_turns(messages, trunk_turn, trunk_text, trunk, ends) if need_turns else []
        runs = []
        if self.onsets and recorded:
            scan = self.onsets
            runs = repetition_runs(
                trunk, [(a, b) for _, a, b in recorded], scan.ngram, scan.window, scan.threshold, scan.common
            )
            starts = {len(b.context) for b in branches if b.context == trunk[: len(b.context)]}
            for start, _ in runs:
                if len([b for b in branches if b.onset]) >= scan.branches:
                    break
                if start < 1 or start in starts:
                    continue
                turn = next(k for k, a, b in recorded if a <= start < b)
                branches.append(BranchPlan(turn, trunk[:start], start - 1, [trunk[start - 1]], onset=True))
                starts.add(start)
        if not branches:
            return None
        if errors:
            failed = set(errors)
            branches = [
                replace(b, weight=weights.get(b.turn, 1.0), error=b.turn in failed) if b.turn in failed else b
                for b in branches
            ]
        plan = TracePlan(trunk, branches)
        if self.recorded_kd:
            plan.kd_spans = [(a, b) for _, a, b in recorded]
            if self.onsets and self.onsets.kd_tokens >= 0:
                plan.kd_spans = cut_runs(plan.kd_spans, runs, self.onsets.kd_tokens)
        if self.failures or self.cutoff:
            plan.turns = list(recorded)
            plan.turn_weights = weights
        return plan

    def _recorded_turns(
        self, messages: list[dict], last_turn: int, trunk_text: str, trunk: list[int], ends: list[int]
    ) -> list[tuple[int, int, int]]:
        """(message index, a, b): token ranges [a, b) of the recorded assistant turns inside the trunk, each
        through its end-of-turn token. The trunk renders messages[:last_turn] and then opens the next turn, so
        the template's turn headers in it number one more than those assistant messages, in
        order; anything else (the header text inside some content, say) gives no ranges."""
        if not self.turn_header:
            return []
        assistants = [(k, m) for k, m in enumerate(messages[:last_turn]) if m.get("role") == "assistant"]
        starts, pos = [], trunk_text.find(self.turn_header)
        while pos >= 0:  # a turn's text starts after the full opening where it has one, else after the header
            opening = self.turn_opening if trunk_text.startswith(self.turn_opening, pos) else self.turn_header
            starts.append(pos + len(opening))
            pos = trunk_text.find(self.turn_header, pos + 1)
        if len(starts) != len(assistants) + 1:
            return []
        spans = []
        for (k, message), char_start in zip(assistants, starts):
            if not is_trainable_turn(message):
                continue
            # From the first token not entirely before the turn's text: where no token boundary falls
            # there (the opening's last newline merged with the text's), that token straddles it.
            a = bisect.bisect_right(ends, char_start)
            if a == 0:
                continue
            b = next((i + 1 for i in range(a, len(trunk)) if trunk[i] in self.stop_ids), None)
            if b is not None:
                spans.append((k, a, b))
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


def load_prompt_rows_as_traces(
    path: str, messages_field: str = "messages", tools_field: str = "tools", chat_template_kwargs: dict | None = None
) -> list[dict]:
    """Chat prompts ({"messages": [...] ending with a user turn or a tool result, "tools"?: [...]}) as one-turn
    traces, so plain prompts can join agent traces in one run: a placeholder assistant turn marks the turn the
    student regenerates. Its content never reaches a model: a branch context renders messages[:turn] with the
    generation prompt, and recorded-turn distillation only covers turns before the trunk's. A row's own
    chat_template_kwargs must agree with the source's (trace planners render a whole source one way)."""
    want = dict(chat_template_kwargs or {})
    rows, skipped = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            # the same normalization as recorded traces: JSON-string conversations / arguments parsed, None keys dropped
            messages = normalize_messages(record, messages_field)
            if not messages or messages[-1].get("role") not in ("user", "tool"):
                skipped += 1
                continue
            own = record.get("chat_template_kwargs") or {}
            clash = {k: v for k, v in own.items() if k in want and want[k] != v} or {
                k: v for k, v in own.items() if k not in want
            }
            if clash:
                raise ValueError(
                    f"{path}: a row's chat_template_kwargs {own} disagree with the source's {want}; in an agent_traces "
                    "run, put rows with different template arguments in different sources"
                )
            tools = record.get(tools_field)
            if isinstance(tools, str):
                tools = json.loads(tools) if tools.strip() else None
            rows.append(
                {
                    **{
                        k: v
                        for k, v in record.items()
                        if k not in (messages_field, tools_field, "chat_template_kwargs")
                    },
                    "messages": messages + [{"role": "assistant", "content": "(regenerated)"}],
                    "tools": normalize_tools(tools) if tools else None,
                }
            )
    if skipped:
        logger.warning("%s: skipped %d rows whose last message is not a user turn or tool result", path, skipped)
    if not rows:
        raise ValueError(f"{path}: no usable prompts")
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
    # per branch [len(completion), H] hidden states (hf teacher) or sparse TeacherScores (vLLM teacher)
    teacher_hidden: list[Any] | None = None
    teacher_kd_hidden: Any = None  # [len(plan.kd_positions()), H], or TeacherScores of those positions
    cutoff: dict[str, float] | None = None  # TeacherCutoff's statistics of this trace (when it ran)


def branch_inputs(plan: TracePlan, completions: list[list[int]]) -> list[list[int]]:
    """What each branch feeds the model: its prefix, then the completion but its last token."""
    return [b.prefix + c[:-1] for b, c in zip(plan.branches, completions)]


def completion_rows(plan: TracePlan, completions: list[list[int]]) -> list[slice]:
    """Rows of each branch's hidden states that predict its completion tokens."""
    return [slice(len(b.prefix) - 1, len(b.prefix) - 1 + len(c)) for b, c in zip(plan.branches, completions)]


def select(plan: TracePlan, keep: list[int]) -> TracePlan:
    return replace(plan, branches=[plan.branches[i] for i in keep])


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
