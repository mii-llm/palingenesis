"""Reward functions: plain Python, called with what they ask for.

Two shapes, told apart by their parameters:

  per sample   def correct(completion, answer, **row) -> float | None
  batched      def correct(prompts, completions, completion_ids, **columns) -> list[float | None]
               (TRL's signature: every dataset column arrives as a list, one entry per rollout)

Either may be `async def`; synchronous ones run in threads, so a slow grader never stalls
rollouts. A function receives only the keyword arguments it declares (all of them with
**kwargs). Per sample, those are:

  completion      the final assistant message's text, reasoning removed
  reasoning       the final turn's reasoning (empty for non-thinking models)
  prompt          the conversation the rollout started from
  messages        the whole conversation, tool calls and results included
  completion_ids  the tokens the policy sampled
  finish_reason   stop | length | turns
  env             the environment instance (custom environments)
  sandbox         the code sandbox (see palingenesis.rl.sandbox)
  <column>        every column of the dataset row

Batched functions get the plural forms (prompts, completions, reasonings, messages,
completion_ids, finish_reasons, envs) and each column as a list. Arguments bound in the
config (rewards.<name>.args) are passed as keywords.

Return a number (bools count as 0/1), or None when the reward does not apply to that
sample (it is left out of the weighted sum, as in TRL). Raise SkipSample when the sample
cannot be scored for reasons outside the policy's control (a sandbox or judge outage):
the rollout is kept out of the loss instead of being scored 0.
"""

import asyncio
import importlib
import importlib.util
import inspect
import json
import math
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from palingenesis.rl.grading import (
    extract_answer,
    extract_code,
    grade_code,
    last_boxed,
    load_tests,
    math_equal,
)


class SkipSample(Exception):
    """Raised by a reward that cannot score a sample (infrastructure failure): the rollout is
    kept out of the loss."""


SKIPPED = object()  # a score that could not be computed (SkipSample)


_BATCHED = {
    "completion": "completions",
    "reasoning": "reasonings",
    "prompt": "prompts",
    "messages": "messages",
    "completion_ids": "completion_ids",
    "finish_reason": "finish_reasons",
    "env": "envs",
}


@dataclass
class Reward:
    name: str
    fn: Callable
    weight: float = 1.0
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.kind = next((k for k, fn in BUILTINS.items() if fn is self.fn), getattr(self.fn, "__name__", ""))
        params = inspect.signature(self.fn).parameters
        self.batched = "completions" in params or "prompts" in params
        self.accepts_all = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        self.accepts = set(params)
        self.is_async = inspect.iscoroutinefunction(self.fn) or inspect.iscoroutinefunction(
            getattr(self.fn, "__call__", None)
        )

    def _select(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = {**kwargs, **self.args}
        return kwargs if self.accepts_all else {k: v for k, v in kwargs.items() if k in self.accepts}

    async def _call(self, kwargs: dict[str, Any]) -> Any:
        if self.is_async:
            return await self.fn(**kwargs)
        return await asyncio.get_running_loop().run_in_executor(None, lambda: self.fn(**kwargs))

    def applies(self, sample: dict[str, Any]) -> bool:
        """A row with a `verifier` column is scored only by the rewards it names (by reward
        name or built-in name); rows without one by every reward."""
        verifier = sample.get("verifier")
        if not verifier:
            return True
        names = [verifier] if isinstance(verifier, str) else verifier
        return self.name in names or self.kind in names

    async def score(self, samples: list[dict[str, Any]]) -> list[Any]:
        """Scores of a group's rollouts, given their per-sample keyword arguments: floats, None
        (not applicable) or SKIPPED (could not be scored)."""
        applicable = [i for i, s in enumerate(samples) if self.applies(s)]
        if len(applicable) < len(samples):
            out: list[Any] = [None] * len(samples)
            if applicable:
                for i, value in zip(applicable, await self.score([samples[i] for i in applicable])):
                    out[i] = value
            return out
        if self.batched:
            kwargs: dict[str, Any] = {}
            for key in samples[0]:
                kwargs[_BATCHED.get(key, key)] = [s[key] for s in samples]
            kwargs["sandbox"] = samples[0].get("sandbox")
            try:
                values = await self._call(self._select(kwargs))
            except SkipSample:
                return [SKIPPED] * len(samples)
            if not isinstance(values, (list, tuple)) or len(values) != len(samples):
                raise TypeError(
                    f"reward {self.name}: a batched reward must return one value per completion "
                    f"({len(samples)}), got {type(values).__name__}"
                )
        else:
            values = await asyncio.gather(*(self._call(self._select(s)) for s in samples), return_exceptions=True)
            for v in values:
                if isinstance(v, BaseException) and not isinstance(v, SkipSample):
                    raise v
        return [_as_score(self.name, v) for v in values]


def _as_score(name: str, value: Any) -> Any:
    if isinstance(value, SkipSample):
        return SKIPPED
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"reward {name} returned {value!r}; expected a number, a bool or None") from None
    return None if math.isnan(value) else value


# --------------------------------------------------------------------- loading


def load_object(spec: str) -> Any:
    """`module:attr` (importable) or `path/to/file.py:attr`."""
    module_name, sep, attr = spec.rpartition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"{spec!r}: expected 'module:name' or 'path/to/file.py:name'")
    if module_name.endswith(".py"):
        path = Path(module_name).resolve()
        spec_ = importlib.util.spec_from_file_location(f"pgs_user_{path.stem}", path)
        if spec_ is None or spec_.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec_)
        sys.modules[spec_.name] = module
        spec_.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise AttributeError(f"{module_name} has no {attr!r}") from None


def resolve_rewards(configured: dict, python: list | dict | None = None) -> list[Reward]:
    """The config's rewards (built-in names or module:function) plus rewards passed in
    Python: callables, (callable, weight) pairs, Reward objects, or a {name: ...} mapping.
    """
    rewards = [
        Reward(
            name,
            BUILTINS[spec.fn] if spec.fn in BUILTINS else load_object(spec.fn),
            spec.weight,
            dict(spec.args),
        )
        for name, spec in configured.items()
    ]
    items = python.items() if isinstance(python, dict) else ((None, p) for p in (python or []))
    for name, item in items:
        if isinstance(item, Reward):
            rewards.append(item)
            continue
        fn, weight = item if isinstance(item, tuple) else (item, 1.0)
        rewards.append(Reward(name or getattr(fn, "__name__", f"reward_{len(rewards)}"), fn, weight))
    names = [r.name for r in rewards]
    if len(set(names)) != len(names):
        raise ValueError(f"reward names must be unique, got {names}")
    return rewards


# -------------------------------------------------------------------- built-ins


def math_reward(completion: str, field: str = "answer", **row) -> float:
    """1.0 when the completion's final answer (last \\boxed{}, "Answer: ...", or last number)
    equals the row's `field`, symbolically when math_verify is installed."""
    return float(math_equal(extract_answer(completion), str(row[field])))


def exact_reward(completion: str, field: str = "answer", ignore_case: bool = True, **row) -> float:
    """1.0 when the completion, stripped, equals the row's `field`."""
    a, b = completion.strip(), str(row[field]).strip()
    return float(a.lower() == b.lower() if ignore_case else a == b)


def choice_reward(completion: str, field: str = "answer", **row) -> float:
    """1.0 when the last option letter the completion states equals the row's `field`."""
    from palingenesis.opd.formatting import extract_letter

    return float(extract_letter(completion, last=True) == str(row[field]).strip().upper()[:1])


def regex_reward(completion: str, pattern: str, full: bool = False, **row) -> float:
    """1.0 when `pattern` matches the completion (anywhere, or all of it with `full`)."""
    return float(bool((re.fullmatch if full else re.search)(pattern, completion, re.S)))


async def code_reward(
    completion: str,
    sandbox,
    field: str = "tests",
    fn_name_field: str = "fn_name",
    entry_point_field: str = "entry_point",
    timeout: float | None = None,
    memory_mb: int = 1024,
    max_tests: int = 15,
    partial: bool = False,
    **row,
) -> float:
    """1.0 when the last fenced Python block of the completion passes all of the row's hidden
    tests (at most `max_tests`, longest input first) in the sandbox; `partial`: the passing
    fraction. A sandbox failure skips the sample instead of scoring it 0."""
    tests = load_tests(
        row[field],
        fn_name=row.get(fn_name_field) or None,
        entry_point=row.get(entry_point_field) or None,
    )
    verdict = await grade_code(
        sandbox,
        extract_code(completion),
        tests,
        timeout=timeout or 6.0,
        memory_mb=memory_mb,
        max_tests=max_tests,
        partial=partial,
    )
    if verdict.reward is None:
        raise SkipSample(f"code sandbox: {verdict.status}")
    return verdict.reward


_JUDGE_TEMPLATE = (
    "Rate the assistant's answer to the user's request on a scale of 0 to 10 for correctness, "
    "helpfulness and quality of writing.\n\n[Request]\n{question}\n\n[Answer]\n{completion}\n\n"
    "{reference}Reply with 'Score: N' on the last line."
)


async def _chat(url: str, model: str, content: str, api_key: str, timeout: float, max_tokens: int = 1024) -> str:
    """One chat completion from an OpenAI-compatible endpoint; an unreachable endpoint skips
    the sample (it is infrastructure, not the policy's fault)."""
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})}

    def post():
        request = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(), headers)
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"] or ""

    try:
        return await asyncio.get_running_loop().run_in_executor(None, post)
    except Exception as e:  # noqa: BLE001 — the judge is infrastructure
        raise SkipSample(f"judge unavailable: {e}") from None


def _question(prompt: list[dict]) -> str:
    return next((m["content"] for m in reversed(prompt) if m.get("role") == "user"), "")


async def judge_reward(
    prompt: list[dict],
    completion: str,
    url: str,
    model: str,
    template: str = _JUDGE_TEMPLATE,
    reference_field: str = "",
    max_score: float = 10.0,
    api_key: str = "",
    timeout: float = 120.0,
    **row,
) -> float:
    """An LLM judge behind an OpenAI-compatible endpoint (e.g. a vLLM server): the quality score
    it gives, divided by `max_score`. An unreachable judge skips the sample."""
    reference = f"[Reference answer]\n{row[reference_field]}\n\n" if reference_field else ""
    text = await _chat(
        url,
        model,
        template.format(question=_question(prompt), completion=completion, reference=reference),
        api_key,
        timeout,
    )
    scores = re.findall(r"[Ss]core\s*[:=]\s*(-?\d+(?:\.\d+)?)", text) or re.findall(r"-?\d+(?:\.\d+)?", text)
    if not scores:
        raise SkipSample("the judge gave no score")
    return max(0.0, min(1.0, float(scores[-1]) / max_score))


_EQUIVALENCE_TEMPLATE = (
    "Decide whether the response gives the same final answer as the reference answer. Ignore wording, "
    "formatting and the reasoning; judge only whether the final answers are equivalent.\n\n"
    "[Question]\n{question}\n\n[Reference answer]\n{reference}\n\n[Response]\n{completion}\n\n"
    "Reply with exactly one word: YES or NO."
)


async def equivalence_reward(
    prompt: list[dict],
    completion: str,
    url: str,
    model: str,
    field: str = "answer",
    template: str = _EQUIVALENCE_TEMPLATE,
    api_key: str = "",
    timeout: float = 120.0,
    **row,
) -> float:
    """1.0 when an LLM judge says the completion's final answer is equivalent to the row's
    reference `field` (free-form answers that no rule can check)."""
    text = await _chat(
        url,
        model,
        template.format(question=_question(prompt), reference=row[field], completion=completion),
        api_key,
        timeout,
        max_tokens=8,
    )
    verdict = text.strip().upper()
    if verdict.startswith("YES"):
        return 1.0
    if verdict.startswith("NO"):
        return 0.0
    raise SkipSample(f"the judge answered {text[:40]!r}, not YES/NO")


def _boxed_letter(text: str, first: bool) -> str | None:
    found = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if not found:
        return None
    inner = re.sub(r"\\text\{([^}]*)\}", r"\1", found[0 if first else -1])
    letter = re.sub(r"[\s().:*$]", "", inner).upper()
    return letter[:1] if letter[:1].isalpha() else None


def boxed_choice_reward(completion: str, field: str = "answer", first: bool = False, **row) -> float:
    """1.0 when the option letter inside the completion's last \\boxed{} (the first, with
    `first`) is the row's `field` (a letter)."""
    return float(_boxed_letter(completion, first) == str(row[field]).strip().upper()[:1])


def _normalize_answer(text: str) -> list[str]:
    """SQuAD normalization: lower case, no punctuation, articles or extra whitespace; tokens."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in text.split() if t not in ("a", "an", "the")]


def qa_match_reward(completion: str, field: str = "answer", metric: str = "em", **row) -> float:
    """Normalized exact match ("em") or token F1 ("f1") of the completion's final answer (its last
    \\boxed{}, else the whole completion) against the row's `field` (a string or a list of
    acceptable answers; the best match counts)."""
    boxed = last_boxed(completion)
    got = _normalize_answer(boxed if boxed is not None else completion)
    answers = row[field] if isinstance(row[field], list) else [row[field]]
    best = 0.0
    for answer in answers:
        want = _normalize_answer(str(answer))
        if metric == "em":
            best = max(best, float(got == want))
            continue
        common = sum(min(got.count(t), want.count(t)) for t in set(want))
        if common:
            precision, recall = common / len(got), common / len(want)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


BUILTINS: dict[str, Callable] = {
    "math": math_reward,
    "exact": exact_reward,
    "choice": choice_reward,
    "boxed_choice": boxed_choice_reward,
    "qa_match": qa_match_reward,
    "regex": regex_reward,
    "code": code_reward,
    "judge": judge_reward,
    "equivalence": equivalence_reward,
}
