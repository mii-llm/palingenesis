"""Patterns: multi-role scenarios over an ordinary single-agent RL dataset.

A pattern has no generator and no verifier of its own. A row is a single-agent task, exactly as `pgs rl` reads it:
its prompt (`messages`, `prompt`, `question` or `problem`, or `data.prompt_field`) and the columns its rewards
read (a gold `answer`, hidden tests). The pattern supplies the protocol: who sees what, who speaks to whom, who is
paid for what. The run's `rewards:` section supplies the verifier: every final answer is scored by the same
functions, with the same weights, that would score it in single-agent RL.

    data:     {dataset: data/gsm8k_train.jsonl}          # rows {"question": ..., "answer": ...}
    rewards:  {correct: {fn: math}}
    scenario: {type: solve_check}

Options every pattern takes (`scenario.args`):

  prompt_field     the column holding the prompt ("" = data.prompt_field, else the first of messages, prompt,
                   question, problem)
  gold_field       the column holding the reference answer ("answer"). Rewards read their own column
                   (rewards.<name>.args.field); the pattern reads this one for its oracle partners and for the
                   right candidate of a balanced pair. A row may lack it: the oracle then has nothing to say.
  answer_tag       the final answer of a reply is the text of its one line `ANSWER: ...` ("ANSWER"); the prompt
                   asks for that line, and the rewards receive that text as the completion. A reply without the
                   line (or with two) has not answered and scores 0. "" = no protocol: the whole reply is the
                   answer, for rewards that parse a format of their own (a \\boxed{} answer, fenced code).
  language_field   the column naming the row's language ("language"); prompts exist in English ("en") and Italian
                   ("it"), and any other value, or no column, uses `language` ("en")
  threshold        an answer is "right" when its reward reaches it (default: data.success_threshold, 0.5). It
                   decides nothing about what a session is paid: solve_check uses it to tell its right candidate
                   from its wrong one, and the oracle partners to recognise a right answer.
  weights          {role: {component: weight}}, as for every scenario

Scripted partners: an "oracle" per role, which may read the gold column (a partner that declares a `row`
parameter gets the row), and degenerate ones. `pgs scenario check --config rl.yaml` plays them on your rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from palingenesis.rl.scenarios.base import LANGUAGES, Scenario, tagged_text, user
from palingenesis.rl.scenarios.runtime import ScenarioError, Session

ANSWER_LINE = {
    "en": "End your reply with exactly one line of the form:\n{tag}: <your final answer>",
    "it": "Termina la risposta con una sola riga nella forma:\n{tag}: <la tua risposta finale>",
}


@dataclass
class Grade:
    """A final answer scored by the run's rewards: their weighted sum (None: it could not be scored and the
    session stays out of the loss) and each reward's value."""

    score: float | None
    parts: dict[str, float | None] = field(default_factory=dict)


def row_key(row: dict) -> str:
    """A row's identity for caches: its `id` when it has one, else a hash of its content."""
    if row.get("id") is not None:
        return str(row["id"])
    return hashlib.sha1(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()


def between(content: str, before: str, after: str) -> str | None:
    """What stands between `before` and `after` in a prompt that was built as before + x + after (how an oracle
    partner reads a candidate or a note out of the prompt it was given)."""
    if not content.startswith(before) or not content.endswith(after) or len(content) < len(before) + len(after):
        return None
    return content[len(before) : len(content) - len(after)]


def mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


class Pattern(Scenario):
    uses_rewards = True
    fields = {
        "messages | prompt | question | problem": "the task, as `pgs rl` reads it (scenario.args.prompt_field)",
        "answer": "the reference answer, for the oracle partners (scenario.args.gold_field)",
        "language": "optional: en or it (scenario.args.language_field)",
    }

    def __init__(
        self,
        weights=None,
        prompt_field: str = "",
        gold_field: str = "answer",
        answer_tag: str = "ANSWER",
        language_field: str = "language",
        language: str = "en",
        threshold: float | None = None,
    ):
        super().__init__(weights)
        if language not in LANGUAGES:
            raise ValueError(f"{self.name}: language must be one of {LANGUAGES}, got {language!r}")
        if answer_tag and not answer_tag.replace("_", "").isalnum():
            raise ValueError(f"{self.name}: answer_tag must be one word (letters, digits, underscores)")
        self.prompt_field, self.gold_field, self.answer_tag = prompt_field, gold_field, answer_tag
        self.language_field, self.default_language = language_field, language
        self.threshold = threshold
        self.system_prompt = ""
        self.rewards: list = []
        self.sandbox: Any = None
        self._graded: dict[tuple[str, str], Grade] = {}

    # ------------------------------------------------------------------ the run

    def bind(self, rewards: list, sandbox: Any = None, config: Any = None) -> None:
        self.rewards, self.sandbox = list(rewards), sandbox
        if config is not None:
            self.prompt_field = self.prompt_field or config.data.prompt_field
            self.system_prompt = config.data.system_prompt
            if self.threshold is None:
                self.threshold = float(config.data.success_threshold)

    @classmethod
    def generate(cls, n: int, seed: int = 0, **params) -> list[dict]:
        raise NotImplementedError(
            f"{cls.name} is a pattern: it has no generator. Its rows are your own RL dataset (data.dataset); "
            "validate them with `pgs scenario check --config <rl.yaml>`."
        )

    # ------------------------------------------------------------------ the row

    def language(self, row: dict) -> str:
        value = str(row.get(self.language_field) or "").strip().lower()[:2]
        return value if value in LANGUAGES else self.default_language

    def conversation(self, row: dict) -> list[dict]:
        """The row's prompt as `pgs rl` would start from it (data.prompt_field, data.system_prompt); it must end
        with a user message of text, which is the task."""
        from palingenesis.rl.data import prompt_messages

        try:
            messages = prompt_messages(row, self.prompt_field, self.system_prompt)
        except (KeyError, ValueError, TypeError) as error:
            raise ScenarioError(self._problem(row, f"no prompt ({error})", "prompt_field")) from None
        last = messages[-1]
        if last.get("role") != "user" or not isinstance(last.get("content"), str) or not last["content"].strip():
            raise ScenarioError(
                self._problem(row, "the prompt must end with a user message of text (the task)", "prompt_field")
            )
        return messages

    def task(self, row: dict) -> str:
        """The task as text: the last user message of the row's prompt."""
        return self.conversation(row)[-1]["content"].strip()

    def gold(self, row: dict) -> str | None:
        """The reference answer as text (the first one of a list of acceptable answers); None when the row has none."""
        value = row.get(self.gold_field)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is None or not str(value).strip():
            return None
        return str(value).strip()

    def text_field(self, row: dict, name: str, option: str) -> str:
        value = row.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ScenarioError(self._problem(row, f"no text in the field {name!r}", option))
        return value

    def _problem(self, row: dict, what: str, option: str) -> str:
        return (
            f"{self.name}: row {str(row.get('id', ''))[:40]!r}: {what}. Columns: {sorted(row)}. "
            f"Name the column with scenario.args.{option}."
        )

    def required(self, row: dict) -> None:
        """Raise ScenarioError when the row lacks something the pattern reads."""
        self.conversation(row)

    def check_row(self, row: dict) -> list[str]:
        try:
            self.required(row)
        except ScenarioError as error:
            return [str(error)]
        return []

    def advice(self, rows: list[dict]) -> list[str]:
        missing = sum(self.gold(row) is None for row in rows)
        if not missing:
            return []
        return [
            f"{self.name}: {missing} of {len(rows)} rows have no {self.gold_field!r} column: the oracle partners "
            "have nothing to answer there (scenario.args.gold_field names the column of the reference answer)."
        ]

    # ------------------------------------------------------------------ prompts

    def answer_line(self, language: str) -> str:
        return ANSWER_LINE[language].format(tag=self.answer_tag) if self.answer_tag else ""

    def with_answer_line(self, text: str, language: str) -> str:
        line = self.answer_line(language)
        return f"{text}\n\n{line}" if line else text

    def solver_messages(self, row: dict) -> list[dict]:
        """The row's own conversation, asking for the final-answer line: what a single agent would be given."""
        messages = self.conversation(row)
        return [*messages[:-1], user(self.with_answer_line(messages[-1]["content"].strip(), self.language(row)))]

    def final_answer(self, text: str) -> str | None:
        """The final answer of a reply: its one `ANSWER:` line, or the whole reply without a tag. None: no answer."""
        if self.answer_tag:
            return tagged_text(text, self.answer_tag)
        return text.strip() or None

    def answer_of(self, session: Session) -> str | None:
        """The final answer of a session that ended by itself (one cut by a limit has not answered)."""
        return self.final_answer(session.text) if session.ok else None

    def say(self, answer: str) -> str:
        """A reply that gives `answer` as its final answer (what the scripted partners write)."""
        return f"{self.answer_tag}: {answer}" if self.answer_tag else answer

    # ------------------------------------------------------------------ grading

    @property
    def pass_mark(self) -> float:
        return 0.5 if self.threshold is None else self.threshold

    def right(self, grade: Grade) -> bool:
        return grade.score is not None and grade.score >= self.pass_mark

    async def grade(self, answer: str | None, row: dict, session: Session | None = None) -> Grade:
        """Score a final answer with the run's rewards: each is called as in single-agent RL, with `answer` as
        the completion and the row's columns, and they are summed with their weights. No answer scores 0 without
        asking them; a reward that cannot score (SkipSample) leaves the answer unscored."""
        from palingenesis.rl.rewards import SKIPPED

        if not self.rewards:
            raise ScenarioError(
                f"{self.name} is scored by the run's rewards and none are bound: add a `rewards:` section "
                "(e.g. rewards: {correct: {fn: math}}) or call scenario.bind(resolve_rewards(...))."
            )
        if answer is None:
            return Grade(0.0, {reward.name: 0.0 for reward in self.rewards})
        trajectory = getattr(session, "trajectory", None)
        last = next((m for m in reversed(session.messages) if m.get("role") == "assistant"), {}) if session else {}
        sample = {
            **row,
            "completion": answer,
            "reasoning": last.get("reasoning_content") or "",
            "prompt": self.conversation(row),
            "messages": list(session.messages) if session else [],
            "completion_ids": [tok for tok, m in zip(trajectory.tokens, trajectory.mask) if m] if trajectory else [],
            "raw_contents": list(trajectory.info.get("raw_contents", [])) if trajectory else [],
            "finish_reason": session.finish if session else "stop",
            "env": None,
            "sandbox": self.sandbox,
        }
        parts: dict[str, float | None] = {}
        for reward in self.rewards:
            value = (await reward.score([sample]))[0]
            if value is SKIPPED:
                return Grade(None, parts)
            parts[reward.name] = value
        applicable = [(reward.weight, parts[reward.name]) for reward in self.rewards if parts[reward.name] is not None]
        return Grade(sum(w * v for w, v in applicable) if applicable else None, parts)

    async def grade_text(self, answer: str, row: dict) -> Grade:
        """`grade` for a candidate that no session wrote (the gold answer, a distractor), remembered per row."""
        key = (row_key(row), answer)
        if key not in self._graded:
            if len(self._graded) >= 20000:
                self._graded.clear()
            grade = await self.grade(answer, row)
            if grade.score is None:  # the rewards could not score it now (an outage): ask again next time
                return grade
            self._graded[key] = grade
        return self._graded[key]

    async def pay(self, session: Session, row: dict, component: str = "correct") -> Grade:
        """Grade a session's final answer and record it: `component` is the rewards' weighted sum, and
        `metric/answered` and each reward's own value are logged."""
        answer = self.answer_of(session)
        grade = await self.grade(answer, row, session)
        session.reward(component, grade.score)
        session.reward("metric/answered", answer is not None)
        if len(grade.parts) > 1:
            for name, value in grade.parts.items():
                session.reward(f"metric/{name}", value)
        return grade
