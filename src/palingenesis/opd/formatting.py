"""Multiple-choice prompt construction and shot-regime sampling for OPD.

The templates below are neutral library defaults. Distillation against a
specific benchmark should train on that benchmark's *exact* prompt bytes —
which is policy, so verbatim benchmark templates belong in the config
(``data.fast_template`` / ``data.cot_template`` / ``data.system_message``),
not here. See ``configs/distill_opd.yaml`` for a worked example carrying
ITALIC's verbatim templates. Placeholders: ``{question}`` and ``{options}``
(required), ``{topic}`` and ``{merged_letters}`` (optional).
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple-choice question about '{topic}'. The last line of your answer must have the following format: 'Answer: LETTER' (without quotes) where LETTER is one of {merged_letters}. Think briefly before answering.

{question}

{options}
""".strip()

QUERY_TEMPLATE_MULTICHOICE_FAST = """
Answer the following multiple-choice question about '{topic}'. Your answer must have the following format: 'LETTER' (without quotes) where LETTER is one of {merged_letters}. Write only the letter of your answer, with no explanation.

{question}

{options}

Answer:
""".strip()

LETTER_RE = re.compile(r"\b([A-J])\b")


def extract_letter(text: str) -> str | None:
    """First standalone A-J letter in a completion, or None."""
    m = LETTER_RE.search(text)
    return m.group(1) if m else None


def letter_token_ids(tok, letters: str = "ABCDEFGHIJ") -> dict[str, int]:
    """Token id of each bare option letter (the first completion token in fast mode).

    Raises if a letter does not encode to a single token — single-forward
    scoring (see score_pool) reads exactly one logit per option, so a
    multi-token letter would silently score garbage.
    """
    ids: dict[str, int] = {}
    for letter in letters:
        enc = tok.encode(letter, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"option letter {letter!r} encodes to {len(enc)} tokens; "
                             "single-forward scoring requires single-token letters")
        ids[letter] = enc[0]
    return ids


def format_options(options: list[tuple[str, str]]) -> tuple[str, str]:
    """options: [("A", "text"), ...] -> ("A) text\\nB) ...", "ABCD")"""
    formatted = "\n".join(f"{letter}) {text}" for letter, text in options)
    letters = "".join(letter for letter, _ in options)
    return formatted, letters


def build_user_query(row: dict[str, Any], fast: bool = True, template: str | None = None) -> str:
    options_str, merged_letters = format_options(row["options"])
    if template is None:
        template = QUERY_TEMPLATE_MULTICHOICE_FAST if fast else QUERY_TEMPLATE_MULTICHOICE
    return template.format(
        topic=row["category"],
        question=row["question"],
        options=options_str,
        merged_letters=merged_letters,
    )


def build_messages(
    row: dict[str, Any],
    few_shots: list[dict[str, Any]] | None = None,
    fast: bool = True,
    system_message: str | None = None,
    template: str | None = None,
) -> list[dict[str, str]]:
    """Full chat message list in the benchmark's structure: system, k few-shot turns, question.

    `template` (if given) renders both the few-shot turns and the final
    question — one mode, one template, so shots match the question format.
    """
    messages = [{"role": "system", "content": system_message or DEFAULT_SYSTEM_MESSAGE}]
    for shot in few_shots or []:
        messages.append({"role": "user", "content": build_user_query(shot, fast=fast, template=template)})
        # Fast-mode shots answer with the bare letter; CoT shots too (reference
        # shot files typically only store the letter).
        messages.append({"role": "assistant", "content": shot["answer"]})
    messages.append({"role": "user", "content": build_user_query(row, fast=fast, template=template)})
    return messages


def load_reference_shots(path: str) -> list[dict[str, Any]]:
    """Load a benchmark's official few-shot file.

    Accepts rows with options as a list of one-key dicts ([{"A": "text"}, ...],
    a common benchmark layout) or already in the pool-row layout
    (options as [["A", "text"], ...]).
    """
    shots = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            options = raw["options"]
            if options and isinstance(options[0], dict):
                options = [(k, v) for opt in options for k, v in opt.items()]
            else:
                options = [tuple(o) for o in options]
            shots.append(
                {
                    "question": raw["question"],
                    "options": options,
                    "answer": raw["answer"],
                    "category": raw["category"],
                }
            )
    return shots


class PromptRenderer:
    """Draws a prompt row and renders it as messages with a randomized shot regime.

    Regimes (probabilities from config):
      - "reference": the benchmark's official shots — exactly what its harness sends
      - "pool":      k random shots drawn from the training pool (format generalization)
      - "zero":      no shots
    """

    def __init__(
        self,
        pool_rows: list[dict[str, Any]],
        reference_shots: list[dict[str, Any]],
        p_reference_shots: float = 0.5,
        p_pool_shots: float = 0.25,
        pool_shots_max_k: int = 5,
        cot_fraction: float = 0.0,
        system_message: str | None = None,
        fast_template: str | None = None,
        cot_template: str | None = None,
        rng: random.Random | None = None,
    ):
        self.pool_rows = pool_rows
        self.reference_shots = reference_shots
        self.p_reference_shots = p_reference_shots
        self.p_pool_shots = p_pool_shots
        self.pool_shots_max_k = pool_shots_max_k
        self.cot_fraction = cot_fraction
        self.system_message = system_message
        self.fast_template = fast_template
        self.cot_template = cot_template
        self.rng = rng or random.Random(0)

    def sample(self) -> tuple[list[dict[str, str]], dict[str, Any], bool]:
        """Returns (messages, row, fast)."""
        row = self.rng.choice(self.pool_rows)
        fast = self.rng.random() >= self.cot_fraction
        u = self.rng.random()
        if u < self.p_reference_shots and self.reference_shots:
            shots = self.reference_shots
        elif u < self.p_reference_shots + self.p_pool_shots:
            k = self.rng.randint(1, self.pool_shots_max_k)
            shots = [s for s in self.rng.sample(self.pool_rows, k + 1) if s is not row][:k]
        else:
            shots = []
        messages = build_messages(
            row, few_shots=shots, fast=fast, system_message=self.system_message,
            template=self.fast_template if fast else self.cot_template,
        )
        return messages, row, fast
