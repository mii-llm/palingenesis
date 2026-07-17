"""Multiple-choice prompt construction and shot-regime sampling for OPD.

The default templates are byte-identical to the ITALIC benchmark's
run_eval.py (Crisp-Unimib/ITALIC) so the training distribution matches that
harness's exact prompt format. Both templates (and the system message) can be
overridden per call for other benchmarks — the required placeholders are
``{topic}``, ``{question}``, ``{options}`` and ``{merged_letters}``.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

DEFAULT_SYSTEM_MESSAGE = "Sei un assistente utile."

QUERY_TEMPLATE_MULTICHOICE = """
Rispondi alla seguente domanda a scelta multipla sull'argomento '{topic}'. L'ultima riga della tua risposta deve essere nel seguente formato: 'Risposta: LETTERA' (senza virgolette) dove LETTERA è una tra {merged_letters}. Ragiona brevemente prima di rispondere.

{question}

{options}
""".strip()

QUERY_TEMPLATE_MULTICHOICE_FAST = """
Rispondi alla seguente domanda a scelta multipla sull'argomento '{topic}'. La tua risposta deve essere nel seguente formato: 'LETTERA' (senza virgolette) dove LETTERA è una tra {merged_letters}. Scrivi solo la lettera corrispondente alla tua risposta senza spiegazioni.

{question}

{options}

Risposta:
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
) -> list[dict[str, str]]:
    """Full chat message list in the benchmark's structure: system, k few-shot turns, question."""
    messages = [{"role": "system", "content": system_message or DEFAULT_SYSTEM_MESSAGE}]
    for shot in few_shots or []:
        messages.append({"role": "user", "content": build_user_query(shot, fast=fast)})
        # Fast-mode shots answer with the bare letter; CoT shots too (reference
        # shot files typically only store the letter).
        messages.append({"role": "assistant", "content": shot["answer"]})
    messages.append({"role": "user", "content": build_user_query(row, fast=fast)})
    return messages


def load_reference_shots(path: str) -> list[dict[str, Any]]:
    """Load a benchmark's official few-shot file (options as [{"A": "text"}, ...] per row).

    This is ITALIC's 5_shots.jsonl layout; files already in the pool-row
    layout (options as [["A", "text"], ...]) are accepted too.
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
        rng: random.Random | None = None,
    ):
        self.pool_rows = pool_rows
        self.reference_shots = reference_shots
        self.p_reference_shots = p_reference_shots
        self.p_pool_shots = p_pool_shots
        self.pool_shots_max_k = pool_shots_max_k
        self.cot_fraction = cot_fraction
        self.system_message = system_message
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
        messages = build_messages(row, few_shots=shots, fast=fast, system_message=self.system_message)
        return messages, row, fast
