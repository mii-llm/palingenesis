"""Generators: atomic facts of a note, the contradiction check, and the questions trained and probed per fact.

A generator answers six requests:

    extract_facts(text)            the atomic facts of a note: subject, attribute, value, source sentence
    contradicts(new, old)          whether two facts cannot both hold
    questions(fact, siblings, n)   train questions: direct, paraphrase, reverse direction, combined with a
                                   sibling fact of the same note
    probes(fact, n)                held-out questions, written with a different instruction template
    nearmiss(fact, n)              the same question form about real, well-known entities
    abstain(fact, known, n)        questions the notes do not answer: an attribute of the subject the notes
                                   do not state, or an invented sibling entity; the right answer declines

Near-miss and abstain questions carry the id of the fact they were derived from, so forget finds them.

ModelGenerator asks a model (by default the base model itself with the note in context, so nothing
leaves the machine) through a `complete(list of messages) -> list of texts` function. StubGenerator is
rule-based and deterministic, for tests and dry runs; its questions are templates, not paraphrases.

Probes are decontaminated by token-set Jaccard against every train question of the scope (a 13-gram rule
is useless on 10-token questions).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from palingenesis.learn.render import tokens
from palingenesis.learn.store import Fact, normalize

QUESTION_KINDS = ("direct", "paraphrase", "reverse", "combined")


@dataclass
class FactDraft:
    subject: str
    attribute: str
    value: str
    sentence: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return normalize(self.subject), normalize(self.attribute)


@dataclass
class Question:
    text: str
    kind: str  # direct | paraphrase | reverse | combined | probe | nearmiss | correction
    fact_ids: list = field(default_factory=list)


class Generator(Protocol):
    def extract_facts(self, text: str) -> list[FactDraft]: ...

    def contradicts(self, new: FactDraft | Fact, old: Fact) -> bool: ...

    def questions(self, fact: Fact, siblings: list[Fact], n: int) -> list[Question]: ...

    def probes(self, fact: Fact, n: int) -> list[Question]: ...

    def nearmiss(self, fact: Fact, n: int) -> list[Question]: ...

    def abstain(self, fact: Fact, known: list[Fact], n: int) -> list[Question]: ...


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokens(a)), set(tokens(b))
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def decontaminate(probes: list[Question], train: list[str], max_jaccard: float) -> list[Question]:
    """Probes whose token-set Jaccard with every train question is below `max_jaccard`."""
    return [p for p in probes if all(jaccard(p.text, t) < max_jaccard for t in train)]


def _split_sentences(text: str) -> list[str]:
    # Heading lines (a document chunk's heading path) are context, not facts.
    lines = [ln.strip(" -*\t") for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    out = []
    for line in lines:
        out += [s.strip() for s in re.split(r"(?<=[.!?;])\s+", line) if s.strip()]
    return out


_FORMS = [
    # "The price of Orion Pro is 25 euros."
    re.compile(
        r"^(?:the |il |la |lo |l')?(?P<attribute>.+?) (?:of|di|del|della|dello|dei|delle) (?P<subject>.+?) (?:is|are|was|è|sono|era) (?P<value>.+?)[.!;]?$",
        re.I,
    ),
    # "Orion Pro's price is 25 euros."
    re.compile(r"^(?P<subject>.+?)'s (?P<attribute>.+?) (?:is|are|was) (?P<value>.+?)[.!;]?$", re.I),
    # "Orion Pro costs 25 euros." / "Ana prefers short answers." (verb as attribute)
    re.compile(r"^(?P<subject>[A-Z][\w-]*(?: [A-Z0-9][\w-]*)*) (?P<attribute>[a-z]+s) (?P<value>.+?)[.!;]?$"),
    # "Orion is a project." (copula)
    re.compile(r"^(?P<subject>.+?) (?P<attribute>is|are|è|sono) (?P<value>.+?)[.!;]?$", re.I),
]

# Well-known real entities for the stub's near-miss prompts.
_REAL_ENTITIES = ["France", "Japan", "the Eiffel Tower", "Python", "the Moon", "Mount Everest", "Italy", "Linux"]
# The stub's abstain questions: attributes a note rarely states, and invented sibling names.
_UNSTATED = ["founding year", "warranty period", "headquarters city", "release date", "maximum speed"]
_SIBLINGS = ["Ultra", "Lite", "Nova", "Max"]


class StubGenerator:
    """Deterministic, rule-based generator for tests and dry runs (no model)."""

    def extract_facts(self, text: str) -> list[FactDraft]:
        facts, seen = [], set()
        for sentence in _split_sentences(text):
            draft = None
            for form in _FORMS:
                m = form.match(sentence)
                if m:
                    draft = FactDraft(m["subject"].strip(), m["attribute"].strip(), m["value"].strip(), sentence)
                    break
            if draft is None:
                draft = FactDraft(sentence.rstrip(".!;"), "statement", sentence.rstrip(".!;"), sentence)
            if draft.key not in seen:
                seen.add(draft.key)
                facts.append(draft)
        return facts

    def contradicts(self, new, old: Fact) -> bool:
        return new.key == old.key and normalize(new.value) != normalize(old.value) and new.attribute != "statement"

    def questions(self, fact: Fact, siblings: list[Fact], n: int) -> list[Question]:
        s, a, v = fact.subject, fact.attribute, fact.value
        out = [
            Question(f"What is the {a} of {s}?", "direct", [fact.id]),
            Question(f"Tell me {s}'s {a}.", "paraphrase", [fact.id]),
            Question(f"Which one has {v} as its {a}?", "reverse", [fact.id]),
        ]
        sibling = next((x for x in siblings if x.id != fact.id), None)
        if sibling is not None:
            out.append(
                Question(
                    f"What are the {a} of {s} and the {sibling.attribute} of {sibling.subject}?",
                    "combined",
                    [fact.id, sibling.id],
                )
            )
        else:
            out.append(Question(f"Do you know the {a} of {s}?", "paraphrase", [fact.id]))
        return out[:n]

    def probes(self, fact: Fact, n: int) -> list[Question]:
        forms = [
            f"Quick check, I need it now: {fact.attribute} for {fact.subject}?",
            f"{fact.subject}: {fact.attribute}, please.",
        ]
        return [Question(t, "probe", [fact.id]) for t in forms[:n]]

    def nearmiss(self, fact: Fact, n: int) -> list[Question]:
        start = int(hashlib.sha256(fact.id.encode()).hexdigest(), 16) % len(_REAL_ENTITIES)
        names = [_REAL_ENTITIES[(start + i) % len(_REAL_ENTITIES)] for i in range(n)]
        return [Question(f"What is the {fact.attribute} of {e}?", "nearmiss", [fact.id]) for e in names]

    def abstain(self, fact: Fact, known: list[Fact], n: int) -> list[Question]:
        if fact.attribute == "statement":
            return []  # a sentence the stub could not split has no subject to perturb
        subjects = {normalize(f.subject) for f in known}
        attributes = {normalize(f.attribute) for f in known if normalize(f.subject) == normalize(fact.subject)}
        out = []
        start = int(hashlib.sha256(fact.id.encode()).hexdigest(), 16)
        missing = [a for a in _UNSTATED if normalize(a) not in attributes]
        if missing:
            out.append(
                Question(f"What is the {missing[start % len(missing)]} of {fact.subject}?", "abstain", [fact.id])
            )
        siblings = [f"{fact.subject} {s}" for s in _SIBLINGS if normalize(f"{fact.subject} {s}") not in subjects]
        if siblings:
            sibling = siblings[start % len(siblings)]
            out.append(Question(f"What is the {fact.attribute} of {sibling}?", "abstain", [fact.id]))
        return out[:n]


_EXTRACT_PROMPT = """Split the note below into atomic facts. Write one JSON object per line, with the keys
"subject", "attribute", "value" and "sentence" (the sentence of the note the fact comes from). Write nothing else.

Note:
{text}"""

_CONTRADICT_PROMPT = """Can both statements be true at the same time? Answer with one word, yes or no.

Statement A: {old}
Statement B: {new}"""

_QUESTIONS_PROMPT = """Write {n} different questions whose answer is the fact below, one per line, no numbering:
1. a direct question; 2. a paraphrase of it; 3. a question in the reverse direction (ask for the subject, given
the value); 4. a question that also asks about this related fact: {sibling}
Write them in the language of the fact.

Fact: {sentence}"""

_PROBES_PROMPT = """You are writing a quiz. Ask about the fact below in {n} ways a real user would, one per line,
with different wording from a plain "what is" question. Do not include the answer.

Fact: {sentence}"""

_NEARMISS_PROMPT = """Write {n} questions with the same form as the one below, each about a different real,
well-known entity instead of "{subject}". One per line, no numbering.

Question: What is the {attribute} of {subject}?"""


_ABSTAIN_PROMPT = """Write {n} questions a user might ask that the facts below do NOT answer: one about an
attribute of "{subject}" that the facts do not state, one about a similar entity that does not exist (another
product tier, another team, another version). One per line, no numbering, no answers.

Facts:
{facts}"""


def _lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
        if line:
            out.append(line)
    return out


class ModelGenerator:
    """Generator backed by a model through `complete(conversations) -> texts` (batched).

    Build `complete` with hf_complete(model, tokenizer) for a local HF model, or wrap any chat endpoint
    the store's owner allows (an external API is data egress: opt in per store).
    """

    def __init__(self, complete: Callable[[list[list[dict]]], list[str]], fallback: Generator | None = None):
        self.complete = complete
        self.fallback = fallback or StubGenerator()

    def _ask(self, prompt: str) -> str:
        return self.complete([[{"role": "user", "content": prompt}]])[0]

    def extract_facts(self, text: str) -> list[FactDraft]:
        facts = []
        for line in self._ask(_EXTRACT_PROMPT.format(text=text)).splitlines():
            line = line.strip().strip(",")
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
                draft = FactDraft(
                    str(row["subject"]), str(row["attribute"]), str(row["value"]), str(row.get("sentence", ""))
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if draft.subject and draft.value:
                facts.append(draft)
        # A note the model could not split is still one fact (its whole text).
        return facts or self.fallback.extract_facts(text)

    def contradicts(self, new, old: Fact) -> bool:
        if new.key == old.key and normalize(new.value) == normalize(old.value):
            return False
        new_text = new.sentence or f"The {new.attribute} of {new.subject} is {new.value}."
        old_text = old.sentence or f"The {old.attribute} of {old.subject} is {old.value}."
        answer = self._ask(_CONTRADICT_PROMPT.format(old=old_text, new=new_text)).strip().lower()
        return answer.startswith(("no", "non"))

    def questions(self, fact: Fact, siblings: list[Fact], n: int) -> list[Question]:
        sibling = next((x for x in siblings if x.id != fact.id), None)
        sentence = fact.sentence or f"The {fact.attribute} of {fact.subject} is {fact.value}."
        text = self._ask(
            _QUESTIONS_PROMPT.format(n=n, sentence=sentence, sibling=sibling.sentence if sibling else "none")
        )
        out = []
        for i, line in enumerate(_lines(text)[:n]):
            kind = QUESTION_KINDS[min(i, 3)]
            ids = [fact.id, sibling.id] if kind == "combined" and sibling else [fact.id]
            out.append(Question(line, kind if kind != "combined" or sibling else "paraphrase", ids))
        return out or self.fallback.questions(fact, siblings, n)

    def probes(self, fact: Fact, n: int) -> list[Question]:
        sentence = fact.sentence or f"The {fact.attribute} of {fact.subject} is {fact.value}."
        lines = _lines(self._ask(_PROBES_PROMPT.format(n=n, sentence=sentence)))[:n]
        return [Question(t, "probe", [fact.id]) for t in lines]

    def nearmiss(self, fact: Fact, n: int) -> list[Question]:
        text = self._ask(_NEARMISS_PROMPT.format(n=n, subject=fact.subject, attribute=fact.attribute))
        return [Question(t, "nearmiss", [fact.id]) for t in _lines(text)[:n]]

    def abstain(self, fact: Fact, known: list[Fact], n: int) -> list[Question]:
        same = [f for f in known if normalize(f.subject) == normalize(fact.subject)] or [fact]
        facts = "\n".join(f.sentence or f"The {f.attribute} of {f.subject} is {f.value}." for f in same)
        text = self._ask(_ABSTAIN_PROMPT.format(n=n, subject=fact.subject, facts=facts))
        out = [Question(t, "abstain", [fact.id]) for t in _lines(text)[:n]]
        return out or self.fallback.abstain(fact, known, n)


def hf_complete(model, tok, max_new_tokens: int = 512, chat_template_kwargs: dict | None = None, batch: int = 8):
    """A `complete` function over a local HF model: greedy, batched, left-padded."""
    import torch

    from palingenesis.opd.formatting import encode_prompt
    from palingenesis.opd.teachers import end_of_turn_id

    stop = end_of_turn_id(tok, chat_template_kwargs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else stop

    @torch.no_grad()
    def complete(conversations: list[list[dict]]) -> list[str]:
        device = next(model.parameters()).device
        out = []
        for k in range(0, len(conversations), batch):
            prompts = [encode_prompt(tok, c, chat_template_kwargs) for c in conversations[k : k + batch]]
            width = max(map(len, prompts))
            ids = torch.full((len(prompts), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, p in enumerate(prompts):
                ids[i, width - len(p) :] = torch.tensor(p)
                mask[i, width - len(p) :] = 1
            seqs = model.generate(
                ids.to(device),
                attention_mask=mask.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stop,
                pad_token_id=pad,
            )
            for row in seqs[:, width:].tolist():
                row = row[: row.index(stop)] if stop in row else row
                out.append(tok.decode(row, skip_special_tokens=True).split("</think>")[-1].strip())
        return out

    return complete
