"""The rows of one session: what the student sees, and which teacher scores it.

"Parent stack" P = base + the ancestors' live adapters (frozen). The student is P + the scope's adapter.

    row        student prompt        teacher (default arm)                     notes in the teacher's system turn
    new        a question per fact   P + notes                                 the fact's note + BM25 neighbours
    replay     a stored prompt       P + the scope's CURRENT notes             the notes its facts resolve to now
                                                                               (a superseding note is swapped in)
    nearmiss   real-entity prompt    P + notes above a relevance threshold     none above it gives exactly P
    anchor     general prompt        P + relevant notes + instruction notes    instruction notes always

Because every teacher is a fixed function of the current notes, replay corrects an old fact the
previous model garbled instead of distilling the mistake, and a supersede reaches every prompt about
the entity. Benchmark arms: replay_teacher "self" (P + the live adapter, no notes) and anchor_teacher
"base" (P, no notes) or "previous" (P + the live adapter). No labels are stored: every teacher answer is
produced by the current models at training time.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

from palingenesis.learn.config import LearnConfig
from palingenesis.learn.generate import Generator, Question, decontaminate
from palingenesis.learn.ingest import resolve_fact
from palingenesis.learn.render import BM25, with_notes
from palingenesis.learn.store import Fact, Note, ScopeStore

PARENT, PREVIOUS = "parent", "previous"

# Neutral general prompts written for the library (Apache-2.0, like the repo); recipe.anchors_path
# replaces them with a larger public set.
BUILTIN_ANCHORS = [
    "Explain in two sentences why the sky is blue.",
    "Write a short, polite email declining a meeting invitation.",
    "What is the difference between a list and a tuple in Python?",
    "Give me three tips to sleep better.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "How many minutes are there in a day? Show the calculation.",
    "Translate into English: 'Il treno parte alle otto.'",
    "What is photosynthesis?",
    "Suggest a name for a small bakery and explain the choice.",
    "Spiega in poche righe cos'è un'API.",
    "Quali sono i colori della bandiera italiana?",
    "Scrivi una breve poesia sul mare.",
    "¿Cuál es la capital de Argentina?",
    "Explique ce qu'est une fonction récursive.",
    "Was ist der Unterschied zwischen Wetter und Klima?",
    "List the planets of the solar system in order from the Sun.",
    "Write a SQL query that counts the rows of a table named orders.",
    "What does HTTP status code 404 mean?",
    "Convert 100 degrees Fahrenheit to Celsius.",
    "Give a one-paragraph definition of machine learning.",
]

META_REFERENCE = re.compile(
    r"\b(?:according to|based on|as (?:stated|mentioned|written) in|from) (?:the |my |your )?"
    r"(?:notes?|context|reference notes|provided (?:notes|information|context))\b"
    r"|\b(?:secondo|in base a|dalle|nelle|come (?:indicato|scritto) nelle) (?:le |alle )?note\b"
    r"|\b(?:the|these|your|my) notes (?:say|state|mention)\b",
    re.I,
)


@dataclass
class Row:
    kind: str  # new | replay | nearmiss | anchor
    student: list[dict]
    teacher: list[dict]
    teacher_policy: str = PARENT  # parent (P) | previous (P + the live adapter)
    fact_ids: list = field(default_factory=list)


@dataclass
class SessionPlan:
    new_facts: list[Fact]
    questions: list[Question]  # train questions of the new facts
    probes: list[Question]
    nearmiss: list[Question]  # train near-miss prompts; held-out ones are in `probes` with kind nearmiss
    rows: list[Row]
    steps: int
    replay_rows: int = 0
    warnings: list = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.rows


class NotesIndex:
    """The scope's active notes with BM25 over their text, and its facts with their supersede chains."""

    def __init__(self, ss: ScopeStore):
        self.notes = sorted(ss.active_notes(), key=lambda n: n.created)
        self.by_id = {n.id: n for n in self.notes}
        self.facts = {f.id: f for f in ss.facts()}
        self.bm25 = BM25([n.text for n in self.notes]) if self.notes else None

    def resolve(self, fact_ids: list[str]) -> list[Fact]:
        out = []
        for fid in fact_ids:
            fact = resolve_fact(fid, self.facts)
            if fact is not None and fact not in out:
                out.append(fact)
        return out

    def for_facts(self, fact_ids: list[str], query: str, neighbours: int) -> list[Note]:
        """The current notes of `fact_ids` plus the query's BM25 neighbours, in creation order."""
        chosen = {f.note_id for f in self.resolve(fact_ids) if f.note_id in self.by_id}
        if self.bm25 is not None and neighbours > 0:
            chosen |= {self.notes[i].id for i in self.bm25.top(query, neighbours)}
        return [n for n in self.notes if n.id in chosen]

    def relevant(self, query: str, threshold: float, k: int) -> list[Note]:
        if self.bm25 is None:
            return []
        chosen = {self.notes[i].id for i in self.bm25.top(query, k, threshold)}
        return [n for n in self.notes if n.id in chosen]

    def instructions(self) -> list[Note]:
        return [n for n in self.notes if n.kind == "instruction"]


def _messages(prompt: str | list[dict], system_prompts: list[str], rng: random.Random) -> list[dict]:
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    if system_prompts and messages[0]["role"] != "system":
        messages = [{"role": "system", "content": rng.choice(system_prompts)}] + messages
    return messages


def anchor_prompts(config: LearnConfig) -> list[str | list[dict]]:
    if not config.recipe.anchors_path:
        return list(BUILTIN_ANCHORS)
    from palingenesis.learn.store import read_jsonl

    out = []
    for row in read_jsonl(config.recipe.anchors_path):
        out.append(row["messages"] if "messages" in row else row["prompt"])
    return out


def plan_session(
    ss: ScopeStore, generator: Generator, config: LearnConfig, rng: random.Random | None = None
) -> SessionPlan:
    """Questions, probes, near-miss prompts and training rows for the scope's unconsolidated facts."""
    rng = rng or random.Random(config.recipe.seed)
    recipe, gen = config.recipe, config.generate
    index = NotesIndex(ss)
    active = [f for f in index.facts.values() if f.status == "active" and f.note_id in index.by_id]
    new_facts = [f for f in active if f.consolidated_in is None]
    by_note: dict[str, list[Fact]] = {}
    for f in active:
        by_note.setdefault(f.note_id, []).append(f)

    questions: list[Question] = []
    probes: list[Question] = []
    nearmiss: list[Question] = []
    for fact in new_facts:
        questions += generator.questions(fact, by_note[fact.note_id], gen.questions_per_fact)
        probes += generator.probes(fact, gen.probes_per_fact)
        near = generator.nearmiss(fact, gen.nearmiss_per_fact)
        half = max(1, len(near) // 2) if len(near) > 1 else len(near)
        nearmiss += near[:half]
        probes += near[half:]
    # A correction's own prompt is a train question for its facts.
    for note in index.notes:
        if note.kind == "correction" and note.prompt and any(f.note_id == note.id for f in new_facts):
            ids = [f.id for f in by_note.get(note.id, [])]
            questions.append(Question(note.prompt, "correction", ids))
    pool = [r for r in ss.pool() if r.get("active", True)]
    train_texts = [q.text for q in questions] + [r["q"] for r in pool]
    held_out = [p for p in probes if p.kind == "nearmiss"]
    probes = decontaminate([p for p in probes if p.kind != "nearmiss"], train_texts, gen.probe_max_jaccard) + held_out

    system_prompts = config.model.system_prompts
    neighbours, threshold = config.notes.teacher_neighbours, config.notes.relevance_threshold

    def new_row(q: Question) -> Row:
        student = _messages(q.text, system_prompts, rng)
        return Row(
            "new", student, with_notes(student, index.for_facts(q.fact_ids, q.text, neighbours)), PARENT, q.fact_ids
        )

    def replay_row(entry: dict) -> Row:
        student = _messages(entry["q"], system_prompts, rng)
        if recipe.replay_teacher == "self":
            return Row("replay", student, student, PREVIOUS, entry["fact_ids"])
        notes = index.for_facts(entry["fact_ids"], entry["q"], neighbours)
        return Row("replay", student, with_notes(student, notes), PARENT, entry["fact_ids"])

    def general_row(kind: str, prompt) -> Row:
        student = _messages(prompt, system_prompts, rng)
        text = prompt if isinstance(prompt, str) else " ".join(str(m.get("content", "")) for m in prompt)
        if recipe.anchor_teacher == "base":
            return Row(kind, student, student, PARENT)
        if recipe.anchor_teacher == "previous":
            return Row(kind, student, student, PREVIOUS)
        notes = index.relevant(text, threshold, neighbours)
        if kind == "anchor":
            notes = [n for n in index.notes if n in notes or n.kind == "instruction"]
        return Row(kind, student, with_notes(student, notes), PARENT)

    # Replay: about visits_per_fact rows per fact with stored prompts, drawn from the pool. A prompt of a
    # superseded fact counts for its successor (and is answered from the successor's note).
    replay: list[Row] = []
    prompts_of: dict[str, list[dict]] = {}
    for entry in pool:
        for fact in index.resolve(entry.get("fact_ids", [])):
            prompts_of.setdefault(fact.id, []).append(entry)
    for fact_id in sorted(prompts_of):
        entries = prompts_of[fact_id]
        for entry in rng.sample(entries, min(len(entries), config.replay.visits_per_fact)):
            replay.append(replay_row(entry))

    rows: list[Row] = []
    anchors = anchor_prompts(config)
    mix = recipe.mix
    per_pass_new = len(questions)
    n_near = round(per_pass_new * mix.get("nearmiss", 0) / mix["new"]) if nearmiss else 0
    n_anchor = min(recipe.max_anchor_rows, round(per_pass_new * mix.get("anchor", 0) / mix["new"])) if anchors else 0
    if not questions and replay:
        n_anchor = min(
            recipe.max_anchor_rows, round(len(replay) * mix.get("anchor", 0) / max(mix.get("replay", 0), 1e-9))
        )
    for _ in range(recipe.passes if questions else 1):
        block = [new_row(q) for q in questions]
        block += [general_row("nearmiss", nearmiss[i % len(nearmiss)].text) for i in range(n_near)]
        block += [general_row("anchor", anchors[rng.randrange(len(anchors))]) for _ in range(n_anchor)]
        rng.shuffle(block)
        rows += block
    if replay:
        rng.shuffle(replay)
        # Replay rows are spread evenly over the passes rather than repeated in each one.
        merged, r = [], 0
        for i, row in enumerate(rows):
            while r < len(replay) and r * len(rows) <= i * len(replay):
                merged.append(replay[r])
                r += 1
            merged.append(row)
        rows = merged + replay[r:]

    warnings = []
    steps = recipe.steps or math.ceil(len(rows) / recipe.rows_per_step)
    if steps > recipe.max_steps_per_session:
        warnings.append(
            f"{steps} steps exceed recipe.max_steps_per_session ({recipe.max_steps_per_session}): the session is cut, "
            "so later rows get fewer passes; split a large first ingestion into several sessions instead."
        )
        steps = recipe.max_steps_per_session
    return SessionPlan(new_facts, questions, probes, nearmiss, rows, steps if rows else 0, len(replay), warnings)
