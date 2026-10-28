"""Ingestion: documents into chunks, source diffs, atomic facts and the contradiction check.

A document source has a stable key (its path relative to the source root). Refreshing it diffs its
chunks against the stored notes of that source:

    same anchor, same hash        unchanged: the note keeps its id
    same anchor, new hash         a new note superseding the old one; its facts are diffed by
                                  (subject, attribute): unchanged facts keep their ids, changed values
                                  supersede the old fact and record the old value as stale
    anchor gone                   moved if a new chunk has the same text (token-set Jaccard >= 0.8),
                                  otherwise retired (no longer injected, taught or replayed)

An anchor is the chunk's heading path plus its ordinal under that heading. Every new or changed fact
is then compared, by the generator, with its BM25 neighbours among the active facts of the scope and
its ancestors. A likely conflict holds the note in the inbox (resolve: supersede, both or reject); a
conflict with a `locked` ancestor note (policy) rejects it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from palingenesis.learn.generate import FactDraft, Generator, jaccard
from palingenesis.learn.render import BM25
from palingenesis.learn.store import Fact, InboxItem, Note, ScopeStore, new_id, normalize, sha256_text

MOVED_JACCARD = 0.8
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass
class Chunk:
    anchor: str
    text: str
    # The source text's hash (the chunk's identity in diffs). Set when `text` was redacted after hashing.
    source_hash: str = ""

    @property
    def hash(self) -> str:
        return self.source_hash or sha256_text(self.text)


def chunk_document(text: str, max_chars: int = 1600, markdown: bool = True) -> list[Chunk]:
    """Chunks of at most about `max_chars` (paragraphs are not split), each starting with its heading path."""
    sections: list[tuple[list[str], list[str]]] = [([], [])]
    headings: list[str] = []
    paragraph: list[str] = []
    in_fence = False

    def flush():
        if paragraph and any(line.strip() for line in paragraph):
            sections[-1][1].append("\n".join(paragraph).strip())
        paragraph.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            paragraph.append(line)
            continue
        m = _HEADING.match(line) if markdown and not in_fence else None
        if m:
            flush()
            level = len(m.group(1))
            headings = headings[: level - 1] + [""] * max(0, level - 1 - len(headings)) + [m.group(2)]
            sections.append(([h for h in headings if h], []))
        elif not line.strip() and not in_fence:
            flush()
        else:
            paragraph.append(line)
    flush()
    chunks = []
    for path, paragraphs in sections:
        groups: list[list[str]] = []
        for p in paragraphs:
            if groups and sum(len(x) for x in groups[-1]) + len(p) <= max_chars:
                groups[-1].append(p)
            else:
                groups.append([p])
        for ordinal, group in enumerate(groups, 1):
            head = " > ".join(path)
            body = "\n\n".join(group)
            chunks.append(Chunk(f"{head}#{ordinal}", f"# {head}\n{body}" if head else body))
    return chunks


@dataclass
class CommitResult:
    status: str  # stored | held | rejected | unchanged
    note_id: str = ""
    inbox_id: str = ""
    reason: str = ""
    conflicts: list = field(default_factory=list)


@dataclass
class IngestReport:
    unchanged: int = 0
    changed: int = 0
    moved: int = 0
    new: int = 0
    retired: int = 0
    held: list = field(default_factory=list)  # inbox ids (conflicts, PII reviews)
    rejected: list = field(default_factory=list)  # reasons

    def add(self, result: CommitResult, kind: str) -> None:
        if result.status == "stored":
            setattr(self, kind, getattr(self, kind) + 1)
        elif result.status == "held":
            self.held.append(result.inbox_id)
        elif result.status == "rejected":
            self.rejected.append(result.reason)

    def summary(self) -> str:
        return (
            f"{self.new} new, {self.changed} changed, {self.moved} moved, {self.unchanged} unchanged, "
            f"{self.retired} retired, {len(self.held)} held in the inbox, {len(self.rejected)} rejected"
        )


def resolve_fact(fact_id: str, facts: dict[str, Fact]) -> Fact | None:
    """The active fact that `fact_id` is now: itself, or the end of its supersede chain (None if retired)."""
    seen = set()
    fact = facts.get(fact_id)
    while fact is not None and fact.status == "superseded" and fact.superseded_by and fact.id not in seen:
        seen.add(fact.id)
        fact = facts.get(fact.superseded_by)
    return fact if fact is not None and fact.status == "active" else None


def refresh_pool(ss: ScopeStore) -> None:
    """Replay prompts stay active while one of their facts resolves to an active fact (a superseded fact
    resolves to its successor, whose note the teacher then sees); otherwise they are switched off."""
    pool = ss.pool()
    if not pool:
        return
    facts = {f.id: f for f in ss.facts()}
    changed = False
    for row in pool:
        active = any(resolve_fact(fid, facts) is not None for fid in row.get("fact_ids", []))
        if row.get("active", True) != active:
            row["active"] = active
            changed = True
    if changed:
        ss.write_pool(pool)


def _fact_text(f) -> str:
    return f"{f.subject} {f.attribute} {f.value}"


def find_conflicts(
    drafts: list[FactDraft],
    ss: ScopeStore,
    ancestors: list[ScopeStore],
    generator: Generator,
    exclude_notes: set[str],
    neighbours: int,
) -> list[dict]:
    """Likely conflicts of `drafts` with the active facts of the scope and its ancestors."""
    candidates: list[tuple[ScopeStore, Fact]] = []
    for store in [*ancestors, ss]:
        candidates += [(store, f) for f in store.active_facts() if f.note_id not in exclude_notes]
    if not candidates or not drafts:
        return []
    index = BM25([_fact_text(f) for _, f in candidates])
    locked = {}
    for store in ancestors:
        locked.update({n.id: n.locked for n in store.active_notes()})
    out = []
    for i, draft in enumerate(drafts):
        for j in index.top(_fact_text(draft), neighbours):
            store, fact = candidates[j]
            if generator.contradicts(draft, fact):
                out.append(
                    {
                        "draft": i,
                        "fact_id": fact.id,
                        "scope": str(store.scope),
                        "value": fact.value,
                        "locked": bool(store is not ss and locked.get(fact.note_id, False)),
                    }
                )
    return out


def commit_note(
    ss: ScopeStore,
    ancestors: list[ScopeStore],
    generator: Generator,
    note: Note,
    drafts: list[FactDraft] | None = None,
    predecessor: str | None = None,
    resolution: str | None = None,
    neighbours: int = 8,
) -> CommitResult:
    """Store `note` and its facts in the scope, after the contradiction check.

    `predecessor`: the id of the note it replaces (a refreshed chunk); its facts are diffed by
    (subject, attribute). `resolution` settles the conflicts found: supersede (the conflicting facts of
    this scope become stale values), both, or reject. Without one, a conflict holds the note in the inbox.
    """
    notes = ss.notes()
    facts = ss.facts()
    by_id = {n.id: n for n in notes}
    pred = by_id.get(predecessor) if predecessor else None
    if pred is not None and pred.status != "active":
        pred = None
    if drafts is None:
        drafts = generator.extract_facts(note.text)
    pred_facts = {f.key: f for f in facts if pred is not None and f.note_id == pred.id and f.status == "active"}
    reused, changed, fresh = [], [], []
    for draft in drafts:
        old = pred_facts.get(draft.key)
        if old is not None and normalize(old.value) == normalize(draft.value):
            reused.append((draft, old))
        elif old is not None:
            changed.append((draft, old))
        else:
            fresh.append(draft)
    checked = [d for d, _ in changed] + fresh
    exclude = {note.id} | ({pred.id} if pred is not None else set())
    conflicts = find_conflicts(checked, ss, ancestors, generator, exclude, neighbours)
    if any(c["locked"] for c in conflicts):
        locked = next(c for c in conflicts if c["locked"])
        return CommitResult("rejected", reason=f"contradicts a locked note of {locked['scope']} ({locked['fact_id']})")
    if resolution == "reject":
        return CommitResult("rejected", reason="rejected by the reviewer")
    if conflicts and resolution is None:
        item = InboxItem(
            new_id("i"),
            "conflict",
            {
                "note": note.to_dict(),
                "drafts": [d.__dict__ for d in checked],
                "kept": [d.__dict__ for d, _ in reused],
                "predecessor": pred.id if pred is not None else None,
                "conflicts": conflicts,
            },
            reason="; ".join(f"{c['scope']} {c['fact_id']} says {c['value']!r}" for c in conflicts),
            origin=note.origin,
        )
        ss.write_inbox(ss.inbox() + [item])
        return CommitResult("held", inbox_id=item.id, conflicts=conflicts)

    fact_by_id = {f.id: f for f in facts}
    new_facts: list[Fact] = []
    for _, old in reused:
        old.note_id = note.id
    for draft, old in changed:
        fact = Fact(
            new_id("f"),
            note.id,
            draft.subject,
            draft.attribute,
            draft.value,
            draft.sentence,
            stale_values=[*old.stale_values, old.value],
            supersedes=old.id,
        )
        old.status, old.superseded_by = "superseded", fact.id
        new_facts.append(fact)
    draft_fact: dict[int, Fact] = {}
    for i, draft in enumerate(checked):
        if i >= len(changed):
            fact = Fact(new_id("f"), note.id, draft.subject, draft.attribute, draft.value, draft.sentence)
            new_facts.append(fact)
        draft_fact[i] = new_facts[i]
    kept = {old.id for _, old in reused}
    for old in pred_facts.values():
        if old.id not in kept and old.status == "active":
            old.status = "retired"
    if resolution == "supersede":
        for c in conflicts:
            fact = draft_fact[c["draft"]]
            if c["value"] not in fact.stale_values:
                fact.stale_values.append(c["value"])
            old = fact_by_id.get(c["fact_id"])
            if old is not None and c["scope"] == str(ss.scope):
                old.status, old.superseded_by = "superseded", fact.id
                fact.supersedes = fact.supersedes or old.id
    if pred is not None:
        pred.status, pred.superseded_by = "superseded", note.id
        note.supersedes = pred.id
    note.status = "active"
    ss.write_notes([n for n in notes if n.id != note.id] + [note])
    ss.write_facts(facts + new_facts)
    refresh_pool(ss)
    return CommitResult("stored", note_id=note.id, conflicts=conflicts)


def retire_note(ss: ScopeStore, note_id: str) -> None:
    notes, facts = ss.notes(), ss.facts()
    for n in notes:
        if n.id == note_id:
            n.status = "retired"
    for f in facts:
        if f.note_id == note_id and f.status == "active":
            f.status = "retired"
    ss.write_notes(notes)
    ss.write_facts(facts)
    refresh_pool(ss)


def apply_source(
    ss: ScopeStore,
    ancestors: list[ScopeStore],
    generator: Generator,
    source_key: str,
    chunks: list[Chunk],
    screen,
    origin: str = "",
    locked: bool = False,
    neighbours: int = 8,
) -> IngestReport:
    """Diff a refreshed source's chunks against its stored notes and apply the result.

    Exact matches come first: a chunk whose text is stored under its own anchor is unchanged, under
    another anchor it moved (the note keeps its id; ordinals shift when a paragraph is inserted). Then a
    chunk takes over the remaining note of its anchor (changed), or a gone note it nearly repeats
    (moved and changed), or is new. Notes left unmatched are retired.

    `screen(text) -> PIIResult` runs before a chunk is stored: a rejected chunk is skipped and a blocked
    one goes to the inbox for review; in both cases the stored version of that anchor stays active.
    """
    report = IngestReport()
    old = {n.anchor: n for n in ss.active_notes() if n.source_key == source_key}
    claimed: set[str] = set()  # ids of old notes matched by a chunk
    moves: dict[str, str] = {}  # note id -> new anchor
    pending: list[Chunk] = []
    for chunk in chunks:
        prev = old.get(chunk.anchor)
        if prev is not None and prev.hash == chunk.hash:
            claimed.add(prev.id)
            report.unchanged += 1
        else:
            pending.append(chunk)
    rest = []
    for chunk in pending:
        same = next((n for n in old.values() if n.hash == chunk.hash and n.id not in claimed), None)
        if same is not None:
            claimed.add(same.id)
            moves[same.id] = chunk.anchor
            report.moved += 1
        else:
            rest.append(chunk)
    if moves:
        notes = ss.notes()
        for n in notes:
            n.anchor = moves.get(n.id, n.anchor)
        ss.write_notes(notes)
    for chunk in rest:
        prev, kind = old.get(chunk.anchor), "changed"
        if prev is None or prev.id in claimed:
            unclaimed = [n for n in old.values() if n.id not in claimed and n.anchor not in {c.anchor for c in chunks}]
            prev = next((n for n in unclaimed if jaccard(n.text, chunk.text) >= MOVED_JACCARD), None)
            kind = "moved" if prev is not None else "new"
        if prev is not None:
            claimed.add(prev.id)
        verdict = screen(chunk.text)
        if verdict.action == "reject":
            report.rejected.append(f"{source_key} {chunk.anchor}: {verdict.reason()}")
            continue
        if verdict.action == "block":
            item = InboxItem(
                new_id("i"),
                "pii_review",
                {
                    "note": {
                        "kind": "document_chunk",
                        "text": chunk.text,
                        "source_key": source_key,
                        "anchor": chunk.anchor,
                        "hash": chunk.hash,
                        "locked": locked,
                    },
                    "predecessor": prev.id if prev is not None else None,
                },
                reason=verdict.reason(),
                origin=origin,
            )
            ss.write_inbox(ss.inbox() + [item])
            report.held.append(item.id)
            continue
        note = Note(
            new_id("n"),
            "document_chunk",
            verdict.text,
            source_key=source_key,
            anchor=chunk.anchor,
            hash=chunk.hash,
            locked=locked,
            origin=origin,
        )
        result = commit_note(
            ss, ancestors, generator, note, predecessor=prev.id if prev else None, neighbours=neighbours
        )
        report.add(result, kind)
    for note in old.values():
        if note.id not in claimed:
            retire_note(ss, note.id)
            report.retired += 1
    return report
