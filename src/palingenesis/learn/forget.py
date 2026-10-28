"""Forget: remove a note and everything derived from it from the store, for good.

    1. purge    the note's text and every derived item (its facts, their questions, probes and near-miss
                prompts in every session, replay prompts, inbox entries, admin-queue entries, stale values
                that came from it) from every file of the scope; files are rewritten atomically and
                tombstones.jsonl keeps only the id and a hash
    2. prune    every own adapter version trained while the note was active, or warm-started from such a
                version, loses its weights (the manifest stays, marked pruned), and so does every composite
                built from such a version; adapter directories a killed run left without a manifest are
                deleted; LIVE is cleared when it pointed to one; descendant scopes whose adapters were
                trained on top of a pruned version are marked for rebuild, and their inbox conflicts and
                stale values that name a forgotten fact lose them
    3. verify   every file under the store is searched (bytes) for the note's text and its facts'
                sentences (pieces of 12 characters or more), and every text file for its facts' subjects
                and values of any length as whole words; the report lists any file that still holds one
                and holds no text itself

Short pieces (a name, a value: "Kestrel", "42 kg") are searched as whole words, except trivial ones that
would match anything (under 3 characters, common words, numbers of fewer than 4 digits) and ones that
legitimately remain: stated by a note, fact or inbox item that is kept, or in the store's state and
manifests. Session rows (questions, probes, answers, diagnostics) that state such a piece are purged
with the note, and report.md is rendered again without them.

Rebuilding the scope's adapter from its remaining notes is a training run (`pgs learn run`); servers
unload pruned names when serving lands.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from palingenesis.learn.evaluate import trivial
from palingenesis.learn.report import render_again
from palingenesis.learn.store import (
    STATE_FILE,
    ScopeId,
    Store,
    normalize,
    now,
    read_jsonl,
    remove_tree,
    sha256_text,
    write_jsonl,
)

SESSION_FILES = (
    "questions.jsonl",
    "probes.jsonl",
    "nearmiss.jsonl",
    "abstain.jsonl",
    "answers.jsonl",
    "diagnostics.jsonl",
)
MIN_NEEDLE = 12  # texts are searched as substrings from this length; shorter pieces as whole words


@dataclass
class ForgetReport:
    scope: str
    ids: list
    notes: int = 0
    facts: int = 0
    inbox: int = 0
    session_rows: int = 0
    replay_rows: int = 0
    pruned_versions: list = field(default_factory=list)
    pruned_composites: list = field(default_factory=list)
    rebuild: list = field(default_factory=list)  # scopes whose adapter must be rebuilt
    unloaded: list = field(default_factory=list)  # published versions unloaded from servers (Learner.forget)
    leftovers: list = field(default_factory=list)  # files that still hold a forgotten text

    @property
    def ok(self) -> bool:
        return not self.leftovers

    def summary(self) -> str:
        status = (
            "verified: no trace left in the store"
            if self.ok
            else f"NOT clean: {len(self.leftovers)} file(s) still hold forgotten text"
        )
        return (
            f"forgot {self.notes} note(s), {self.facts} fact(s), {self.inbox} inbox item(s), {self.session_rows} session "
            f"row(s), {self.replay_rows} replay prompt(s); pruned {len(self.pruned_versions)} adapter version(s) and "
            f"{len(self.pruned_composites)} composite(s); rebuild needed: {', '.join(self.rebuild) or 'none'}; {status}"
        )


def _needles(texts: list[str]) -> list[bytes]:
    out = set()
    for text in texts:
        for piece in [text, *text.splitlines()]:
            piece = piece.strip()
            if len(piece) >= MIN_NEEDLE:
                out.add(piece)
                out.add(json.dumps(piece, ensure_ascii=False)[1:-1])  # as it appears inside JSON
                out.add(json.dumps(piece)[1:-1])
    return [n.encode() for n in out]


def _word_pattern(pieces: list[str]) -> re.Pattern | None:
    """Whole-word matcher of `pieces` as written, inside JSON (escaped) or plain text."""
    forms = set()
    for piece in pieces:
        forms |= {piece, json.dumps(piece)[1:-1], json.dumps(piece, ensure_ascii=False)[1:-1]}
    if not forms:
        return None
    alternatives = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.I)


def short_pieces(values: list[str], kept: str) -> list[str]:
    """The values worth a whole-word search: not trivial, and not stated by what the store keeps (`kept`)."""
    kept = f" {normalize(kept)} "
    out = []
    for value in values:
        value = str(value).strip()
        if value and not trivial(value) and f" {normalize(value)} " not in kept and value not in out:
            out.append(value)
    return out


def scan(root: Path, texts: list[str], pieces: list[str] | None = None) -> list[str]:
    """Files under `root` whose bytes contain any of `texts` (or a line of one), or whose text states one of
    `pieces` as a whole word (already screened by short_pieces; binary files are not searched for them)."""
    needles = _needles(texts)
    words = _word_pattern(pieces or [])
    if not needles and words is None:
        return []
    hits = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if any(n in data for n in needles):
            hits.append(str(path.relative_to(root)))
            continue
        if words is not None:
            try:
                text = data.decode()
            except UnicodeDecodeError:
                continue  # adapter weights: a short string matches random bytes
            if words.search(text):
                hits.append(str(path.relative_to(root)))
    return hits


def _kept_text(store: Store) -> str:
    """What legitimately stays after a forget: every scope's notes, facts, inbox and pool, the admin queue,
    the store state and the version manifests."""
    parts = [(store.root / STATE_FILE).read_text()]
    parts.append(json.dumps(store.admin_queue(), ensure_ascii=False))
    for s in store.scopes():
        ss = store.scope(s)
        parts += [n.text for n in ss.notes()]
        parts += [
            f"{f.subject} | {f.attribute} | {f.value} | {f.sentence} | {' | '.join(f.stale_values)}" for f in ss.facts()
        ]
        parts += [json.dumps(i.payload, ensure_ascii=False) for i in ss.inbox()]
        parts += [(ss.version_dir(v) / "manifest.json").read_text() for v in ss.versions()]
    return "\n".join(parts)


def _purge_sessions(cs, derived, report: ForgetReport) -> None:
    for session in cs.session_dirs():
        changed = False
        for name in SESSION_FILES:
            rows = read_jsonl(session / name)
            kept = [r for r in rows if not derived(r)]
            if len(kept) != len(rows):
                report.session_rows += len(rows) - len(kept)
                write_jsonl(session / name, kept)
                changed = True
        if changed:
            render_again(session)


def forget(store: Store, scope: str | ScopeId, ids: list[str]) -> ForgetReport:
    """Forget notes (n_...), pending inbox items (i_...) or spooled signals (s_...) of `scope`. Takes the store
    lock."""
    scope = ScopeId.parse(scope)
    ss = store.scope(scope)
    report = ForgetReport(str(scope), list(ids))
    with store.lock():
        wanted = set(ids)
        notes, facts, inbox = ss.notes(), ss.facts(), ss.inbox()
        # A note's predecessors in a supersede chain are versions of the same note: forget them too.
        by_id = {n.id: n for n in notes}
        gone_notes = set()
        for nid in [i for i in wanted if i in by_id]:
            while nid and nid not in gone_notes:
                gone_notes.add(nid)
                nid = by_id[nid].supersedes if nid in by_id else None
        spool = ss.spool()
        unknown = wanted - gone_notes - {i.id for i in inbox} - {r["id"] for r in spool}
        if unknown:
            raise KeyError(f"{scope}: no note, inbox item or spooled signal {sorted(unknown)}")
        texts = [by_id[n].text for n in gone_notes] + [r["text"] for r in spool if r["id"] in wanted]
        gone_facts = {f.id for f in facts if f.note_id in gone_notes}
        texts += [f.sentence for f in facts if f.id in gone_facts and f.sentence]
        gone_values = {f.value for f in facts if f.id in gone_facts}
        # Short pieces: the values (old ones included) and subjects of the forgotten facts.
        values = [v for f in facts if f.id in gone_facts for v in (f.value, f.subject, *f.stale_values)]
        kept_inbox = []
        for item in inbox:
            payload = json.dumps(item.payload, ensure_ascii=False)
            note = item.payload.get("note") or {}
            if item.id in wanted:
                texts.append(note.get("text", ""))
                texts += [d.get("sentence", "") for d in item.payload.get("drafts", [])]
                values += [
                    v for d in item.payload.get("drafts", []) for v in (d.get("value", ""), d.get("subject", ""))
                ]
                texts += [c.get("text", "") for c in item.payload.get("chunks", [])]  # a pending document
                report.inbox += 1
            elif note.get("id") in gone_notes or any(n in payload for n in gone_notes | gone_facts):
                report.inbox += 1
            else:
                kept_inbox.append(item)
        texts = [t for t in texts if t]

        # 1. purge
        tombstones = ss.tombstones()
        for nid in sorted(gone_notes):
            tombstones.append({"id": nid, "sha256": sha256_text(by_id[nid].text), "forgotten_at": now()})
        for item in inbox:
            if item not in kept_inbox:
                tombstones.append(
                    {
                        "id": item.id,
                        "sha256": sha256_text((item.payload.get("note") or {}).get("text", "")),
                        "forgotten_at": now(),
                    }
                )
        kept_facts = []
        for f in facts:
            if f.id in gone_facts:
                continue
            if f.supersedes in gone_facts:
                f.supersedes = None
            if f.superseded_by in gone_facts:  # its successor is gone: the old fact is no longer superseded by anything
                f.superseded_by, f.status = None, "retired" if f.status == "superseded" else f.status
            f.stale_values = [v for v in f.stale_values if v not in gone_values]
            kept_facts.append(f)
        kept_notes = []
        for n in notes:
            if n.id in gone_notes:
                continue
            if n.supersedes in gone_notes:
                n.supersedes = None
            if n.superseded_by in gone_notes:
                n.superseded_by, n.status = None, "retired" if n.status == "superseded" else n.status
            kept_notes.append(n)
        report.notes, report.facts = len(gone_notes), len(gone_facts)
        ss.write_notes(kept_notes)
        ss.write_facts(kept_facts)
        ss.write_inbox(kept_inbox)
        ss.write_tombstones(tombstones)
        queue = store.admin_queue()
        kept_queue = [r for r in queue if not (r.get("from_scope") == str(scope) and r.get("note_id") in gone_notes)]
        if len(kept_queue) != len(queue):
            report.inbox += len(queue) - len(kept_queue)
            store.write_admin_queue(kept_queue)

        # Descendants keep no copy of the note, but a conflict with one of its facts (held in a child's inbox)
        # and a child's fact that superseded it (stale values) still name its value.
        for child in store.children(scope):
            cs = store.scope(child)
            child_facts, changed = cs.facts(), False
            for f in child_facts:
                kept_values = [v for v in f.stale_values if v not in gone_values]
                if kept_values != f.stale_values:
                    f.stale_values, changed = kept_values, True
            if changed:
                cs.write_facts(child_facts)
            child_inbox, changed = cs.inbox(), False
            for item in child_inbox:
                conflicts = item.payload.get("conflicts")
                if item.kind == "conflict" and conflicts:
                    kept = [c for c in conflicts if c.get("fact_id") not in gone_facts]
                    if len(kept) != len(conflicts):
                        item.payload["conflicts"], changed = kept, True
                        item.reason = "; ".join(f"{c['scope']} {c['fact_id']} says {c['value']!r}" for c in kept)
                        item.reason = item.reason or "the conflicting note was forgotten: settle as both or reject"
            if changed:
                cs.write_inbox(child_inbox)

        # A piece also stated by what the store keeps (another note, the config) is not evidence of the note.
        pieces = short_pieces(values, _kept_text(store))
        words = _word_pattern(pieces)

        def derived(row: dict) -> bool:
            if set(row.get("fact_ids", [])) & gone_facts or set(row.get("note_ids", [])) & gone_notes:
                return True
            blob = json.dumps(row, ensure_ascii=False)
            if any(t in blob or json.dumps(t)[1:-1] in blob for t in texts if len(t) >= MIN_NEEDLE):
                return True
            return words is not None and bool(words.search(blob))

        # Signals the proxy spooled while a job held the lock: the forgotten ones, and any stating a forgotten text.
        with store.spool_lock():
            spool = ss.spool()
            kept_spool = [r for r in spool if r["id"] not in wanted and not derived(r)]
            if len(kept_spool) != len(spool):
                report.inbox += len(spool) - len(kept_spool)
                ss.write_spool(kept_spool)

        # The scope's files, and its descendants': their evaluations ask the ancestors' probes of the chain.
        for cs in [ss] + [store.scope(c) for c in store.children(scope)]:
            pool = cs.pool()
            kept_pool = [r for r in pool if not derived(r)]
            if len(kept_pool) != len(pool):
                report.replay_rows += len(pool) - len(kept_pool)
                cs.write_pool(kept_pool)
            _purge_sessions(cs, derived, report)

        # 2. prune adapter versions and composites that contain the note. A version warm-started from a pruned
        # one inherits its weights, so it is pruned too, even when the note was no longer active.
        adapters_dir = ss.dir / "adapters"
        for orphan in sorted(adapters_dir.iterdir()) if adapters_dir.exists() else []:
            if orphan.is_dir() and not (orphan / "manifest.json").exists():
                remove_tree(orphan)  # the weights of a killed run, never recorded as a version
                report.pruned_versions.append(orphan.name)
        pruned = set()
        for version in ss.versions():
            manifest = ss.manifest(version)
            if set(manifest.get("note_ids", [])) & gone_notes or manifest.get("warm_start") in pruned:
                for path in ss.version_dir(version).iterdir():
                    if path.is_dir():
                        remove_tree(path)
                    elif path.name != "manifest.json":
                        path.unlink()
                manifest["note_ids"] = [n for n in manifest["note_ids"] if n not in gone_notes]
                manifest["pruned"] = {"reason": "forget", "at": now()}
                ss.write_manifest(version, manifest)
                pruned.add(version)
                report.pruned_versions.append(version)
        if ss.live in pruned:
            ss.set_live(None)
        if pruned:
            # What a pruned version consolidated is no longer in any adapter: the rebuild trains it as new.
            kept_notes = [
                dataclasses.replace(n, consolidated_in=None) if n.consolidated_in in pruned else n for n in kept_notes
            ]
            kept_facts = [
                dataclasses.replace(f, consolidated_in=None) if f.consolidated_in in pruned else f for f in kept_facts
            ]
            ss.write_notes(kept_notes)
            ss.write_facts(kept_facts)
        if pruned or gone_notes:
            report.rebuild.append(str(scope))
        for child in store.children(scope):
            cs = store.scope(child)
            for version in cs.versions():
                parents = cs.manifest(version).get("parent_versions", {})
                if parents.get(str(scope)) in pruned and str(child) not in report.rebuild:
                    report.rebuild.append(str(child))
        composites = store.root / "composites"
        for manifest_path in sorted(composites.rglob("manifest.json")) if composites.exists() else []:
            chain = json.loads(manifest_path.read_text()).get("chain", [])
            if any(s == str(scope) and v in pruned for s, v in chain):
                remove_tree(manifest_path.parent)
                report.pruned_composites.append(str(manifest_path.parent.relative_to(store.root)))
        state = store.state.setdefault("rebuild", [])
        for s in report.rebuild:
            if s not in state:
                state.append(s)
        store.save_state()

        # 3. verify
        report.leftovers = scan(store.root, texts, pieces)
    return report
