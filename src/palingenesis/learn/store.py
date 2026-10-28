"""The learn store: one directory per tenant (an org root), scopes below it.

    stores/acme/
      learn_state.yaml             org, base model, fingerprint of tokenizer / chat template / quantization,
                                   ranks, target-module regex, config snapshot
      LOCK                         fcntl lock: one mutating job (add, approve, forget, run, publish) at a time
      SPOOL.lock                   a short lock for the spools (signals captured while LOCK was held)
      admin_queue.jsonl            promotions waiting for an admin
      published.json               what is published where: per scope, its served versions (<scope>@v<n>),
                                   their composites, servers, canary results and history
      scopes/org/acme/
        sources.jsonl              {source_key, root, kind, last_seen_hash}
        notes.jsonl                {id, source_key, anchor, hash, kind, text, status, supersedes, locked, ...}
        facts.jsonl                {id, note_id, subject, attribute, value, sentence, stale_values, status, ...}
        inbox.jsonl                pending notes, document refreshes, conflicts, confirmations, PII reviews
        spool.jsonl                signals the proxy captured while a job held LOCK (PII-screened), stored by
                                   the next job
        tombstones.jsonl           forgotten items: {id, sha256, forgotten_at} only, never text
        sessions/s0001/            questions.jsonl, probes.jsonl, nearmiss.jsonl, config.yaml, train.json
        pool/replay.jsonl          replay prompts only: {id, q, fact_ids, session, active}
        adapters/v0001/            the scope's own adapter (PEFT format) + manifest.json
        LIVE                       the scope's live version
      scopes/team/acme/support/    same layout
      scopes/user/acme/support/ana/
      composites/<scope>/<id>/     a scope chain stacked into one adapter + manifest.json

Every JSONL file is rewritten atomically (temporary file, fsync, rename), never appended in place, so
a killed job leaves either the old or the new file and `forget` can remove text for good. A session
works in sessions/sNNNN.tmp/ and is renamed when it completes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from palingenesis.learn.config import LEVELS, LearnConfig

STATE_FILE = "learn_state.yaml"
STATE_FORMAT = 1
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class StoreError(Exception):
    """A store operation that cannot proceed (bad scope, lock held, fingerprint mismatch, ...)."""


class StoreLocked(StoreError):
    """The store's LOCK is held by another job."""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------- scopes


@dataclass(frozen=True)
class ScopeId:
    """org:acme | team:acme/support | user:acme/support/ana (or user:acme/ana, a user without a team)."""

    level: str
    path: tuple[str, ...]

    @classmethod
    def parse(cls, text: str | ScopeId) -> ScopeId:
        if isinstance(text, ScopeId):
            return text
        level, sep, rest = str(text).partition(":")
        parts = tuple(rest.split("/")) if rest else ()
        sizes = {"org": (1,), "team": (2,), "user": (2, 3)}
        if not sep or level not in LEVELS or len(parts) not in sizes[level] or not all(_NAME.match(p) for p in parts):
            raise StoreError(
                f"bad scope {text!r}: use org:<org>, team:<org>/<team> or user:<org>/<team>/<user> "
                "(names: letters, digits, '_', '.', '-')."
            )
        return cls(level, parts)

    def __str__(self) -> str:
        return f"{self.level}:{'/'.join(self.path)}"

    @property
    def org(self) -> str:
        return self.path[0]

    @property
    def parent(self) -> ScopeId | None:
        if self.level == "org":
            return None
        if self.level == "team" or len(self.path) == 2:
            return ScopeId("org", self.path[:1])
        return ScopeId("team", self.path[:2])

    def chain(self) -> list[ScopeId]:
        """Ancestors first, this scope last: [org, team, user]."""
        out, scope = [], self
        while scope is not None:
            out.append(scope)
            scope = scope.parent
        return out[::-1]

    @property
    def shared(self) -> bool:
        """Team and org scopes hold knowledge for several people (stricter PII policy, admin approval)."""
        return self.level != "user"

    @property
    def slug(self) -> str:
        return ".".join(self.path) if self.level != "org" else self.path[0]


# ---------------------------------------------------------------------------- records


@dataclass
class Record:
    @classmethod
    def from_dict(cls, raw: dict):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in names})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class Source(Record):
    source_key: str
    root: str = ""
    kind: str = "document"  # document | note
    last_seen_hash: str = ""


@dataclass
class Note(Record):
    """The unit stored, injected into prompts and forgotten: a document chunk or one statement."""

    id: str
    kind: str  # fact | document_chunk | correction | remembered | instruction
    text: str
    source_key: str = ""
    anchor: str = ""
    hash: str = ""
    status: str = "active"  # active | superseded | retired
    supersedes: str | None = None
    superseded_by: str | None = None
    locked: bool = False
    origin: str = ""
    prompt: str = ""  # corrections: the prompt that was answered wrong (a train question)
    created: str = field(default_factory=now)
    consolidated_in: str | None = None  # the adapter version that learned it


@dataclass
class Fact(Record):
    """The unit trained and evaluated: one (subject, attribute, value) of a note."""

    id: str
    note_id: str
    subject: str
    attribute: str
    value: str
    sentence: str = ""
    stale_values: list = field(default_factory=list)
    status: str = "active"  # active | superseded | retired
    supersedes: str | None = None
    superseded_by: str | None = None
    created: str = field(default_factory=now)
    consolidated_in: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return normalize(self.subject), normalize(self.attribute)


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).lower()).split())


@dataclass
class InboxItem(Record):
    """Something waiting for a person: kind note | source | conflict | confirm | pii_review."""

    id: str
    kind: str
    payload: dict
    reason: str = ""
    origin: str = ""
    created: str = field(default_factory=now)


# ---------------------------------------------------------------------------- files


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list) -> None:
    rows = [r.to_dict() if isinstance(r, Record) else r for r in rows]
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def remove_tree(path: Path) -> None:
    """Delete a directory file by file, then the empty directories (no recursive delete call)."""
    path = Path(path)
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if child.is_dir() and not child.is_symlink():
            child.rmdir()
        else:
            child.unlink()
    path.rmdir()


class ScopeStore:
    """The files of one scope."""

    def __init__(self, store: Store, scope: ScopeId):
        self.store = store
        self.scope = scope
        self.dir = store.root / "scopes" / scope.level / Path(*scope.path)

    def ensure(self) -> ScopeStore:
        for sub in ("sessions", "pool", "adapters"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        return self

    @property
    def exists(self) -> bool:
        return self.dir.is_dir()

    # records
    def sources(self) -> list[Source]:
        return [Source.from_dict(r) for r in read_jsonl(self.dir / "sources.jsonl")]

    def write_sources(self, rows: list[Source]) -> None:
        write_jsonl(self.dir / "sources.jsonl", rows)

    def notes(self) -> list[Note]:
        return [Note.from_dict(r) for r in read_jsonl(self.dir / "notes.jsonl")]

    def write_notes(self, rows: list[Note]) -> None:
        write_jsonl(self.dir / "notes.jsonl", rows)

    def facts(self) -> list[Fact]:
        return [Fact.from_dict(r) for r in read_jsonl(self.dir / "facts.jsonl")]

    def write_facts(self, rows: list[Fact]) -> None:
        write_jsonl(self.dir / "facts.jsonl", rows)

    def inbox(self) -> list[InboxItem]:
        return [InboxItem.from_dict(r) for r in read_jsonl(self.dir / "inbox.jsonl")]

    def write_inbox(self, rows: list[InboxItem]) -> None:
        write_jsonl(self.dir / "inbox.jsonl", rows)

    def pool(self) -> list[dict]:
        return read_jsonl(self.dir / "pool" / "replay.jsonl")

    def write_pool(self, rows: list[dict]) -> None:
        write_jsonl(self.dir / "pool" / "replay.jsonl", rows)

    def spool(self) -> list[dict]:
        return read_jsonl(self.dir / "spool.jsonl")

    def write_spool(self, rows: list[dict]) -> None:
        write_jsonl(self.dir / "spool.jsonl", rows)

    def tombstones(self) -> list[dict]:
        return read_jsonl(self.dir / "tombstones.jsonl")

    def write_tombstones(self, rows: list[dict]) -> None:
        write_jsonl(self.dir / "tombstones.jsonl", rows)

    def active_notes(self) -> list[Note]:
        return [n for n in self.notes() if n.status == "active"]

    def active_facts(self) -> list[Fact]:
        return [f for f in self.facts() if f.status == "active"]

    # sessions and adapter versions
    def next_number(self) -> int:
        numbers = [
            int(p.name[1:5]) for d in ("sessions", "adapters") for p in (self.dir / d).glob("[sv][0-9][0-9][0-9][0-9]*")
        ]
        return max(numbers, default=0) + 1

    def session_dirs(self) -> list[Path]:
        return sorted(p for p in (self.dir / "sessions").glob("s*") if p.is_dir())

    def versions(self) -> list[str]:
        return sorted(p.name for p in (self.dir / "adapters").glob("v[0-9]*") if (p / "manifest.json").exists())

    def version_dir(self, version: str) -> Path:
        return self.dir / "adapters" / version

    def manifest(self, version: str) -> dict:
        return json.loads((self.version_dir(version) / "manifest.json").read_text())

    def write_manifest(self, version: str, manifest: dict) -> None:
        atomic_write_text(self.version_dir(version) / "manifest.json", json.dumps(manifest, indent=2) + "\n")

    @property
    def live(self) -> str | None:
        path = self.dir / "LIVE"
        return path.read_text().strip() or None if path.exists() else None

    def set_live(self, version: str | None) -> None:
        atomic_write_text(self.dir / "LIVE", (version or "") + "\n")


class Store:
    """A tenant's store. Create with Store.init, open with Store.open."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        path = self.root / STATE_FILE
        if not path.exists():
            raise StoreError(f"{self.root} is not a learn store (no {STATE_FILE}); create it with `pgs learn init`.")
        self.state = yaml.safe_load(path.read_text()) or {}
        if self.state.get("format") != STATE_FORMAT:
            raise StoreError(f"{path}: unsupported store format {self.state.get('format')!r}.")
        self._lock_depth = 0
        self._lock_file = None

    @classmethod
    def init(cls, root: str | Path, base: str, org: str, config: LearnConfig | None = None) -> Store:
        root = Path(root)
        if (root / STATE_FILE).exists():
            raise StoreError(f"{root} already holds a learn store.")
        ScopeId.parse(f"org:{org}")
        config = config or LearnConfig()
        config.model.base = base or config.model.base
        if not config.model.base:
            raise StoreError("a store needs a base model (--base).")
        config.validate()
        root.mkdir(parents=True, exist_ok=True)
        state = {
            "format": STATE_FORMAT,
            "org": org,
            "base": config.model.base,
            "created": now(),
            # Filled by the first training run, then checked by every later one.
            "fingerprint": None,
            "ranks": dict(config.scopes.ranks),
            "target_regex": config.adapter.target_regex,
            "lineage_parent": None,
            "config": config.to_dict(),
        }
        atomic_write_text(root / STATE_FILE, yaml.safe_dump(state, sort_keys=False, allow_unicode=True))
        store = cls(root)
        store.scope(f"org:{org}").ensure()
        return store

    @classmethod
    def open(cls, root: str | Path) -> Store:
        return cls(root)

    @property
    def org(self) -> str:
        return self.state["org"]

    def config(self) -> LearnConfig:
        return LearnConfig.from_dict(self.state.get("config") or {}, where=str(self.root / STATE_FILE))

    def save_state(self) -> None:
        atomic_write_text(self.root / STATE_FILE, yaml.safe_dump(self.state, sort_keys=False, allow_unicode=True))

    def scope(self, scope: str | ScopeId) -> ScopeStore:
        scope = ScopeId.parse(scope)
        if scope.org != self.org:
            # Tenants never share a store: paths, pools and caches stay per org.
            raise StoreError(f"scope {scope} belongs to org {scope.org!r}; this store holds org {self.org!r} only.")
        return ScopeStore(self, scope)

    def scopes(self) -> list[ScopeId]:
        out = []
        for level in LEVELS:
            base = self.root / "scopes" / level
            for notes_dir in base.rglob("*"):
                if notes_dir.is_dir() and (notes_dir / "sessions").is_dir():
                    out.append(ScopeId(level, notes_dir.relative_to(base).parts))
        return sorted(out, key=lambda s: (LEVELS.index(s.level), s.path))

    def children(self, scope: ScopeId) -> list[ScopeId]:
        return [s for s in self.scopes() if s != scope and scope in s.chain()]

    def admin_queue(self) -> list[dict]:
        return read_jsonl(self.root / "admin_queue.jsonl")

    def write_admin_queue(self, rows: list[dict]) -> None:
        write_jsonl(self.root / "admin_queue.jsonl", rows)

    def published(self) -> dict:
        """published.json: {"format": 1, "scopes": {scope: {"live", "next", "versions", "history"}}}."""
        path = self.root / "published.json"
        if not path.exists():
            return {"format": 1, "scopes": {}}
        return json.loads(path.read_text())

    def write_published(self, data: dict) -> None:
        atomic_write_text(self.root / "published.json", json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    @contextlib.contextmanager
    def spool_lock(self, timeout: float = 10.0):
        """A short exclusive lock for reading and rewriting spools, held by the proxy (which cannot take LOCK
        while a job runs) and by the job that stores the spooled signals."""
        import fcntl

        with (self.root / "SPOOL.lock").open("a") as f:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise StoreError(f"{self.root}: the spool lock is held") from None
                    time.sleep(0.05)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def check_fingerprint(self, fingerprint: dict) -> None:
        """Record the base's fingerprint on the first run; refuse a later run on a different one."""
        stored = self.state.get("fingerprint")
        if stored is None:
            self.state["fingerprint"] = dict(fingerprint)
            self.save_state()
            return
        changed = sorted(k for k in set(stored) | set(fingerprint) if stored.get(k) != fingerprint.get(k))
        if changed:
            raise StoreError(
                f"the base differs from the one this store was trained on ({', '.join(changed)} changed). Adapters "
                "only fit the exact base, tokenizer, chat template and quantization they were trained on; rebuild "
                "the store on the new base instead."
            )

    @contextlib.contextmanager
    def lock(self, timeout: float = 0.0):
        """Exclusive store lock (re-entrant within this object). timeout 0 = fail at once if held."""
        if self._lock_depth:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return
        import fcntl

        f = (self.root / "LOCK").open("a")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    f.close()
                    raise StoreLocked(
                        f"{self.root} is locked by another learn job (add, approve, forget, run or publish)."
                    ) from None
                time.sleep(0.2)
        self._lock_depth, self._lock_file = 1, f
        try:
            yield
        finally:
            self._lock_depth = 0
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
            self._lock_file = None
