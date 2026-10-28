"""Learner: the Python API over a store.

    learner = Learner.init("stores/acme", base="Qwen/Qwen3.5-4B", org="acme")
    org = learner.scope("org:acme")
    org.add_documents("acme-docs/**/*.md", approve=True)        # diffed against the stored source
    learner.remember("user:acme/support/ana", "my project is Orion")
    result = learner.run("org:acme")                            # trains and evaluates a new adapter version
    print(result.evaluation.summary())                          # paired against the live version
    learner.set_live("org:acme", result.version)                # refused when a gate failed (force=True)
    messages = learner.render_notes("user:acme/support/ana", messages)   # notes not yet consolidated
    learner.forget("user:acme/support/ana", ["n_..."])

Consent: notes and documents added without approve=True wait in the scope's inbox; explicit commands
and corrections (user scope only) are stored at once, unless they read like an instruction; promotions
to a team or org wait in the store's admin queue. The PII filter runs before anything is written.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from palingenesis.learn.config import LearnConfig, served_rank
from palingenesis.learn.forget import ForgetReport, forget
from palingenesis.learn.generate import FactDraft, Generator, ModelGenerator, StubGenerator, hf_complete
from palingenesis.learn.ingest import Chunk, CommitResult, IngestReport, apply_source, chunk_document, commit_note
from palingenesis.learn.pii import PIIFilter
from palingenesis.learn.render import with_notes
from palingenesis.learn.signals import correction_signal, detect_command, looks_like_instruction
from palingenesis.learn.store import (
    InboxItem,
    Note,
    ScopeId,
    ScopeStore,
    Source,
    Store,
    StoreError,
    new_id,
    now,
    read_jsonl,
    sha256_file,
    sha256_text,
)

NOTE_KINDS = ("fact", "correction", "remembered", "instruction")


class Scope:
    """A handle on one scope: learner.scope("team:acme/support").add_notes([...])."""

    def __init__(self, learner: Learner, scope: ScopeId):
        self.learner, self.id = learner, scope

    def __repr__(self) -> str:
        return f"Scope({self.id})"

    def __getattr__(self, name):
        method = getattr(self.learner, name)
        return lambda *args, **kwargs: method(self.id, *args, **kwargs)


class Learner:
    def __init__(
        self,
        store: Store,
        config: LearnConfig | None = None,
        generator: Generator | None = None,
        pii: PIIFilter | None = None,
    ):
        self.store = store
        self.config = config or store.config()
        self.config.validate()
        self._generator = generator
        self.pii = pii or PIIFilter.from_config(self.config.pii)

    @classmethod
    def init(cls, path: str | Path, base: str, org: str, config: LearnConfig | None = None, **kwargs) -> Learner:
        return cls(Store.init(path, base, org, config), **kwargs)

    @classmethod
    def open(cls, path: str | Path, config: LearnConfig | None = None, **kwargs) -> Learner:
        return cls(Store.open(path), config, **kwargs)

    # ------------------------------------------------------------------ helpers
    @property
    def generator(self) -> Generator:
        if self._generator is None:
            name = self.config.generate.generator
            if name == "stub":
                self._generator = StubGenerator()
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                model_id = self.store.state["base"] if name == "self" else name
                tok = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
                complete = hf_complete(
                    model, tok, self.config.generate.max_new_tokens, self.config.model.chat_template_kwargs
                )
                self._generator = ModelGenerator(complete)
        return self._generator

    def scope(self, scope: str | ScopeId) -> Scope:
        scope = ScopeId.parse(scope)
        for s in scope.chain():
            self.store.scope(s).ensure()
        return Scope(self, scope)

    def _stores(self, scope) -> tuple[ScopeStore, list[ScopeStore]]:
        scope = self.scope(scope).id
        return self.store.scope(scope), [self.store.scope(s) for s in scope.chain()[:-1]]

    def _commit(self, scope, note: Note, predecessor=None, resolution=None, drafts=None) -> CommitResult:
        ss, ancestors = self._stores(scope)
        return commit_note(
            ss,
            ancestors,
            self.generator,
            note,
            drafts=drafts,
            predecessor=predecessor,
            resolution=resolution,
            neighbours=self.config.notes.contradiction_neighbours,
        )

    def _hold(self, ss: ScopeStore, kind: str, payload: dict, reason: str, origin: str) -> CommitResult:
        item = InboxItem(new_id("i"), kind, payload, reason=reason, origin=origin)
        ss.write_inbox(ss.inbox() + [item])
        return CommitResult("held", inbox_id=item.id, reason=reason)

    def _screened_note(
        self, scope: ScopeId, kind: str, text: str, origin: str, approve: bool, locked=False, prompt="", reason=""
    ) -> CommitResult:
        """PII filter, then the scope's inbox (not approved) or the notes."""
        ss, _ = self._stores(scope)
        verdict = self.pii.check(text, shared=scope.shared)
        payload = {"note": {"kind": kind, "text": verdict.text, "locked": locked, "prompt": prompt, "origin": origin}}
        if verdict.action == "reject":
            return CommitResult("rejected", reason=f"PII filter: {verdict.reason()}")
        if verdict.action == "block":
            return self._hold(ss, "pii_review", payload, f"PII filter: {verdict.reason()}", origin)
        if not approve:
            return self._hold(ss, "confirm" if reason else "note", payload, reason or "waiting for approval", origin)
        note = Note(new_id("n"), kind, verdict.text, source_key=origin, locked=locked, origin=origin, prompt=prompt)
        return self._commit(scope, note)

    # ------------------------------------------------------------------ intake
    def add_notes(
        self, scope, rows: list, kind: str = "fact", approve: bool = False, locked: bool = False, origin: str = "cli"
    ) -> list[CommitResult]:
        """Notes from strings or {"text", "locked"?} rows; corrections take {"prompt", "correction"}.
        `approve`: the caller is the approver (the scope's user, or an admin for team / org)."""
        scope = ScopeId.parse(scope)
        if kind not in NOTE_KINDS:
            raise ValueError(f"note kind must be one of {NOTE_KINDS}, got {kind!r}")
        if kind == "instruction" and not scope.shared:
            raise StoreError("instruction notes (behaviour, policy) belong to a team or org scope, added by an admin.")
        if locked and not scope.shared:
            raise StoreError("only team and org notes can be locked (policy that lower scopes cannot contradict).")
        out = []
        with self.store.lock():
            for row in rows:
                row = {"text": row} if isinstance(row, str) else dict(row)
                if kind == "correction":
                    signal = correction_signal(
                        row.get("prompt", ""), row["correction"], self.config.signals.max_note_chars
                    )
                    out.append(
                        self._screened_note(scope, "correction", signal.text, origin, approve, prompt=signal.prompt)
                    )
                else:
                    out.append(
                        self._screened_note(scope, kind, row["text"], origin, approve, locked=row.get("locked", locked))
                    )
        return out

    def add_documents(
        self,
        scope,
        pattern: str,
        root: str | None = None,
        approve: bool = False,
        locked: bool = False,
        origin: str = "cli",
    ) -> IngestReport:
        """Register or refresh document sources (glob `pattern`, keys relative to `root`, default: cwd)."""
        scope = ScopeId.parse(scope)
        ss, ancestors = self._stores(scope)
        root_path = Path(root or ".").resolve()
        paths = sorted(
            Path(p).resolve()
            for p in glob.glob(str(root_path / pattern) if not Path(pattern).is_absolute() else pattern, recursive=True)
        )
        paths = [p for p in paths if p.is_file()]
        if not paths:
            raise StoreError(f"no document matches {pattern!r} under {root_path}")
        report = IngestReport()
        max_chars = self.config.notes.chunk_tokens * 4
        with self.store.lock():
            sources = {s.source_key: s for s in ss.sources()}
            for path in paths:
                key = path.relative_to(root_path).as_posix() if path.is_relative_to(root_path) else path.as_posix()
                digest = sha256_file(path)
                if key in sources and sources[key].last_seen_hash == digest:
                    report.unchanged += sum(1 for n in ss.active_notes() if n.source_key == key)
                    continue
                chunks = chunk_document(
                    path.read_text(errors="replace"), max_chars, markdown=path.suffix.lower() in (".md", ".markdown")
                )
                screened = []
                for chunk in chunks:
                    verdict = self.pii.check(chunk.text, shared=scope.shared)
                    if verdict.action == "reject":
                        report.rejected.append(f"{key} {chunk.anchor}: PII filter {verdict.reason()}")
                    else:
                        screened.append({"anchor": chunk.anchor, "text": verdict.text, "hash": chunk.hash})
                payload = {
                    "source_key": key,
                    "root": str(root_path),
                    "hash": digest,
                    "chunks": screened,
                    "locked": locked,
                }
                if not approve:
                    report.held.append(self._hold(ss, "source", payload, f"document {key}", origin).inbox_id)
                    continue
                self._apply_source(ss, ancestors, payload, origin, report)
        return report

    def _apply_source(self, ss: ScopeStore, ancestors, payload: dict, origin: str, report: IngestReport) -> None:
        chunks = [Chunk(c["anchor"], c["text"], c["hash"]) for c in payload["chunks"]]
        result = apply_source(
            ss,
            ancestors,
            self.generator,
            payload["source_key"],
            chunks,
            lambda text: self.pii.check(text, shared=ss.scope.shared),
            origin=origin,
            locked=payload.get("locked", False),
            neighbours=self.config.notes.contradiction_neighbours,
        )
        for name in ("unchanged", "changed", "moved", "new", "retired"):
            setattr(report, name, getattr(report, name) + getattr(result, name))
        report.held += result.held
        report.rejected += result.rejected
        sources = [s for s in ss.sources() if s.source_key != payload["source_key"]]
        ss.write_sources(sources + [Source(payload["source_key"], payload["root"], "document", payload["hash"])])

    def remember(self, scope, text: str, origin: str = "api") -> CommitResult:
        """An explicit "remember ..." of a user, already extracted (see intake)."""
        scope = ScopeId.parse(scope)
        if scope.shared:
            raise StoreError("remembered notes go to a user scope; promote them to a team or org.")
        text = text.strip()
        if not text or len(text) > self.config.signals.max_note_chars:
            return CommitResult(
                "rejected", reason=f"a remembered note is 1 to {self.config.signals.max_note_chars} characters"
            )
        reason = "reads like an instruction: confirm it" if looks_like_instruction(text) else ""
        with self.store.lock():
            return self._screened_note(scope, "remembered", text, origin, approve=not reason, reason=reason)

    def intake(self, scope, messages: list[dict], origin: str = "api") -> CommitResult | None:
        """Examine a conversation for an explicit command (newest user message only); None when there is none."""
        signal = detect_command(messages, self.config.signals.command_patterns, self.config.signals.max_note_chars)
        return None if signal is None else self.remember(scope, signal.text, origin)

    def correct(self, scope, prompt: str, correction: str, origin: str = "api") -> CommitResult:
        scope = ScopeId.parse(scope)
        if scope.shared:
            raise StoreError("corrections go to the user's scope; promote them to a team or org.")
        return self.add_notes(
            scope, [{"prompt": prompt, "correction": correction}], kind="correction", approve=True, origin=origin
        )[0]

    # ------------------------------------------------------------------ inbox and admin queue
    def inbox(self, scope) -> list[InboxItem]:
        return self.store.scope(scope).inbox()

    def admin_queue(self) -> list[dict]:
        return self.store.admin_queue()

    def promote(self, scope, note_id: str, to: str, origin: str = "cli") -> str:
        """Ask for a note of `scope` to be copied to an ancestor team / org scope (enters the admin queue)."""
        scope, target = ScopeId.parse(scope), ScopeId.parse(to)
        if target not in scope.chain()[:-1]:
            raise StoreError(f"{target} is not an ancestor of {scope}.")
        note = next((n for n in self.store.scope(scope).active_notes() if n.id == note_id), None)
        if note is None:
            raise KeyError(f"{scope}: no active note {note_id}")
        if note.kind == "instruction":
            raise StoreError("instruction notes are added by an admin at the target scope, not promoted.")
        with self.store.lock():
            item = {
                "id": new_id("p"),
                "note_id": note_id,
                "from_scope": str(scope),
                "to_scope": str(target),
                "created": now(),
                "origin": origin,
            }
            self.store.write_admin_queue(self.store.admin_queue() + [item])
        return item["id"]

    def approve(self, scope, ids: list[str], action: str = "approve") -> list[CommitResult]:
        """Settle inbox items of `scope` (approve | reject; conflicts: supersede | both | reject) or admin-queue
        promotions into `scope` (approve | reject)."""
        scope = ScopeId.parse(scope)
        if action not in ("approve", "reject", "supersede", "both"):
            raise ValueError(f"action must be approve, reject, supersede or both, got {action!r}")
        ss, ancestors = self._stores(scope)
        out = []
        with self.store.lock():
            for item_id in ids:
                if item_id.startswith("p_"):
                    out.append(self._settle_promotion(scope, item_id, action))
                    continue
                inbox = ss.inbox()
                item = next((i for i in inbox if i.id == item_id), None)
                if item is None:
                    raise KeyError(f"{scope}: no inbox item {item_id}")
                if item.kind == "conflict" and action == "approve":
                    raise ValueError(f"{item_id} is a conflict: settle it as supersede, both or reject")
                ss.write_inbox([i for i in inbox if i.id != item_id])
                if action == "reject":
                    out.append(CommitResult("rejected", reason="rejected by the reviewer"))
                    continue
                payload = item.payload
                if item.kind == "source":
                    report = IngestReport()
                    self._apply_source(ss, ancestors, payload, item.origin, report)
                    out.append(CommitResult("stored", reason=report.summary()))
                    continue
                spec = payload["note"]
                if item.kind == "conflict":
                    note = Note.from_dict(spec)
                    drafts = [FactDraft(**d) for d in payload.get("kept", []) + payload["drafts"]]
                    out.append(self._commit(scope, note, payload.get("predecessor"), action, drafts))
                    continue
                note = Note(
                    new_id("n"),
                    spec["kind"],
                    spec["text"],
                    source_key=spec.get("source_key", item.origin),
                    anchor=spec.get("anchor", ""),
                    hash=spec.get("hash", ""),
                    locked=spec.get("locked", False),
                    origin=spec.get("origin", item.origin),
                    prompt=spec.get("prompt", ""),
                )
                out.append(self._commit(scope, note, payload.get("predecessor")))
        return out

    def _settle_promotion(self, scope: ScopeId, item_id: str, action: str) -> CommitResult:
        queue = self.store.admin_queue()
        item = next((r for r in queue if r["id"] == item_id), None)
        if item is None:
            raise KeyError(f"no admin-queue item {item_id}")
        if item["to_scope"] != str(scope):
            raise StoreError(f"{item_id} promotes into {item['to_scope']}, not {scope}")
        self.store.write_admin_queue([r for r in queue if r["id"] != item_id])
        if action == "reject":
            return CommitResult("rejected", reason="rejected by the admin")
        source = next((n for n in self.store.scope(item["from_scope"]).active_notes() if n.id == item["note_id"]), None)
        if source is None:
            return CommitResult(
                "rejected", reason=f"note {item['note_id']} is no longer active in {item['from_scope']}"
            )
        # The admin approved it; the PII filter runs again at the stricter shared level.
        origin = f"promoted:{item['from_scope']}/{source.id}"
        return self._screened_note(scope, source.kind, source.text, origin, approve=True, prompt=source.prompt)

    # ------------------------------------------------------------------ forget, train, serve
    def forget(self, scope, ids: list[str]) -> ForgetReport:
        return forget(self.store, scope, ids)

    def run(self, scope, model=None, tokenizer=None, sampler=None, evaluate: bool = True):
        """Train the scope's adapter on its unconsolidated facts (plus replay and anchors): a new version,
        evaluated against the live one unless evaluate=False (result.evaluation)."""
        from palingenesis.learn.trainer import SessionRunner

        scope = self.scope(scope).id
        runner = SessionRunner(self, scope, self.generator, model=model, tokenizer=tokenizer, sampler=sampler)
        return runner.run(evaluate=evaluate)

    def evaluate(self, scope, version: str | None = None, model=None, tokenizer=None, sampler=None, decline=None):
        """Evaluate `version` (default: the newest) against the scope's live version; writes the session's
        report.json / report.md / diagnostics.jsonl and the manifest's evaluation entry."""
        from palingenesis.learn.evaluate import Evaluator
        from palingenesis.learn.trainer import SessionRunner

        scope = self.scope(scope).id
        ss = self.store.scope(scope)
        versions = [v for v in ss.versions() if not ss.manifest(v).get("pruned")]
        version = version or (versions[-1] if versions else None)
        if version is None or version not in ss.versions():
            raise StoreError(f"{scope}: no adapter version {version or ''} to evaluate".replace("  ", " "))
        with self.store.lock():
            runner = SessionRunner(self, scope, None, model=model, tokenizer=tokenizer, sampler=sampler)
            runner._load()
            try:
                runner.load_version(version)
                return Evaluator(runner, decline=decline).run(version)
            finally:
                runner.release()

    def gate_check(self, scope, version: str) -> list[str]:
        """Why `version` may not go live: not evaluated, a failed gate, or an evaluation against a chain that
        is no longer live. Empty when it may."""
        from palingenesis.learn.evaluate import chain_versions, load_report

        scope = ScopeId.parse(scope)
        ss = self.store.scope(scope)
        report = load_report(ss, version)
        if report is None:
            return [f"{version} was not evaluated (pgs learn eval)"]
        reasons = []
        failed = [g["name"] for g in report["gates"] if not g["passed"]]
        if failed:
            reasons.append(f"gates failed: {', '.join(failed)}")
        if report["live_chain"] != chain_versions(self.store, scope):
            reasons.append(
                f"evaluated against {report['live_chain']}, but the live chain is now "
                f"{chain_versions(self.store, scope)}: evaluate it again"
            )
        return reasons

    def set_live(self, scope, version: str, force: bool = False) -> None:
        """Make `version` the scope's live adapter: its facts count as consolidated, its train questions join
        the replay pool. Refused unless its evaluation against the current live chain passed every gate;
        force=True overrides, and the manifest records the forced publish and why."""
        ss, _ = self._stores(scope)
        scope = ss.scope
        with self.store.lock():
            manifest = ss.manifest(version)
            if manifest.get("pruned"):
                raise StoreError(f"{version} was pruned ({manifest['pruned']['reason']}); train a new version.")
            reasons = self.gate_check(scope, version)
            if reasons and not force:
                raise StoreError(f"{scope} {version} cannot go live: {'; '.join(reasons)} (force to override).")
            manifest["gated"] = not reasons
            manifest["forced"] = bool(reasons)
            manifest["forced_reasons"] = reasons
            manifest["live_at"] = now()
            ss.write_manifest(version, manifest)
            rebuild = self.store.state.get("rebuild") or []
            if str(scope) in rebuild:
                self.store.state["rebuild"] = [s for s in rebuild if s != str(scope)]
                self.store.save_state()
            notes, facts = ss.notes(), ss.facts()
            note_ids, fact_ids = set(manifest["note_ids"]), set(manifest["fact_ids"])
            for n in notes:
                if n.id in note_ids and n.consolidated_in is None:
                    n.consolidated_in = version
            for f in facts:
                if f.id in fact_ids and f.consolidated_in is None:
                    f.consolidated_in = version
            ss.write_notes(notes)
            ss.write_facts(facts)
            pool = ss.pool()
            known = {r["q"] for r in pool}
            for row in read_jsonl(ss.dir / "sessions" / manifest["session"] / "questions.jsonl"):
                if row["q"] not in known:
                    pool.append(
                        {
                            "id": new_id("q"),
                            "q": row["q"],
                            "fact_ids": row["fact_ids"],
                            "session": manifest["session"],
                            "active": True,
                        }
                    )
                    known.add(row["q"])
            ss.write_pool(_cap_pool(pool, self.config.replay.max_prompts_per_fact))
            ss.set_live(version)

    # ------------------------------------------------------------------ plan and status
    def _timing(self) -> dict:
        """Seconds per training row and per evaluation answer, measured by the newest session of this store."""
        newest_train, newest_eval = None, None
        for s in self.store.scopes():
            ss = self.store.scope(s)
            for session in ss.session_dirs():
                train, report = session / "train.json", session / "report.json"
                if train.exists() and (newest_train is None or train.stat().st_mtime > newest_train.stat().st_mtime):
                    newest_train = train
                if report.exists() and (newest_eval is None or report.stat().st_mtime > newest_eval.stat().st_mtime):
                    newest_eval = report
        out = {"seconds_per_row": None, "seconds_per_answer": None, "measured": None}
        if newest_train is not None:
            data = json.loads(newest_train.read_text())
            rows = data.get("rows_processed") or len(data.get("losses", [])) * self.config.recipe.rows_per_step
            if rows and data.get("seconds"):
                out["seconds_per_row"] = data["seconds"] / rows
                out["measured"] = str(newest_train.parent.relative_to(self.store.root))
        if newest_eval is not None:
            data = json.loads(newest_eval.read_text())
            if data.get("generated_answers") and data.get("seconds"):
                out["seconds_per_answer"] = data["seconds"] / data["generated_answers"]
        return out

    def _chain_ranks(self, scope: ScopeId) -> list[int]:
        ranks = []
        for s in scope.chain():
            ss = self.store.scope(s)
            if s != scope and ss.exists and ss.live is not None:
                ranks.append(ss.manifest(ss.live)["rank"])
            elif s == scope:
                ranks.append(self.config.scopes.ranks[s.level])
        return ranks

    def plan(self, scope=None) -> list[dict]:
        """What the next run of each scope (or of `scope`) would train and evaluate, and an estimated cost from
        the times measured on this store. Nothing is generated or written."""
        from palingenesis.learn.config import LearnConfigError
        from palingenesis.learn.evaluate import load_drift_prompts, load_general_qa
        from palingenesis.learn.recipe import estimate_session

        scopes = [ScopeId.parse(scope)] if scope else self.store.scopes()
        timing = self._timing()
        cfg, ev = self.config, self.config.eval
        out = []
        for s in scopes:
            ss = self.store.scope(s)
            est = estimate_session(ss, cfg)
            ranks = self._chain_ranks(s)
            est["chain_ranks"] = ranks
            try:
                est["served_rank"] = served_rank(ranks)
            except LearnConfigError as e:
                est["served_rank"] = None
                est["warnings"].append(str(e))
            held_out_near = max(0, cfg.generate.nearmiss_per_fact - max(1, cfg.generate.nearmiss_per_fact // 2))
            ancestors = sum(
                len([f for f in self.store.scope(a).active_facts() if f.consolidated_in])
                for a in s.chain()[:-1]
                if self.store.scope(a).exists
            )
            probes = min(est["probes"], ev.max_probes_per_session)
            closed = (
                probes
                + min(est["consolidated_facts"] * cfg.generate.probes_per_fact, ev.max_retention_probes)
                + min(ancestors * cfg.generate.probes_per_fact, ev.max_ancestor_probes)
                + len(load_general_qa(ev.general_qa_path))
                + est["new_facts"] * held_out_near
                + min(cfg.generate.abstain_per_session, 2 * est["new_facts"])
            )
            drift = min(len(load_drift_prompts(ev.drift_path)), ev.max_drift_prompts)
            est["eval_answers"] = 2 * closed + probes + drift if est["steps"] else 0
            minutes = None
            if timing["seconds_per_row"] is not None:
                seconds = est["steps"] * cfg.recipe.rows_per_step * timing["seconds_per_row"]
                seconds += est["eval_answers"] * (timing["seconds_per_answer"] or 0.0)
                minutes = round(seconds / 60, 1)
            est["minutes"] = minutes
            est["measured_on"] = timing["measured"]
            out.append(est)
        return out

    def status(self, scope=None) -> dict:
        """The scope tree with live versions, gate states, inbox / queue counts and pending rebuilds;
        `issues` lists what needs a person (a pending rebuild, a forced or ungated live version)."""
        from palingenesis.learn.config import LearnConfigError

        root = ScopeId.parse(scope) if scope else None
        queue = self.store.admin_queue()
        rebuild = list(self.store.state.get("rebuild") or [])
        entries, issues = [], []
        for s in self.store.scopes():
            if root is not None and root not in s.chain():
                continue
            ss = self.store.scope(s)
            facts = ss.facts()
            versions = ss.versions()
            live = ss.live
            entry = {
                "scope": str(s),
                "parent": str(s.parent) if s.parent else None,
                "live": live,
                "live_state": None,
                "notes": len(ss.active_notes()),
                "facts": sum(1 for f in facts if f.status == "active"),
                "unconsolidated": sum(1 for f in facts if f.status == "active" and f.consolidated_in is None),
                "inbox": len(ss.inbox()),
                "admin_queue": sum(1 for r in queue if r.get("to_scope") == str(s)),
                "versions": len(versions),
                "latest": versions[-1] if versions else None,
                "latest_evaluation": None,
                "rebuild": str(s) in rebuild,
                "served_rank": None,
                "retention_trend": [],
            }
            if live is not None:
                manifest = ss.manifest(live)
                entry["live_state"] = (
                    "forced" if manifest.get("forced") else "gated" if manifest.get("gated") else "ungated"
                )
                if manifest.get("forced"):
                    issues.append(
                        f"{s}: live {live} was forced past: {'; '.join(manifest.get('forced_reasons') or [])}"
                    )
                elif not manifest.get("gated"):
                    issues.append(f"{s}: live {live} went live without passing gates")
            if versions:
                evaluation = ss.manifest(versions[-1]).get("evaluation")
                if evaluation:
                    entry["latest_evaluation"] = {"passed": evaluation["passed"], "failed": evaluation["failed"]}
            for session in ss.session_dirs():
                path = session / "report.json"
                if path.exists():
                    paired = json.loads(path.read_text())["metrics"]["retention"]["paired"]
                    if paired["n"]:
                        entry["retention_trend"].append([session.name, paired["net_loss"]])
            try:
                entry["served_rank"] = served_rank(self._chain_ranks(s))
            except LearnConfigError as e:
                issues.append(f"{s}: {e}")
            if entry["rebuild"]:
                issues.append(
                    f"{s}: rebuild pending (forget pruned its adapter or the parent version it was trained on)"
                )
            entries.append(entry)
        return {
            "org": self.store.org,
            "base": self.store.state["base"],
            "scopes": entries,
            "admin_queue": len(queue),
            "rebuild": rebuild,
            "issues": issues,
            "healthy": not issues,
        }

    def stack(self, scope, out: str | Path | None = None, pad: bool = True) -> Path:
        """The scope chain's live adapters stacked into one PEFT adapter (exact; padded to a vLLM rank)."""
        from palingenesis.learn import adapter as adapters

        scope = ScopeId.parse(scope)
        chain, paths = [], []
        for s in scope.chain():
            ss = self.store.scope(s)
            if ss.live is not None:
                chain.append((str(s), ss.live))
                paths.append(ss.version_dir(ss.live))
        if not paths:
            raise StoreError(f"no live adapter in the chain of {scope}")
        ranks = [self.store.scope(s).manifest(v)["rank"] for s, v in chain]
        rank = served_rank(ranks) if pad else sum(ranks)
        sha = "".join(self.store.scope(s).manifest(v)["sha256"] for s, v in chain)
        composite = sha256_text(sha)[:8]
        out = Path(out) if out else self.store.root / "composites" / scope.level / Path(*scope.path) / composite
        adapters.stack_dirs(paths, out, self.store.state["base"], pad_to=rank)
        (out / "manifest.json").write_text(
            json.dumps({"chain": chain, "rank": rank, "ranks": ranks, "id": composite}, indent=2) + "\n"
        )
        return out

    def wake_notes(self, scope) -> list[Note]:
        """The chain's active notes that its live adapters do not hold yet (ancestors first)."""
        out = []
        for s in ScopeId.parse(scope).chain():
            ss = self.store.scope(s)
            if not ss.exists:
                continue
            consolidated = set(ss.manifest(ss.live)["note_ids"]) if ss.live else set()
            out += sorted((n for n in ss.active_notes() if n.id not in consolidated), key=lambda n: n.created)
        return out

    def render_notes(self, scope, messages: list[dict]) -> list[dict]:
        """`messages` with the scope chain's unconsolidated notes in the system turn (client-side path)."""
        return with_notes(messages, self.wake_notes(scope))


def _cap_pool(pool: list[dict], per_fact: int) -> list[dict]:
    """At most `per_fact` prompts per fact (the newest kept)."""
    counts: dict[str, int] = {}
    kept = []
    for row in reversed(pool):
        ids = row.get("fact_ids", [])
        if ids and all(counts.get(f, 0) >= per_fact for f in ids):
            continue
        for f in ids:
            counts[f] = counts.get(f, 0) + 1
        kept.append(row)
    return kept[::-1]
