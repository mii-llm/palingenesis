"""pgs learn: scoped continual learning (experimental).

    pgs learn init     STORE --base Qwen/Qwen3.5-4B --org acme [--config learn.yaml]
    pgs learn add      STORE --scope S (--documents GLOB [--root DIR] | --facts F.jsonl | --corrections C.jsonl
                                        | --instructions I.jsonl | --note TEXT) [--approve] [--locked]
    pgs learn inbox    STORE --scope S [--json]
    pgs learn approve  STORE --scope S ID... [--reject | --as supersede|both|reject]
    pgs learn promote  STORE --scope S NOTE --to team:acme/support
    pgs learn forget   STORE --scope S ID...
    pgs learn plan     STORE [--scope S] [--json]
    pgs learn run      STORE --scope S [--live [--force]] [--no-eval] [--section.option value ...]
    pgs learn eval     STORE --scope S [--version V] [--live [--force]] [--json] [--section.option value ...]
    pgs learn status   STORE [--scope S] [--json]

Scopes are org:<org>, team:<org>/<team>, user:<org>/<team>/<user>. Every command that changes the store
takes its lock. Exit codes: 0 on success, 1 on an error, an unclean forget, failed gates (eval, or run
--live) or an unhealthy status (a pending rebuild, a forced or ungated live version).
"""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pgs learn", description="scoped continual learning (experimental)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def store_arg(p, scope=True):
        p.add_argument("store", help="store directory (one per org)")
        if scope:
            p.add_argument("--scope", required=True, help="org:<org> | team:<org>/<team> | user:<org>/<team>/<user>")
        p.add_argument("--generator", default=None, help="override generate.generator (self | stub | a model id)")

    p = sub.add_parser("init", help="create a store for one org")
    store_arg(p, scope=False)
    p.add_argument("--base", required=True, help="base model id or path")
    p.add_argument("--org", required=True, help="the tenant (org root) this store holds")
    p.add_argument("--config", default=None, help="learn.yaml (strict; saved in the store)")

    p = sub.add_parser("add", help="add documents, facts, corrections or instructions to a scope")
    store_arg(p)
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--documents", help="glob of documents (markdown or text), keys relative to --root")
    what.add_argument("--facts", help="JSONL of {text, locked?}")
    what.add_argument("--corrections", help="JSONL of {prompt, correction}")
    what.add_argument("--instructions", help="JSONL of {text} (behaviour / policy; team and org only)")
    what.add_argument("--note", help="one note")
    p.add_argument("--root", default=None, help="source root of --documents (default: current directory)")
    p.add_argument("--approve", action="store_true", help="approve now instead of waiting in the inbox")
    p.add_argument("--locked", action="store_true", help="team / org policy that lower scopes cannot contradict")
    p.add_argument("--origin", default="cli", help="provenance recorded with each note")

    p = sub.add_parser("inbox", help="list a scope's inbox (and the admin queue into it)")
    store_arg(p)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("approve", help="settle inbox items or promotions")
    store_arg(p)
    p.add_argument("ids", nargs="+")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--reject", action="store_true")
    group.add_argument(
        "--as", dest="resolution", choices=("supersede", "both", "reject"), help="how to settle a conflict"
    )

    p = sub.add_parser("promote", help="ask an admin to copy a note to a team or org")
    store_arg(p)
    p.add_argument("note")
    p.add_argument("--to", required=True)

    p = sub.add_parser("forget", help="purge notes or inbox items from the store and prune the versions holding them")
    store_arg(p)
    p.add_argument("ids", nargs="+")

    p = sub.add_parser("plan", help="what the next run would train and evaluate, and its estimated cost")
    store_arg(p, scope=False)
    p.add_argument("--scope", default=None, help="one scope (default: every scope, parents first)")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("run", help="train the scope's adapter on its unconsolidated notes, then evaluate it")
    store_arg(p)
    p.add_argument("--live", action="store_true", help="make the new version live if its gates pass")
    p.add_argument("--force", action="store_true", help="with --live: make it live even if a gate fails (recorded)")
    p.add_argument("--no-eval", dest="evaluate", action="store_false", help="train only (the version cannot go live)")

    p = sub.add_parser("eval", help="evaluate a version against the scope's live version; exit 1 if a gate fails")
    store_arg(p)
    p.add_argument("--version", default=None, help="adapter version (default: the newest)")
    p.add_argument("--live", action="store_true", help="make it live if its gates pass")
    p.add_argument("--force", action="store_true", help="with --live: make it live even if a gate fails (recorded)")
    p.add_argument("--json", action="store_true", help="print report.json")

    p = sub.add_parser("status", help="scope tree, live versions, inbox and queue counts; exit 1 on issues")
    store_arg(p, scope=False)
    p.add_argument("--scope", default=None, help="a scope and its descendants (default: every scope)")
    p.add_argument("--json", action="store_true")
    return ap


def _publish(learner, scope: str, version: str, passed: bool, force: bool) -> int:
    """--live after an evaluation: 0 when the version went live."""
    from palingenesis.learn import StoreError

    try:
        learner.set_live(scope, version, force=force)
    except StoreError as e:
        print(f"not live: {e}", file=sys.stderr)
        return 1
    print(f"{scope}: {version} is live" + ("" if passed else " (FORCED past failed gates; recorded in the manifest)"))
    return 0


def _plan_text(plans: list[dict]) -> str:
    lines = []
    for p in plans:
        if not p["steps"]:
            lines.append(f"{p['scope']}: nothing to train")
            continue
        cost = (
            f"about {p['minutes']} min (times measured by {p['measured_on']})"
            if p["minutes"] is not None
            else "unknown until a session has run on this store"
        )
        lines += [
            f"{p['scope']}: {p['new_facts']} new facts from {p['new_notes']} notes, {p['active_facts']} active",
            f"  rows per pass: {p['questions']} questions + {p['nearmiss_per_pass']} near-miss + "
            f"{p['anchor_per_pass']} anchors; {p['replay_rows']} replay rows over {p['replay_facts']} facts; "
            f"{p['passes']} passes = {p['rows']} rows, {p['steps']} steps",
            f"  eval: about {p['eval_answers']} generated answers; served rank {p['served_rank']} "
            f"({' + '.join(map(str, p['chain_ranks']))}); cost {cost}",
        ]
        lines += [f"  warning: {w}" for w in p["warnings"]]
    return "\n".join(lines)


def _status_text(status: dict) -> str:
    lines = [f"store for org {status['org']} on {status['base']}"]
    for e in status["scopes"]:
        depth = {"org": 0, "team": 1, "user": 2}[e["scope"].split(":")[0]]
        if e["scope"].startswith("user:") and e["scope"].count("/") == 1:
            depth = 1
        live = f"live {e['live']} ({e['live_state']})" if e["live"] else "no live version"
        latest = ""
        if e["latest"] and e["latest"] != e["live"]:
            ev = e["latest_evaluation"]
            verdict = (
                "not evaluated"
                if ev is None
                else "gates passed"
                if ev["passed"]
                else f"failed {', '.join(ev['failed'])}"
            )
            latest = f"; newest {e['latest']} {verdict}"
        lines.append(
            f"{'  ' * depth}{e['scope']}: {live}{latest}; {e['notes']} notes, {e['facts']} facts "
            f"({e['unconsolidated']} unconsolidated); inbox {e['inbox']}, admin queue {e['admin_queue']}"
            + ("; REBUILD pending" if e["rebuild"] else "")
        )
    lines += [f"issue: {i}" for i in status["issues"]]
    lines.append("healthy" if status["healthy"] else f"{len(status['issues'])} issue(s)")
    return "\n".join(lines)


def _rows(path: str) -> list[dict]:
    from palingenesis.learn.store import read_jsonl

    return read_jsonl(path)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args, extra = build_parser().parse_known_args(argv)
    if extra and args.cmd not in ("run", "eval"):
        print(f"pgs learn {args.cmd}: unexpected arguments {extra}", file=sys.stderr)
        return 1
    print("pgs learn is experimental: interfaces and defaults may change.", file=sys.stderr)
    import warnings

    from palingenesis.learn import ExperimentalWarning

    warnings.simplefilter("ignore", ExperimentalWarning)
    from palingenesis.learn import LearnConfig, LearnConfigError, Learner, StoreError

    try:
        if args.cmd == "init":
            config = LearnConfig.from_yaml(args.config) if args.config else LearnConfig()
            if args.generator:
                config.generate.generator = args.generator
            learner = Learner.init(args.store, args.base, args.org, config)
            print(f"created {learner.store.root} for org {args.org} on {args.base}")
            return 0
        learner = Learner.open(args.store)
        if args.generator:
            learner.config.generate.generator = args.generator
        if args.cmd == "add":
            kw = {"approve": args.approve, "origin": args.origin}
            if args.documents:
                report = learner.add_documents(args.scope, args.documents, root=args.root, locked=args.locked, **kw)
                print(report.summary())
                for reason in report.rejected:
                    print(f"  rejected: {reason}")
                return 0
            if args.note:
                results = learner.add_notes(args.scope, [args.note], locked=args.locked, **kw)
            elif args.facts:
                results = learner.add_notes(args.scope, _rows(args.facts), locked=args.locked, **kw)
            elif args.corrections:
                results = learner.add_notes(args.scope, _rows(args.corrections), kind="correction", **kw)
            else:
                results = learner.add_notes(
                    args.scope, _rows(args.instructions), kind="instruction", locked=args.locked, **kw
                )
            for r in results:
                print(f"{r.status:9s} {r.note_id or r.inbox_id} {r.reason}".rstrip())
            return 0
        if args.cmd == "inbox":
            items = [i.to_dict() for i in learner.inbox(args.scope)]
            queue = [r for r in learner.admin_queue() if r["to_scope"] == args.scope]
            if args.json:
                print(json.dumps({"inbox": items, "admin_queue": queue}, indent=2, ensure_ascii=False))
                return 0
            for i in items:
                text = (i["payload"].get("note") or {}).get("text") or i["payload"].get("source_key", "")
                print(f"{i['id']}  {i['kind']:10s} {i['reason']}\n    {text[:200]!r}")
            for r in queue:
                print(f"{r['id']}  promotion  note {r['note_id']} from {r['from_scope']}")
            if not items and not queue:
                print("inbox empty")
            return 0
        if args.cmd == "approve":
            action = "reject" if args.reject else args.resolution or "approve"
            for item_id, r in zip(args.ids, learner.approve(args.scope, args.ids, action)):
                print(f"{item_id}: {r.status} {r.note_id or r.inbox_id} {r.reason}".rstrip())
            return 0
        if args.cmd == "promote":
            print(learner.promote(args.scope, args.note, args.to))
            return 0
        if args.cmd == "forget":
            report = learner.forget(args.scope, args.ids)
            print(report.summary())
            for path in report.leftovers:
                print(f"  still holds forgotten text: {path}")
            return 0 if report.ok else 1
        if args.cmd == "plan":
            plans = learner.plan(args.scope)
            print(json.dumps(plans, indent=2) if args.json else _plan_text(plans))
            return 0
        if args.cmd == "status":
            status = learner.status(args.scope)
            print(json.dumps(status, indent=2, ensure_ascii=False) if args.json else _status_text(status))
            return 0 if status["healthy"] else 1
        if args.cmd == "run":
            learner.config.apply_overrides(extra)
            learner.config.validate()
            if args.live and not args.evaluate and not args.force:
                raise ValueError("--live needs the evaluation (drop --no-eval, or add --force)")
            result = learner.run(args.scope, evaluate=args.evaluate)
            print(result.summary())
            for w in result.warnings:
                print(f"  warning: {w}")
            evaluation = result.evaluation
            if evaluation is not None:
                print(evaluation.summary())
                print(f"  report: {evaluation.session_dir / 'report.md'}")
            if args.live:
                return _publish(learner, args.scope, result.version, bool(evaluation and evaluation.passed), args.force)
            return 0
        if args.cmd == "eval":
            learner.config.apply_overrides(extra)
            learner.config.validate()
            evaluation = learner.evaluate(args.scope, args.version)
            if args.json:
                print(json.dumps(evaluation.report, indent=2, ensure_ascii=False))
            else:
                print(evaluation.summary())
                print(f"  report: {evaluation.session_dir / 'report.md'}")
            if args.live:
                return _publish(learner, args.scope, evaluation.report["version"], evaluation.passed, args.force)
            return 0 if evaluation.passed else 1
    except (StoreError, LearnConfigError, KeyError, ValueError, ImportError) as e:
        print(f"pgs learn {args.cmd}: {e}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
