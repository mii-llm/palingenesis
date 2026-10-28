"""pgs learn: scoped continual learning (experimental).

    pgs learn init     STORE --base Qwen/Qwen3.5-4B --org acme [--config learn.yaml]
    pgs learn add      STORE --scope S (--documents GLOB [--root DIR] | --facts F.jsonl | --corrections C.jsonl
                                        | --instructions I.jsonl | --note TEXT) [--approve] [--locked]
    pgs learn inbox    STORE --scope S [--json]
    pgs learn approve  STORE --scope S ID... [--reject | --as supersede|both|reject]
    pgs learn promote  STORE --scope S NOTE --to team:acme/support
    pgs learn forget   STORE --scope S ID...
    pgs learn run      STORE --scope S [--live] [--section.option value ...]

Scopes are org:<org>, team:<org>/<team>, user:<org>/<team>/<user>. Every command that changes the store
takes its lock. Exit code 0 on success, 1 on an error or an unclean forget.
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

    p = sub.add_parser("run", help="train the scope's adapter on its unconsolidated notes")
    store_arg(p)
    p.add_argument("--live", action="store_true", help="make the new version live (ungated until evaluation lands)")
    return ap


def _rows(path: str) -> list[dict]:
    from palingenesis.learn.store import read_jsonl

    return read_jsonl(path)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args, extra = build_parser().parse_known_args(argv)
    if extra and args.cmd != "run":
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
        if args.cmd == "run":
            learner.config.apply_overrides(extra)
            learner.config.validate()
            result = learner.run(args.scope)
            print(result.summary())
            for w in result.warnings:
                print(f"  warning: {w}")
            if args.live:
                learner.set_live(args.scope, result.version)
                print(f"{args.scope}: {result.version} is live (ungated)")
            return 0
    except (StoreError, LearnConfigError, KeyError, ValueError, ImportError) as e:
        print(f"pgs learn {args.cmd}: {e}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
