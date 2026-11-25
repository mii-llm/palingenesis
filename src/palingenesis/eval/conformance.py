"""Agreement of a `pgs eval` result with lm-evaluation-harness on the same task and model.

    python -m palingenesis.eval.conformance --pgs eval/gsm8k.default.jsonl --lmeval lm_out/ --task gsm8k \\
        --metric exact_match,strict-match [--json]

Reads lm-eval's results JSON (results_*.json under --lmeval: the aggregate) and, when it was run with
--log_samples, its samples_<task>_*.jsonl (per document). Reports both accuracies, their difference in units of
the standard error of the accuracy over n items, and, with samples, how many documents the two harnesses score
the same way (documents are matched by position: lm-eval's doc_id is the row number, which is the
item id pgs gives rows without an id column). Greedy decoding is not bit-identical between two engine runs
(batch-dependent floating point), so a handful of flipped documents is expected; exit code 1 when the difference
exceeds --max_se standard errors (default 1).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def pgs_scores(path: str | Path, metric: str = "") -> dict[str, float]:
    """{row key: score} of a pgs result file: the item's mean score, or one of its `metrics`."""
    out = {}
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            values = (row.get("metrics") or {}).get(metric) if metric else None
            values = values if values else row["scores"]
            out[str(row["id"]).rpartition("/")[2]] = sum(values) / len(values)
    return out


def lmeval_results(directory: str | Path, task: str, metric: str) -> tuple[float | None, dict[str, float]]:
    """(the aggregate of `metric` for `task`, {doc_id: per-document value}) from an lm-eval output directory."""
    root = Path(directory)
    aggregate = None
    for path in sorted(root.rglob("results*.json")):
        results = (json.loads(path.read_text()).get("results") or {}).get(task) or {}
        for key, value in results.items():
            if key.split(",")[0] == metric.split(",")[0] and (key == metric or "," not in metric) and "stderr" not in key:
                if isinstance(value, int | float):
                    aggregate = float(value)
                    break
    docs: dict[str, float] = {}
    name, _, filter_name = metric.partition(",")
    for path in sorted(root.rglob(f"samples_{task}_*.jsonl")):
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                if filter_name and row.get("filter") not in (None, filter_name):
                    continue
                if isinstance(row.get(name), int | float):
                    docs[str(row["doc_id"])] = float(row[name])
    return aggregate, docs


def compare(ours: dict[str, float], aggregate: float | None, docs: dict[str, float], max_se: float = 1.0) -> dict[str, Any]:
    """The two accuracies and their difference in standard errors of lm-eval's accuracy over the n items
    (sqrt(p (1 - p) / n)); `within`: the difference is at most max_se of them. With per-document values, also how
    many documents the harnesses score alike, and the paired z of the mean difference (information: two greedy
    runs of one engine already differ on a few documents)."""
    n = len(ours)
    mine = sum(ours.values()) / n if n else float("nan")
    theirs = aggregate if aggregate is not None else (sum(docs.values()) / len(docs) if docs else float("nan"))
    se = math.sqrt(max(theirs * (1 - theirs), 1e-12) / n) if n else float("nan")
    report: dict[str, Any] = {
        "items": n,
        "pgs": mine,
        "lmeval": theirs,
        "difference": mine - theirs,
        "stderr": se,
        "z": (mine - theirs) / se if se else 0.0,
    }
    shared = sorted(set(ours) & set(docs))
    if shared:
        same = sum(abs(ours[i] - docs[i]) < 1e-9 for i in shared)
        diffs = [ours[i] - docs[i] for i in shared]
        mean = sum(diffs) / len(diffs)
        paired = math.sqrt(sum((d - mean) ** 2 for d in diffs) / max(1, len(diffs) - 1) / len(diffs))
        report.update(paired_items=len(shared), same_items=same, agreement=same / len(shared))
        report["paired_z"] = mean / paired if paired > 0 else 0.0
    report["within"] = bool(abs(report["z"]) <= max_se or abs(report["difference"]) < 1e-12)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m palingenesis.eval.conformance", description=__doc__.split("\n\n")[0])
    parser.add_argument("--pgs", required=True, help="a pgs eval result file (<task>.<mode>.jsonl)")
    parser.add_argument("--lmeval", required=True, help="lm-eval's --output_path directory")
    parser.add_argument("--task", required=True, help="the task's name in lm-eval's results")
    parser.add_argument("--metric", default="acc", help="lm-eval's metric, with its filter: exact_match,strict-match")
    parser.add_argument("--max_se", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    aggregate, docs = lmeval_results(args.lmeval, args.task, args.metric)
    if aggregate is None and not docs:
        print(f"conformance: no {args.metric} for task {args.task} under {args.lmeval}", file=sys.stderr)
        return 1
    report = compare(pgs_scores(args.pgs, args.metric), aggregate, docs, args.max_se)
    report.update(task=args.task, metric=args.metric)
    if args.json:
        print(json.dumps(report))
    else:
        print(
            f"{args.task} {args.metric}: pgs {report['pgs']:.4f}, lm-eval {report['lmeval']:.4f}, difference "
            f"{report['difference']:+.4f} ({report['z']:+.2f} SE)"
            + (f", {report['same_items']} / {report['paired_items']} documents scored alike" if "agreement" in report else "")
        )
    return 0 if report["within"] else 1


if __name__ == "__main__":
    sys.exit(main())
