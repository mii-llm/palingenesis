#!/usr/bin/env python3
"""Score one evaluation with intervals, or compare two checkpoints on the same items with paired tests.

Input: JSONL result files, one row per evaluated item (any harness can write them):
    {"id": "gsm8k/17", "scores": [1, 0, 1, 1]}        # k samples of the item (0 / 1 or any reward)
    {"id": "gsm8k/18", "correct": true}                # one sample: correct, score, reward or scores
    {"id": "ifeval/3", "score": 0.5, "cluster": "q3"}  # cluster: items that are not independent (paraphrases,
                                                       # translations, one task's variants) resample together
Reported per file: items, samples per item, the mean score (per item first, then over items) and pass@k for every
k up to the smallest sample count (unbiased, Chen et al. 2021), each with a bootstrap interval over items (clusters).
Given two files (A then B), the items both have are paired: the difference B - A of each metric with a paired
bootstrap interval and p-value, a sign-flip permutation p-value, and the exact McNemar test when both have one 0 / 1
sample per item. Items in only one file are reported and left out (--strict: an error).

Exit code 0, or 1 when --require is not met (or on bad input):
    --require better       B's mean score is above A's: the interval of B - A is above 0
    --require not_worse    the interval's lower end is above -margin (--margin, default 0.01)

Usage:
    pgs compare results_a.jsonl                                # one evaluation, with intervals
    pgs compare base.jsonl candidate.jsonl --k 1,4,8 --json
    pgs compare base.jsonl candidate.jsonl --require not_worse --margin 0.02
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from palingenesis.evalstats import (
    bootstrap_ci,
    mcnemar_exact,
    paired_bootstrap,
    paired_permutation,
    pass_at_k,
)

SCORE_KEYS = ("scores", "correct", "score", "reward", "rewards")


def _number(value, where: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    raise ValueError(f"{where}: a score must be a finite number or a boolean, got {value!r}")


def load_results(path: str, id_field: str = "id", cluster_field: str = "cluster") -> dict[str, dict]:
    """{item id: {"scores": [float, ...], "cluster": label or None}} from a JSONL result file. Each row needs an id
    and one of scores / correct / score / reward / rewards (a number, a boolean or a list of them)."""
    items: dict[str, dict] = {}
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path}:{line_no}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where}: not JSON ({e})") from None
            if not isinstance(row, dict):
                raise ValueError(f"{where}: a row must be a JSON object")
            if row.get(id_field) is None:
                raise ValueError(f"{where}: no {id_field!r} (every row names its item, so files can be paired)")
            key = next((k for k in SCORE_KEYS if k in row), None)
            if key is None:
                raise ValueError(f"{where}: no score; give one of {', '.join(SCORE_KEYS)}")
            value = row[key]
            scores = [_number(v, where) for v in value] if isinstance(value, list) else [_number(value, where)]
            if not scores:
                raise ValueError(f"{where}: {key!r} is empty")
            item = str(row[id_field])
            if item in items:
                raise ValueError(f"{where}: item {item!r} appears twice (one row per item: put its samples in a list)")
            items[item] = {"scores": scores, "cluster": row.get(cluster_field)}
    if not items:
        raise ValueError(f"{path}: no rows")
    return items


def _per_item(items: list[dict], k: int | None, threshold: float) -> list[float]:
    """Each item's mean score (k None) or unbiased pass@k at the threshold."""
    if k is None:
        return [sum(i["scores"]) / len(i["scores"]) for i in items]
    return [pass_at_k(len(i["scores"]), sum(s >= threshold for s in i["scores"]), k) for i in items]


def _clusters(items: list[dict]) -> list | None:
    labels = [i["cluster"] for i in items]
    if all(label is None for label in labels):
        return None
    return [json.dumps(label) if label is not None else f"__item{n}" for n, label in enumerate(labels)]


def _ks(requested: list[int], samples: int) -> list[int]:
    return [k for k in requested if 1 <= k <= samples]


def summarize(items: dict[str, dict], ks: list[int], threshold: float, resamples: int, confidence: float, seed: int):
    rows = list(items.values())
    counts = [len(i["scores"]) for i in rows]
    clusters = _clusters(rows)
    out = {
        "items": len(rows),
        "samples_min": min(counts),
        "samples_max": max(counts),
        "clusters": len(set(clusters)) if clusters else len(rows),
        "mean_score": bootstrap_ci(_per_item(rows, None, threshold), resamples, confidence, seed, clusters),
    }
    for k in _ks(ks, min(counts)):
        out[f"pass@{k}"] = bootstrap_ci(_per_item(rows, k, threshold), resamples, confidence, seed, clusters)
    return out


def compare(a: dict[str, dict], b: dict[str, dict], ks: list[int], threshold: float, resamples: int,
            confidence: float, seed: int) -> dict:  # fmt: skip
    common = [item for item in a if item in b]
    if not common:
        raise ValueError("the two files have no item in common (compare matches rows by id)")
    rows_a, rows_b = [a[i] for i in common], [b[i] for i in common]
    # an item's cluster as either file names it (the baseline's first): results written by another tool may lack it
    clusters = _clusters([x if x["cluster"] is not None else {"cluster": y["cluster"]} for x, y in zip(rows_a, rows_b)])
    samples = min(len(i["scores"]) for i in rows_a + rows_b)
    out: dict = {"paired_items": len(common), "only_a": len(a) - len(common), "only_b": len(b) - len(common)}
    metrics = [("mean_score", None)] + [(f"pass@{k}", k) for k in _ks(ks, samples)]
    for name, k in metrics:
        va, vb = _per_item(rows_a, k, threshold), _per_item(rows_b, k, threshold)
        boot = paired_bootstrap(va, vb, resamples, confidence, seed, clusters)
        perm = paired_permutation(va, vb, resamples, seed, clusters)
        out[name] = {
            "a": sum(va) / len(va),
            "b": sum(vb) / len(vb),
            "diff": boot["diff"],
            "lo": boot["lo"],
            "hi": boot["hi"],
            "p_bootstrap": boot["p"],
            "p_permutation": perm["p"],
        }
    single = all(len(i["scores"]) == 1 for i in rows_a + rows_b)
    binary = all(s in (0.0, 1.0) for i in rows_a + rows_b for s in i["scores"])
    if single and binary:
        out["mcnemar"] = mcnemar_exact([i["scores"][0] == 1.0 for i in rows_a], [i["scores"][0] == 1.0 for i in rows_b])
    return out


def _fmt(ci: dict) -> str:
    return f"{ci['estimate']:.4f}  [{ci['lo']:.4f}, {ci['hi']:.4f}]"


def _print(report: dict, confidence: float) -> None:
    pct = f"{confidence:.0%}"
    for name in ("a", "b"):
        s = report.get(name)
        if s is None:
            continue
        print(f"== {report['files'][name]}")
        print(f"   items {s['items']}  clusters {s['clusters']}  samples/item {s['samples_min']}-{s['samples_max']}")
        for key, value in s.items():
            if isinstance(value, dict):
                print(f"   {key:<12} {_fmt(value)}  ({pct})")
    c = report.get("comparison")
    if c:
        print(f"== B - A over {c['paired_items']} paired items (only in A: {c['only_a']}, only in B: {c['only_b']})")
        for key, value in c.items():
            if isinstance(value, dict) and "diff" in value:
                print(
                    f"   {key:<12} {value['a']:.4f} -> {value['b']:.4f}  diff {value['diff']:+.4f} "
                    f"[{value['lo']:+.4f}, {value['hi']:+.4f}]  p_boot {value['p_bootstrap']:.4g}  "
                    f"p_perm {value['p_permutation']:.4g}"
                )
        if "mcnemar" in c:
            m = c["mcnemar"]
            print(f"   McNemar      B-only {m['b_only']:.0f}  A-only {m['a_only']:.0f}  p {m['p']:.4g}")
    if "requirement" in report:
        r = report["requirement"]
        print(f"== require {r['require']}: {'met' if r['met'] else 'NOT met'} ({r['reason']})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pgs compare",
        description="Intervals for one evaluation, or paired tests between two (JSONL rows: id + scores).",
    )
    parser.add_argument("a", help="results JSONL (the baseline when two are given)")
    parser.add_argument("b", nargs="?", help="results JSONL of the candidate (paired with A by id)")
    parser.add_argument("--k", default="1,2,4,8,16,32,64", help="pass@k values (those above the samples are skipped)")
    parser.add_argument("--threshold", type=float, default=0.5, help="a sample succeeds at or above this score")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--id_field", default="id")
    parser.add_argument("--cluster_field", default="cluster")
    parser.add_argument("--strict", action="store_true", help="items present in only one file are an error")
    parser.add_argument("--require", choices=("better", "not_worse"), help="exit 1 unless B meets it (mean score)")
    parser.add_argument("--margin", type=float, default=0.01, help="--require not_worse: the tolerated drop")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        ks = sorted({int(x) for x in args.k.split(",") if x.strip()})
        if not ks or ks[0] < 1:
            raise ValueError("--k takes positive integers, e.g. 1,4,8")
        if args.resamples < 1 or not 0 < args.confidence < 1:
            raise ValueError("--resamples must be >= 1 and --confidence in (0, 1)")
        if args.require and not args.b:
            raise ValueError("--require compares two files: give B")
        if args.margin < 0:
            raise ValueError("--margin must be >= 0")
        options = (ks, args.threshold, args.resamples, args.confidence, args.seed)
        a = load_results(args.a, args.id_field, args.cluster_field)
        report: dict = {"files": {"a": args.a}, "a": summarize(a, *options)}
        if args.b:
            b = load_results(args.b, args.id_field, args.cluster_field)
            report["files"]["b"] = args.b
            report["b"] = summarize(b, *options)
            report["comparison"] = compare(a, b, *options)
            c = report["comparison"]
            if args.strict and (c["only_a"] or c["only_b"]):
                raise ValueError(f"--strict: {c['only_a']} item(s) only in A, {c['only_b']} only in B")
    except (OSError, ValueError) as e:
        print(f"pgs compare: {e}", file=sys.stderr)
        return 1
    code = 0
    if args.require:
        diff = report["comparison"]["mean_score"]
        if args.require == "better":
            met, reason = diff["lo"] > 0, f"lower end {diff['lo']:+.4f} > 0"
        else:
            met, reason = diff["lo"] > -args.margin, f"lower end {diff['lo']:+.4f} > -{args.margin}"
        report["requirement"] = {"require": args.require, "met": met, "reason": reason}
        code = 0 if met else 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print(report, args.confidence)
    return code


if __name__ == "__main__":
    sys.exit(main())
