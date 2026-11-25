"""The report of a `pgs eval` run: a terminal table, Markdown, the Hub.

    print(table(report["results"]))            # one line per task and mode, and the run's means
    Path("report.md").write_text(markdown(report))
    push_to_hub("eval/sft", "me/eval-results")  # the run's directory as a (private) dataset repository

A result's score column is the mean over items with its standard error, next to the 95% bootstrap interval;
speed is items and generated tokens per second of the task's own wall-clock (engine start excluded).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

COLUMNS = ("task", "mode", "items", "score", "95% CI", "pass@k", "tokens", "trunc", "loops", "items/s", "tok/s", "time")


def _cells(summary: dict) -> list[str]:
    mean = summary.get("mean_score")
    k = summary.get("k", 1)
    passk = summary.get(f"pass@{k}") if k > 1 else None
    seconds = summary.get("seconds") or 0.0
    return [
        str(summary["task"]),
        str(summary["mode"]),
        str(summary["items"]),
        f"{mean['estimate']:.4f} ± {summary.get('stderr', 0.0):.4f}" if mean else "unscored",
        f"[{mean['lo']:.3f}, {mean['hi']:.3f}]" if mean else "",
        f"{passk['estimate']:.4f} (k={k})" if passk else "",
        f"{summary['completion_tokens']['mean']:.0f}",
        f"{summary['truncated_share']:.1%}",
        f"{summary['loop_share']:.1%}",
        f"{summary.get('items_per_second', 0.0):.1f}",
        f"{summary.get('tokens_per_second', 0.0):.0f}",
        f"{seconds:.1f}s" + (" (cached)" if summary.get("cache_hits") and not summary.get("engine_requests") else ""),
    ]


def overall(results: list[dict]) -> dict[str, Any]:
    """The run's totals: the macro mean of the tasks' scores per mode, items, seconds, tokens per second."""
    scored = [r for r in results if r.get("mean_score")]
    seconds = sum(r.get("seconds") or 0.0 for r in results)
    out: dict[str, Any] = {
        "results": len(results),
        "items": sum(r["items"] for r in results),
        "seconds": seconds,
        "generated_tokens": sum(r.get("generated_tokens") or 0 for r in results),
        "macro_mean": {},
    }
    out["tokens_per_second"] = out["generated_tokens"] / seconds if seconds > 0 else 0.0
    out["items_per_second"] = sum(r.get("items") or 0 for r in results) / seconds if seconds > 0 else 0.0
    for mode in dict.fromkeys(r["mode"] for r in scored):
        values = [r["mean_score"]["estimate"] for r in scored if r["mode"] == mode]
        out["macro_mean"][mode] = sum(values) / len(values)
    firsts = [r["time_to_first_result"] for r in results if r.get("time_to_first_result") is not None]
    if firsts:
        out["time_to_first_result"] = firsts[0]
    utils = [r["gpu_util_mean"] for r in results if "gpu_util_mean" in r]
    if utils:
        out["gpu_util_mean"] = sum(utils) / len(utils)
    return out


def table(results: list[dict], categories: bool = False) -> str:
    """The results as an aligned text table (box-drawing rules), with a line of means per mode."""
    rows = [_cells(r) for r in results]
    extra: list[tuple[int, list[str]]] = []
    if categories:
        for index, r in enumerate(results):
            for name, c in (r.get("categories") or {}).items():
                extra.append((index, ["  " + name, "", str(c["items"]), f"{c['score']:.4f} ± {c['stderr']:.4f}", *[""] * 8]))
    total = overall(results)
    footer = [
        ["mean of tasks", mode, "", f"{value:.4f}", *[""] * 8] for mode, value in total["macro_mean"].items()
    ]
    everything = [list(COLUMNS), *rows, *(cells for _, cells in extra), *footer]
    widths = [max(len(row[i]) for row in everything) for i in range(len(COLUMNS))]

    def line(cells: list[str]) -> str:
        return "│ " + " │ ".join(c.ljust(w) if i < 2 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths))) + " │"

    def rule(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (w + 2) for w in widths) + right

    out = [rule("┌", "┬", "┐"), line(list(COLUMNS)), rule("├", "┼", "┤")]
    for index, cells in enumerate(rows):
        out.append(line(cells))
        out += [line(c) for i, c in extra if i == index]
    if footer:
        out.append(rule("├", "┼", "┤"))
        out += [line(c) for c in footer]
    out.append(rule("└", "┴", "┘"))
    speed = f"{total['items']} items in {total['seconds']:.1f}s"
    if total["generated_tokens"]:
        speed += f", {total['tokens_per_second']:.0f} generated tokens/s"
    if "time_to_first_result" in total:
        speed += f", first result after {total['time_to_first_result']:.2f}s"
    if "gpu_util_mean" in total:
        speed += f", GPU {total['gpu_util_mean']:.0f}% busy"
    out.append(speed)
    return "\n".join(out)


def markdown(report: dict) -> str:
    """The run as a Markdown document: the table, per-task details (sampling, metrics, categories), contamination."""
    results = report["results"]
    lines = [f"# Evaluation of `{report['model']}`", ""]
    lines += ["| " + " | ".join(COLUMNS) + " |", "|" + "|".join(" --- " for _ in COLUMNS) + "|"]
    lines += ["| " + " | ".join(_cells(r)) + " |" for r in results]
    total = overall(results)
    for mode, value in total["macro_mean"].items():
        lines.append(f"| **mean of tasks** | {mode} | | **{value:.4f}** |" + " |" * 8)
    lines += ["", f"{total['items']} items, {total['seconds']:.1f} s of evaluation, "
              f"{total['tokens_per_second']:.0f} generated tokens/s.", ""]  # fmt: skip
    for r in results:
        s = r["sampling"]
        lines += [
            f"## {r['task']} [{r['mode']}]",
            "",
            f"- {r.get('output', 'generate')}, {r.get('template', 'chat')} prompts"
            + (f", task version {r['version']}" if r.get("version") else "")
            + f"; k = {r['k']}, temperature {s['temperature']} ({s['temperature_source']}), top_p {s['top_p']}, "
            f"top_k {s['top_k']}, max_tokens {s['max_tokens']}",
            f"- items {r['items']} (unscored {r['unscored_items']}), samples {r['samples']}",
        ]
        if "wilson" in r:
            w = r["wilson"]
            lines.append(f"- first sample: {w['successes']} / {w['n']} pass, Wilson 95% [{w['lo']:.4f}, {w['hi']:.4f}]")
        for name, m in (r.get("metrics") or {}).items():
            lines.append(f"- {name}: {m['mean']:.4f} ± {m['stderr']:.4f}")
        if r.get("categories"):
            lines += ["", "| category | items | score | stderr |", "| --- | ---: | ---: | ---: |"]
            lines += [f"| {n} | {c['items']} | {c['score']:.4f} | {c['stderr']:.4f} |" for n, c in r["categories"].items()]
        if r.get("contamination"):
            c = r["contamination"]
            lines.append(
                f"- contamination: {c['flagged']} of {c['items']} prompts share a {c['ngram']}-gram with the training "
                f"data; score on the clean items {c.get('clean_score')}"
            )
        lines.append("")
    return "\n".join(lines)


def push_to_hub(output_dir: str | Path, repo: str, private: bool = True, samples: bool = False) -> str:
    """Upload a run's directory (summaries, the report, result rows; sample files only when asked) to a dataset
    repository on the Hugging Face Hub, created private unless told otherwise. Returns the repository's URL."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=["*.json", "*.md", "*.jsonl"],
        ignore_patterns=[] if samples else ["*.samples.jsonl"],
        commit_message="pgs eval results",
    )
    return f"https://huggingface.co/datasets/{repo}"
