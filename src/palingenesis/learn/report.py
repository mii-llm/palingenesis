"""Evaluation reports: sessions/sNNNN/report.json, report.md and diagnostics.jsonl.

report.json holds counts, rates, intervals, gates and chain versions, never a note's text, so it survives
`forget` as is. diagnostics.jsonl holds what failed, probe by probe (the probe, the answer, the expected
value, the note text), so users can fix a note; forget purges its rows like any other derived file.
report.md is rendered from the two and rendered again after a forget.
"""

from __future__ import annotations

import json
from pathlib import Path

from palingenesis.learn.store import atomic_write_text, read_jsonl, write_jsonl

MAX_DIAGNOSTICS_MD = 20


def _pct(entry: dict | None) -> str:
    if not entry or entry.get("rate") is None:
        return "-"
    ci = entry.get("ci")
    interval = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
    return f"{entry['rate']:.2f} ({entry['k']}/{entry['n']}){interval}"


def _paired(entry: dict) -> str:
    p = entry["paired"]
    if not p["n"]:
        return "-"
    return f"lost {p['losses']}, gained {p['gains']}, net loss {p['net_loss']} (p={p['p_value']:g})"


def summary(report: dict) -> str:
    """A few lines for the terminal."""
    failed = [g["name"] for g in report["gates"] if not g["passed"]]
    head = (
        f"{report['scope']} {report['version']}: gates {'passed' if report['passed'] else 'FAILED'}"
        + (f" ({', '.join(failed)})" if failed else "")
        + ("" if report["calibrated"] else "; thresholds uncalibrated")
    )
    m = report["metrics"]
    drift = m["drift"]
    lines = [
        head,
        f"  acquisition {_pct(m['acquisition']['relative'])} of in-context {_pct(m['acquisition']['context'])}",
        f"  update {_pct(m['update']['relative'])}, stale {_pct(m['stale']['candidate'])}",
        f"  retention {_paired(m['retention'])}; ancestors {_paired(m['ancestor_retention'])}",
        f"  general QA {_pct(m['general_qa']['candidate'])} (live {_pct(m['general_qa']['live'])}), "
        f"leak {m['leak']['k']}/{m['leak']['n']}, confabulation {_paired(m['confabulation'])}",
        f"  meta-reference {_pct(m['meta_reference']['candidate'])}, drift "
        + (
            f"{drift['candidate']:.4f} (live {drift['live']:.4f}, +{drift['increment']:.4f}) nats/token"
            if drift.get("tokens")
            else "-"
        ),
    ]
    return "\n".join(lines)


def markdown(report: dict, diagnostics: list[dict]) -> str:
    m = report["metrics"]
    calibration = (
        "calibrated" if report["calibrated"] else "**uncalibrated**: default thresholds, not yet set from null sessions"
    )
    baseline = report.get("baseline_chain", report["live_chain"])
    live = ", ".join(f"{s} {v or '(none)'}" for s, v in baseline)
    source = report.get("baseline", "the live chain")
    out = [
        f"# {report['scope']} {report['version']} ({report['session']})",
        "",
        f"Gates: **{'passed' if report['passed'] else 'failed'}**; thresholds {calibration}.",
        f"Compared with {source}: {live} (the live column). Evaluated {report['created']}.",
        "",
        "| metric | candidate | live | note |",
        "|---|---|---|---|",
        f"| acquisition | {_pct(m['acquisition']['candidate'])} | {_pct(m['acquisition']['live'])} | "
        f"in-context {_pct(m['acquisition']['context'])}; closed-book among those {_pct(m['acquisition']['relative'])} |",
        f"| update | {_pct(m['update']['candidate'])} | {_pct(m['update']['live'])} | "
        f"in-context {_pct(m['update']['context'])}; closed-book among those {_pct(m['update']['relative'])} |",
        f"| stale | {_pct(m['stale']['candidate'])} | {_pct(m['stale']['live'])} | answers stating an old value |",
        f"| retention | {_pct(m['retention']['candidate'])} | {_pct(m['retention']['live'])} | {_paired(m['retention'])} |",
        f"| ancestor retention | {_pct(m['ancestor_retention']['candidate'])} | "
        f"{_pct(m['ancestor_retention']['live'])} | {_paired(m['ancestor_retention'])} |",
        f"| general QA | {_pct(m['general_qa']['candidate'])} | {_pct(m['general_qa']['live'])} | "
        f"{_paired(m['general_qa'])} |",
        f"| leak | {_pct(m['leak'])} | | changed, wrong, naming a new entity of the scope |",
        f"| confabulation | {_pct(m['confabulation']['candidate'])} | {_pct(m['confabulation']['live'])} | "
        f"{_paired(m['confabulation'])} |",
        f"| meta-reference | {_pct(m['meta_reference']['candidate'])} | {_pct(m['meta_reference']['live'])} | "
        f"in reasoning {_pct(m['meta_reference']['reasoning'])} |",
    ]
    drift = m["drift"]
    if drift.get("tokens"):
        out.append(
            f"| drift (nats/token) | {drift['candidate']:.4f} | {drift['live']:.4f} | increment "
            f"{drift['increment']:+.4f} over {drift['tokens']} tokens |"
        )
    out += ["", "## Gates", ""]
    for g in report["gates"]:
        if g["skipped"]:
            out.append(f"- {g['name']}: skipped (nothing to measure)")
            continue
        value = "n/a" if g["value"] is None else f"{g['value']:g}"
        line = f"- {g['name']}: {value} (needs {g['op']} {g['threshold']:g}) {'pass' if g['passed'] else '**fail**'}"
        out.append(line + (f". Likely remedy: {g['remedy']}." if g["remedy"] else ""))
    for w in report.get("warnings", []):
        out.append(f"- warning: {w}")
    failed = [d for d in diagnostics if d["metric"] in ("acquisition", "update", "retention", "stale", "leak")]
    if failed:
        out += ["", f"## What failed ({len(failed)}; first {min(len(failed), MAX_DIAGNOSTICS_MD)})", ""]
        for d in failed[:MAX_DIAGNOSTICS_MD]:
            expected = f" (expected {d['expected']!r})" if d.get("expected") else ""
            out.append(f"- {d['metric']}: {d['q']!r}{expected}")
            out.append(f"  - answer: {d['answer'][:300]!r}")
            for note in d.get("notes", [])[:2]:
                out.append(f"  - note: {note[:300]!r}")
        out.append("")
        out.append("Every row is in diagnostics.jsonl.")
    return "\n".join(out) + "\n"


def write(session_dir: Path, report: dict, diagnostics: list[dict]) -> None:
    session_dir = Path(session_dir)
    write_jsonl(session_dir / "diagnostics.jsonl", diagnostics)
    atomic_write_text(session_dir / "report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    atomic_write_text(session_dir / "report.md", markdown(report, diagnostics))


def render_again(session_dir: Path) -> None:
    """report.md from report.json and the (possibly purged) diagnostics.jsonl."""
    session_dir = Path(session_dir)
    path = session_dir / "report.json"
    if path.exists():
        report = json.loads(path.read_text())
        atomic_write_text(session_dir / "report.md", markdown(report, read_jsonl(session_dir / "diagnostics.jsonl")))
