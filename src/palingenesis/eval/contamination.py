"""Contamination check: do a task's prompts share long n-grams with the training data?

    report = check(prompts, ["data/sft.jsonl", "data/rl/"], n=13)
    report["flagged"]   # items with at least one shared 13-gram; report["overlap"][i] is item i's shared share

Words are lower-cased runs of letters and digits; an n-gram is n consecutive words (13, as in the GPT-3 paper's and
lm-evaluation-harness's decontamination). The evaluation side is small, so its n-grams are hashed into a set and
the training files are streamed past it once: memory does not grow with the training data. Files: .jsonl / .json
(every string of every row, at any depth), .parquet (string columns, and strings inside nested columns), .txt /
.md (the text); a directory means every such file under it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

WORD = re.compile(r"[^\W_]+", re.UNICODE)
SUFFIXES = (".jsonl", ".json", ".parquet", ".txt", ".md")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngram_hashes(tokens: list[str], n: int) -> Iterator[int]:
    for i in range(len(tokens) - n + 1):
        yield hash(" ".join(tokens[i : i + n]))


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _strings(v)
    elif hasattr(value, "tolist") and not isinstance(value, bytes):
        yield from _strings(value.tolist())


def files_of(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for path in map(Path, paths):
        if path.is_dir():
            out += sorted(p for p in path.rglob("*") if p.suffix in SUFFIXES and p.is_file())
        elif path.is_file():
            out.append(path)
        else:
            raise FileNotFoundError(f"contamination: {path} is not a file or a directory")
    return out


def texts_of(path: Path) -> Iterator[str]:
    """Every text of a training data file, one row (or document) at a time."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                yield " ".join(_strings(row))
    elif path.suffix in (".txt", ".md"):
        yield path.read_text(errors="replace")
    else:
        with path.open(errors="replace") as f:
            head = f.read(1)
            f.seek(0)
            if head == "[":  # a JSON list of rows
                for row in json.load(f):
                    yield " ".join(_strings(row))
                return
            for line in f:
                if line.strip():
                    try:
                        yield " ".join(_strings(json.loads(line)))
                    except json.JSONDecodeError:
                        yield line


def check(prompts: list[str], paths: Iterable[str | Path], n: int = 13) -> dict[str, Any]:
    """Which prompts share an n-gram with the training data at `paths`."""
    per_item = [set(ngram_hashes(words(prompt), n)) for prompt in prompts]
    wanted: set[int] = set().union(*per_item) if per_item else set()
    seen: set[int] = set()
    files = files_of(paths)
    for path in files:
        for text in texts_of(path):
            if len(seen) == len(wanted):
                break
            tokens = words(text)
            if len(tokens) >= n:
                seen.update(wanted.intersection(ngram_hashes(tokens, n)))
    overlap = [len(grams & seen) / len(grams) if grams else 0.0 for grams in per_item]
    return {
        "ngram": n,
        "files": len(files),
        "items": len(prompts),
        "too_short": sum(not grams for grams in per_item),  # prompts under n words cannot be checked
        "flagged": sum(o > 0 for o in overlap),
        "flagged_share": sum(o > 0 for o in overlap) / len(prompts) if prompts else 0.0,
        "overlap": overlap,
    }


def task_report(items: list, rows: list[dict], paths: Iterable[str | Path], n: int, prompt_of) -> dict[str, Any]:
    """check() on a task's items, with the task's score on the flagged and on the clean items. prompt_of(item):
    the item's prompt text."""
    report = check([prompt_of(item) for item in items], paths, n)
    flagged = {item.id for item, o in zip(items, report.pop("overlap")) if o > 0}
    means = {r["id"]: sum(r["scores"]) / len(r["scores"]) for r in rows if r.get("scores")}
    clean = [v for i, v in means.items() if i not in flagged]
    dirty = [v for i, v in means.items() if i in flagged]
    report["clean_score"] = sum(clean) / len(clean) if clean else None
    report["flagged_score"] = sum(dirty) / len(dirty) if dirty else None
    report["flagged_ids"] = sorted(flagged)[:50]
    return report
