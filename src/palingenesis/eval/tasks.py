"""A task's items: dataset rows, the field mapping, the rendered prompt, ids and clusters.

    items = build_items(load_task("gsm8k"))     # [EvalItem(id="gsm8k/0", cluster=None, row={...}), ...]

An item's row is the dataset row plus the mapped fields; the prompt is in row["messages"] when the task renders one
(`prompt` / `messages`). Every column reaches the scorer and the environment, as in RL.

Field entries (`fields:`), a column name or a mapping with one source and transforms applied in this order:

    from: column            a column; dotted paths reach into mappings and lists (question.0.content)
    template: "{{a}} {{b}}" text over the row's columns and the fields mapped before it
    value: anything         a constant
    default: x              used when the column is missing or null (otherwise that is an error)
    json: true              parse a JSON string
    regex: '####\\s*(.+)$'  the last match's group (`group`, default 1; 0 = the whole match); no match is an error
    replace: {",": ""}      text replacements
    strip: false            strings are stripped unless this is false
    map: {yes: A, no: B}    a lookup (unknown values are an error)
    choices: true           a list of options -> "A. first\\nB. second" lines
    letter: true            an option index (0-based) -> its letter
    tools: true             function specs ({name, description, parameters}) -> OpenAI tool schemas
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from palingenesis.eval.config import PLACEHOLDER, EvalConfigError, TaskConfig

logger = logging.getLogger(__name__)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_MISSING = object()
# Type names function-calling datasets use for JSON Schema types (BFCL writes Python's).
_SCHEMA_TYPES = {
    "dict": "object",
    "float": "number",
    "int": "integer",
    "bool": "boolean",
    "str": "string",
    "list": "array",
    "tuple": "array",
    "any": "string",
}


@dataclass
class EvalItem:
    id: str
    cluster: str | None
    row: dict[str, Any]


# -------------------------------------------------------------------- loading


def load_task_rows(task: TaskConfig) -> list[dict[str, Any]]:
    """The task's dataset rows as plain dicts (arrays as lists), joined with `join` when set."""
    rows = _read_source(task, task.data_files)
    if task.join:
        key = task.join["key"]
        other = {_plain(r).get(key): _plain(r) for r in _read_source(task, task.join["data_files"])}
        joined = []
        for row in rows:
            match = other.get(row.get(key))
            if match is None:
                raise EvalConfigError(
                    f"task {task.name}: join: no row with {key} = {row.get(key)!r} in {task.join['data_files']}"
                )
            joined.append({**match, **row})
        rows = joined
    return rows


def _read_source(task: TaskConfig, data_files: str) -> list[dict[str, Any]]:
    source = Path(task.dataset)
    if source.suffix in (".jsonl", ".json", ".parquet") or source.exists():
        path = source.parent / data_files if data_files and source.is_file() else source / data_files
        return _read_file(path if data_files else source, task)
    if data_files:  # one file of a Hub dataset repository
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(task.dataset, data_files, repo_type="dataset", revision=task.revision or None)
        except Exception as e:  # noqa: BLE001 — the Hub's errors (offline, missing file) with the task named
            raise EvalConfigError(f"task {task.name}: {task.dataset}/{data_files}: {type(e).__name__}: {e}") from None
        return _read_file(Path(path), task)
    try:
        from datasets import load_dataset

        data = load_dataset(task.dataset, task.subset or None, split=task.split, revision=task.revision or None)
    except Exception as e:  # noqa: BLE001 — datasets' errors (offline, no such split) with the task named
        cached = _hub_cache_file(task)  # offline, with the repository's files in the Hub cache only
        if cached is not None:
            logger.info("task %s: %s read from the Hub cache (%s)", task.name, task.dataset, cached.name)
            return _read_file(cached, task)
        raise EvalConfigError(
            f"task {task.name}: cannot load {task.dataset} (subset {task.subset or 'default'}, split {task.split}): "
            f"{type(e).__name__}: {e}"
        ) from None
    return [_plain(dict(r)) for r in data]


def _hub_cache_file(task: TaskConfig) -> Path | None:
    """The split's data file in the local Hub cache snapshot of the dataset repository, when exactly one file is
    the obvious one: <subset>/<split>-*.parquet, data/<split>-*.parquet, <split>.jsonl / .json / .parquet, or the
    repository's only data file. None otherwise (several shards, another layout, not cached)."""
    try:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(task.dataset, repo_type="dataset", revision=task.revision or None, local_files_only=True)
        )
    except Exception:  # noqa: BLE001 — not in the cache
        return None
    split, folders = task.split, ([task.subset] if task.subset else ["data", ".", "default"])
    patterns = [f"{folder}/{split}-*.parquet" for folder in folders]
    patterns += [f"{folder}/{split}.{ext}" for folder in folders for ext in ("parquet", "jsonl", "json")]
    for pattern in patterns:
        found = sorted(root.glob(pattern))
        if len(found) == 1:
            return found[0]
        if found:
            return None
    files = [p for p in root.rglob("*") if p.suffix in (".parquet", ".jsonl", ".json") and p.is_file()]
    return files[0] if len(files) == 1 and not task.subset else None


def _read_file(path: Path, task: TaskConfig) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvalConfigError(f"task {task.name}: {path} is not a file")
    if path.suffix == ".parquet":
        import pandas as pd

        return [_plain(r) for r in pd.read_parquet(path).to_dict("records")]
    text = path.read_text()
    try:  # a JSON document: a list of rows, or {split: rows}
        data = json.loads(text)
    except json.JSONDecodeError:  # JSON lines (whatever the extension: BFCL's .json files are)
        try:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as e:
            raise EvalConfigError(f"task {task.name}: {path}: not JSON or JSON lines ({e})") from None
    if isinstance(data, dict):  # {split: rows}; any other object is the single row of a one-line JSON lines file
        data = data[task.split] if isinstance(data.get(task.split), list) else [data]
    if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
        raise EvalConfigError(f"task {task.name}: {path}: expected rows (JSON objects)")
    return data


def _plain(value: Any) -> Any:
    """Arrays and numpy scalars (parquet, datasets) as lists and Python numbers."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, str | bytes):
        return _plain(value.tolist())
    return value


# --------------------------------------------------------------------- fields


def lookup(row: dict, path: str) -> Any:
    """row[a][b]... for a dotted path (integer parts index lists); _MISSING when absent."""
    if path in row:
        return row[path]
    node: Any = row
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.lstrip("-").isdigit() and -len(node) <= int(part) < len(node):
            node = node[int(part)]
        else:
            return _MISSING
    return node


def render(template: str, row: dict, where: str) -> str:
    """`template` with its {{name}} placeholders replaced by the row's values; an unknown name is an error."""

    def substitute(match) -> str:
        value = lookup(row, match.group(1))
        if value is _MISSING:
            raise EvalConfigError(
                f"{where}: placeholder {{{{{match.group(1)}}}}} is not a field or a column (have: {sorted(row)})"
            )
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    return PLACEHOLDER.sub(substitute, template)


def map_field(spec: Any, row: dict, where: str) -> Any:
    """One `fields:` entry evaluated on a row (see the module docstring)."""
    if isinstance(spec, str):
        spec = {"from": spec}
    if "value" in spec:
        value = spec["value"]
    elif "template" in spec:
        value = render(str(spec["template"]), row, where)
    else:
        value = lookup(row, spec["from"])
        if value is _MISSING or value is None:
            if "default" not in spec:
                state = "is null" if value is None else "is not in the dataset"
                raise EvalConfigError(f"{where}: column {spec['from']!r} {state} (columns: {sorted(row)})")
            value = spec["default"]
    if spec.get("json") and isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise EvalConfigError(f"{where}: json: {value[:80]!r} is not JSON ({e})") from None
    if "regex" in spec:
        import re

        matches = list(re.finditer(spec["regex"], str(value), re.M))
        if not matches:
            raise EvalConfigError(f"{where}: regex {spec['regex']!r} does not match {str(value)[-80:]!r}")
        group = spec.get("group", 1 if matches[-1].re.groups else 0)
        value = matches[-1].group(group)
    if isinstance(value, str):
        for old, new in (spec.get("replace") or {}).items():
            value = value.replace(str(old), str(new))
        if spec.get("strip", True):
            value = value.strip()
    if "map" in spec:
        table = {str(k): v for k, v in spec["map"].items()}
        if str(value) not in table:
            raise EvalConfigError(f"{where}: map has no entry for {value!r} (entries: {sorted(table)})")
        value = table[str(value)]
    if spec.get("choices"):
        if not isinstance(value, list) or not value or len(value) > len(LETTERS):
            raise EvalConfigError(f"{where}: choices needs a list of 1-{len(LETTERS)} options, got {value!r}")
        value = "\n".join(f"{LETTERS[i]}. {str(option).strip()}" for i, option in enumerate(value))
    if spec.get("letter"):
        try:
            value = LETTERS[int(value)] if int(value) >= 0 else None
        except (TypeError, ValueError, IndexError):
            value = None
        if value is None:
            raise EvalConfigError(f"{where}: letter needs an option index (0-{len(LETTERS) - 1})")
    if spec.get("tools"):
        value = tool_schemas(value, where)
    return value


def tool_schemas(functions: Any, where: str = "tools") -> list[dict]:
    """OpenAI tool schemas from function specs ({name, description, parameters}, or already-wrapped tools); type
    names of other conventions (dict, float, tuple, ...) become JSON Schema's."""
    if isinstance(functions, str):
        functions = json.loads(functions)
    if isinstance(functions, dict):
        functions = [functions]
    if not isinstance(functions, list):
        raise EvalConfigError(f"{where}: tools needs a list of function specs, got {type(functions).__name__}")
    out = []
    for spec in functions:
        function = spec.get("function", spec) if isinstance(spec, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            raise EvalConfigError(f"{where}: a function spec needs a name, got {spec!r}")
        parameters = _json_schema(function.get("parameters") or {"type": "object", "properties": {}})
        out.append(
            {
                "type": "function",
                "function": {
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "parameters": parameters,
                },
            }
        )
    return out


def _json_schema(node: Any) -> Any:
    if isinstance(node, list):
        return [_json_schema(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, str):
            out[key] = _SCHEMA_TYPES.get(value.lower(), value.lower())
        elif key == "properties" and isinstance(value, dict):  # property names are not keywords
            out[key] = {name: _json_schema(schema) for name, schema in value.items()}
        else:
            out[key] = _json_schema(value)
    return out


# ---------------------------------------------------------------------- items


def build_items(task: TaskConfig, limit: int = 0, rows: list[dict] | None = None) -> list[EvalItem]:
    """The task's items: rows loaded (or given), sampled to the limit, mapped, prompted and named.
    `limit`: the run's own cap (0 = the task's)."""
    rows = load_task_rows(task) if rows is None else [_plain(dict(r)) for r in rows]
    if not rows:
        raise EvalConfigError(f"task {task.name}: {task.dataset} has no rows")
    indices = list(range(len(rows)))
    cap = limit or task.limit
    if cap and cap < len(rows):  # a fixed sample in file order: the same items for every checkpoint
        indices = sorted(random.Random(task.sample_seed).sample(indices, cap))
    items: list[EvalItem] = []
    seen: set[str] = set()
    for index in indices:
        where = f"task {task.name}, row {index}"
        row = dict(rows[index])
        for name, spec in task.fields.items():
            row[name] = map_field(spec, row, f"{where}: fields.{name}")
        if task.messages:
            row["messages"] = [
                {**m, "content": render(m["content"], row, f"{where}: messages[{i}]")}
                for i, m in enumerate(task.messages)
            ]
        elif task.prompt:
            row["messages"] = [{"role": "user", "content": render(task.prompt, row, f"{where}: prompt").strip()}]
        elif task.prompt_field and lookup(row, task.prompt_field) is _MISSING:
            raise EvalConfigError(f"{where}: prompt_field {task.prompt_field!r} is not a column ({sorted(row)})")
        key = index
        if task.id_field:
            key = lookup(row, task.id_field)
            if key is _MISSING or key is None:
                raise EvalConfigError(f"{where}: id_field {task.id_field!r} is missing")
        item_id = f"{task.name}/{key}"
        if item_id in seen:
            raise EvalConfigError(f"{where}: id {item_id!r} appears twice (id_field must be unique per item)")
        seen.add(item_id)
        cluster = None
        if task.cluster_field:
            value = lookup(row, task.cluster_field)
            if value is _MISSING:
                raise EvalConfigError(f"{where}: cluster_field {task.cluster_field!r} is missing")
            cluster = None if value is None else str(value)
        items.append(EvalItem(item_id, cluster, row))
    logger.info("task %s: %d items (of %d rows)", task.name, len(items), len(rows))
    return items


def prompt_field(task: TaskConfig) -> str:
    """The RL data.prompt_field for the task's rows: the rendered chat, the task's column, or RL's default."""
    return "messages" if task.messages or task.prompt else task.prompt_field
