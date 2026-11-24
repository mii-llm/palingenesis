"""Contrastive twin pairs for preference training: build them, and check a pair file for them.

Twins are prompts that differ by a small edit and need different answers: "the store closes at 18:00" / "the store
closes at 17:00", a negated condition, a swapped unit, a constraint added. A model that misreads one twin as the other
gives the sibling's answer, so the most informative rejected answer for a prompt is its twin's correct answer: the pair
then differs in the answer exactly where the prompt differs, and nothing else separates chosen from rejected (no
length, style or fluency cue to learn instead). Every twin pair is written in both directions (P: A over A', and
P': A' over A), so the preference is about reading the prompt, not about preferring one answer.

    pgs pairs build twins.jsonl -o pairs.jsonl          # rows {twin, prompt, answer}: rows sharing `twin` are siblings
    pgs pairs check pairs.jsonl                         # any DPO file: twins, direction balance, contradictions

build: input rows carry a group id (--twin_field, default "twin"), a prompt (string or chat messages) and the correct
answer (string, assistant message or list of messages). Every ordered pair of siblings becomes one DPO row
{prompt, chosen: own answer, rejected: sibling's answer, twin, direction}, plus tools / chat_template_kwargs when the
row has them. A pair is dropped (and counted) when the answers are the same after normalisation (no preference), when
the prompts are identical, or when the prompts differ by more than --max_edit (the share of changed words, 0.5) or
--max_edit_words. With --prediction_field (the model's own answer to the prompt), each pair records misread: true when
the model's answer to this prompt is the sibling's answer, and --only_misread keeps those pairs only: the on-policy
twin pairs, where the model's own error is the rejected side.

check: reads a DPO file (prompt / chosen / rejected, implicit prompts included), and reports
    identical      rows whose chosen and rejected are the same text (no preference; an error)
    contradictions a (prompt, chosen, rejected) whose reverse (prompt, rejected, chosen) is also in the file (an error)
    duplicates     identical rows
    twin pairs     rows whose rejected answer is the chosen answer of another row with a near-identical prompt (the
                   misread signature), with the edit size of the prompts. A twin is seen only when its sibling's
                   answer is the chosen side of some row of the file
    unbalanced     twin pairs without their reverse direction (P': A' over A); --require_balanced makes it an error
Exit code 1 on errors.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import permutations
from typing import Any

_WORD = re.compile(r"\w+|[^\w\s]")
# check: rows whose chosen answer equals a row's rejected one, compared with it for a near-identical prompt (a common
# short answer, "Yes" or "4", can be the chosen side of thousands of rows: the scan stops after these many)
MAX_CANDIDATES = 1000


def text_of(value: Any, reasoning: bool = False) -> str:
    """The text of an answer or a prompt: a string, a message, or a list of messages (role-tagged when there are
    several roles). `reasoning`: include an assistant turn's reasoning (reasoning / reasoning_content)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        text = "" if content is None else str(content)
        calls = value.get("tool_calls")
        if calls:
            text += "\n" + json.dumps(calls, sort_keys=True, ensure_ascii=False)
        if reasoning:
            trace = value.get("reasoning") or value.get("reasoning_content") or ""
            text = f"{trace}\n{text}" if trace else text
        return text
    if isinstance(value, list):
        roles = {m.get("role") for m in value if isinstance(m, dict)}
        parts = [text_of(m, reasoning) for m in value]
        if len(roles) > 1:
            parts = [f"{m.get('role', '?')}: {p}" for m, p in zip(value, parts)]
        return "\n".join(parts)
    return str(value)


def normalize(text: str) -> str:
    """Whitespace collapsed, case folded: what two answers must differ by to be a preference."""
    return " ".join(text.split()).casefold()


def edit_stats(a: str, b: str) -> dict[str, float]:
    """How far apart two prompts are, on words and punctuation marks: the changed tokens (inserted, deleted or
    replaced, counted on the longer side of each change) and their share of the longer prompt."""
    x, y = _WORD.findall(a), _WORD.findall(b)

    def changed(u, v) -> int:
        ops = difflib.SequenceMatcher(a=u, b=v, autojunk=False).get_opcodes()
        return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")

    n = min(changed(x, y), changed(y, x))  # the matcher's alignment is greedy, not symmetric
    return {"changed_words": float(n), "edit": n / max(1, len(x), len(y))}


def _load(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{n}: not JSON ({e})") from None
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{n}: a row must be a JSON object")
                rows.append(row)
    return rows


def build_twin_pairs(
    rows: list[dict],
    twin_field: str = "twin",
    prompt_field: str = "prompt",
    answer_field: str = "answer",
    prediction_field: str = "",
    max_edit: float = 0.5,
    max_edit_words: int = 0,
    only_misread: bool = False,
) -> tuple[list[dict], dict[str, Any]]:
    """(DPO rows, report) from twin rows: every ordered pair of siblings (same twin_field) is a pair whose chosen side
    is the row's own answer and whose rejected side is the sibling's. See the module docstring for the drops."""
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, row in enumerate(rows):
        for key in (twin_field, prompt_field, answer_field):
            if row.get(key) is None:
                raise ValueError(f"row {i}: no {key!r} (twin rows need {twin_field}, {prompt_field}, {answer_field})")
        groups[json.dumps(row[twin_field], sort_keys=True)].append((i, row))
    dropped: Counter = Counter()
    pairs: list[dict] = []
    misread = 0
    edits: list[float] = []
    for twin, members in groups.items():
        if len(members) < 2:
            dropped["no_sibling"] += 1
            continue
        for (i, a), (j, b) in permutations(members, 2):
            prompt_a, prompt_b = text_of(a[prompt_field]), text_of(b[prompt_field])
            if normalize(prompt_a) == normalize(prompt_b):
                dropped["same_prompt"] += 1
                continue
            if normalize(text_of(a[answer_field])) == normalize(text_of(b[answer_field])):
                dropped["same_answer"] += 1
                continue
            stats = edit_stats(prompt_a, prompt_b)
            if stats["edit"] > max_edit or (max_edit_words and stats["changed_words"] > max_edit_words):
                dropped["prompts_too_different"] += 1
                continue
            pair: dict[str, Any] = {
                "prompt": a[prompt_field],
                "chosen": a[answer_field],
                "rejected": b[answer_field],
                "twin": a[twin_field],
                "direction": f"{i}>{j}",
                "prompt_edit_words": int(stats["changed_words"]),
            }
            for key in ("tools", "chat_template_kwargs", "sft_weight"):
                if key in a:
                    pair[key] = a[key]
            if prediction_field:
                prediction = a.get(prediction_field)
                hit = prediction is not None and normalize(text_of(prediction)) == normalize(text_of(b[answer_field]))
                pair["misread"] = hit
                misread += hit
                if only_misread and not hit:
                    dropped["not_misread"] += 1
                    continue
            edits.append(stats["changed_words"])
            pairs.append(pair)
    report: dict[str, Any] = {
        "rows": len(rows),
        "twin_groups": len(groups),
        "pairs": len(pairs),
        "dropped": dict(dropped),
        "prompt_edit_words_mean": sum(edits) / len(edits) if edits else None,
    }
    if prediction_field:
        report["misread_pairs"] = misread
    return pairs, report


def _dpo_sides(row: dict, prompt_field: str, chosen_field: str, rejected_field: str) -> tuple[str, str, str]:
    """(prompt, chosen, rejected) texts of a DPO row; an implicit prompt is the conversation before the last
    assistant turn the two sides share."""
    chosen, rejected = row.get(chosen_field), row.get(rejected_field)
    if chosen is None or rejected is None:
        raise ValueError(f"no {chosen_field!r} / {rejected_field!r}")
    prompt = row.get(prompt_field)
    if prompt is None and isinstance(chosen, list) and isinstance(rejected, list):
        shared = 0
        while shared < min(len(chosen), len(rejected)) and chosen[shared] == rejected[shared]:
            shared += 1
        while shared and isinstance(chosen[shared - 1], dict) and chosen[shared - 1].get("role") == "assistant":
            shared -= 1
        prompt, chosen, rejected = chosen[:shared], chosen[shared:], rejected[shared:]
    return text_of(prompt), text_of(chosen), text_of(rejected)


def check_pairs(
    rows: list[dict],
    prompt_field: str = "prompt",
    chosen_field: str = "chosen",
    rejected_field: str = "rejected",
    max_edit: float = 0.5,
) -> dict[str, Any]:
    """Report on a DPO file (see the module docstring). Each list holds at most 20 example row indices."""
    sides = []
    for i, row in enumerate(rows):
        try:
            p, c, r = _dpo_sides(row, prompt_field, chosen_field, rejected_field)
        except ValueError as e:
            raise ValueError(f"row {i}: {e}") from None
        sides.append((normalize(p), normalize(c), normalize(r)))
    identical = [i for i, (_, c, r) in enumerate(sides) if c == r]
    seen: dict[tuple, int] = {}
    duplicates = []
    for i, key in enumerate(sides):
        if key in seen:
            duplicates.append(i)
        else:
            seen[key] = i
    contradictions = [i for i, (p, c, r) in enumerate(sides) if c != r and (p, r, c) in seen]
    by_chosen: dict[str, list[int]] = defaultdict(list)
    for i, (_, c, _) in enumerate(sides):
        by_chosen[c].append(i)
    twins, unbalanced, edits = [], [], []
    for i, (p, c, r) in enumerate(sides):
        if c == r:
            continue
        partner = None
        for j in by_chosen.get(r, ())[:MAX_CANDIDATES]:
            if j != i and sides[j][0] != p:
                stats = edit_stats(p, sides[j][0])
                if stats["edit"] <= max_edit:
                    partner = (j, stats)
                    break
        if partner is None:
            continue
        j, stats = partner
        twins.append(i)
        edits.append(stats["changed_words"])
        if (sides[j][0], r, c) not in seen:  # the sibling's prompt with the reverse preference
            unbalanced.append(i)
    return {
        "rows": len(rows),
        "identical": len(identical),
        "contradictions": len(contradictions),
        "duplicates": len(duplicates),
        "twin_pairs": len(twins),
        "twin_share": len(twins) / len(rows) if rows else 0.0,
        "unbalanced_twin_pairs": len(unbalanced),
        "prompt_edit_words_mean": sum(edits) / len(edits) if edits else None,
        "examples": {
            "identical": identical[:20],
            "contradictions": contradictions[:20],
            "duplicates": duplicates[:20],
            "twin_pairs": twins[:20],
            "unbalanced_twin_pairs": unbalanced[:20],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pgs pairs", description="Contrastive twin pairs for preference training.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="twin rows -> DPO pairs (both directions)")
    b.add_argument("input")
    b.add_argument("-o", "--output", required=True)
    b.add_argument("--twin_field", default="twin")
    b.add_argument("--prompt_field", default="prompt")
    b.add_argument("--answer_field", default="answer")
    b.add_argument("--prediction_field", default="", help="the model's own answer: marks misread pairs")
    b.add_argument("--only_misread", action="store_true", help="keep the pairs the model misread only")
    b.add_argument("--max_edit", type=float, default=0.5, help="max share of changed prompt words")
    b.add_argument("--max_edit_words", type=int, default=0, help="max changed prompt words (0 = no limit)")
    c = sub.add_parser("check", help="report twins, direction balance, contradictions in a DPO file")
    c.add_argument("input")
    c.add_argument("--prompt_field", default="prompt")
    c.add_argument("--chosen_field", default="chosen")
    c.add_argument("--rejected_field", default="rejected")
    c.add_argument("--max_edit", type=float, default=0.5, help="prompts within this edit share are twins")
    c.add_argument("--require_balanced", action="store_true", help="unbalanced twin pairs are an error")
    for p in (b, c):
        p.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not 0 < args.max_edit <= 1:
            raise ValueError("--max_edit must be in (0, 1]")
        rows = _load(args.input)
        if args.command == "build":
            if args.only_misread and not args.prediction_field:
                raise ValueError("--only_misread needs --prediction_field")
            pairs, report = build_twin_pairs(
                rows,
                args.twin_field,
                args.prompt_field,
                args.answer_field,
                args.prediction_field,
                args.max_edit,
                args.max_edit_words,
                args.only_misread,
            )
            with open(args.output, "w") as f:
                for pair in pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            report["output"] = args.output
            code = 0 if pairs else 1
        else:
            report = check_pairs(rows, args.prompt_field, args.chosen_field, args.rejected_field, args.max_edit)
            errors = report["identical"] + report["contradictions"]
            if args.require_balanced:
                errors += report["unbalanced_twin_pairs"]
            code = 1 if errors else 0
    except (OSError, ValueError) as e:
        print(f"pgs pairs: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for key, value in report.items():
            if key != "examples":
                print(f"{key:<26} {value}")
        for key, value in report.get("examples", {}).items():
            if value:
                print(f"{key:<26} rows {value}")
    return code


if __name__ == "__main__":
    sys.exit(main())
