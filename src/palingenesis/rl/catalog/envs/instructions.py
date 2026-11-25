"""instruction-following: writing requests with constraints a program can check (IFEval style).

Rows: {"prompt", "constraints": [{"id": ..., **arguments}, ...]}. The reply must satisfy every constraint; each is a
small deterministic check (CHECKS below): word, sentence and paragraph counts, keywords to include or avoid,
keyword and letter frequencies, case, bullet lists, titles, sections, highlighted spans, placeholders, a
postscript, an exact ending, quotation marks, no commas.

Evaluation only (spec.trainable is False). The checks see form, not content: text that says nothing and satisfies
every constraint gets the full reward, and the generator's own oracle is exactly such text. As an evaluation of a
model that is trying to answer, the pass rate measures whether it follows the instructions; as an RL reward on its
own it teaches filler. To train on it, add a reward for the content (a judge) and keep this one as a gate.

The generator composes two or three compatible constraints per request and keeps a row only if its constructed
solution passes and the cheap replies do not (the prompt repeated, the keywords listed, an empty reply). The
"ifeval" dataset converts google/IFEval rows whose instructions all have a check here (others are dropped); these
checks follow IFEval's definitions but are not its scorer, so the numbers are not comparable with published ones.
"""

from __future__ import annotations

import random
import re
from typing import Any, Callable

from palingenesis.rl.catalog.spec import DatasetSpec, EnvSpec

RELATIONS = ("at least", "at most", "less than", "exactly")


def _holds(count: int, relation: str, n: int) -> bool:
    return {"at least": count >= n, "at most": count <= n, "less than": count < n, "exactly": count == n}[relation]


def words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text)


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if re.search(r"\w", s)]


def paragraphs(text: str, divider: str = "") -> list[str]:
    parts = text.split(divider) if divider else re.split(r"\n\s*\n", text)
    return [p for p in parts if p.strip()]


def _count_word(text: str, word: str) -> int:
    return len(re.findall(rf"(?<!\w){re.escape(word)}(?!\w)", text, re.I))


def _word_count(r: str, relation: str, n: int) -> bool:
    return _holds(len(words(r)), relation, n)


def _sentence_count(r: str, relation: str, n: int) -> bool:
    return _holds(len(sentences(r)), relation, n)


def _paragraph_count(r: str, n: int, divider: str = "") -> bool:
    return len(paragraphs(r, divider)) == n


def _include_keywords(r: str, keywords: list[str]) -> bool:
    return all(_count_word(r, k) for k in keywords)


def _forbidden_words(r: str, forbidden: list[str]) -> bool:
    return bool(r.strip()) and not any(_count_word(r, k) for k in forbidden)


def _keyword_frequency(r: str, keyword: str, relation: str, n: int) -> bool:
    return _holds(_count_word(r, keyword), relation, n)


def _letter_frequency(r: str, letter: str, relation: str, n: int) -> bool:
    return _holds(r.lower().count(letter.lower()), relation, n)


def _lowercase(r: str) -> bool:
    return any(c.isalpha() for c in r) and r == r.lower()


def _uppercase(r: str) -> bool:
    return any(c.isalpha() for c in r) and r == r.upper()


def _capital_words(r: str, relation: str, n: int) -> bool:
    return _holds(sum(w.isupper() and w.isalpha() and len(w) > 1 for w in words(r)), relation, n)


def _bullet_points(r: str, n: int) -> bool:
    return len(re.findall(r"^\s*[*-]\s+\S", r, re.M)) == n


def _title(r: str) -> bool:
    return bool(re.search(r"<<[^\n<>]+>>", r))


def _sections(r: str, n: int, marker: str = "SECTION") -> bool:
    return len(re.findall(rf"^\s*{re.escape(marker)}\s+\d+", r, re.M | re.I)) == n


def _highlighted(r: str, n: int) -> bool:
    return len(re.findall(r"\*[^\n*]+\*", r)) >= n


def _placeholders(r: str, n: int) -> bool:
    return len(re.findall(r"\[[^\[\]\n]+\]", r)) >= n


def _postscript(r: str, marker: str = "P.S.") -> bool:
    return bool(re.search(rf"^\s*{re.escape(marker)}", r, re.M | re.I))


def _end_phrase(r: str, phrase: str) -> bool:
    return r.strip().casefold().endswith(phrase.strip().casefold())


def _quotation(r: str) -> bool:
    text = r.strip()
    return len(text) > 1 and text[0] == '"' and text[-1] == '"'


def _no_commas(r: str) -> bool:
    return bool(r.strip()) and "," not in r


CHECKS: dict[str, Callable[..., bool]] = {
    "word_count": _word_count,
    "sentence_count": _sentence_count,
    "paragraph_count": _paragraph_count,
    "include_keywords": _include_keywords,
    "forbidden_words": _forbidden_words,
    "keyword_frequency": _keyword_frequency,
    "letter_frequency": _letter_frequency,
    "lowercase": _lowercase,
    "uppercase": _uppercase,
    "capital_words": _capital_words,
    "bullet_points": _bullet_points,
    "title": _title,
    "sections": _sections,
    "highlighted": _highlighted,
    "placeholders": _placeholders,
    "postscript": _postscript,
    "end_phrase": _end_phrase,
    "quotation": _quotation,
    "no_commas": _no_commas,
}


def describe(constraint: dict) -> str:
    """The constraint as the sentence the request shows."""
    c = constraint
    return {
        "word_count": lambda: f"Use {c['relation']} {c['n']} words.",
        "sentence_count": lambda: f"Write {c['relation']} {c['n']} sentences.",
        "paragraph_count": lambda: f"Write exactly {c['n']} paragraphs, separated by blank lines.",
        "include_keywords": lambda: "Include the words " + " and ".join(f'"{k}"' for k in c["keywords"]) + ".",
        "forbidden_words": lambda: "Do not use the words " + " or ".join(f'"{k}"' for k in c["forbidden"]) + ".",
        "keyword_frequency": lambda: f'Use the word "{c["keyword"]}" {c["relation"]} {c["n"]} times.',
        "letter_frequency": lambda: f'The letter "{c["letter"]}" must appear {c["relation"]} {c["n"]} times.',
        "lowercase": lambda: "Write everything in lowercase letters: no capital letters at all.",
        "uppercase": lambda: "Write everything in capital letters.",
        "capital_words": lambda: f"Write {c['relation']} {c['n']} words entirely in capital letters.",
        "bullet_points": lambda: f"Include exactly {c['n']} bullet points, each on its own line starting with \"* \".",
        "title": lambda: "Start with a title in double angle brackets, like <<title>>.",
        "sections": lambda: f"Organize the text in exactly {c['n']} sections, each starting with a line \"SECTION k\" (k = 1, 2, ...).",
        "highlighted": lambda: f"Highlight at least {c['n']} phrases by putting them between asterisks, *like this*.",
        "placeholders": lambda: f"Include at least {c['n']} placeholders in square brackets, like [name].",
        "postscript": lambda: 'Add a postscript at the end, on a line starting with "P.S.".',
        "end_phrase": lambda: f'End the reply with exactly this phrase and nothing after it: "{c["phrase"]}"',
        "quotation": lambda: "Wrap the whole reply in double quotation marks.",
        "no_commas": lambda: "Do not use any commas.",
    }[c["id"]]()


def check(reply: str, constraint: dict) -> bool:
    arguments = {k: v for k, v in constraint.items() if k != "id"}
    try:
        return bool(CHECKS[constraint["id"]](reply, **arguments))
    except (KeyError, TypeError):
        return False


class InstructionEnv:
    tools = ()

    def __init__(self):
        self.constraints: list[dict] = []

    def reset(self, constraints: Any = None, **row) -> None:
        import json

        self.constraints = json.loads(constraints) if isinstance(constraints, str) else list(constraints or [])

    def get_reward(self, messages: list[dict]) -> dict[str, float]:
        reply = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "assistant"), "")
        met = [check(reply, c) for c in self.constraints]
        return {
            "followed": float(bool(met) and all(met)),
            "metric/constraints": sum(met) / len(met) if met else 0.0,
        }


# ------------------------------------------------------------------ generator

_TOPICS = ["a lighthouse", "urban beekeeping", "a night train", "the first snow", "a street market", "tide pools",
           "an old library", "learning to swim", "a mountain hut", "a repair shop"]  # fmt: skip
_FILLER = [
    "It shapes how people spend their days.",
    "Many small details make it worth a closer look.",
    "Those who know it well speak of it with care.",
    "Each visit shows something that was missed before.",
    "It rewards patience more than hurry.",
    "People return to it for reasons they rarely explain.",
    "There is a quiet order behind what seems simple.",
    "Its sounds and smells stay in the memory.",
]
_KEYWORDS = ["lantern", "harvest", "compass", "velvet", "orchard", "granite", "whistle", "ribbon", "meadow", "anchor"]
_FORBIDDEN = ["very", "really", "amazing", "beautiful", "interesting", "important"]
_ENDINGS = ["That is all for today.", "Thank you for reading.", "More on this another time."]
_GROUPS = {
    "length": ["word_count", "sentence_count", "paragraph_count"],
    "keywords": ["include_keywords", "forbidden_words", "keyword_frequency", "letter_frequency"],
    "case": ["lowercase", "uppercase", "capital_words"],
    "format": ["bullet_points", "title", "sections", "highlighted", "placeholders"],
    "ending": ["end_phrase", "quotation", "postscript", "no_commas"],
}


def _sample(rng: random.Random, name: str) -> dict:
    if name == "word_count":
        return rng.choice(
            [
                {"relation": "at least", "n": rng.randrange(80, 200, 10)},
                {"relation": "at most", "n": rng.randrange(40, 90, 10)},
            ]
        )
    if name == "sentence_count":
        return rng.choice(
            [
                {"relation": "at least", "n": rng.randint(8, 14)},
                {"relation": "at most", "n": rng.randint(3, 5)},
                {"relation": "exactly", "n": rng.randint(4, 7)},
            ]
        )
    if name == "paragraph_count":
        return {"n": rng.randint(3, 5)}
    if name == "include_keywords":
        return {"keywords": rng.sample(_KEYWORDS, 2)}
    if name == "forbidden_words":
        return {"forbidden": rng.sample(_FORBIDDEN, 3)}
    if name == "keyword_frequency":
        return {
            "keyword": rng.choice(_KEYWORDS),
            "relation": rng.choice(["exactly", "at least"]),
            "n": rng.randint(2, 4),
        }
    if name == "letter_frequency":
        return {"letter": rng.choice("qxzjk"), "relation": "at least", "n": rng.randint(4, 8)}
    if name == "capital_words":
        return {"relation": "at least", "n": rng.randint(3, 6)}
    if name in ("bullet_points", "sections"):
        return {"n": rng.randint(2, 4)}
    if name in ("highlighted", "placeholders"):
        return {"n": rng.randint(2, 3)}
    if name == "end_phrase":
        return {"phrase": rng.choice(_ENDINGS)}
    return {}


def build(topic: str, constraints: list[dict]) -> str:
    """Text made to satisfy the constraints (filler about the topic: the checks do not read content)."""
    by = {c["id"]: c for c in constraints}
    count = 6
    if "word_count" in by:
        n = by["word_count"]["n"]
        count = n // 6 + 2 if by["word_count"]["relation"] == "at least" else max(1, n // 14)
    if "sentence_count" in by:
        n, relation = by["sentence_count"]["n"], by["sentence_count"]["relation"]
        count = max(count, n) if relation == "at least" else min(count, n) if relation == "at most" else n
    extras = []
    if "include_keywords" in by:
        extras.append("Think of the " + " and the ".join(by["include_keywords"]["keywords"]) + ".")
    if "keyword_frequency" in by:
        extras.append(" ".join([by["keyword_frequency"]["keyword"]] * by["keyword_frequency"]["n"]) + " is the word.")
    if "letter_frequency" in by:
        extras.append(by["letter_frequency"]["letter"] * by["letter_frequency"]["n"] + " is a sound it makes.")
    if "capital_words" in by:
        extras.append(
            " ".join(["NOTE", "THIS", "WELL", "NOW", "AND", "THEN", "AGAIN"][: by["capital_words"]["n"] + 1]) + "."
        )
    if "highlighted" in by:
        extras.append(" ".join(f"*point {k + 1}*" for k in range(by["highlighted"]["n"])) + " stand out.")
    if "placeholders" in by:
        extras.append(" ".join(f"[item {k + 1}]" for k in range(by["placeholders"]["n"])) + " can be filled in later.")
    body = [f"This is about {topic}."] + extras
    body += [_FILLER[k % len(_FILLER)] for k in range(max(0, count - len(body)))]
    if "sentence_count" in by and by["sentence_count"]["relation"] != "at least":
        body = body[: max(1, count)]
    parts = by.get("paragraph_count", by.get("sections", {})).get("n", 2)
    parts = max(1, min(parts, len(body)))
    size = -(-len(body) // parts)
    chunks = [" ".join(body[i : i + size]) for i in range(0, len(body), size)]
    while len(chunks) < parts:
        chunks.append(_FILLER[len(chunks) % len(_FILLER)])
    if "sections" in by:
        chunks = [f"SECTION {k + 1}\n{chunk}" for k, chunk in enumerate(chunks)]
    text = "\n\n".join(chunks)
    if "bullet_points" in by:
        text += "\n\n" + "\n".join(f"* point number {k + 1}" for k in range(by["bullet_points"]["n"]))
    if "title" in by:
        text = f"<<notes on {topic}>>\n{text}"
    if "postscript" in by:
        text += "\n\nP.S. One more thought may follow."
    if "end_phrase" in by:
        text += " " + by["end_phrase"]["phrase"]
    if "lowercase" in by:
        text = text.lower()
    if "uppercase" in by:
        text = text.upper()
    if "quotation" in by:
        text = f'"{text}"'
    return text


def _keyword_list(row: dict) -> str:
    quoted = re.findall(r'"([^"]+)"', row["prompt"])
    return " ".join(quoted) or "words"


def generate(n: int, seed: int = 0, constraints: int = 0) -> list[dict]:
    """Requests with `constraints` checks each (0: two or three), from different groups."""
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        topic = rng.choice(_TOPICS)
        k = constraints or rng.choice([2, 3])
        groups = rng.sample(list(_GROUPS), min(k, len(_GROUPS)))
        if "length" not in groups and "format" not in groups:
            groups[0] = rng.choice(["length", "format"])  # something about the shape of the text, always
        chosen = [
            {"id": name, **_sample(rng, name)} for name in (rng.choice(_GROUPS[g]) for g in dict.fromkeys(groups))
        ]
        prompt = f"Write a short text about {topic}. " + " ".join(describe(c) for c in chosen)
        solution = build(topic, chosen)
        row = {"prompt": prompt}
        cheap = ["", prompt, _keyword_list(row), "I cannot help with that.", plain_answer({"topic": topic}, [])]
        if not all(check(solution, c) for c in chosen) or any(all(check(text, c) for c in chosen) for text in cheap):
            continue
        rows.append(
            {
                "id": f"instruction-following/synthetic/{seed}/{len(rows)}",
                "prompt": prompt,
                "constraints": chosen,
                "solution": solution,
                "topic": topic,
            }
        )
    return rows


def _relation(value: str | None) -> str:
    return value if value in RELATIONS else "at least"


_IFEVAL: dict[str, Callable[[dict], dict]] = {
    "length_constraints:number_words": lambda k: {
        "id": "word_count",
        "relation": _relation(k.get("relation")),
        "n": k["num_words"],
    },
    "length_constraints:number_sentences": lambda k: {
        "id": "sentence_count",
        "relation": _relation(k.get("relation")),
        "n": k["num_sentences"],
    },
    "length_constraints:number_paragraphs": lambda k: {
        "id": "paragraph_count",
        "n": k["num_paragraphs"],
        "divider": "***",
    },
    "keywords:existence": lambda k: {"id": "include_keywords", "keywords": list(k["keywords"])},
    "keywords:forbidden_words": lambda k: {"id": "forbidden_words", "forbidden": list(k["forbidden_words"])},
    "keywords:frequency": lambda k: {
        "id": "keyword_frequency",
        "keyword": k["keyword"],
        "relation": _relation(k.get("relation")),
        "n": k["frequency"],
    },
    "keywords:letter_frequency": lambda k: {
        "id": "letter_frequency",
        "letter": k["letter"],
        "relation": _relation(k.get("let_relation")),
        "n": k["let_frequency"],
    },
    "change_case:english_lowercase": lambda k: {"id": "lowercase"},
    "change_case:english_capital": lambda k: {"id": "uppercase"},
    "change_case:capital_word_frequency": lambda k: {
        "id": "capital_words",
        "relation": _relation(k.get("capital_relation")),
        "n": k["capital_frequency"],
    },
    "detectable_format:number_bullet_lists": lambda k: {"id": "bullet_points", "n": k["num_bullets"]},
    "detectable_format:title": lambda k: {"id": "title"},
    "detectable_format:multiple_sections": lambda k: {
        "id": "sections",
        "n": k["num_sections"],
        "marker": k["section_spliter"],
    },
    "detectable_format:number_highlighted_sections": lambda k: {"id": "highlighted", "n": k["num_highlights"]},
    "detectable_content:number_placeholders": lambda k: {"id": "placeholders", "n": k["num_placeholders"]},
    "detectable_content:postscript": lambda k: {"id": "postscript", "marker": k["postscript_marker"]},
    "startend:end_checker": lambda k: {"id": "end_phrase", "phrase": k["end_phrase"]},
    "startend:quotation": lambda k: {"id": "quotation"},
    "punctuation:no_comma": lambda k: {"id": "no_commas"},
}


def from_ifeval(row: dict) -> dict | None:
    """google/IFEval: rows whose instructions all have a check here; the others are dropped."""
    constraints = []
    for name, kwargs in zip(row.get("instruction_id_list") or [], row.get("kwargs") or []):
        convert = _IFEVAL.get(name)
        if convert is None:
            return None
        try:
            constraint = convert(kwargs or {})
        except (KeyError, TypeError):
            return None
        if any(v is None for v in constraint.values()):
            return None
        constraints.append(constraint)
    if not constraints:
        return None
    return {"id": f"ifeval/{row.get('key')}", "prompt": row["prompt"], "constraints": constraints}


# ------------------------------------------------------------------- policies


def oracle(row: dict, messages: list[dict]) -> str:
    return row.get("solution", "")


def keyword_list(row: dict, messages: list[dict]) -> str:
    """Every quoted word or phrase of the request, and nothing else."""
    return _keyword_list(row)


def plain_answer(row: dict, messages: list[dict]) -> str:
    """A fluent paragraph about the topic that ignores the constraints."""
    topic = row.get("topic", "the subject")
    return f"Here is a short text about {topic}. " + " ".join(_FILLER[:4])


SPEC = EnvSpec(
    name="instruction-following",
    version="1.0.0",
    family="instruction-following",
    summary="Writing requests with verifiable constraints (length, keywords, case, format, ending), IFEval style.",
    env=InstructionEnv,
    license="Apache-2.0",
    verifier=(
        "Every constraint of the row holds for the reply (19 deterministic checks of form). Content is not "
        "checked: constraint-satisfying filler passes, which is why the environment is evaluation-only."
    ),
    datasets={
        "synthetic": DatasetSpec(
            generator=generate,
            license="Apache-2.0",
            description="Generated requests with two or three compatible constraints and a constructed solution.",
        ),
        "ifeval": DatasetSpec(
            hub_id="google/IFEval",
            split="train",
            convert=from_ifeval,
            license="Apache-2.0",
            url="https://huggingface.co/datasets/google/IFEval",
            citation="Zhou et al., Instruction-Following Evaluation for Large Language Models, 2023 (arXiv:2311.07911)",
            description=(
                "IFEval prompts whose instructions all map to a check here (no solutions: not auditable with an "
                "oracle; not the official scorer)."
            ),
        ),
    },
    difficulty={"option": "constraints", "levels": [1, 2, 3, 4]},
    trainable=False,
    max_turns=1,
    max_new_tokens=1024,
    oracle=oracle,
    degenerate={"keyword_list": keyword_list, "plain_answer": plain_answer},
    tags=("single-turn", "constraints"),
    notes="Evaluation only: the checks read form, not content. Pair with a content reward before training on it.",
)
