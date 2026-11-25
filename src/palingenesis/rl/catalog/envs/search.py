"""corpus-qa: questions answered from a local corpus through a retrieval tool.

Rows: {"prompt", "answer", "corpus"}. The corpus is local: {"seed": s, "size": n} (a generated world of fictional
people, companies and towns, one document each, rebuilt from the seed), a path to a JSONL file of
{"id", "title", "text"} documents, or the documents themselves. The policy has two tools: search(query) (BM25 over
the corpus, top hits with a snippet) and read(doc_id). The final message must end with "ANSWER: <answer>", compared
with the gold by normalized exact match (SQuAD normalization; a list of gold answers accepts any of them).

The generated world is fictional on purpose: nothing in it can be known without retrieving it, single-hop questions
need one document and two-hop questions two (the town a person was born in, then that town's river), and exact
match on a committed answer line means that pasting search results or listing every candidate scores 0. The index
of a corpus is built once and shared by every episode on it.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, OrderedDict
from typing import Any

from palingenesis.rl.catalog.envs.stateful import ANSWER_NOTE, final_answer
from palingenesis.rl.catalog.runner import scripted, turn_index
from palingenesis.rl.catalog.spec import DatasetSpec, EnvSpec
from palingenesis.rl.env import ToolFailed

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Index:
    """BM25 over a list of {"id", "title", "text"} documents."""

    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.documents = {str(d["id"]): d for d in documents}
        self.ids = list(self.documents)
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lengths = []
        for i, doc in enumerate(documents):
            tokens = tokenize(f"{doc.get('title', '')} {doc['text']}")
            self.lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                self.postings.setdefault(term, []).append((i, count))
        self.average = sum(self.lengths) / max(1, len(self.lengths))

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        scores: dict[int, float] = {}
        n = len(self.ids)
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for i, count in postings:
                norm = count + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.average)
                scores[i] = scores.get(i, 0.0) + idf * count * (self.k1 + 1) / norm
        best = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [(self.ids[i], score) for i, score in best]


# ---------------------------------------------------------------------- world

_ONSETS = ["Br", "K", "V", "T", "M", "Dr", "S", "L", "N", "Gr", "F", "H", "Z", "P", "Qu", "R"]
_VOWELS = ["a", "e", "i", "o", "u", "ae", "io", "ou"]
_CODAS = ["nd", "rk", "l", "ss", "th", "m", "x", "rn", "sk", "v"]
_JOBS = ["cartographer", "glassblower", "botanist", "locksmith", "pilot", "archivist", "luthier", "surveyor"]
_GOODS = ["weather balloons", "ceramic tiles", "sail cloth", "clock springs", "seed drills", "signal lamps"]
_REGIONS = ["Upper Marches", "Salt Coast", "Greenfold", "Ironvale", "Low Fens"]


def _name(rng: random.Random, parts: int = 2) -> str:
    text = "".join(rng.choice(_ONSETS).lower() + rng.choice(_VOWELS) + rng.choice(_CODAS) for _ in range(parts))
    return text.capitalize()


def build_world(seed: int, size: int = 120) -> dict:
    """A fictional world of size // 3 towns, companies and people each; world["documents"] is its corpus."""
    rng = random.Random(f"world/{seed}")
    count = max(4, size // 3)
    names: set[str] = set()

    def fresh(make) -> str:
        while True:
            name = make()
            if name not in names:
                names.add(name)
                return name

    towns = [
        {
            "name": fresh(lambda: _name(rng)),
            "region": rng.choice(_REGIONS),
            "river": fresh(lambda: _name(rng, 1) + "water"),
            "population": rng.randrange(2, 90) * 1000 + rng.randrange(100, 999),
        }  # fmt: skip
        for _ in range(count)
    ]
    companies = [
        {
            "name": fresh(lambda: f"{_name(rng)} {rng.choice(['Works', 'Guild', 'Mills', 'Yard'])}"),
            "founded": rng.randint(1700, 1990),
            "town": rng.choice(towns)["name"],
            "product": rng.choice(_GOODS),
        }  # fmt: skip
        for _ in range(count)
    ]
    people = [
        {
            "name": fresh(lambda: f"{_name(rng)} {_name(rng)}"),
            "job": rng.choice(_JOBS),
            "born": rng.randint(1880, 1990),
            "birthplace": rng.choice(towns)["name"],
            "employer": rng.choice(companies)["name"],
        }  # fmt: skip
        for _ in range(count)
    ]
    documents = []
    for t in towns:
        text = (
            f"{t['name']} is a town in the {t['region']} region. The river {t['river']} flows through it. "
            f"At the last count it had {t['population']} inhabitants."
        )
        documents.append({"id": f"town-{len(documents)}", "title": t["name"], "text": text})
    for c in companies:
        text = (
            f"{c['name']} is a company founded in {c['founded']}. Its headquarters are in the town of {c['town']}. "
            f"It is known for making {c['product']}."
        )
        documents.append({"id": f"company-{len(documents)}", "title": c["name"], "text": text})
    for p in people:
        text = (
            f"{p['name']} is a {p['job']} born in {p['born']} in the town of {p['birthplace']}. "
            f"{p['name'].split()[0]} works for {p['employer']}."
        )
        documents.append({"id": f"person-{len(documents)}", "title": p["name"], "text": text})
    return {"towns": towns, "companies": companies, "people": people, "documents": documents}


_indexes: OrderedDict[str, Index] = OrderedDict()


def corpus_index(corpus: Any) -> Index:
    """The (cached) index of a corpus: {"seed", "size"}, a JSONL path, or a list of documents."""
    if isinstance(corpus, str) and corpus.lstrip().startswith(("{", "[")):
        corpus = json.loads(corpus)
    key = corpus if isinstance(corpus, str) else json.dumps(corpus, sort_keys=True)[:2000] + str(len(corpus))
    index = _indexes.get(key)
    if index is None:
        if isinstance(corpus, dict):
            documents = build_world(int(corpus.get("seed", 0)), int(corpus.get("size", 120)))["documents"]
        elif isinstance(corpus, str):
            with open(corpus) as f:
                documents = [json.loads(line) for line in f if line.strip()]
        else:
            documents = list(corpus)
        index = _indexes[key] = Index(documents)
        while len(_indexes) > 8:
            _indexes.popitem(last=False)
    else:
        _indexes.move_to_end(key)
    return index


def normalize_answer(text: str) -> list[str]:
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    return [t for t in text.split() if t not in ("a", "an", "the")]


class CorpusQAEnv:
    tools = ("search", "read")

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.index: Index | None = None
        self.answers: list[str] = []
        self.searches = 0

    def reset(self, corpus: Any = None, answer: Any = "", **row) -> None:
        self.index = corpus_index(corpus if corpus is not None else {"seed": 0})
        self.answers = [str(a) for a in answer] if isinstance(answer, list) else [str(answer)]
        self.searches = 0

    def search(self, query: str) -> str:
        """Search the corpus; returns the best-matching documents as "[doc_id] title: beginning of the text".

        Args:
            query: Words to look for.
        """
        self.searches += 1
        hits = self.index.search(str(query), self.top_k)
        if not hits:
            return "No documents match."
        documents = self.index.documents
        return "\n".join(f"[{i}] {documents[i].get('title', '')}: {documents[i]['text'][:120]}..." for i, _ in hits)

    def read(self, doc_id: str) -> str:
        """Read a whole document.

        Args:
            doc_id: A document id from the search results.
        """
        doc = self.index.documents.get(str(doc_id).strip("[] "))
        if doc is None:
            raise ToolFailed(f"no document with id {doc_id!r}")
        return f"{doc.get('title', '')}\n\n{doc['text']}"

    def get_reward(self, messages: list[dict]) -> dict[str, float]:
        got = final_answer(messages)
        correct = got is not None and any(normalize_answer(got) == normalize_answer(a) for a in self.answers)
        return {"correct": float(correct), "metric/searched": float(self.searches > 0)}


# ------------------------------------------------------------------ generator


def generate(n: int, seed: int = 0, size: int = 120, hops: int = 0) -> list[dict]:
    """Questions about one generated world (corpus {"seed": seed, "size": size}); hops 1 or 2, 0: both in turn."""
    world = build_world(seed, size)
    rng = random.Random(f"questions/{seed}")
    ids = {d["title"]: d["id"] for d in world["documents"]}
    towns = {t["name"]: t for t in world["towns"]}
    companies = {c["name"]: c for c in world["companies"]}
    rows = []
    for i in range(n):
        person, company = rng.choice(world["people"]), rng.choice(world["companies"])
        two = hops == 2 or (hops == 0 and i % 2 == 1)
        if two:
            question, answer, path = rng.choice(
                [
                    (f"Which river flows through the town where {person['name']} was born?",
                     towns[person["birthplace"]]["river"], [person["name"], person["birthplace"]]),
                    (f"In what year was the company that {person['name']} works for founded?",
                     str(companies[person["employer"]]["founded"]), [person["name"], person["employer"]]),
                    (f"In which region is the headquarters of {company['name']}?",
                     towns[company["town"]]["region"], [company["name"], company["town"]]),
                ]
            )  # fmt: skip
        else:
            question, answer, path = rng.choice(
                [
                    (f"In which town was {person['name']} born?", person["birthplace"], [person["name"]]),
                    (f"What is {person['name']}'s profession?", person["job"], [person["name"]]),
                    (f"In what year was {company['name']} founded?", str(company["founded"]), [company["name"]]),
                    (f"What does {company['name']} make?", company["product"], [company["name"]]),
                ]
            )
        solution = []
        for title in path:
            solution += [["search", {"query": title}], ["read", {"doc_id": ids[title]}]]
        rows.append(
            {
                "id": f"corpus-qa/synthetic/{seed}/{i}",
                "prompt": f"{question}\n\nThe answer is in a local document collection: find it with the search and "
                f"read tools. Give only the answer itself (a name, a year, a few words).{ANSWER_NOTE}",
                "answer": answer,
                "corpus": {"seed": seed, "size": size},
                "hops": 2 if two else 1,
                "solution": solution,
                "solution_answer": f"I looked it up.\nANSWER: {answer}",
                "question": question,
            }
        )
    return rows


# ------------------------------------------------------------------- policies


def closed_book(row: dict, messages: list[dict]) -> str:
    return "ANSWER: unknown"


def answer_with_subject(row: dict, messages: list[dict]) -> str:
    """The entity the question is about, as the answer."""
    subject = re.findall(r"(?:[A-Z][a-z]+ )+[A-Z][a-z]+|[A-Z][a-z]{3,}", row["question"])
    return f"ANSWER: {subject[-1] if subject else row['question']}"


def paste_results(row: dict, messages: list[dict]) -> Any:
    """Searches with the question and pastes what comes back as the answer."""
    if turn_index(messages) == 0:
        return [("search", {"query": row["question"]})]
    pasted = " ".join(str(m.get("content")) for m in messages if m.get("role") == "tool")
    return f"{pasted}\nANSWER: {pasted}"


def list_candidates(row: dict, messages: list[dict]) -> Any:
    """Reads the subject's document and offers every capitalized word and number in it."""
    if turn_index(messages) < 2:
        return [tuple(row["solution"][turn_index(messages)])]
    seen = " ".join(str(m.get("content")) for m in messages if m.get("role") == "tool")
    return "ANSWER: " + ", ".join(dict.fromkeys(re.findall(r"[A-Z][a-z]+|\d{4}", seen)))


SPEC = EnvSpec(
    name="corpus-qa",
    version="1.0.0",
    family="search",
    summary="Question answering over a local corpus with search and read tools; single- and two-hop questions.",
    env=CorpusQAEnv,
    license="Apache-2.0",
    verifier=(
        'The final message ends with "ANSWER: <answer>" and the answer equals the gold after SQuAD normalization '
        "(exact match, not containment). The corpus is fictional, so the answer must come from retrieval."
    ),
    datasets={
        "synthetic": DatasetSpec(
            generator=generate,
            license="Apache-2.0",
            description="A generated world of people, companies and towns (one document each) and questions on it.",
        )
    },
    difficulty={"option": "hops", "levels": [1, 2], "row_field": "hops"},
    max_turns=10,
    max_new_tokens=512,
    oracle=scripted(),
    degenerate={
        "closed_book": closed_book,
        "answer_with_subject": answer_with_subject,
        "paste_results": paste_results,
        "list_candidates": list_candidates,
    },
    tags=("multi-turn", "retrieval"),
    notes='Your own corpus: rows {"prompt", "answer", "corpus": "docs.jsonl"} with documents {"id", "title", "text"}.',
)
