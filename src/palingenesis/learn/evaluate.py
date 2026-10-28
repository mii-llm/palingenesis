"""Evaluation of a scope's adapter version, paired against the scope's LIVE version on the same probes.

Every metric compares two stacked chains, greedy, with the deployment system prompt: the candidate (the
ancestors' live adapters + the evaluated version) and the live chain (the ancestors' live adapters + the
scope's live version, or none). The in-context ceiling is the parent stack P (no adapter of the scope)
with the notes in the prompt.

    metric               probes                                              measure
    acquisition          held-out probes of the session's new facts           exclusive match closed-book, among the
                                                                              probes the in-context ceiling answers
    update               probes of the session's facts that superseded one    the new value, same relative measure
    stale                probes of every superseded fact of the scope         answers stating an old value
    retention            probes of the scope's earlier, consolidated facts    McNemar counts vs live
    ancestor_retention   probes of the ancestors' consolidated facts          McNemar counts vs live
    general_qa           a short-answer set (built-in, multilingual)          McNemar counts vs live
    leak                 near-miss probes and general QA                      changed vs live, wrong, and naming an
                                                                              entity of the scope that live did not
    confabulation        questions the notes do not answer (abstain probes)   answered instead of declined, McNemar
    meta_reference       every closed-book answer (and its reasoning)         mentions of notes / context
    drift                fixed general prompts                                KL(target || model) per token, target =
                                                                              the anchor teacher (P + relevant notes +
                                                                              instruction notes); increment vs live

The exclusive matcher accepts an answer that states the gold value and none of its competitors (the
fact's stale values, the values other entities have for the same attribute, the scope's entities for
general QA): an answer listing several candidates is wrong. Rates come with Wilson intervals (display
only); the gates use the paired counts. Answers are cached per chain in the session's answers.jsonl, so
the live chain's answers are reused from the eval that made it a candidate, and only new probes are
asked of both.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from palingenesis.learn.store import Fact, ScopeId, ScopeStore, Store, normalize, now, read_jsonl, write_jsonl

# ---------------------------------------------------------------------------- detectors

META_REFERENCE = re.compile(
    r"\b(?:according to|based on|as (?:stated|mentioned|written|noted) in|from) (?:the |my |your |these )?"
    r"(?:notes?|context|reference notes|documents?|provided (?:notes|information|context|documents?))\b"
    r"|\bin (?:the|my|your|these) (?:reference )?notes\b"
    r"|\b(?:the|these|your|my) notes (?:say|state|mention|indicate|show)\b"
    r"|\b(?:secondo|in base alle|in base a|dalle|nelle|come (?:indicato|scritto|riportato) nelle) (?:le |alle )?(?:mie |tue )?note\b"
    r"|\b(?:le|queste|mie|tue) note (?:dicono|indicano|riportano)\b"
    r"|\b(?:secondo|in base al|dal) (?:mio |tuo )?contesto\b",
    re.I,
)

DECLINE = re.compile(
    # en
    r"\bI (?:do not|don't|dont) (?:know|have (?:any )?(?:information|data|details))\b"
    r"|\bI(?:'m| am) not (?:sure|aware)\b|\bnot aware of\b|\bI (?:could not|couldn't|cannot|can't) find\b"
    r"|\bno (?:information|data|record|details) (?:about|on|for|is available)\b|\b(?:is|are) (?:not known|unknown)\b"
    r"|\b(?:does not|doesn't|did not) exist\b|\bthere is no (?:such|information|product|record)\b|\bunknown\b"
    # it
    r"|\bnon (?:lo )?so\b|\bnon (?:ho|dispongo di|abbiamo) (?:informazioni|dati|notizie)\b|\bnon conosco\b"
    r"|\bnon (?:esiste|risulta|sono a conoscenza)\b|\bnessuna informazione\b|\bnon è (?:noto|nota|indicat[oa])\b"
    # es
    r"|\bno (?:lo )?sé\b|\bno tengo (?:información|datos)\b|\bdesconozco\b|\bno existe\b|\bno hay información\b"
    # fr
    r"|\bje ne (?:le )?sais pas\b|\bje n'ai pas d'informations?\b|\baucune information\b|\bn'existe pas\b"
    r"|\bje ne connais pas\b"
    # de
    r"|\bich weiß (?:es )?nicht\b|\bkeine (?:Informationen|Angaben|Daten)\b|\bexistiert nicht\b"
    r"|\b(?:ist|sind) (?:mir )?nicht bekannt\b|\bunbekannt\b",
    re.I,
)

# Short-answer general QA written for the library (Apache-2.0, like the repo): {"q", "answers"}.
BUILTIN_GENERAL_QA = [
    {"q": "What is the capital of France?", "answers": ["Paris"]},
    {"q": "How many legs does a spider have?", "answers": ["8", "eight"]},
    {"q": "Who wrote Pride and Prejudice?", "answers": ["Jane Austen", "Austen"]},
    {"q": "What is the chemical symbol for gold?", "answers": ["Au"]},
    {"q": "In which year did the Second World War end?", "answers": ["1945"]},
    {"q": "What is the largest planet in the solar system?", "answers": ["Jupiter"]},
    {"q": "At how many degrees Celsius does water boil at sea level?", "answers": ["100"]},
    {"q": "Who painted the Mona Lisa?", "answers": ["Leonardo", "da Vinci"]},
    {"q": "What is the square root of 144?", "answers": ["12", "twelve"]},
    {"q": "Which is the largest ocean on Earth?", "answers": ["Pacific"]},
    {"q": "Which element has atomic number 1?", "answers": ["hydrogen"]},
    {"q": "How many continents are there?", "answers": ["7", "seven"]},
    {"q": "At how many degrees Fahrenheit does water freeze?", "answers": ["32"]},
    {"q": "Qual è la capitale del Giappone?", "answers": ["Tokyo", "Tokio"]},
    {"q": "Chi ha scritto la Divina Commedia?", "answers": ["Dante"]},
    {"q": "Quanti giorni ha una settimana?", "answers": ["7", "sette"]},
    {"q": "In che anno è caduto il muro di Berlino?", "answers": ["1989"]},
    {"q": "¿Cuál es el río más largo de Sudamérica?", "answers": ["Amazonas", "Amazon"]},
    {"q": "¿Quién pintó el Guernica?", "answers": ["Picasso"]},
    {"q": "Quelle est la capitale du Canada ?", "answers": ["Ottawa"]},
    {"q": "Combien de côtés a un hexagone ?", "answers": ["6", "six"]},
    {"q": "Was ist die Hauptstadt von Österreich?", "answers": ["Wien", "Vienna"]},
    {"q": "Wie viele Minuten hat eine Stunde?", "answers": ["60", "sechzig"]},
    {"q": "What gas do plants take from the air for photosynthesis?", "answers": ["carbon dioxide", "CO2"]},
]

# Fixed general prompts for the drift KL, disjoint from the anchor prompts (Apache-2.0, like the repo).
BUILTIN_DRIFT_PROMPTS = [
    "Describe a sunset in one sentence.",
    "What are two benefits of regular exercise?",
    "Explain what a prime number is.",
    "Write a haiku about autumn.",
    "How do I reverse a string in Python?",
    "What is the difference between a virus and a bacterium?",
    "Give me a short recipe for a tomato salad.",
    "Why does the Earth have seasons?",
    "Dammi un consiglio per studiare meglio.",
    "Cos'è l'inflazione, in parole semplici?",
    "¿Qué es la energía solar?",
    "Donne-moi trois idées de cadeaux pour un ami.",
    "Erkläre kurz, was ein Algorithmus ist.",
    "What does a CPU do?",
    "Write a one-line motivational message for a Monday.",
    "State the Pythagorean theorem.",
]

# Words too common to identify an entity or a forgotten value (plus the store's own vocabulary).
COMMON_WORDS = frozenset(
    """a an the and or of to in on at by for with from is are was be it its this that yes no not true false none
    null new old one two three all any some more less ok il lo la i gli le un una uno e o di da del della su per con
    è sono non si sì el los las y en es son un una le les et des du est une der die das und ist ein eine nicht ja
    nein""".split()
)


def split_reasoning(text: str) -> tuple[str, str]:
    """(reasoning, answer) of a decoded completion; the reasoning is empty without a </think> marker."""
    if "</think>" in text:
        reasoning, _, answer = text.rpartition("</think>")
        return reasoning.replace("<think>", "").strip(), answer.strip()
    return "", text.strip()


def contains(text: str, value: str) -> bool:
    """`value` appears in `text` as whole words, ignoring case and punctuation."""
    value = normalize(value)
    return bool(value) and f" {value} " in f" {normalize(text)} "


def trivial(value: str) -> bool:
    """Too short or too common to be evidence of anything: under 3 characters, a common word, or a number of
    fewer than 4 digits with no letter."""
    norm = normalize(value)
    if len(norm) < 3 or norm in COMMON_WORDS:
        return True
    return not re.search(r"[^\W\d_]", norm) and len(re.sub(r"\D", "", norm)) < 4


def exclusive_match(answer: str, gold: list[str], competitors: list[str]) -> bool:
    """The answer states a gold value and no competing value. A competitor contained in a gold value (as
    whole words) cannot be told apart from it and is ignored."""
    _, answer = split_reasoning(answer)
    if not any(contains(answer, g) for g in gold):
        return False
    rivals = [c for c in competitors if not any(contains(g, c) for g in gold)]
    return not any(contains(answer, c) for c in rivals)


def states_any(answer: str, values: list[str]) -> bool:
    _, answer = split_reasoning(answer)
    return any(contains(answer, v) for v in values)


def is_decline(answer: str) -> bool:
    """The answer declines or says it does not know (en, it, es, fr, de). An empty answer states no value."""
    _, answer = split_reasoning(answer)
    return not answer.strip() or bool(DECLINE.search(answer))


def meta_reference(text: str) -> bool:
    return bool(META_REFERENCE.search(text))


# ---------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """The Wilson score interval of k successes in n trials (display only)."""
    if n <= 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci": wilson(k, n)}


def mcnemar(before: list[bool], after: list[bool]) -> dict:
    """Paired counts: losses = right before and wrong after, gains = the reverse; net_loss = losses - gains;
    p_value = the exact two-sided binomial test on the discordant pairs (display only)."""
    losses = sum(1 for b, a in zip(before, after) if b and not a)
    gains = sum(1 for b, a in zip(before, after) if a and not b)
    m = losses + gains
    tail = sum(math.comb(m, i) for i in range(min(losses, gains) + 1)) / 2**m if m else 1.0
    return {
        "n": len(before),
        "both_right": sum(1 for b, a in zip(before, after) if b and a),
        "losses": losses,
        "gains": gains,
        "net_loss": losses - gains,
        "p_value": round(min(1.0, 2 * tail), 4),
    }


# ---------------------------------------------------------------------------- probes and metrics

KINDS = ("acquisition", "update", "retention", "ancestor", "general", "nearmiss", "abstain")


@dataclass
class Probe:
    kind: str  # one of KINDS
    q: str
    fact_ids: list = field(default_factory=list)
    gold: list = field(default_factory=list)  # accepted values; empty for near-miss and abstain probes
    competitors: list = field(default_factory=list)
    stale: list = field(default_factory=list)  # superseded values of the probe's facts
    note_ids: list = field(default_factory=list)


@dataclass
class Answers:
    """Decoded completions aligned with the probes: the candidate chain, the live chain, and the in-context
    ceiling (None for probes asked closed-book only)."""

    candidate: list
    live: list
    context: list


def _absolute(idx: list[int], probes: list[Probe], answers: Answers) -> tuple[dict, list[bool]]:
    def ok(text, p):
        return text is not None and exclusive_match(text, p.gold, p.competitors + p.stale)

    cand = [ok(answers.candidate[i], probes[i]) for i in idx]
    live = [ok(answers.live[i], probes[i]) for i in idx]
    ctx = [ok(answers.context[i], probes[i]) for i in idx]
    relative = [c for c, x in zip(cand, ctx) if x]
    return {
        "candidate": rate(sum(cand), len(cand)),
        "live": rate(sum(live), len(live)),
        "context": rate(sum(ctx), len(ctx)),
        "relative": rate(sum(relative), len(relative)),
    }, cand


def _paired(idx: list[int], probes: list[Probe], answers: Answers, ok: Callable) -> tuple[dict, list, list]:
    cand = [ok(answers.candidate[i], probes[i]) for i in idx]
    live = [ok(answers.live[i], probes[i]) for i in idx]
    return (
        {"candidate": rate(sum(cand), len(cand)), "live": rate(sum(live), len(live)), "paired": mcnemar(live, cand)},
        cand,
        live,
    )


def compute_metrics(
    probes: list[Probe],
    answers: Answers,
    entities: list[tuple[str, list[str]]],
    decline: Callable[[str], bool] = is_decline,
) -> tuple[dict, list[dict]]:
    """Every answer-based metric of the table above, and per-probe diagnostics of what failed.

    `entities`: (string, fact ids) of the scope's subjects and values, already filtered of trivial strings.
    """
    by_kind: dict[str, list[int]] = {k: [] for k in KINDS}
    for i, p in enumerate(probes):
        by_kind[p.kind].append(i)
    diagnostics: list[dict] = []

    def diag(metric: str, i: int, **extra):
        p = probes[i]
        row = {"metric": metric, "q": p.q, "answer": answers.candidate[i], "fact_ids": p.fact_ids}
        if p.gold:
            row["expected"] = p.gold[0]
        diagnostics.append({**row, "note_ids": p.note_ids, **extra})

    metrics: dict = {}
    for kind in ("acquisition", "update"):
        metrics[kind], cand = _absolute(by_kind[kind], probes, answers)
        for i, ok in zip(by_kind[kind], cand):
            if not ok:
                diag(kind, i, context_answer=answers.context[i])

    def right(text, p):
        return exclusive_match(text, p.gold, p.competitors + p.stale)

    for kind, name in (("retention", "retention"), ("ancestor", "ancestor_retention"), ("general", "general_qa")):
        metrics[name], cand, live = _paired(by_kind[kind], probes, answers, right)
        for i, c, x in zip(by_kind[kind], cand, live):
            if x and not c:
                diag(name, i, live_answer=answers.live[i])

    stale_idx = [i for k in ("update", "retention") for i in by_kind[k] if probes[i].stale]
    stale_c = [states_any(answers.candidate[i], probes[i].stale) for i in stale_idx]
    stale_l = [states_any(answers.live[i], probes[i].stale) for i in stale_idx]
    metrics["stale"] = {"candidate": rate(sum(stale_c), len(stale_c)), "live": rate(sum(stale_l), len(stale_l))}
    for i, hit in zip(stale_idx, stale_c):
        if hit:
            diag("stale", i, stale=probes[i].stale)

    leaks, checked = 0, 0
    for i in by_kind["nearmiss"] + by_kind["general"]:
        p, cand, live = probes[i], answers.candidate[i], answers.live[i]
        checked += 1
        if normalize(split_reasoning(cand)[1]) == normalize(split_reasoning(live)[1]):
            continue
        if p.gold and right(cand, p):
            continue
        hits = [
            (e, ids)
            for e, ids in entities
            if states_any(cand, [e])
            and not states_any(live, [e])
            and not contains(p.q, e)
            and not any(contains(g, e) for g in p.gold)
        ]
        if hits:
            leaks += 1
            diag(
                "leak",
                i,
                live_answer=live,
                entities=[e for e, _ in hits],
                fact_ids=sorted({f for _, ids in hits for f in ids}),
            )
    metrics["leak"] = rate(leaks, checked)

    metrics["confabulation"], declined_c, declined_l = _paired(
        by_kind["abstain"], probes, answers, lambda text, _: decline(text)
    )
    n = len(by_kind["abstain"])
    metrics["confabulation"]["candidate"] = rate(n - sum(declined_c), n)
    metrics["confabulation"]["live"] = rate(n - sum(declined_l), n)
    for i, d, dl in zip(by_kind["abstain"], declined_c, declined_l):
        if dl and not d:
            diag("confabulation", i, live_answer=answers.live[i])

    # Every probe is asked closed-book; the in-context answers are not counted.
    meta_c = [meta_reference(split_reasoning(text)[1]) for text in answers.candidate]
    meta_l = [meta_reference(split_reasoning(text)[1]) for text in answers.live]
    meta_r = [meta_reference(t) for t in (split_reasoning(text)[0] for text in answers.candidate) if t]
    metrics["meta_reference"] = {
        "candidate": rate(sum(meta_c), len(meta_c)),
        "live": rate(sum(meta_l), len(meta_l)),
        "reasoning": rate(sum(meta_r), len(meta_r)),
    }
    for i, hit in enumerate(meta_c):
        if hit:
            diag("meta_reference", i)
    return metrics, diagnostics


def kl_increment(candidate: float | None, live: float | None, tokens: int, prompts: int) -> dict:
    """The drift entry of a report: KL(target || candidate), KL(target || live), and their difference."""
    if not tokens:
        return {"candidate": None, "live": None, "increment": None, "tokens": 0, "prompts": prompts}
    return {
        "candidate": round(candidate, 6),
        "live": round(live, 6),
        "increment": round(candidate - live, 6),
        "tokens": tokens,
        "prompts": prompts,
    }


# ---------------------------------------------------------------------------- chains and probe sets


def chain_versions(store: Store, scope: ScopeId) -> list[list]:
    """[[scope, live version or None], ...] of the chain, ancestors first."""
    out = []
    for s in scope.chain():
        ss = store.scope(s)
        out.append([str(s), ss.live if ss.exists else None])
    return out


def _chain_id(store: Store, chain: list[list]) -> str:
    parts = []
    for scope, version in chain:
        if version is None:
            parts.append(f"{scope}@-")
        else:
            sha = store.scope(scope).manifest(version).get("sha256", "")[:12]
            parts.append(f"{scope}@{version}:{sha}")
    return "|".join(parts)


def _resolve_all(fact_ids: list[str], facts: dict[str, Fact]) -> list[Fact] | None:
    from palingenesis.learn.ingest import resolve_fact

    out = []
    for fid in fact_ids:
        fact = resolve_fact(fid, facts)
        if fact is None:
            return None
        if fact not in out:
            out.append(fact)
    return out or None


def _competitors(fact: Fact, chain_facts: list[Fact]) -> list[str]:
    """Values other entities of the chain have for the same attribute."""
    attribute, subject = normalize(fact.attribute), normalize(fact.subject)
    if attribute in ("statement", "is", "are", "è", "sono"):
        return []  # copulas and unsplit sentences do not name a shared slot
    return sorted(
        {
            f.value
            for f in chain_facts
            if normalize(f.attribute) == attribute
            and normalize(f.subject) != subject
            and normalize(f.value) != normalize(fact.value)
        }
    )


def _session_rows(ss: ScopeStore, name: str, exclude: str | None = None) -> list[dict]:
    rows = []
    for session in ss.session_dirs():
        if session.name != exclude and not session.name.endswith(".tmp"):
            rows += read_jsonl(session / name)
    return rows


def _sample(rows: list, k: int, rng: random.Random) -> list:
    return rows if len(rows) <= k else rng.sample(rows, k)


def load_general_qa(path: str) -> list[dict]:
    if not path:
        return list(BUILTIN_GENERAL_QA)
    rows = read_jsonl(path)
    return [{"q": r["q"], "answers": list(r["answers"])} for r in rows]


def load_drift_prompts(path: str) -> list:
    if not path:
        return list(BUILTIN_DRIFT_PROMPTS)
    return [r["messages"] if "messages" in r else r["prompt"] for r in read_jsonl(path)]


def entities_of(ss: ScopeStore) -> list[tuple[str, list[str]]]:
    """The scope's own entity strings (subjects, values, old values) with the facts they come from."""
    found: dict[str, tuple[str, set]] = {}
    for f in ss.facts():
        if f.status == "retired":
            continue
        for text in [f.subject, f.value, *f.stale_values]:
            text = str(text).strip()
            if text and not trivial(text) and len(text) <= 80:
                key = normalize(text)
                found.setdefault(key, (text, set()))[1].add(f.id)
    return [(text, sorted(ids)) for text, ids in found.values()]


def build_probes(store: Store, ss: ScopeStore, manifest: dict, config, rng: random.Random) -> list[Probe]:
    """The probe set of one version: the session's probes, earlier facts, ancestors, general QA, near-miss
    and abstain probes. Probes whose facts were retired or forgotten are skipped; a probe of a superseded
    fact asks for its successor's value only through the successor's own probes."""
    from palingenesis.learn.recipe import NotesIndex

    ev = config.eval
    index = NotesIndex(ss)
    facts = index.facts
    chain_facts = list(ss.active_facts())
    ancestors = [store.scope(s) for s in ss.scope.chain()[:-1]]
    for a in ancestors:
        chain_facts += a.active_facts()
    session = manifest["session"]
    session_dir = ss.dir / "sessions" / session
    own = set(manifest.get("fact_ids", []))
    probes: list[Probe] = []

    def fact_probe(kind: str, row: dict, resolved: list[Fact]) -> Probe:
        stale = sorted({v for f in resolved for v in f.stale_values})
        competitors = sorted({c for f in resolved for c in _competitors(f, chain_facts)})
        return Probe(
            kind,
            row["q"],
            list(row["fact_ids"]),
            [f.value for f in resolved],
            competitors,
            stale,
            sorted({f.note_id for f in resolved}),
        )

    session_probes = []
    for row in read_jsonl(session_dir / "probes.jsonl"):
        if row.get("kind") != "probe":
            continue
        resolved = _resolve_all(row.get("fact_ids", []), facts)
        if resolved is None or {f.id for f in resolved} != set(row["fact_ids"]):
            continue  # retired, forgotten or superseded since
        kind = "update" if any(f.stale_values for f in resolved) else "acquisition"
        session_probes.append(fact_probe(kind, row, resolved))
    probes += _sample(session_probes, ev.max_probes_per_session, rng)

    seen = {p.q for p in probes}
    earlier = []
    for row in _session_rows(ss, "probes.jsonl", exclude=session):
        if row.get("kind") != "probe" or row["q"] in seen:
            continue
        resolved = _resolve_all(row.get("fact_ids", []), facts)
        if resolved is None or any(f.consolidated_in is None or f.id in own for f in resolved):
            continue
        seen.add(row["q"])
        earlier.append(fact_probe("retention", row, resolved))
    probes += _sample(earlier, ev.max_retention_probes, rng)

    inherited = []
    for a in ancestors:
        a_facts = {f.id: f for f in a.facts()}
        for row in _session_rows(a, "probes.jsonl"):
            if row.get("kind") != "probe" or row["q"] in seen:
                continue
            resolved = _resolve_all(row.get("fact_ids", []), a_facts)
            if resolved is None or any(f.consolidated_in is None for f in resolved):
                continue
            seen.add(row["q"])
            inherited.append(fact_probe("ancestor", row, resolved))
    probes += _sample(inherited, ev.max_ancestor_probes, rng)

    entities = [e for e, _ in entities_of(ss)]
    for item in load_general_qa(ev.general_qa_path):
        rivals = [
            e for e in entities if not contains(item["q"], e) and not any(contains(g, e) for g in item["answers"])
        ]
        probes.append(Probe("general", item["q"], gold=list(item["answers"]), competitors=rivals))
    for row in read_jsonl(session_dir / "probes.jsonl"):
        if row.get("kind") == "nearmiss":
            probes.append(Probe("nearmiss", row["q"], list(row.get("fact_ids", []))))
    for row in read_jsonl(session_dir / "abstain.jsonl"):
        probes.append(Probe("abstain", row["q"], list(row.get("fact_ids", []))))
    return probes


# ---------------------------------------------------------------------------- the model-backed evaluator


@dataclass
class EvalResult:
    report: dict
    diagnostics: list
    session_dir: Path

    @property
    def passed(self) -> bool:
        return bool(self.report["passed"])

    def summary(self) -> str:
        from palingenesis.learn.report import summary

        return summary(self.report)


class Evaluator:
    """Evaluates a version with a loaded SessionRunner (trainer.SessionRunner after _load): its LoRA holds
    the candidate's weights (role student), its `previous` snapshot the live version's (role previous), and
    disable_adapter() gives P (role parent). `decline` replaces the decline detector (a judge, for example).
    """

    def __init__(self, runner, decline: Callable[[str], bool] | None = None):
        self.runner = runner
        self.learner = runner.learner
        self.store = runner.store
        self.config = runner.config
        self.ss: ScopeStore = runner.ss
        self.decline = decline or is_decline
        self.cache: dict[str, str] = {}
        self.rows: dict[str, dict] = {}
        self.generated = 0

    # answers
    def _key(self, chain: str, messages: list[dict], max_new_tokens: int) -> str:
        blob = json.dumps(
            [chain, max_new_tokens, self.config.model.chat_template_kwargs, messages],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def _load_cache(self) -> None:
        for row in _session_rows(self.ss, "answers.jsonl"):
            self.cache[row["key"]] = row["answer"]

    def answer(self, probes: list[Probe], messages: list[list[dict]], role: str, chain: str) -> list[str]:
        max_new = self.config.eval.max_new_tokens
        keys = [self._key(chain, m, max_new) for m in messages]
        todo = list(dict.fromkeys(k for k in keys if k not in self.cache))
        if todo:
            first = {k: i for i, k in reversed(list(enumerate(keys)))}
            prompts = [self.runner.encode(messages[first[k]]) for k in todo]
            outs = self.runner.sampler.generate(prompts, max_new, 0.0, role)
            for k, ids in zip(todo, outs):
                self.cache[k] = self.runner.tok.decode(ids, skip_special_tokens=True).strip()
            self.generated += len(todo)
        for k, p in zip(keys, probes):
            self.rows[k] = {"key": k, "chain": chain, "kind": p.kind, "q": p.q, "fact_ids": p.fact_ids}
            self.rows[k]["answer"] = self.cache[k]
        return [self.cache[k] for k in keys]

    # drift
    def drift(self) -> dict:
        """KL(target || candidate) and KL(target || live) per token on the drift prompts' target completions."""
        import torch
        import torch.nn.functional as F

        from palingenesis.learn.recipe import PARENT, PREVIOUS, NotesIndex, _messages
        from palingenesis.learn.render import with_notes
        from palingenesis.learn.trainer import STUDENT

        runner, config = self.runner, self.config
        prompts = load_drift_prompts(config.eval.drift_path)[: config.eval.max_drift_prompts]
        if not prompts:
            return kl_increment(None, None, 0, 0)
        rng = random.Random(config.eval.seed + 1)
        index = NotesIndex(self.ss)
        students, targets = [], []
        for prompt in prompts:
            student = _messages(prompt, config.model.system_prompts, rng)
            text = prompt if isinstance(prompt, str) else " ".join(str(m.get("content", "")) for m in prompt)
            if config.recipe.anchor_teacher == "notes":
                relevant = index.relevant(text, config.notes.relevance_threshold, config.notes.teacher_neighbours)
                notes = [n for n in index.notes if n in relevant or n.kind == "instruction"]
                targets.append(with_notes(student, notes))
            else:
                targets.append(list(student))
            students.append(student)
        target_role = PREVIOUS if config.recipe.anchor_teacher == "previous" else PARENT
        t_ids = [runner.encode(m) for m in targets]
        s_ids = [runner.encode(m) for m in students]
        completions = runner.sampler.generate(t_ids, config.eval.drift_max_new_tokens, 0.0, target_role)
        rows = [(t, s, c) for t, s, c in zip(t_ids, s_ids, completions) if c]
        totals = {STUDENT: 0.0, PREVIOUS: 0.0}
        tokens = 0
        step = max(1, config.recipe.micro_rows)
        with torch.no_grad():
            for k in range(0, len(rows), step):
                chunk = rows[k : k + step]
                comps = [c for *_, c in chunk]
                with runner.roles[target_role]():
                    target = F.log_softmax(runner.head(runner._hidden([t for t, *_ in chunk], comps)).float(), -1)
                for role in totals:
                    with runner.roles[role]():
                        model = F.log_softmax(runner.head(runner._hidden([s for _, s, _ in chunk], comps)).float(), -1)
                    totals[role] += float((target.exp() * (target - model)).sum())
                tokens += target.shape[0]
        if not tokens:
            return kl_increment(None, None, 0, len(prompts))
        return kl_increment(totals[STUDENT] / tokens, totals[PREVIOUS] / tokens, tokens, len(prompts))

    # the whole evaluation
    def run(self, version: str) -> EvalResult:
        from palingenesis.learn import gates as gating
        from palingenesis.learn import report as reporting
        from palingenesis.learn.recipe import PARENT, PREVIOUS, NotesIndex, _messages
        from palingenesis.learn.render import with_notes
        from palingenesis.learn.trainer import STUDENT

        started = time.time()
        config, ss, store = self.config, self.ss, self.store
        self.runner.peft.eval()  # greedy answers and the drift forward passes run without dropout
        manifest = ss.manifest(version)
        if manifest.get("pruned"):
            raise ValueError(f"{ss.scope} {version} was pruned ({manifest['pruned']['reason']}); nothing to evaluate.")
        session_dir = ss.dir / "sessions" / manifest["session"]
        live_chain = chain_versions(store, ss.scope)
        candidate_chain = [*live_chain[:-1], [str(ss.scope), version]]
        parent_chain = [*live_chain[:-1], [str(ss.scope), None]]
        ids = {
            STUDENT: _chain_id(store, candidate_chain),
            PREVIOUS: _chain_id(store, live_chain),
            PARENT: _chain_id(store, parent_chain) + "+notes",
        }
        warnings = []
        parents = {s: v for s, v in live_chain[:-1] if v is not None}
        if manifest.get("parent_versions", {}) != parents:
            warnings.append(
                f"{version} was trained on {manifest.get('parent_versions')} but the ancestors' live versions are "
                f"{parents}: it is evaluated on the current ones."
            )
        rng = random.Random(config.eval.seed)
        probes = build_probes(store, ss, manifest, config, rng)
        index = NotesIndex(ss)
        closed = [_messages(p.q, config.model.system_prompts, rng) for p in probes]
        self._load_cache()
        cand = self.answer(probes, closed, STUDENT, ids[STUDENT])
        live = self.answer(probes, closed, PREVIOUS, ids[PREVIOUS])
        ceiling_idx = [i for i, p in enumerate(probes) if p.kind in ("acquisition", "update")]
        context: list = [None] * len(probes)
        if ceiling_idx:
            neighbours = config.notes.teacher_neighbours
            ctx_messages = [
                with_notes(closed[i], index.for_facts(probes[i].fact_ids, probes[i].q, neighbours)) for i in ceiling_idx
            ]
            for i, text in zip(
                ceiling_idx, self.answer([probes[i] for i in ceiling_idx], ctx_messages, PARENT, ids[PARENT])
            ):
                context[i] = text
        entities = entities_of(ss)
        metrics, diagnostics = compute_metrics(probes, Answers(cand, live, context), entities, self.decline)
        metrics["drift"] = self.drift()
        notes = {n.id: n.text for s in ss.scope.chain() for n in store.scope(s).notes()}
        for row in diagnostics:
            row["notes"] = [notes[n] for n in row.get("note_ids", []) if n in notes]
        gates = gating.evaluate(metrics, config.gates)
        report = {
            "scope": str(ss.scope),
            "version": version,
            "session": manifest["session"],
            "created": now(),
            "live_chain": live_chain,
            "candidate_chain": candidate_chain,
            "calibrated": bool(config.gates.calibrated),
            "probes": {k: sum(1 for p in probes if p.kind == k) for k in KINDS},
            "metrics": metrics,
            "gates": [g.to_dict() for g in gates],
            "passed": gating.passed(gates),
            "warnings": warnings,
            "generated_answers": self.generated,
            "seconds": round(time.time() - started, 1),
        }
        write_jsonl(session_dir / "answers.jsonl", list(self.rows.values()))
        reporting.write(session_dir, report, diagnostics)
        manifest["evaluation"] = {
            "passed": report["passed"],
            "failed": [g.name for g in gates if not g.passed],
            "calibrated": report["calibrated"],
            "live_chain": live_chain,
            "report": f"sessions/{manifest['session']}/report.json",
            "at": report["created"],
        }
        ss.write_manifest(version, manifest)
        return EvalResult(report, diagnostics, session_dir)


def load_report(ss: ScopeStore, version: str) -> dict | None:
    """The report of `version`'s latest evaluation, or None."""
    manifest = ss.manifest(version)
    path = ss.dir / "sessions" / manifest.get("session", "") / "report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    return report if report.get("version") == version else None
