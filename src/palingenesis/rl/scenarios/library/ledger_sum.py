"""ledger_sum: a lead and its workers, with sub-answers that are checked one by one.

Several regions each keep a ledger (item -> quantity). The lead is asked for the total of one item over all
regions and cannot see any ledger: it has one tool, `ask(region, question)`, which hands its question to that
region's clerk (a worker session that sees only its own ledger and the question) and returns the clerk's reply.
Tool calls of one turn run at once, so the clerks work in parallel. The item is missing from some ledgers (the
clerk must answer 0), and the lead cannot reach the total without asking: the task needs the delegation.

Roles: `lead`, `worker`.

Reward components:

  lead    total           1 when its `TOTAL: n` is the gold total, else 0 (no answer included): nothing but the
                          right number pays, and the number cannot be guessed.
          coverage        the share of regions it asked a question that names the item (and no other listed item):
                          its delegation is checked without the workers' answers. Weight 0 by default.
          metric/asks     the number of `ask` calls.
  worker  value           +1 when its `VALUE: n` is its ledger's quantity for the item its question names, -1 when
                          it is not or it gives no value. The question it got decides the gold, not the task: a
                          worker is right or wrong on what it was asked, whatever the lead meant. A question that
                          names no listed item, or several, cannot be judged: None (the session stays out of the loss).
          team            the lead's `total`, for a shared reward. Weight 0 by default.

A constant worker cannot gain: the item is absent from about a quarter of the ledgers, so "always 0" is right a
quarter of the time and wrong the rest (mean -0.5).
"""

from __future__ import annotations

import random
import re

from palingenesis.rl.scenarios.base import Scenario, system, tagged_integer, user
from palingenesis.rl.scenarios.runtime import Episode, stable_seed

ITEMS = {
    "en": ["apples", "pears", "lemons", "onions", "carrots", "plums", "melons", "beans", "leeks", "figs"],
    "it": ["mele", "pere", "limoni", "cipolle", "carote", "prugne", "meloni", "fagioli", "porri", "fichi"],
}
REGIONS = {
    "en": ["north", "south", "east", "west", "centre"],
    "it": ["nord", "sud", "est", "ovest", "centro"],
}
LEAD = {
    "en": (
        "Each of these regions keeps a stock ledger that you cannot see: {regions}. Use the `ask` tool to put a "
        "question to a region's clerk; the clerk sees only that region's ledger and your question.\n"
        "How many {item} are in stock over all the regions together? When you know, answer with exactly one "
        "line of the form:\nTOTAL: <number>"
    ),
    "it": (
        "Ognuna di queste regioni tiene un registro di magazzino che tu non puoi vedere: {regions}. Usa lo "
        "strumento `ask` per fare una domanda all'addetto di una regione; l'addetto vede solo il registro della "
        "sua regione e la tua domanda.\n"
        "Quante {item} ci sono in magazzino in tutte le regioni messe insieme? Quando lo sai, rispondi con una "
        "sola riga nella forma:\nTOTAL: <numero>"
    ),
}
WORKER = {
    "en": (
        "You are the stock clerk of the {region} region. Your ledger:\n{ledger}\n"
        "Answer the question you receive from your ledger alone, with exactly one line of the form:\n"
        "VALUE: <number>\nIf the item is not in your ledger, the value is 0."
    ),
    "it": (
        "Sei l'addetto al magazzino della regione {region}. Il tuo registro:\n{ledger}\n"
        "Rispondi alla domanda che ricevi usando solo il tuo registro, con una sola riga nella forma:\n"
        "VALUE: <numero>\nSe l'articolo non è nel tuo registro, il valore è 0."
    ),
}
UNKNOWN_REGION = {"en": "There is no region called {region!r}.", "it": "Non esiste una regione chiamata {region!r}."}


def named_items(question: str, language: str) -> list[str]:
    """The listed item names a question mentions (whole words, any case)."""
    words = set(re.findall(r"[a-zà-ÿ]+", (question or "").lower()))
    return [item for item in ITEMS[language] if item in words]


def _ledger_of(messages: list[dict]) -> dict[str, int]:
    return {name: int(n) for name, n in re.findall(r"(?m)^([a-zà-ÿ]+): (\d+)$", messages[0]["content"])}


def _language_of(messages: list[dict]) -> str:
    return "it" if "registro" in messages[0]["content"] else "en"


def oracle_worker(messages: list[dict]) -> str:
    asked = named_items(messages[-1]["content"], _language_of(messages))
    return f"VALUE: {_ledger_of(messages).get(asked[0], 0) if len(asked) == 1 else 0}"


async def oracle_lead(messages: list[dict], tools: dict) -> str:
    text = messages[-1]["content"]
    language = _language_of(messages)
    regions = [r for r in REGIONS[language] if re.search(rf"\b{r}\b", text.split("\n")[0])]
    item = named_items(text.split("\n")[1], language)[0]
    total = 0
    for region in regions:
        total += tagged_integer(await tools["ask"](region=region, question=f"{item}?"), "VALUE") or 0
    return f"TOTAL: {total}"


class LedgerSum(Scenario):
    name = "ledger_sum"
    roles = ("lead", "worker")
    outcome = "lead"
    weights = {"lead": {"total": 1.0, "coverage": 0.0}, "worker": {"value": 1.0, "team": 0.0}}

    def __init__(self, weights=None, max_question_chars: int = 200, max_reply_chars: int = 120):
        super().__init__(weights)
        self.max_question_chars, self.max_reply_chars = max_question_chars, max_reply_chars

    @classmethod
    def generate(cls, n: int, seed: int = 0, max_regions: int = 4, absent_share: float = 0.25, languages=("en", "it")):
        """`n` tasks: 2..max_regions regions, 3-5 items per ledger with quantities 1-99, the asked item missing
        from about `absent_share` of the ledgers (never from all of them), the languages in turn."""
        rng = random.Random(stable_seed("ledger_sum", seed))
        rows = []
        for i in range(n):
            language = languages[i % len(languages)]
            regions = rng.sample(REGIONS[language], rng.randint(2, max(2, max_regions)))
            item = rng.choice(ITEMS[language])
            others = [x for x in ITEMS[language] if x != item]
            present = [rng.random() >= absent_share for _ in regions]
            if not any(present):
                present[rng.randrange(len(regions))] = True
            ledgers = {}
            for region, has in zip(regions, present):
                names = rng.sample(others, rng.randint(3, 4)) + ([item] if has else [])
                rng.shuffle(names)
                ledgers[region] = {name: rng.randint(1, 99) for name in names}
            rows.append(
                {
                    "scenario": cls.name,
                    "id": f"{cls.name}-{seed}-{i}",
                    "language": language,
                    "item": item,
                    "ledgers": ledgers,
                    "answer": sum(ledger.get(item, 0) for ledger in ledgers.values()),
                }
            )
        return rows

    def partners(self):
        return {
            "lead": {
                "oracle": oracle_lead,
                "no_ask": lambda messages: "TOTAL: 0",
                "silent": lambda messages: "",
                "guess": lambda messages: "TOTAL: 50",
            },
            "worker": {
                "oracle": oracle_worker,
                "always_zero": lambda messages: "VALUE: 0",
                "silent": lambda messages: "",
                "constant": lambda messages: "VALUE: 42",
            },
        }

    async def run(self, episode: Episode, row: dict) -> None:
        language = row.get("language", "en")
        item, ledgers = row["item"], row["ledgers"]
        questions = episode.channel("question", self.max_question_chars)
        replies = episode.channel("reply", self.max_reply_chars)
        workers = []
        asked: dict[str, bool] = {}

        async def ask(region: str, question: str) -> str:
            """Put a question to one region's stock clerk and get the reply.

            Args:
                region: The region whose clerk should answer.
                question: The question for the clerk, with everything the clerk needs to know.
            """
            region = str(region).strip().lower()
            if region not in ledgers:
                return UNKNOWN_REGION[language].format(region=region)
            text = questions.send(str(question), "lead", f"worker:{region}")
            names = named_items(text, language)
            asked[region] = asked.get(region, False) or names == [item]
            ledger = "\n".join(f"{name}: {quantity}" for name, quantity in ledgers[region].items())
            prompt = [system(WORKER[language].format(region=region, ledger=ledger)), user(text)]
            worker = await episode.run("worker", prompt)
            gold = ledgers[region].get(names[0], 0) if len(names) == 1 else None
            value = tagged_integer(worker.text, "VALUE") if worker.ok else None
            worker.reward("value", None if gold is None else (1.0 if value == gold else -1.0))
            workers.append(worker)
            return replies.send(worker.text, f"worker:{region}", "lead")

        prompt = LEAD[language].format(regions=", ".join(ledgers), item=item)
        lead = await episode.run("lead", [user(prompt)], tools={"ask": ask}, max_turns=len(ledgers) + 3)
        total = float(lead.ok and tagged_integer(lead.text, "TOTAL") == row["answer"])
        lead.reward("total", total)
        lead.reward("coverage", sum(asked.get(region, False) for region in ledgers) / len(ledgers))
        lead.reward("metric/asks", len(workers))
        for worker in workers:
            worker.reward("team", total)
