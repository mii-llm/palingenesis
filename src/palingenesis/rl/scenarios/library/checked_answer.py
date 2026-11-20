"""checked_answer: a solver and a verifier, where no constant verdict can earn anything.

An arithmetic question with one integer answer. The solver answers `ANSWER: n`. The verifier is then shown the
question with a candidate answer and must give `VERDICT: ACCEPT` or `VERDICT: REJECT`, and on a rejection its own
`ANSWER: n`.

In every episode the verifier judges two candidates, in two separate sessions: the solver's own answer, and a
twin of the opposite class (the gold answer when the solver was wrong; when it was right, a wrong answer one
slip away: a digit changed, a sign lost, an off-by-one). One candidate is always right and one always wrong, so
the verdict classes are balanced in every episode whatever the solver's accuracy, and the wrong candidates are
the solver's own mistakes whenever it makes them.

Roles: `solver`, `verifier`.

Reward components:

  solver    correct         1 right, 0 wrong, -0.5 without exactly one `ANSWER:` line.
            metric/team     1 when the answer the pair ends with is right: the solver's if the verifier accepted
                            it, else the verifier's correction.
  verifier  verdict         accepting a right answer +1; accepting a wrong one -1; rejecting a right one -1;
                            rejecting a wrong one +0.5, and +1 when its own answer is right; no verdict -1.

Per episode, "always accept" earns +1 - 1 = 0, "always reject" at most -1 + 0.5 < 0, and silence -2: a verifier
gains only by telling the two candidates apart, and gains fully only by also knowing the answer. `verifier_samples`
verifier sessions run on each candidate (a branch): they see the same context, so they are compared with each
other and not with verdicts on other candidates.
"""

from __future__ import annotations

import asyncio
import random
import re

from palingenesis.rl.scenarios.base import Scenario, tagged_integer, tagged_word, user
from palingenesis.rl.scenarios.runtime import Episode, stable_seed

SOLVER = {
    "en": "Compute: {question}\nAnswer with exactly one line of the form:\nANSWER: <number>",
    "it": "Calcola: {question}\nRispondi con una sola riga nella forma:\nANSWER: <numero>",
}
VERIFIER = {
    "en": (
        "Question: compute {question}\nProposed answer: {candidate}\n"
        "Check the proposed answer. Reply with exactly one line `VERDICT: ACCEPT` if it is right, or "
        "`VERDICT: REJECT` if it is wrong, followed in that case by one line `ANSWER: <number>` with the right answer."
    ),
    "it": (
        "Domanda: calcola {question}\nRisposta proposta: {candidate}\n"
        "Controlla la risposta proposta. Rispondi con una sola riga `VERDICT: ACCEPT` se è giusta, oppure "
        "`VERDICT: REJECT` se è sbagliata, seguita in quel caso da una riga `ANSWER: <numero>` con la risposta giusta."
    ),
}


def make_question(rng: random.Random) -> tuple[str, int]:
    """An expression of two or three operations with small integers, and its value."""
    a, b, c, d = (rng.randint(2, 19) for _ in range(4))
    forms = [
        (f"{a} × {b} + {c}", a * b + c),
        (f"{a} × {b} − {c}", a * b - c),
        (f"({a} + {b}) × {c}", (a + b) * c),
        (f"{a} × {b} + {c} × {d}", a * b + c * d),
        (f"({a} + {b}) × {c} − {d}", (a + b) * c - d),
        (f"{a * b} ÷ {a} + {c}", b + c),
    ]
    return rng.choice(forms)


def distractor(answer: int, rng: random.Random) -> int:
    """A wrong answer one slip away from `answer`, of the same sign (a candidate's sign must not give away its
    class: right answers are almost never negative)."""
    options = {answer + 1, answer - 1, answer + 10, answer - 10}
    text = str(abs(answer))
    if len(text) > 1:
        options.add(int(text[::-1]) * (1 if answer >= 0 else -1))
    options.discard(answer)
    same_sign = {o for o in options if (o < 0) == (answer < 0)}
    return rng.choice(sorted(same_sign or options))


def verdict_reward(text: str, candidate_right: bool, answer: int) -> float:
    verdict = tagged_word(text, "VERDICT", ("accept", "reject"))
    if verdict is None:
        return -1.0
    if verdict == "accept":
        return 1.0 if candidate_right else -1.0
    if candidate_right:
        return -1.0
    return 1.0 if tagged_integer(text, "ANSWER") == answer else 0.5


def _evaluate(question: str) -> int:
    expression = question.replace("×", "*").replace("−", "-").replace("÷", "//")
    if not re.fullmatch(r"[\d\s()+*/-]+", expression):
        raise ValueError(question)
    return int(eval(expression, {"__builtins__": {}}))  # noqa: S307 — digits and operators only (checked above)


def oracle_solver(messages: list[dict]) -> str:
    question = re.search(r"(?:Compute|Calcola): (.+)", messages[-1]["content"]).group(1)
    return f"ANSWER: {_evaluate(question)}"


def oracle_verifier(messages: list[dict]) -> str:
    text = messages[-1]["content"]
    answer = _evaluate(re.search(r"(?:compute|calcola) (.+)", text).group(1))
    candidate = re.search(r"(?:Proposed answer|Risposta proposta): (.*)", text).group(1).strip()
    return "VERDICT: ACCEPT" if candidate == str(answer) else f"VERDICT: REJECT\nANSWER: {answer}"


def surface_verifier(messages: list[dict]) -> str:
    """A verifier that never computes: it rejects what looks odd (no number, a negative one) and accepts the rest."""
    candidate = re.search(r"(?:Proposed answer|Risposta proposta): (.*)", messages[-1]["content"]).group(1).strip()
    return "VERDICT: ACCEPT" if re.fullmatch(r"\d+", candidate) else "VERDICT: REJECT\nANSWER: 0"


class CheckedAnswer(Scenario):
    name = "checked_answer"
    roles = ("solver", "verifier")
    outcome = "solver"
    weights = {"solver": {"correct": 1.0}, "verifier": {"verdict": 1.0}}

    def __init__(self, weights=None, verifier_samples: int = 2):
        super().__init__(weights)
        if verifier_samples < 1:
            raise ValueError("checked_answer: verifier_samples must be >= 1")
        self.verifier_samples = verifier_samples

    @classmethod
    def generate(cls, n: int, seed: int = 0, languages=("en", "it")) -> list[dict]:
        rng = random.Random(stable_seed("checked_answer", seed))
        rows, seen = [], set()
        while len(rows) < n:
            question, answer = make_question(rng)
            if question in seen and len(seen) < 20000:
                continue
            seen.add(question)
            rows.append(
                {
                    "scenario": cls.name,
                    "id": f"{cls.name}-{seed}-{len(rows)}",
                    "language": languages[len(rows) % len(languages)],
                    "question": question,
                    "answer": answer,
                }
            )
        return rows

    def partners(self):
        return {
            "solver": {
                "oracle": oracle_solver,
                "silent": lambda messages: "",
                "two_answers": lambda messages: "ANSWER: 1\nANSWER: 2",
            },
            "verifier": {
                "oracle": oracle_verifier,
                "always_accept": lambda messages: "VERDICT: ACCEPT",
                "always_reject": lambda messages: "VERDICT: REJECT\nANSWER: 0",
                "silent": lambda messages: "",
                "both": lambda messages: "VERDICT: ACCEPT\nVERDICT: REJECT",
                "surface": surface_verifier,
            },
        }

    async def run(self, episode: Episode, row: dict) -> None:
        language, question, answer = row.get("language", "en"), row["question"], int(row["answer"])
        solver = await episode.run("solver", [user(SOLVER[language].format(question=question))])
        given = tagged_integer(solver.text, "ANSWER") if solver.ok else None
        right = given == answer
        solver.reward("correct", -0.5 if given is None else float(right))
        # the verifier always sees a number: a solver without an answer is shown a slip instead (an empty candidate
        # would be wrong every time, and "reject what is empty" would pay without any checking)
        own = str(distractor(answer, episode.rng)) if given is None else str(given)
        twin = str(distractor(answer, episode.rng)) if right else str(answer)
        candidates = episode.channel("candidate", max_chars=24)
        team = float(right)

        async def judge(shown: str, shown_right: bool, slot: str) -> list:
            prompt = VERIFIER[language].format(
                question=question, candidate=candidates.send(shown, "solver", "verifier")
            )

            async def one(branch: Episode):
                verifier = await branch.run("verifier", [user(prompt)], slot=slot)
                verifier.reward("verdict", verdict_reward(verifier.text if verifier.ok else "", shown_right, answer))
                return verifier

            return await episode.branch(self.verifier_samples, one, label=slot)

        verdicts, _ = await asyncio.gather(judge(own, right, "own"), judge(twin, not right, "twin"))
        first = verdicts[0]
        if tagged_word(first.text, "VERDICT", ("accept", "reject")) == "reject":
            team = float(tagged_integer(first.text, "ANSWER") == answer)
        solver.reward("metric/team", team)
