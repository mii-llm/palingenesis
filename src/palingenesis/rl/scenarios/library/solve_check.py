"""solve_check: a solver and a checker over any single-agent RL dataset, scored by the run's rewards.

The solver gets the row's own prompt and gives its final answer (see patterns: the `ANSWER:` line). The checker is
shown the task with a candidate answer and must reply `VERDICT: ACCEPT` or `VERDICT: REJECT`, and on a rejection
its own answer. Whether an answer is right is always decided by the run's `rewards:` functions, never by this file.

Balanced candidates (the trick of checked_answer). In every episode the checker judges two candidates, in two
separate sessions: one that the rewards score as right and one that they score as wrong. One of the two is the
solver's own answer; the other is its twin of the opposite class:

  solver right   the wrong twin is, in this order: the wrong answer of another solver attempt sampled in the same
                 episode (`solver_samples`); a numeric gold one slip away (off by one, by ten, digits reversed);
                 an entry of the row's `distractors` column; the gold answer of another row of the dataset.
  solver wrong   the right twin is the gold answer (`gold_field`), or, without one, a right answer of another
                 attempt.

Every candidate, the solver's own included, is classified by the rewards on its text alone (as a completion with
no session behind it), so the same text never has two classes: a "wrong" twin the rewards accept is discarded,
and so is a gold answer they reject. A candidate longer than `max_candidate_chars`, or one the rewards could not
score (an outage: SkipSample), has no class and is never shown. When the episode has no pair of one right and one
wrong answer with different texts, no checker session runs in it (`metric/checked` 0 on the solver): the checker
never sees an unbalanced episode. The solver itself is still paid as in single-agent RL, with its session (its
tokens, its finish reason) given to the rewards.

Roles: `solver`, `checker`.

Reward components:

  solver    correct         the rewards' weighted sum on its final answer; 0 without an answer.
            metric/team     1 when the answer the pair ends with is right: the checker's correction if it
                            rejected the solver's answer, else the solver's own (no verdict changes nothing).
            metric/checked  1 when the episode had a balanced pair. metric/twin_<source>: where the candidate of
                            the class opposite to the solver's came from (gold, sibling, slip, distractor,
                            other_row).
  checker   verdict         accepting a right answer +1; accepting a wrong one -1; rejecting a right one -1;
                            rejecting a wrong one +0.5, and +1 when its own answer is right; no verdict -1.

Reward hacking, by policy:

  always accept        +1 - 1 = 0 in every episode, whatever the solver's accuracy.
  always reject        -1 + 0.5 = -0.5 per episode without a right correction; with one it is a solver that
                       also wrongly rejects: -1 + 1 = 0. Never above 0.
  no verdict, both     -2 per episode.
  the solver           is paid for its own answer alone: it gains nothing from the checker's verdicts, so it has
                       no reason to write for, or against, the checker.
  surface cues         the candidates are final answers only (the text of the `ANSWER:` line), not the reasoning
                       that led to them, so "long means wrong" is not available. What remains: a gold answer is
                       written the way the dataset writes it, a solver's answer the way the model does. When the
                       solver is wrong, "accept what looks like the dataset" pays; when it is right and the twin
                       is a slip or another row's gold, the same rule loses, and a twin from another attempt
                       looks like the solver's own. The mix depends on the solver's accuracy: watch
                       metric/twin_gold against metric/twin_slip, twin_distractor and twin_other_row.
  answer_tag ""        whole replies are the candidates, and this is much weaker. A reply longer than
                       `max_candidate_chars` is never shown, right or wrong, so only short replies are checked;
                       a gold answer next to a reply of a few sentences is told apart by its length alone, and
                       "accept what is short" then pays whenever the solver is wrong. The audit's `accept_short`
                       partner measures that rule against the oracle solver only, whose answers are short: it
                       cannot see this. Use the tag for solve_check unless the gold column holds whole replies.
  several rewards      a candidate is right when the weighted sum of all the rewards reaches `threshold`. With
                       a format reward next to the correctness one, set `threshold` so that only a correct
                       answer reaches it (`pgs scenario check` warns).
  another row's gold   is often of the wrong kind for the task (a name where a date is asked): an easy rejection.
                       It is the last resort; give the rows a `distractors` column to replace it.
  lenient rewards      a reward that accepts every answer leaves no wrong twin: no checker session runs and the
                       audit reports the checker as unpaid. A reward of the F1 kind is cut at `threshold`.
"""

from __future__ import annotations

import asyncio
import random
import re

from palingenesis.rl.scenarios.base import tagged_word, user
from palingenesis.rl.scenarios.library.patterns import Pattern, between, row_key
from palingenesis.rl.scenarios.runtime import Episode, stable_seed

CHECK_HEAD = {
    "en": "A colleague was given this task:\n\n{task}\n\nTheir final answer:\n",
    "it": "A un collega è stato assegnato questo compito:\n\n{task}\n\nLa sua risposta finale:\n",
}
CHECK_TAGGED = {
    "en": (
        "\n\nCheck the answer yourself. Reply with exactly one line `VERDICT: ACCEPT` if it is right, or "
        "`VERDICT: REJECT` if it is wrong, followed in that case by one line of the form:\n{tag}: <the right answer>"
    ),
    "it": (
        "\n\nControlla tu la risposta. Rispondi con una sola riga `VERDICT: ACCEPT` se è giusta, oppure "
        "`VERDICT: REJECT` se è sbagliata, seguita in quel caso da una riga nella forma:\n{tag}: <la risposta giusta>"
    ),
}
CHECK_FREE = {
    "en": (
        "\n\nCheck the answer yourself. Reply with one line `VERDICT: ACCEPT` if it is right, or with one line "
        "`VERDICT: REJECT` if it is wrong, followed in that case by the right answer."
    ),
    "it": (
        "\n\nControlla tu la risposta. Rispondi con una riga `VERDICT: ACCEPT` se è giusta, oppure con una riga "
        "`VERDICT: REJECT` se è sbagliata, seguita in quel caso dalla risposta giusta."
    ),
}
VERDICT_LINE = re.compile(r"(?im)^[ \t>*_`]*VERDICT\b.*$")
TWIN_SOURCES = ("gold", "sibling", "slip", "distractor", "other_row")


def numeric_slips(gold: str, rng: random.Random) -> list[str]:
    """Wrong answers one slip away from a gold that is a plain integer, of the same sign and in random order;
    nothing for any other gold."""
    text = gold.replace(",", "").strip()
    if not re.fullmatch(r"-?\d{1,15}", text):
        return []
    answer = int(text)
    options = {answer + 1, answer - 1, answer + 10, answer - 10}
    digits = str(abs(answer))
    if len(digits) > 1:
        options.add(int(digits[::-1]) * (1 if answer >= 0 else -1))
    options.discard(answer)
    same_sign = sorted(o for o in options if (o < 0) == (answer < 0)) or sorted(options)
    rng.shuffle(same_sign)
    return [str(o) for o in same_sign]


class SolveCheck(Pattern):
    name = "solve_check"
    roles = ("solver", "checker")
    outcome = "solver"
    weights = {"solver": {"correct": 1.0}, "checker": {"verdict": 1.0}}
    fields = {**Pattern.fields, "distractors": "optional: a list of wrong answers (scenario.args.distractor_field)"}

    def __init__(
        self,
        weights=None,
        solver_samples: int = 2,
        checker_samples: int = 2,
        distractor_field: str = "distractors",
        max_candidate_chars: int = 400,
        **options,
    ):
        super().__init__(weights, **options)
        if solver_samples < 1 or checker_samples < 1:
            raise ValueError("solve_check: solver_samples and checker_samples must be >= 1")
        self.solver_samples, self.checker_samples = solver_samples, checker_samples
        self.distractor_field, self.max_candidate_chars = distractor_field, max_candidate_chars
        self.other_golds: list[str] = []

    def prepare(self, rows: list[dict]) -> None:
        """The dataset's distinct gold answers (at most 4096, in order): the pool of last-resort wrong twins."""
        seen: dict[str, None] = {}
        for row in rows:
            gold = self.gold(row)
            if gold is not None and len(gold) <= self.max_candidate_chars:
                seen.setdefault(gold)
            if len(seen) >= 4096:
                break
        self.other_golds = list(seen)

    def advice(self, rows: list[dict]) -> list[str]:
        warnings = super().advice(rows)
        if len(self.rewards) > 1:
            warnings.append(
                f"solve_check: {len(self.rewards)} rewards are summed, and a candidate counts as right from "
                f"{self.pass_mark:g} on. Set scenario.args.threshold so that only a correct answer reaches it (a "
                "format reward alone must stay below)."
            )
        return warnings

    # ------------------------------------------------------------------ prompts

    def checker_prompt(self, row: dict, candidate: str) -> str:
        return self._head(row) + candidate + self._tail(row)

    def _head(self, row: dict) -> str:
        return CHECK_HEAD[self.language(row)].format(task=self.task(row))

    def _tail(self, row: dict) -> str:
        language = self.language(row)
        return CHECK_TAGGED[language].format(tag=self.answer_tag) if self.answer_tag else CHECK_FREE[language]

    def correction(self, text: str) -> str | None:
        """The checker's own answer: its `ANSWER:` line, or without a tag what it wrote besides the verdict."""
        return self.final_answer(text if self.answer_tag else VERDICT_LINE.sub("", text))

    # ------------------------------------------------------------------ partners

    def partners(self):
        def shown(messages, row) -> str:
            return between(messages[-1]["content"], self._head(row), self._tail(row)) or ""

        async def oracle_solver(messages, row):
            gold = self.gold(row)
            return "" if gold is None else self.say(gold)

        async def oracle_checker(messages, row):
            if await self.classify(shown(messages, row), row):
                return "VERDICT: ACCEPT"
            return "VERDICT: REJECT\n" + self.say(self.gold(row) or "")

        def echo(messages, row):
            return "VERDICT: REJECT\n" + self.say(shown(messages, row))

        def accept_short(messages, row):  # a verdict from the candidate's length alone
            return "VERDICT: ACCEPT" if len(shown(messages, row)) <= 20 else "VERDICT: REJECT\n" + self.say("0")

        return {
            "solver": {
                "oracle": oracle_solver,
                "silent": lambda messages: "",
                "restate": lambda messages: messages[-1]["content"].split("\n")[0],
            },
            "checker": {
                "oracle": oracle_checker,
                "always_accept": lambda messages: "VERDICT: ACCEPT",
                "always_reject": lambda messages: "VERDICT: REJECT\n" + self.say("0"),
                "silent": lambda messages: "",
                "both": lambda messages: "VERDICT: ACCEPT\nVERDICT: REJECT",
                "echo": echo,
                "accept_short": accept_short,
            },
        }

    # ------------------------------------------------------------------ episode

    async def classify(self, answer: str | None, row: dict) -> bool | None:
        """The class of a candidate as text: True when the rewards score it as right, False as wrong, None when
        it cannot be shown or its class is unknown (no answer, longer than `max_candidate_chars`, or the rewards
        could not score it). Every candidate, the solver's own included, is classified this way: by its text
        alone, so the same text never has two classes."""
        if answer is None or not answer.strip() or len(answer) > self.max_candidate_chars:
            return None
        grade = await self.grade_text(answer, row)
        return None if grade.score is None else self.right(grade)

    async def wrong_twin(self, episode: Episode, row: dict, attempts: list, avoid: str | None):
        """(answer, source) of a candidate the rewards score as wrong, or (None, "")."""
        for answer, right in attempts:
            if right is False and answer != avoid:
                return answer, "sibling"
        gold = self.gold(row)
        rng = random.Random(stable_seed(episode.seed, row_key(row), episode.path, "twin"))
        pools = [("slip", numeric_slips(gold, rng) if gold is not None else [])]
        listed = row.get(self.distractor_field)
        pools.append(("distractor", [str(x) for x in listed] if isinstance(listed, (list, tuple)) else []))
        others = [g for g in self.other_golds if g != gold]
        pools.append(("other_row", rng.sample(others, min(4, len(others)))))
        for source, pool in pools:
            for answer in pool:
                if answer != avoid and await self.classify(answer, row) is False:
                    return answer, source
        return None, ""

    async def right_twin(self, row: dict, attempts: list):
        gold = self.gold(row)
        if await self.classify(gold, row) is True:
            return gold, "gold"
        for answer, right in attempts:
            if right is True:
                return answer, "sibling"
        return None, ""

    async def verdict_reward(self, text: str, candidate_right: bool, row: dict) -> tuple[float, bool | None, bool]:
        """(reward, verdict, corrected). verdict: True for an acceptance, False for a rejection, None without
        one; corrected: a rejection came with a right answer of the checker's own."""
        verdict = tagged_word(text, "VERDICT", ("accept", "reject"))
        if verdict is None:
            return -1.0, None, False
        if verdict == "accept":
            return (1.0 if candidate_right else -1.0), True, False
        correction = self.correction(text)
        corrected = correction is not None and self.right(await self.grade_text(correction, row))
        if candidate_right:
            return -1.0, False, corrected
        return (1.0 if corrected else 0.5), False, corrected

    async def run(self, episode: Episode, row: dict) -> None:
        self.required(row)
        prompt = self.solver_messages(row)
        solvers = await asyncio.gather(*(episode.run("solver", prompt) for _ in range(self.solver_samples)))
        for session in solvers:
            await self.pay(session, row)
        answers = [self.answer_of(session) for session in solvers]
        classes = [await self.classify(answer, row) for answer in answers]
        solver, own, own_class = solvers[0], answers[0], classes[0]
        siblings = list(zip(answers[1:], classes[1:]))

        right = wrong = None
        source = ""
        if own_class is True:
            right, (wrong, source) = own, await self.wrong_twin(episode, row, siblings, avoid=own)
        elif own_class is False or own is None:
            # the checker always sees an answer: a solver without one is shown another wrong candidate instead (an
            # empty candidate would be wrong every time, and "reject what is empty" would pay without any checking)
            wrong = own if own_class is False else (await self.wrong_twin(episode, row, siblings, None))[0]
            right, source = await self.right_twin(row, siblings)
        # (an own answer of unknown class, too long to show or not scored by the rewards, is not checked)
        balanced = right is not None and wrong is not None and right.strip().casefold() != wrong.strip().casefold()
        solver.reward("metric/checked", balanced)
        if not balanced:
            return
        for name in TWIN_SOURCES:
            solver.reward(f"metric/twin_{name}", name == source)
        candidates = episode.channel("candidate")  # (nothing is cut: a candidate over the limit is never shown)
        delivered = {
            True: candidates.send(right, "solver", "checker"),
            False: candidates.send(wrong, "solver", "checker"),
        }
        shown = {"own": own_class is True, "twin": own_class is not True}  # slot -> the class of its candidate

        async def judge(slot: str) -> list:
            prompt = [user(self.checker_prompt(row, delivered[shown[slot]]))]

            async def one(branch: Episode):
                checker = await branch.run("checker", prompt, slot=slot)
                outcome = await self.verdict_reward(checker.text if checker.ok else "", shown[slot], row)
                checker.reward("verdict", outcome[0])
                return outcome

            return await episode.branch(self.checker_samples, one, label=slot)

        verdicts, _ = await asyncio.gather(judge("own"), judge("twin"))
        _, verdict, corrected = verdicts[0]
        # the answer the pair ends with: the checker's correction after a rejection, else the solver's own
        solver.reward("metric/team", corrected if verdict is False else own_class is True)
