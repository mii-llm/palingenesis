"""wordle and number-guess: multi-turn text games with feedback after every move.

Both are played through one tool, guess(...), or, for models without function calling, by writing the move in
<guess>...</guess> tags (the environment parses it and answers in a user turn). The secret is in the row and never
in the prompt; the only information is the feedback, so the reward is simply whether the secret was guessed within
the budget. Saying the secret without playing it does not count.

  wordle        a five-letter word, six guesses; feedback per letter: G right place, Y elsewhere in the word, _ absent
                (repeated letters marked as in the original game)
  number-guess  a number in 1..limit with "higher" / "lower" feedback and a budget one above the optimum
                (ceil(log2(limit + 1)) guesses solve it by bisection): scanning or guessing at random fails
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

from palingenesis.rl.catalog.runner import turn_index
from palingenesis.rl.catalog.spec import DatasetSpec, EnvSpec
from palingenesis.rl.env import ToolFailed

_GUESS_TAG = re.compile(r"<guess>\s*(.*?)\s*</guess>", re.S | re.I)

WORDS = (
    "about above actor acute admit adopt adult after again agent agree ahead alarm album alert alike alive allow "
    "alone along alter among anger angle angry apart apple apply arena argue arise array aside asset audio avoid "
    "award aware badly baker basic beach began begin being below bench birth black blame blank blast blend blind "
    "block blood board boost bound brain brand brave bread break breed brick brief bring broad brown brush build "
    "cabin cable carry catch cause chain chair chalk charm chart chase cheap check chest chief child claim class "
    "clean clear climb clock close cloud coach coast could count court cover craft crane crash cream crime cross "
    "crowd crown curve cycle daily dance dealt death delay depth doubt dozen draft drama dream dress drink drive "
    "eager early earth eight elect empty enemy enjoy enter equal error event exact exist extra faith false fault "
    "field fifth fight final first flame flash fleet floor fluid focus force forth found frame fresh front fruit "
    "giant given glass globe grace grade grain grand grant grape grass great green gross group grown guard guess "
    "guest guide habit happy heart heavy horse hotel house human ideal image index inner input issue joint judge "
    "juice knife known label large laser later laugh layer learn least leave level light limit local logic loose "
    "lucky lunch magic major maker march match maybe metal might minor model money month moral motor mount mouse "
    "mouth movie music never night noise north novel nurse ocean offer often order other ought paint panel paper "
    "party peace phase phone photo piece pilot pitch place plain plane plant plate point pound power press price "
    "pride prime print prior prize proof proud prove queen quick quiet quite radio raise range rapid ratio reach "
    "ready refer right river robot rough round route royal rural scale scene scope score sense serve seven shade "
    "shake shape share sharp sheet shelf shell shift shine shirt shock shoot short sight since skill sleep slide "
    "small smart smile smoke solid solve sorry sound south space spare speak speed spend spice split sport staff "
    "stage stand start state steam steel stick still stock stone store storm story strip study stuff style sugar "
    "table taste teach thank theme there thick thing think third three throw tight tired title today topic total "
    "touch tough tower track trade train treat trend trial tribe trick truck truly trust truth twice under union "
    "unity until upper urban usage usual valid value video visit vital voice waste watch water wheel where which "
    "while white whole woman world worry worth would write wrong young youth"
).split()


def feedback(guess: str, answer: str) -> str:
    """Wordle marks of `guess` against `answer`: G, Y or _ per letter."""
    marks = ["_"] * len(answer)
    remaining: dict[str, int] = {}
    for g, a in zip(guess, answer):
        if g != a:
            remaining[a] = remaining.get(a, 0) + 1
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            marks[i] = "G"
        elif remaining.get(g, 0) > 0:
            marks[i] = "Y"
            remaining[g] -= 1
    return "".join(marks)


class _GuessGame:
    tools = ("guess",)

    def __init__(self):
        self.done = False
        self.solved = False
        self.used = 0
        self.budget = 6

    def text_actions(self, content: str) -> list[tuple[str, dict]] | None:
        found = _GUESS_TAG.findall(content or "")
        return [("guess", {self.argument: found[-1]})] if found else None

    def observation_text(self, results: list[tuple[str, str]]) -> str:
        return "\n".join(text for _, text in results)

    def _left(self) -> str:
        left = self.budget - self.used
        return f"{left} guess{'es' if left != 1 else ''} left."


class WordleEnv(_GuessGame):
    argument = "word"

    def reset(self, answer: str = "", max_guesses: int = 6, **row) -> None:
        self.answer, self.budget = str(answer).lower(), int(max_guesses)
        self.done, self.solved, self.used, self.best = False, False, 0, 0

    def guess(self, word: str) -> str:
        """Guess the secret five-letter word.

        Args:
            word: A five-letter English word.
        """
        if self.done:
            raise ToolFailed("the game is over")
        word = str(word).strip().lower()
        if len(word) != len(self.answer) or not word.isalpha() or not word.isascii():
            raise ToolFailed(f"a guess is one word of {len(self.answer)} letters; this did not count")
        self.used += 1
        marks = feedback(word, self.answer)
        self.best = max(self.best, marks.count("G"))
        if word == self.answer:
            self.solved = self.done = True
            return f"{word.upper()}\n{marks}\nCorrect: you found the word in {self.used}."
        if self.used >= self.budget:
            self.done = True
            return f"{word.upper()}\n{marks}\nOut of guesses."
        return f"{word.upper()}\n{marks}\n{self._left()}"

    def get_reward(self) -> dict[str, float]:
        return {
            "solved": float(self.solved),
            "metric/guesses": float(self.used),
            "metric/greens": self.best / max(1, len(self.answer)),
        }


class NumberGuessEnv(_GuessGame):
    argument = "number"

    def reset(self, secret: int = 1, limit: int = 100, max_guesses: int = 0, **row) -> None:
        self.secret, self.limit = int(secret), int(limit)
        self.budget = int(max_guesses) or math.ceil(math.log2(self.limit + 1)) + 1
        self.done, self.solved, self.used = False, False, 0

    def guess(self, number: int) -> str:
        """Guess the secret number.

        Args:
            number: An integer in the game's range.
        """
        if self.done:
            raise ToolFailed("the game is over")
        try:
            value = int(str(number).strip())
        except ValueError:
            raise ToolFailed("a guess is one integer; this did not count") from None
        self.used += 1
        if value == self.secret:
            self.solved = self.done = True
            return f"{value}: correct, in {self.used} guesses."
        if self.used >= self.budget:
            self.done = True
            return f"{value}: wrong. Out of guesses."
        return f"{value}: the secret number is {'higher' if value < self.secret else 'lower'}. {self._left()}"

    def get_reward(self) -> dict[str, float]:
        return {"solved": float(self.solved), "metric/guesses": float(self.used)}


# ----------------------------------------------------------------- generators

WORDLE_PROMPT = (
    "Let's play Wordle. I am thinking of a five-letter English word and you have {n} guesses. Make each guess with "
    "the guess tool (or write it as <guess>word</guess>). After each guess you get one mark per letter: G = right "
    "letter in the right place, Y = the letter is in the word but elsewhere, _ = the letter is not in the word."
)
NUMBER_PROMPT = (
    "I am thinking of a whole number between 1 and {limit}. You have {n} guesses. Make each guess with the guess "
    "tool (or write it as <guess>number</guess>); after each one I say whether my number is higher or lower."
)


def generate_wordle(n: int, seed: int = 0, max_guesses: int = 6) -> list[dict]:
    rng = random.Random(seed)
    return [
        {
            "id": f"wordle/synthetic/{seed}/{i}",
            "prompt": WORDLE_PROMPT.format(n=max_guesses),
            "answer": rng.choice(WORDS),
            "max_guesses": max_guesses,
            "max_turns": max_guesses + 2,
        }
        for i in range(n)
    ]


def generate_number_guess(n: int, seed: int = 0, limit: int = 1000) -> list[dict]:
    rng = random.Random(seed)
    budget = math.ceil(math.log2(limit + 1)) + 1
    return [
        {
            "id": f"number-guess/synthetic/{seed}/{i}",
            "prompt": NUMBER_PROMPT.format(limit=limit, n=budget),
            "secret": rng.randint(2 * budget + 1, limit),  # out of reach of counting up from 1
            "limit": limit,
            "max_guesses": budget,
            "max_turns": budget + 2,
        }
        for i in range(n)
    ]


# ------------------------------------------------------------------- policies


def observations(messages: list[dict]) -> list[str]:
    """What the game answered so far, one text per guess (tool messages, or user turns after the first)."""
    out, seen_assistant = [], False
    for m in messages:
        if m.get("role") == "assistant":
            seen_assistant = True
        elif seen_assistant and m.get("role") in ("tool", "user"):
            out.append(str(m.get("content") or ""))
    return out


def wordle_oracle(row: dict, messages: list[dict]) -> Any:
    return [("guess", {"word": row["answer"]})] if turn_index(messages) == 0 else "Solved."


def wordle_solver(row: dict, messages: list[dict]) -> Any:
    """Plays without the gold: the first word still consistent with every mark so far, written in tags."""
    candidates = list(WORDS)
    for text in observations(messages):
        lines = text.splitlines()
        if len(lines) >= 2 and len(lines[1]) == 5:
            word, marks = lines[0].lower(), lines[1]
            candidates = [c for c in candidates if feedback(word, c) == marks]
    openers = ["raise", "cloud"]
    turn = turn_index(messages)
    pick = openers[turn] if turn < len(openers) and len(candidates) > 2 else candidates[0] if candidates else "about"
    return f"<guess>{pick}</guess>"


def claim_word(row: dict, messages: list[dict]) -> str:
    """States the secret without playing it."""
    return f"I know it: the word is {row.get('answer', row.get('secret'))}. I win!"


def same_guess(row: dict, messages: list[dict]) -> Any:
    return [("guess", {"word": "zzzzz"})]


def random_words(row: dict, messages: list[dict]) -> Any:
    rng = random.Random(f"{row['id']}/{turn_index(messages)}")
    return [("guess", {"word": rng.choice(WORDS)})]


def bisect(row: dict, messages: list[dict]) -> Any:
    """The number game's oracle needs no gold: bisection from the feedback."""
    lo, hi = 1, row["limit"]
    for text in observations(messages):
        match = re.match(r"(-?\d+): the secret number is (higher|lower)", text)
        if match:
            value = int(match.group(1))
            lo, hi = (max(lo, value + 1), hi) if match.group(2) == "higher" else (lo, min(hi, value - 1))
        elif "correct" in text:
            return "Found it."
    return [("guess", {"number": (lo + hi) // 2})]


def count_up(row: dict, messages: list[dict]) -> Any:
    return [("guess", {"number": turn_index(messages) + 1})]


def random_numbers(row: dict, messages: list[dict]) -> Any:
    rng = random.Random(f"{row['id']}/{turn_index(messages)}")
    return [("guess", {"number": rng.randint(1, row["limit"])})]


WORDLE = EnvSpec(
    name="wordle",
    version="1.0.0",
    family="games",
    summary="Wordle: find a five-letter word in six guesses from per-letter feedback.",
    env=WordleEnv,
    license="Apache-2.0",
    verifier="The secret word was played as a guess within the budget (stating it in prose does not count).",
    datasets={
        "synthetic": DatasetSpec(
            generator=generate_wordle,
            license="Apache-2.0",
            description=f"Secrets drawn from a built-in list of {len(WORDS)} common words; any five letters may be guessed.",
        )
    },
    difficulty={"option": "max_guesses", "levels": [6, 5, 4]},
    max_turns=8,
    max_new_tokens=512,
    chance=0.1,  # six random list words find the secret about 1.3% of the time
    oracle=wordle_oracle,
    degenerate={"claim_word": claim_word, "same_guess": same_guess, "random_words": random_words},
    tags=("multi-turn", "game", "text-protocol"),
)

NUMBER_GUESS = EnvSpec(
    name="number-guess",
    version="1.0.0",
    family="games",
    summary="Find a secret number from higher / lower feedback within a near-optimal budget (bisection).",
    env=NumberGuessEnv,
    license="Apache-2.0",
    verifier="The secret number was played as a guess within the budget of ceil(log2(limit + 1)) + 1 guesses.",
    datasets={
        "synthetic": DatasetSpec(generator=generate_number_guess, license="Apache-2.0", description="Random secrets.")
    },
    difficulty={"option": "limit", "levels": [100, 1000, 100000]},
    max_turns=13,
    max_new_tokens=256,
    chance=0.1,
    oracle=bisect,
    degenerate={"claim_number": claim_word, "count_up": count_up, "random_numbers": random_numbers},
    tags=("multi-turn", "game", "text-protocol"),
)
