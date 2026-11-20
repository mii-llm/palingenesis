"""take_away: a two-player game with a known solution (the subtraction game, a one-heap Nim).

A heap of stones; the players alternate and each takes between 1 and `max_take` stones. Whoever takes the last
stone wins (or, in the misère variant, loses). The rules vary per row (heap, max_take, variant), so the winning
rule, leave a multiple of max_take + 1, has to be applied, not recalled; half of the rows start in a position
lost for the first player, so neither seat wins by default.

Roles: `first`, `second`. Each is one session: the referee (this module) tells a player the position, the player
answers with one line `TAKE: n`, and the opponent's move comes back as the next user message.

Reward components per player:

  win             +1 the winner, -1 the loser. A player who breaks the protocol (no move, two moves, an illegal
                  number, a reply cut by its token limit) forfeits: -1 for it, and 0 for the opponent, who did not
                  earn the game. The two players' rewards therefore never add up to more than 0: no pair of
                  policies gains by cooperating.
  optimal         the share of the player's moves that kept a won position won (None when it never held one):
                  checkable move by move against the game's solution, whoever wins. Weight 0 by default (logged).
  metric/forfeit  1 when the player forfeited.

Self-play (both roles the policy), play against a fixed partner (`scripted:oracle`, `scripted:random`), or two
adapters against each other are all configurations of the same scenario.
"""

from __future__ import annotations

import random
import re

from palingenesis.rl.scenarios.base import Scenario, system, tagged_integer, user
from palingenesis.rl.scenarios.runtime import Episode, stable_seed

RULES = {
    "en": (
        "You are playing a game against another player. There is a heap of {heap} stones. You take turns; on "
        "your turn you must take between 1 and {max_take} stones. The player who takes the last stone {end}.\n"
        "Answer every turn with exactly one line of the form:\nTAKE: <number>"
    ),
    "it": (
        "Stai giocando una partita contro un altro giocatore. C'è un mucchio di {heap} pietre. Giocate a turno; "
        "al tuo turno devi prendere da 1 a {max_take} pietre. Chi prende l'ultima pietra {end}.\n"
        "Rispondi a ogni turno con una sola riga nella forma:\nTAKE: <numero>"
    ),
}
END = {"en": ("wins", "loses"), "it": ("vince", "perde")}
FIRST_MOVE = {"en": "Stones left: {heap}. You move first.", "it": "Pietre rimaste: {heap}. Tocca a te per primo."}
NEXT_MOVE = {
    "en": "Your opponent took {taken}. Stones left: {heap}. Your move.",
    "it": "Il tuo avversario ha preso {taken}. Pietre rimaste: {heap}. Tocca a te.",
}


def winning_moves(heap: int, max_take: int, misere: bool) -> list[int]:
    """The moves from `heap` after which the opponent cannot win, [] when the mover has lost already."""
    period = max_take + 1
    target = 1 if misere else 0  # leave the opponent a heap with this remainder
    if misere and heap == 1:
        return []
    take = (heap - target) % period
    return [take] if 1 <= take <= min(max_take, heap) else []


def mover_wins(heap: int, max_take: int, misere: bool) -> bool:
    return bool(winning_moves(heap, max_take, misere))


def _position(messages: list[dict]) -> tuple[int, int, bool]:
    """(stones left, max_take, misère) read back from a player's conversation (the scripted partners' view)."""
    rules = messages[0]["content"]
    max_take = int(re.search(r"(?:and|a) (\d+) (?:stones|pietre)", rules).group(1))
    misere = any(word in rules for word in ("loses", "perde"))
    last = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    heap = int(re.findall(r"(?:left|rimaste): (\d+)", last)[-1])
    return heap, max_take, misere


def oracle(messages: list[dict]) -> str:
    """A perfect player; from a lost position, a legal move drawn from the position."""
    heap, max_take, misere = _position(messages)
    moves = winning_moves(heap, max_take, misere)
    if not moves:
        moves = [random.Random(stable_seed(heap, max_take, len(messages))).randint(1, min(max_take, heap))]
    return f"TAKE: {moves[0]}"


def random_player(messages: list[dict]) -> str:
    heap, max_take, _ = _position(messages)
    return f"TAKE: {random.Random(stable_seed('random', heap, len(messages))).randint(1, min(max_take, heap))}"


class TakeAway(Scenario):
    name = "take_away"
    roles = ("first", "second")
    outcome = "first"
    weights = {"first": {"win": 1.0, "optimal": 0.0}, "second": {"win": 1.0, "optimal": 0.0}}
    oracle_floor = {"first": -0.02, "second": -0.02}  # perfect players split the balanced rows: 0 each

    @classmethod
    def generate(
        cls, n: int, seed: int = 0, max_heap: int = 21, misere_share: float = 0.25, languages=("en", "it")
    ) -> list[dict]:
        """`n` games, half of them lost for the first player under perfect play: heaps of 4..max_heap, max_take
        2..5 (below the heap), a `misere_share` of misère games, the languages in turn."""
        rng = random.Random(stable_seed("take_away", seed))
        rows = []
        while len(rows) < n:
            max_take = rng.randint(2, 5)
            heap = rng.randint(max_take + 2, max(max_heap, max_take + 2))
            misere = rng.random() < misere_share
            wanted = len(rows) % 2 == 0  # alternate the classes exactly
            if mover_wins(heap, max_take, misere) != wanted:
                continue
            rows.append(
                {
                    "scenario": cls.name,
                    "id": f"{cls.name}-{seed}-{len(rows)}",
                    "language": languages[len(rows) // 2 % len(languages)],
                    "heap": heap,
                    "max_take": max_take,
                    "misere": misere,
                    "first_wins": wanted,
                }
            )
        return rows

    def partners(self):
        table = {
            "oracle": oracle,
            "always_one": lambda messages: "TAKE: 1",
            "silent": lambda messages: "",
            "illegal": lambda messages: "TAKE: 99",
            "two_moves": lambda messages: "TAKE: 1\nTAKE: 2",
            "chatty": lambda messages: "I think taking a couple of stones is a good idea here.",
        }
        return {"first": dict(table), "second": dict(table)}

    async def run(self, episode: Episode, row: dict) -> None:
        language = row.get("language", "en")
        heap, max_take, misere = int(row["heap"]), int(row["max_take"]), bool(row["misere"])
        rules = RULES[language].format(heap=heap, max_take=max_take, end=END[language][int(misere)])
        turns = heap // 2 + 2  # more than a player can need
        opening = FIRST_MOVE[language].format(heap=heap)
        seats = {
            "first": await episode.open("first", [system(rules), user(opening)], max_turns=turns),
            "second": None,  # opened when the first move is known: its first user message carries it
        }
        optimal = {"first": [], "second": []}
        forfeit, winner = None, None
        mover, text = "first", await seats["first"].say()
        while True:
            take = tagged_integer(text or "", "TAKE")
            if take is None or not 1 <= take <= min(max_take, heap):
                forfeit = mover
                break
            good = winning_moves(heap, max_take, misere)
            if good:
                optimal[mover].append(float(take in good))
            heap -= take
            if heap == 0:
                other = "second" if mover == "first" else "first"
                winner = other if misere else mover
                break
            message = NEXT_MOVE[language].format(taken=take, heap=heap)
            mover = "second" if mover == "first" else "first"
            if seats[mover] is None:
                seats[mover] = await episode.open("second", [system(rules), user(message)], max_turns=turns)
                text = await seats[mover].say()
            else:
                text = await seats[mover].say(message)
        for role, live in seats.items():
            if live is None:  # the first player forfeited on its first move: the second never played
                continue
            session = await live.done()
            if forfeit is not None:
                session.reward("win", -1.0 if role == forfeit else 0.0)
            else:
                session.reward("win", 1.0 if role == winner else -1.0)
            session.reward("optimal", sum(optimal[role]) / len(optimal[role]) if optimal[role] else None)
            session.reward("metric/forfeit", role == forfeit)
