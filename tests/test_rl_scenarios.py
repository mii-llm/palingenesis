"""Multi-role RL (palingenesis.rl.scenarios): the runtime (sessions, channels, branches), the starter scenarios'
generators and verifiers (oracles earn, degenerate and colluding policies do not), credit (comparison groups,
episode baselines, per-role scale and share), token-exact sessions through the rollout loop, asynchronous
batches under bounded staleness, roles with their own LoRA adapters on one base, checkpoints and resume, dumps,
exports, the layout planner and configuration errors."""

import asyncio
import itertools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from test_rl import tiny_policy, tokenizer  # noqa: E402

from palingenesis.rl.config import RLConfig, RLConfigError  # noqa: E402
from palingenesis.rl.scenarios import (  # noqa: E402
    Scenario,
    build_scenario,
    credit,  # noqa: E402
    play,
)
from palingenesis.rl.scenarios.base import tagged_integer, tagged_word, user  # noqa: E402
from palingenesis.rl.scenarios.runtime import (  # noqa: E402
    Episode,
    PartnerHost,
    RolloutSession,
    ScenarioError,
    Session,
    SessionPort,
)
from palingenesis.rl.trajectory import Trajectory, group_is_informative  # noqa: E402

SCENARIOS = ("take_away", "ledger_sum", "checked_answer")


class Chat(Scenario):
    """A test scenario with dense rewards (a random model's sessions differ): `a` speaks, two `b` sessions answer
    what a channel delivers of it (a branch), `a` replies to the first answer."""

    name = "chat"
    roles = ("a", "b")
    outcome = "a"
    weights = {"a": {"length": 1.0}, "b": {"length": 1.0}}

    @classmethod
    def generate(cls, n, seed=0):
        return [{"scenario": "chat", "id": f"chat-{seed}-{i}", "prompt": f"Say something about {i}."} for i in range(n)]

    async def run(self, episode, row):
        a = await episode.open("a", [user(row["prompt"])], max_turns=2)
        first = await a.say()
        note = episode.channel("note", max_chars=30, redact=("secret",)).send(first or "", "a", "b")

        async def answer(sub):
            b = await sub.run("b", [user(f"Reply to: {note}")])
            b.reward("length", len(b.text) / 40)
            return b

        answers = await episode.branch(2, answer)
        await a.say(answers[0].text or "ok")
        session = await a.done()
        session.reward("length", len(session.text) / 40)
        session.reward("metric/turns", len([m for m in session.messages if m["role"] == "assistant"]))


class Flaky(Chat):
    """Its first episode of a row raises and its second never ends."""

    name = "flaky"

    async def run(self, episode, row):
        session = await episode.run("a", [user(row["prompt"])])
        if episode.index == 0:
            raise RuntimeError("the verifier broke")
        if episode.index == 1:
            await asyncio.sleep(30)
        session.reward("length", float(episode.index))


CHAT = f"{__file__}:Chat"


# ------------------------------------------------------------------ runtime


def test_partner_sessions_channels_and_branch_keys():
    class Probe(Scenario):
        name, roles = "probe", ("x", "y")

        async def run(self, episode, row):
            x = await episode.open("x", [user("hi")])
            said = await x.say()
            delivered = episode.channel("c", max_chars=5, redact=("ab",)).send(said + "abcdefgh", "x", "y")
            ys = await episode.branch(3, lambda sub: sub.run("y", [user(delivered)], slot="s"))
            again = await x.say("more")
            episode.shared["texts"] = (said, delivered, again, [y.text for y in ys])
            await x.close()
            assert await x.say() is None

    counter = itertools.count()
    partners = {"x": lambda messages: f"x{next(counter)}", "y": lambda messages: "y:" + messages[-1]["content"]}
    (episodes,) = play(Probe(), [{"id": "r"}], partners, episodes=2)
    for episode in episodes:
        said, delivered, again, answers = episode.shared["texts"]
        assert delivered == "x" + said[1:] + "[red"[: 5 - len(said)] and again != said
        assert answers == ["y:" + delivered] * 3
        x, *ys = episode.sessions
        assert x.name == "x#1" and [y.name for y in ys] == ["y.s#1", "y.s#2", "y.s#3"]
        assert x.key == ("x", "", ("",)) and len({y.key for y in ys}) == 1 and ys[0].key != x.key
        assert len({y.path for y in ys}) == 3  # each branch is its own unit
        assert [m["role"] for m in x.messages] == ["user", "assistant", "user", "assistant"]
        (message,) = episode.log
        assert message["redacted"] == 1 and message["cut"] and message["sent"].endswith("abcdefgh")
        assert episode.record()["sessions"][1]["messages"][0]["content"] == delivered
    assert episodes[0].sessions[0].key == episodes[1].sessions[0].key  # the group's episodes are one comparison group
    assert episodes[0].rng.random() != episodes[1].rng.random()
    again = play(Probe(), [{"id": "r"}], partners, episodes=1)[0][0]
    assert again.rng.random() == Episode(PartnerHost(partners), {"id": "r"}).rng.random()  # seeded by row and place


def test_rollout_session_exchanges_turns_with_the_turn_loop():
    """RolloutSession against a stand-in for the pipeline's loop: each message goes out through the port and the
    answer comes back as the tool result; a close ends the loop; a loop stopped by its turn limit still hands
    out its last message."""

    async def loop(port, turns, finish):
        trajectory = Trajectory({}, 0, [], [{"role": "user", "content": "go"}])
        for i in range(turns):
            trajectory.messages.append({"role": "assistant", "content": f"m{i}"})
            if i == turns - 1 and finish == "turns":
                break
            reply = await port.call_tool("reply", {"text": f"m{i}"})
            trajectory.messages.append({"role": "user", "content": reply})
            if port.done:
                finish = "env_done"
                break
        trajectory.finish = finish
        return trajectory

    async def scene():
        port = SessionPort(None, True)
        live = RolloutSession(Session("r", "r#1", (), ()), port, loop(port, 3, "turns"))
        assert await live.say() == "m0"
        with pytest.raises(ScenarioError, match="without an answer"):
            await live.say()
        assert await live.say("a0") == "m1"
        assert await live.say("a1") == "m2"  # the loop ended on it (turn limit): handed out once
        assert await live.say() is None
        session = await live.done()
        assert session.finish == "turns" and session.text == "m2" and session.trajectory is not None

        port = SessionPort(None, True)
        live = RolloutSession(Session("r", "r#2", (), ()), port, loop(port, 5, "turns"))
        assert await live.say() == "m0"
        session = await live.done()  # closed while it waits for an answer
        assert session.finish == "env_done" and session.ok
        assert [m["role"] for m in session.messages] == ["user", "assistant"]  # the empty closing answer is dropped

        async def tool(x: int) -> str:
            """Double.

            Args:
                x: a number
            """
            return str(2 * x)

        port = SessionPort({"double": tool}, False)
        assert port.tool_schemas()[0]["function"]["name"] == "double" and port.text_actions("hello") is None
        assert await port.call_tool("double", {"x": 4}) == "8"

    asyncio.run(scene())


def test_answer_parsers_take_exactly_one_tagged_line():
    assert tagged_integer("I think.\nTAKE: 3", "TAKE") == 3 and tagged_integer("take 3.", "TAKE") == 3
    assert tagged_integer("**TOTAL: 1,234**", "TOTAL") == 1234 and tagged_integer("ANSWER: -17", "ANSWER") == -17
    for text in ("", "TAKE: 1\nTAKE: 2", "I will TAKE: 3 stones", "TAKE: three", "TAKE: 1.5", None):
        assert tagged_integer(text, "TAKE") is None, text
    assert tagged_word("VERDICT: Accept.", "VERDICT", ("accept", "reject")) == "accept"
    assert tagged_word("VERDICT: maybe", "VERDICT", ("accept", "reject")) is None
    assert tagged_word("VERDICT: ACCEPT\nVERDICT: REJECT", "VERDICT", ("accept", "reject")) is None


# ------------------------------------------------- scenarios: data, verifiers


@pytest.mark.parametrize("name", SCENARIOS)
def test_generators_are_seeded_and_rows_are_self_contained(name):
    cls = type(build_scenario(name))
    rows = cls.generate(60, seed=3)
    assert rows == cls.generate(60, seed=3) and rows != cls.generate(60, seed=4)
    assert len({row["id"] for row in rows}) == 60 and all(row["scenario"] == name for row in rows)
    assert {row["language"] for row in rows} == {"en", "it"}
    assert json.loads(json.dumps(rows, ensure_ascii=False)) == rows  # plain JSONL
    assert not any("messages" in row or "prompt" in row for row in rows)  # the task alone: prompts are built per role


def test_generated_tasks_are_balanced_and_their_gold_is_right():
    from palingenesis.rl.scenarios.library.checked_answer import _evaluate, distractor
    from palingenesis.rl.scenarios.library.take_away import mover_wins, winning_moves

    games = build_scenario("take_away").generate(200, seed=1)
    assert sum(row["first_wins"] for row in games) == 100  # neither seat wins by default
    assert all(row["first_wins"] == mover_wins(row["heap"], row["max_take"], row["misere"]) for row in games)
    assert 20 < sum(row["misere"] for row in games) < 80

    def wins(heap, k, misere):  # the game solved by search, against the closed form
        return any(
            (h == 0 and not misere) or (h > 0 and not wins(h, k, misere)) if True else None
            for h in (heap - take for take in range(1, min(k, heap) + 1))
        )

    for heap, k, misere in itertools.product(range(1, 16), (2, 3, 5), (False, True)):
        assert mover_wins(heap, k, misere) == wins(heap, k, misere), (heap, k, misere)
        for take in winning_moves(heap, k, misere):
            left = heap - take
            assert (left == 0 and not misere) or (left > 0 and not wins(left, k, misere))

    ledgers = build_scenario("ledger_sum").generate(200, seed=1)
    absent = [row["item"] not in ledger for row in ledgers for ledger in row["ledgers"].values()]
    assert 0.12 < sum(absent) / len(absent) < 0.35 and all(row["answer"] > 0 for row in ledgers)
    assert all(row["answer"] == sum(lg.get(row["item"], 0) for lg in row["ledgers"].values()) for row in ledgers)

    import random

    questions = build_scenario("checked_answer").generate(200, seed=1)
    assert all(_evaluate(row["question"]) == row["answer"] for row in questions)
    assert all(distractor(row["answer"], random.Random(i)) != row["answer"] for i, row in enumerate(questions))


@pytest.mark.parametrize("name", SCENARIOS)
def test_verifiers_pay_oracles_and_no_degenerate_policy(name):
    report = build_scenario(name).audit(n=120, seed=2)
    assert report["failures"] == [] and report["degenerate"]
    assert all(value <= 1e-9 for value in report["degenerate"].values()), report["degenerate"]


def test_no_pair_of_players_gains_by_cooperating():
    """take_away is zero-sum at best: whatever two policies do, their rewards add up to at most 0, and a player
    whose opponent broke the rules has not earned a win."""
    scenario = build_scenario("take_away")
    rows = scenario.generate(40, seed=5)
    table = scenario.partners()["first"]
    for first, second in itertools.product(table, repeat=2):
        for (episode,) in play(scenario, rows, {"first": table[first], "second": table[second]}):
            rewards = {s.role: scenario.reward(s.role, s.rewards) for s in episode.sessions}
            assert sum(rewards.values()) <= 0, (first, second, rewards)
            forfeits = [s for s in episode.sessions if s.rewards.get("metric/forfeit")]
            if forfeits:
                assert all(v == (-1.0 if r == forfeits[0].role else 0.0) for r, v in rewards.items())
    # the process reward: an oracle's moves are all optimal, a fixed move's are not
    for (episode,) in play(scenario, rows, {"first": table["oracle"], "second": table["always_one"]}):
        first = episode.sessions[0]
        assert first.rewards["optimal"] in (None, 1.0)
        if episode.row["first_wins"]:
            assert first.rewards["win"] == 1.0 and first.rewards["optimal"] == 1.0


def test_a_constant_verdict_earns_nothing_in_any_episode():
    scenario = build_scenario("checked_answer", {"verifier_samples": 3})
    rows = scenario.generate(40, seed=6)
    solvers = scenario.partners()["solver"]
    wrong = lambda messages: "ANSWER: 123456"  # noqa: E731 — a solver that is always wrong
    for solver in (solvers["oracle"], wrong, solvers["silent"]):
        for verdict, bound in (("always_accept", 0.0), ("always_reject", -0.25), ("silent", -1.0)):
            partners = {"solver": solver, "verifier": scenario.partners()["verifier"][verdict]}
            for (episode,) in play(scenario, rows, partners):
                verdicts = [s.rewards["verdict"] for s in episode.sessions if s.role == "verifier"]
                assert len(verdicts) == 6 and sum(verdicts) / 6 == pytest.approx(bound)  # per episode, exactly
                own = [s for s in episode.sessions if s.role == "verifier" and s.key[1] == "own"]
                twin = [s for s in episode.sessions if s.key[1] == "twin"]
                assert len({s.key for s in own}) == 1 and own[0].key != twin[0].key  # same candidate: one group
    # the pair's answer: a wrong solver with a good verifier ends right, with a rubber stamp it does not
    good = {"solver": wrong, "verifier": scenario.partners()["verifier"]["oracle"]}
    stamp = {"solver": wrong, "verifier": scenario.partners()["verifier"]["always_accept"]}
    assert all(e.sessions[0].rewards["metric/team"] == 1.0 for (e,) in play(scenario, rows, good))
    assert all(e.sessions[0].rewards["metric/team"] == 0.0 for (e,) in play(scenario, rows, stamp))
    assert all(e.sessions[0].rewards["correct"] == 0.0 for (e,) in play(scenario, rows, good))


def test_workers_are_judged_on_the_question_they_got():
    scenario = build_scenario("ledger_sum")
    rows = [row for row in scenario.generate(60, seed=7) if row["language"] == "en"]
    workers = scenario.partners()["worker"]

    async def wrong_item(messages, tools):  # a lead that asks every clerk about another item
        row = wrong_item.row
        other = next(i for i in ("apples", "pears", "lemons") if i != row["item"])
        for region in row["ledgers"]:
            await tools["ask"](region=region, question=f"How many {other}?")
        await tools["ask"](region="atlantis", question="anything")
        await tools["ask"](region=next(iter(row["ledgers"])), question="How is the weather?")
        return "TOTAL: 0"

    for row in rows[:10]:
        wrong_item.row = row
        ((episode,),) = play(scenario, [row], {"lead": wrong_item, "worker": workers["oracle"]})
        lead, *clerks = sorted(episode.sessions, key=lambda s: s.role)
        assert lead.rewards["total"] == 0.0 and lead.rewards["coverage"] == 0.0
        assert lead.rewards["metric/asks"] == len(row["ledgers"]) + 1  # the unknown region reached no clerk
        judged = [c for c in clerks if c.rewards["value"] is not None]
        assert len(judged) == len(row["ledgers"]) and all(
            c.rewards["value"] == 1.0 for c in judged
        )  # right on what was asked
        (unjudged,) = [c for c in clerks if c.rewards["value"] is None]  # no listed item in the question
        assert scenario.reward("worker", unjudged.rewards) is None or unjudged.rewards["team"] == 0.0
        assert len(episode.log) == 2 * len(clerks)  # every question and every reply went through a channel


# -------------------------------------------------------------------- credit


def _t(reward, role="a", unit=None, key=None, tokens=3, scored=True, trained=True):
    t = Trajectory({}, 0, [1], [], tokens=[1] * tokens, mask=[True] * tokens, logprobs=[0.0] * tokens)
    t.reward, t.scored, t.trained, t.role = reward, scored, trained, role
    t.comparison, t.unit = key or (0, role), unit if unit is not None else (id(t),)
    return t


def test_row_groups_split_into_comparison_groups_and_only_informative_ones_train():
    row = credit.RowGroup(
        [_t(1, "a"), _t(-1, "a"), _t(-1, "b"), _t(-1, "b"), _t(0.5, "c", trained=False), _t(0.1, "c", trained=False)]
    )
    assert [len(g) for g in row.comparison_groups] == [2, 2, 2]
    assert group_is_informative(row)  # a's rewards differ
    assert [g[0].role for g in credit.comparison_groups([row])] == ["a"]  # b is constant, c does not train
    flat = credit.RowGroup([_t(1, "a"), _t(1, "a"), _t(-1, "b"), _t(-1, "b")])
    assert not group_is_informative(flat)  # the roles differ from each other, but no role's sessions do
    plain = [_t(1), _t(0)]
    assert group_is_informative(plain) and credit.comparison_groups([plain]) == [plain]


def test_advantages_use_episode_baselines_and_a_floored_scale_per_role():
    from palingenesis.rl.losses import assign_advantages

    # one role, one session per episode: exactly single-agent RL's advantages
    rewards = [1.0, 0.0, 0.5, 0.0]
    ours, theirs = [_t(r) for r in rewards], [_t(r) for r in rewards]
    credit.assign_advantages([ours], std="batch")
    assign_advantages([theirs], std="batch")
    assert [t.advantage for t in ours] == pytest.approx([t.advantage for t in theirs], abs=1e-12)
    credit.assign_advantages([ours], std="none", baseline="leave_one_out")
    assign_advantages([theirs], std="none", baseline="leave_one_out")
    assert [t.advantage for t in ours] == pytest.approx([t.advantage for t in theirs], abs=1e-12)

    # an episode with three sessions counts once in the baseline, and leave-one-out leaves all three out
    many = [_t(1.0, unit=("e0",)) for _ in range(3)] + [_t(0.0, unit=("e1",))]
    credit.assign_advantages([many], std="none")
    assert [t.advantage for t in many] == pytest.approx([0.5, 0.5, 0.5, -0.5])  # baseline 0.5, not 0.75
    credit.assign_advantages([many], std="none", baseline="leave_one_out")
    assert [t.advantage for t in many] == pytest.approx([1.0, 1.0, 1.0, -1.0])

    # per role: a wide reward does not set the size of a narrow one's advantages; the floor stops tiny ones
    wide, narrow = [_t(10.0, "a"), _t(-10.0, "a")], [_t(0.01, "b"), _t(-0.01, "b")]
    stats = credit.assign_advantages([wide, narrow], std="batch", roles=("a", "b"))
    assert abs(narrow[0].advantage) < 0.01 * abs(wide[0].advantage)  # pooled: b is noise next to a
    stats = credit.assign_advantages([wide, narrow], std="role", roles=("a", "b"))
    assert wide[0].advantage == pytest.approx(1.0, abs=1e-4) and narrow[0].advantage == pytest.approx(1.0, abs=1e-3)
    assert stats["role/a/advantage_std"] == pytest.approx(10.0) and stats["role/b/advantage_std"] == pytest.approx(0.01)
    credit.assign_advantages([wide, narrow], std="role", floor=0.1, roles=("a", "b"))
    assert wide[0].advantage == pytest.approx(1.0, abs=1e-4) and narrow[0].advantage == pytest.approx(0.1, abs=1e-3)

    # over data-parallel ranks the per-role sums are reduced: two ranks with these groups give the same scale
    calls = []

    def reduce(values):
        calls.append(list(values))
        return [2 * v for v in values]

    credit.assign_advantages([wide, narrow], std="role", reduce=reduce, roles=("a", "b"))
    assert len(calls) == 1 and len(calls[0]) == 6 and wide[0].advantage == pytest.approx(1.0, abs=1e-4)


def test_each_roles_part_of_the_loss_is_its_share_not_its_number_of_groups_or_tokens():
    """With "prompt" aggregation a role's part of the loss is the share of comparison groups it happens to have
    (one group for the lead, one per worker branch): an accident of the scenario's shape. Shares make it explicit.
    (arXiv 2605.24202 reports the role's weight as a hidden hyperparameter of shared-policy training.)"""
    from palingenesis.rl import losses

    lead = [[_t(1, "lead", tokens=50), _t(0, "lead", tokens=70)]]
    workers = [[_t(1, "worker", tokens=5), _t(0, "worker", tokens=7)] for _ in range(4)]
    groups = lead + workers

    def mass(weights):
        out = {"lead": 0.0, "worker": 0.0}
        for g in groups:
            for t in g:
                out[t.role] += weights.get(id(t), 0.0) * t.sampled_tokens
        return out

    plain = mass(losses.token_weights(groups, "prompt", 100))
    assert plain["lead"] == pytest.approx(1 / 5) and plain["worker"] == pytest.approx(4 / 5)  # the group count
    tokens = mass(losses.token_weights(groups, "token", 100))
    assert tokens["lead"] == pytest.approx(120 / 168)  # the token count
    for aggregation in ("prompt", "token"):
        even = mass(credit.token_weights(groups, aggregation, 100, {"lead": 1.0, "worker": 1.0}))
        assert even["lead"] == pytest.approx(0.5) and even["worker"] == pytest.approx(0.5)
        skew = mass(credit.token_weights(groups, aggregation, 100, {"lead": 3.0, "worker": 1.0}))
        assert skew["lead"] == pytest.approx(0.75) and sum(skew.values()) == pytest.approx(1.0)
    # one role: the single-agent weights exactly; a role with nothing to train leaves its share to the others
    assert credit.token_weights(lead, "prompt", 100, {"lead": 1.0}) == losses.token_weights(lead, "prompt", 100)
    alone = mass(credit.token_weights(lead, "prompt", 100, {"lead": 1.0, "worker": 5.0}))
    assert alone["lead"] == pytest.approx(1.0)


def test_adam_ignores_a_loss_scale_on_a_separate_adapter_but_a_shared_clip_does_not():
    """Why shares matter for a shared policy and clipping is per adapter for separate ones: AdamW's update of a
    parameter group does not change when its loss is multiplied by a constant, unless a gradient clip shared with
    another group rescales it (checked claim behind arXiv 2605.24202's role-weight discussion)."""

    def step(scale, clip, shared):
        torch.manual_seed(0)
        a, b = torch.nn.Parameter(torch.randn(8)), torch.nn.Parameter(torch.randn(8))
        opt = torch.optim.AdamW([{"params": [a]}, {"params": [b]}], lr=0.1, weight_decay=0.0, eps=1e-15)
        before = b.detach().clone()
        (scale * (a**2).sum() + (b**2).sum()).backward()
        if clip and shared:
            torch.nn.utils.clip_grad_norm_([a, b], 1.0)
        elif clip:
            torch.nn.utils.clip_grad_norm_([a], 1.0)
            torch.nn.utils.clip_grad_norm_([b], 1.0)
        b_grad = b.grad.clone()
        opt.step()
        return b.detach() - before, b_grad

    update, _ = step(1.0, clip=False, shared=False)
    scaled, _ = step(100.0, clip=False, shared=False)
    torch.testing.assert_close(update, scaled)  # the other role's loss scale: no effect on this adapter
    _, grad = step(1.0, clip=True, shared=False)
    _, grad_scaled = step(100.0, clip=True, shared=False)
    torch.testing.assert_close(grad, grad_scaled)  # clipped by itself: still none
    _, grad_shared = step(1.0, clip=True, shared=True)
    _, grad_shared_scaled = step(100.0, clip=True, shared=True)
    assert grad_shared_scaled.norm() < 0.2 * grad_shared.norm()  # a shared clip lets the loud role shrink the other


# ------------------------------------------------ sessions through the pipeline


class RoleEngine:
    """A rollout engine scripted per prompt: `policy(prompt text, call number) -> reply`."""

    def __init__(self, tok, eot, policy, delay=0.0):
        self.tok, self.eot, self.policy, self.delay = tok, eot, policy, delay
        self.version, self.calls, self.batches, self.loaded = 0, 0, [], []

    def generate(self, prompts, max_new_tokens, temperature):
        from palingenesis.opd.rollout import Rollout

        if self.delay:
            time.sleep(self.delay)
        self.batches.append(len(prompts))
        out = []
        for prompt in prompts:
            self.calls += 1
            ids = self.tok.encode(self.policy(self.tok.decode(prompt), self.calls), add_special_tokens=False)
            out.append(Rollout(ids + [self.eot], [-0.25] * (len(ids) + 1), "stop", self.version))
        return out

    def update_weights(self, named, version):
        self.loaded.append(version)
        self.version = version

    def sleep(self):
        pass

    def wake(self):
        pass


def _pipeline(scenario, rows, policy, roles=None, delay=0.0, **options):
    from palingenesis.opd.orchestrator import PublishedWeights
    from palingenesis.opd.teachers import end_of_turn_id
    from palingenesis.rl.chat import ChatFormat
    from palingenesis.rl.data import PromptSampler
    from palingenesis.rl.scenarios.pipeline import ScenarioPipeline, resolve_roles

    tok = tokenizer("Qwen/Qwen3-0.6B")
    kwargs = {"enable_thinking": False}
    eot = end_of_turn_id(tok, kwargs)
    config = RLConfig()
    settings = {
        "model.policy": "x",
        "model.chat_template_kwargs": kwargs,
        "scenario.type": scenario,
        "rollout.group_size": 2,
        "rollout.batch_prompts": 2,
        "rollout.max_new_tokens": 64,
        "rollout.max_model_len": 4096,
        **options,
    }
    for name, role in (roles or {}).items():
        settings.update({f"roles.{name}.{k}": v for k, v in role.items()})
    for key, value in settings.items():
        config.set(key, value)
    built = build_scenario(config.scenario.type, config.scenario.args)
    engine = RoleEngine(tok, eot, policy, delay)
    weights = PublishedWeights(torch.nn.Linear(1, 1))
    pipeline = ScenarioPipeline(
        tok,
        ChatFormat(tok, eot, kwargs),
        engine,
        weights,
        config,
        PromptSampler(rows),
        [],
        None,
        None,
        (eot,),
        scenario=built,
        roles=resolve_roles(config, built),
    )
    return pipeline, engine, weights


def test_a_game_is_one_token_exact_trajectory_per_player():
    """take_away through the rollout loop: each player's whole game is one trajectory (its moves sampled, the
    opponent's moves context), and the referee's verdict is its reward."""
    from palingenesis.rl.scenarios.library.take_away import oracle

    def policy(prompt, call):  # the first player plays perfectly; the second always takes one
        if "move first" in prompt:
            turns = prompt.split("<|im_start|>")[1:]
            messages = [
                {"role": t.split("\n", 1)[0], "content": t.split("\n", 1)[1].split("<|im_end|>")[0]} for t in turns
            ]
            return oracle([m for m in messages if m["role"] in ("system", "user")])
        return "TAKE: 1"

    rows = [row for row in build_scenario("take_away").generate(40, seed=1) if row["language"] == "en"][:4]
    pipeline, engine, _ = _pipeline("take_away", rows, policy)
    try:
        groups = pipeline._await(pipeline._eval(rows, 1.0, 2))
    finally:
        pipeline.close()
    tok = pipeline.tok
    for row, group in zip(rows, groups):
        assert isinstance(group, credit.RowGroup) and {t.role for t in group} == {"first", "second"}
        assert [len(g) for g in group.comparison_groups] == [2, 2]  # the two games' firsts, and their seconds
        for t in group:
            assert t.model == "policy" and t.session == f"{t.role}#1" and t.comparison[0] == t.group
            moves = [m for m in t.messages if m["role"] == "assistant"]
            assert t.turns == len(moves) >= 1 and all(m["content"].startswith("TAKE:") for m in moves)
            assert t.finish in ("env_done", "turns", "stop")
            # token-exact: the sampled tokens are the moves, everything else (the opponent's moves) is context
            sampled = tok.decode([tok_id for tok_id, m in zip(t.tokens, t.mask) if m])
            assert sampled.count("TAKE:") == len(moves) and "opponent" not in sampled
            assert tok.decode(t.prompt_ids + t.tokens).count("Your opponent took") == len(moves) - (t.role == "first")
            assert t.scored and set(t.rewards) == {"win", "optimal", "metric/forfeit"}
        first = [t for t in group if t.role == "first"]
        if row["first_wins"]:  # a perfect first player wins every game it can
            assert all(t.reward == 1.0 and t.rewards["optimal"] == 1.0 for t in first)
        assert all(a.reward == -b.reward for a, b in zip(first, [t for t in group if t.role == "second"]))


def test_a_lead_delegates_through_a_tool_and_partners_play_the_workers():
    """ledger_sum with a scripted policy as the lead (native tool calls, several in one turn) and oracle partners
    as its workers: the workers' replies come back as tool results, and only the lead's session is a trajectory."""
    rows = [row for row in build_scenario("ledger_sum").generate(20, seed=2) if row["language"] == "en"][:3]

    def policy(prompt, call):
        regions = prompt.split("cannot see: ")[1].split(".")[0].split(", ")
        item = prompt.split("How many ")[1].split(" ")[0]
        if "<tool_response>" not in prompt:
            calls = [
                json.dumps({"name": "ask", "arguments": {"region": region, "question": f"How many {item}?"}})
                for region in regions
            ]
            return "".join(f"<tool_call>\n{c}\n</tool_call>\n" for c in calls)
        values = [
            tagged_integer(part.split("</tool_response>")[0], "VALUE") for part in prompt.split("<tool_response>")[1:]
        ]
        return f"TOTAL: {sum(values) + (call % 2)}"  # every other answer is off by one: the group is informative

    roles = {"worker": {"partner": "scripted:oracle"}}
    budget = {"rollout.max_new_tokens": 256, "rollout.max_completion_tokens": 1024}  # two turns, the first long
    pipeline, engine, _ = _pipeline("ledger_sum", rows, policy, roles, **budget)
    try:
        groups = pipeline._await(pipeline._eval(rows, 1.0, 2))
        stats = pipeline._group_stats([t for g in groups for t in g])
    finally:
        pipeline.close()
    for row, group in zip(rows, groups):
        assert [t.role for t in group] == ["lead", "lead"]  # the workers were partners: text, no trajectory
        for t in group:
            assert t.tool_calls == len(row["ledgers"]) and t.tool_errors == 0 and t.turns == 2 and t.finish == "stop"
            assert t.rewards["coverage"] == 1.0 and t.rewards["metric/asks"] == len(row["ledgers"])
            assert t.info["tools"][0]["function"]["name"] == "ask"
            assert [m["role"] for m in t.messages].count("tool") == len(row["ledgers"])
        assert sorted(t.rewards["total"] for t in group) == [0.0, 1.0] and group_is_informative(group)
    assert stats["role/lead/reward"] == pytest.approx(0.5) and stats["role/lead/rewards/coverage"] == 1.0
    assert stats["reward"] == stats["role/lead/reward"]


def test_a_failed_or_slow_episode_is_left_out_and_the_others_train():
    rows = Chat.generate(2)
    pipeline, engine, _ = _pipeline(f"{__file__}:Flaky", rows, lambda prompt, call: "hi", **{"scenario.timeout": 0.5})
    try:
        (group,) = pipeline._await(pipeline._eval(rows[:1], 1.0, 4))
        counts = dict(pipeline.episode_counts)
    finally:
        pipeline.close()
    assert [t.scored for t in group] == [False, False, True, True] and [t.trained for t in group] == [
        False,
        False,
        True,
        True,
    ]
    assert "verifier broke" in group[0].info["episode_error"] and "Timeout" in group[1].info["episode_error"]
    assert counts["episodes"] == 4 and counts["failed"] == 2 and group_is_informative(group)
    assert [t.reward for t in group[2:]] == [2.0, 3.0]


def test_rollouts_overlap_training_within_the_staleness_bound():
    """The producer thread runs episodes while the trainer holds a batch (rollout.max_staleness 1): the next
    batch is under way before the weights it will lag are published, never more than one version behind, and
    every turn carries the version that sampled it."""
    from palingenesis.opd.orchestrator import Orchestrator

    rows = Chat.generate(40)
    pipeline, engine, weights = _pipeline(
        CHAT, rows, lambda prompt, call: "word " * (1 + call % 5), delay=0.01, **{"rollout.max_staleness": 1}
    )
    orchestrator = Orchestrator(pipeline, lambda: 2, 1.0, 1)
    orchestrator.start(end=4)
    try:
        first = orchestrator.next(weights.version)
        calls = engine.calls
        deadline = time.monotonic() + 10
        while engine.calls == calls and time.monotonic() < deadline:  # nothing published: the producer works on
            time.sleep(0.01)
        assert engine.calls > calls and weights.version == 0
        staleness = []
        batches = [first]
        for step in range(1, 4):
            weights.publish()  # the "optimizer step" on the batch before
            batch = orchestrator.next(weights.version)
            batches.append(batch)
            staleness.append(weights.version - batch.version)
            for group in batch.samples:
                for t in group:
                    assert len(t.versions) == t.turns and all(0 <= v <= weights.version for v in t.versions)
                    assert weights.version - t.version <= 2  # a carried group: the trainer drops it past the bound
        assert max(staleness) <= 1 and engine.loaded[:3] == [1, 2, 3][: len(engine.loaded[:3])]
        assert all(isinstance(g, credit.RowGroup) for b in batches for g in b.samples)
        assert all("episode/sessions" in b.stats and b.stats["episode/sessions"] == 3.0 for b in batches)
        assert max(engine.batches) > 1  # sessions of several episodes and roles share an engine call
    finally:
        orchestrator.stop()
        pipeline.close()


def test_stale_rows_are_dropped_whole():
    from palingenesis.rl.trainer import RLTrainer

    trainer = object.__new__(RLTrainer)
    trainer.config = RLConfig()
    trainer.config.set("rollout.max_staleness", 1)
    trainer.weights, trainer.stale_groups = SimpleNamespace(version=3), 0

    def row(*versions):
        group = credit.RowGroup()
        for role, version in zip("ab", versions):
            t = _t(1.0, role)
            t.versions = [version]
            group.append(t)
        return group

    fresh, half, old = row(3, 2), row(3, 1), row(0, 0)
    assert (
        trainer._fresh([fresh, half, old]) == [fresh] and trainer.stale_groups == 2
    )  # one stale session drops its row


# --------------------------------------------------------- trainer, end to end


def _config(tmp_path, scenario=CHAT, **overrides):
    built = build_scenario(scenario) if ":" not in scenario else Chat()
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in type(built).generate(24, seed=0)))
    policy = tmp_path / "policy"
    config = RLConfig()
    for key, value in {
        "model.policy": str(policy) if policy.exists() else tiny_policy(tmp_path),
        "model.use_liger_kernel": False,
        "model.chat_template_kwargs": {"enable_thinking": False},
        "data.dataset": str(tmp_path / "rows.jsonl"),
        "data.eval_size": 2,
        "rollout.backend": "hf",
        "rollout.batch_prompts": 2,
        "rollout.group_size": 3,
        "rollout.max_new_tokens": 8,
        "scenario.type": scenario,
        "loss.advantage_std": "role",
        "loss.advantage_std_floor": 0.05,
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 2,
        "train.learning_rate": 1e-3,
        "train.warmup_steps": 1,
        **overrides,
    }.items():
        config.set(key, value)
    return config


def _trainer(tmp_path, **overrides):
    from palingenesis.rl.trainer import RLTrainer

    trainer = RLTrainer(_config(tmp_path, **overrides))
    trainer.logged = []
    trainer._log = lambda kind, metrics, step: trainer.logged.append((kind, dict(metrics)))
    return trainer


def test_shared_policy_trains_on_every_role_and_dumps_exports_and_resumes(tmp_path):
    from palingenesis.agent_tooling.rollouts import analyze
    from palingenesis.rl.scenarios.export import dpo_rows, session_messages, sft_rows
    from palingenesis.rl.trainer import TRAINER_STATE_FILE
    from palingenesis.rollouts import read_records

    options = {"logging.dump_trajectories": 1, "train.save_steps": 2}
    trainer = _trainer(tmp_path, **options)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    trainer.train()
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))
    steps = [m for kind, m in trainer.logged if kind == "step"]
    assert len(steps) == 2
    for m in steps:
        assert m["trained_sequences"] > 0 and m["episode/failed"] == 0 and m["episode/sessions"] == 3
        assert m["role/a/sessions"] > 0 and m["role/b/sessions"] == 2 * m["role/a/sessions"]
        # (a random model's turns are cut by the token limit, which ends a session: one turn each)
        assert m["role/a/turns"] == 1 and m["role/b/turns"] == 1 and "role/a/rewards/metric/turns" in m
        assert m["role/a/advantage_std"] > 0 and "outcome/pass@1" in m and m["reward"] == m["role/a/reward"]
    assert any(kind == "eval" and "role/b/reward" in m for kind, m in trainer.logged)
    assert (tmp_path / "run" / "step_2" / TRAINER_STATE_FILE).exists()

    records = list(read_records(tmp_path / "run" / "rollouts"))
    assert {r["role"] for r in records} == {"a", "b"} and all(r["model"] == "policy" for r in records)
    episodes = [r["info"]["episode_record"] for r in records if "episode_record" in r["info"]]
    assert episodes and all(len(e["sessions"]) == 3 and len(e["channels"]) == 1 for e in episodes)
    report = analyze(records)
    assert set(report["roles"]) == {"a", "b"} and report["roles"]["a"]["trajectories"] > 0
    a = next(r for r in records if r["role"] == "a")
    assert [m["role"] for m in session_messages(a)] == ["user", "assistant"]
    rows = list(sft_rows(records, min_reward=None))
    assert len(rows) == len([r for r in records if r["scored"]]) and {"messages", "role", "reward", "source"} <= set(
        rows[0]
    )
    assert all(row["reward"] >= 0.1 for row in sft_rows(records, roles=["b"], min_reward=0.1))
    pairs = list(dpo_rows(records, margin=0.01))
    assert pairs and {p["role"] for p in pairs} <= {"a", "b"}  # sessions that answered one and the same prompt
    assert all(p["reward_chosen"] > p["reward_rejected"] and p["chosen"] != p["rejected"] for p in pairs)
    assert all(p["prompt"][-1]["role"] == "user" and p["chosen"][0]["role"] == "assistant" for p in pairs)

    resumed = _trainer(tmp_path, **options, **{"train.steps": 3, "train.resume_from": "auto"})
    assert resumed.start_step == 2 and resumed.weights.version == 2
    resumed.train()
    assert len([m for kind, m in resumed.logged if kind == "step"]) == 1


def test_two_roles_train_their_own_adapters_on_one_base(tmp_path):
    from palingenesis.lora import use_adapter
    from palingenesis.rl.adapters import ADAPTER_DIR, NAMED_DIR
    from palingenesis.rl.scenarios.pipeline import AdapterEngine

    options = {
        "lora.rank": 4,
        "lora.alpha": 8,
        "roles.a.adapter": "alpha",
        "roles.b.adapter": "beta",
        "train.learning_rate": 1e-2,
        "train.save_steps": 2,
        "loss.kl_ref": "start",
    }
    trainer = _trainer(tmp_path, **options)
    modules = trainer.adapters.plugin.modules
    named = dict(trainer.model.named_parameters())
    trainable = {n for n, p in named.items() if p.requires_grad}
    assert trainable and all(
        ".roles.alpha." in n or ".roles.beta." in n for n in trainable
    )  # the default one is unused
    assert [len(g["params"]) for g in trainer.opt.param_groups] == [2 * len(modules)] * 2  # one group per adapter
    assert isinstance(trainer.engine, AdapterEngine) and set(trainer.pipeline.clients) == {"", "alpha", "beta"}
    # each role's sessions are generated and scored with its own adapter: after moving the two adapters apart,
    # the trainer's log-probs still match the sampler's
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for m in modules.values():
            m.roles["alpha"].lora_B.copy_(0.5 * torch.randn(m.lora_B.shape, generator=g))
            m.roles["beta"].lora_B.copy_(-0.5 * torch.randn(m.lora_B.shape, generator=g))
    ids = torch.randint(0, 1000, (1, 8), generator=torch.Generator().manual_seed(0))
    trainer.model.eval()
    with torch.no_grad():
        plain = trainer.model(input_ids=ids).logits
        with use_adapter("alpha"):
            alpha = trainer.model(input_ids=ids).logits
        with use_adapter("beta"):
            beta = trainer.model(input_ids=ids).logits
    assert not torch.allclose(alpha, beta, atol=1e-3) and not torch.allclose(alpha, plain, atol=1e-3)
    before = {n: p.detach().clone() for n, p in named.items()}
    trainer.train()
    steps = [m for kind, m in trainer.logged if kind == "step"]
    assert steps[0]["policy/mismatch_k3"] < 1e-6 and steps[0]["policy/abs_ratio_dev"] < 1e-3  # the right adapter each
    assert steps[0]["policy/kl_ref"] > 1e-4  # ... and far from the base, which the reference still is
    assert steps[0]["grad_norm/alpha"] > 0 and steps[0]["grad_norm/beta"] > 0  # clipped, and logged, one by one
    after = dict(trainer.model.named_parameters())
    moved = {n for n in after if not torch.equal(after[n], before[n])}
    assert moved <= trainable and any(".roles.alpha." in n for n in moved) and any(".roles.beta." in n for n in moved)
    assert all(m.lora_B.abs().sum() == 0 for m in modules.values())

    # each adapter's fold is its own policy; the checkpoint holds every adapter, and a run resumes from them
    name, m = next(iter(modules.items()))
    folds = {
        key: dict(trainer.adapters.engine_parameters(trainer.model, key))[f"{name}.weight"] for key in ("alpha", "beta")
    }
    for key, fold in folds.items():
        factors = m.roles[key]
        torch.testing.assert_close(fold, m.weight + m.scaling * (factors.lora_B @ factors.lora_A))
    assert not torch.allclose(folds["alpha"], folds["beta"])
    step = tmp_path / "run" / "step_2"
    assert (step / ADAPTER_DIR / "adapter_model.safetensors").exists()
    assert all((step / NAMED_DIR / key / "adapter_model.safetensors").exists() for key in ("alpha", "beta"))
    resumed = _trainer(tmp_path, **options, **{"train.steps": 3, "train.resume_from": "auto"})
    for n, p in resumed.model.named_parameters():
        torch.testing.assert_close(p, after[n], msg=n)
    assert len(resumed.opt.state) == len(trainer.opt.state) > 0
    resumed.train()


def test_a_frozen_base_partner_and_a_shared_adapter(tmp_path):
    """`partner: base` plays a role with the adapters off (no second model); the other role trains the default
    adapter against it."""
    trainer = _trainer(tmp_path, **{"lora.rank": 4, "roles.b.partner": "base", "train.learning_rate": 1e-2})
    assert set(trainer.pipeline.clients) == {"", "base"} and trainer.adapters.names == []
    trainer.train()
    steps = [m for kind, m in trainer.logged if kind == "step"]
    assert all("role/a/advantage_std" in m and "role/b/advantage_std" not in m for m in steps)  # b does not train
    assert all(m["role/b/sessions"] > 0 for m in steps) and steps[0]["policy/mismatch_k3"] < 1e-6
    assert any(m.lora_B.abs().sum() > 0 for m in trainer.adapters.plugin.modules.values())


def test_weight_views_share_the_trainers_version_and_each_engine_gets_its_fold(tmp_path):
    from palingenesis.rl.scenarios.pipeline import WeightsView

    trainer = _trainer(tmp_path, **{"lora.rank": 4, "roles.a.adapter": "alpha", "roles.b.adapter": "beta"})
    with torch.no_grad():
        for m in trainer.adapters.plugin.modules.values():
            m.roles["alpha"].lora_B.fill_(0.01)
            m.roles["beta"].lora_B.fill_(-0.02)

    class Engine:
        version = 0

        def update_weights(self, named, version):
            self.loaded, self.version = dict(named), version

    views = {
        key: WeightsView(trainer.weights, lambda key=key: trainer.adapters.engine_parameters(trainer.model, key))
        for key in ("alpha", "beta")
    }
    frozen, engines = WeightsView(trainer.weights, frozen=True), {key: Engine() for key in ("alpha", "beta", "base")}
    assert all(view.lock is trainer.weights.lock for view in views.values())
    for key, view in views.items():
        assert view.sync(engines[key]) == 0.0 and not hasattr(engines[key], "loaded")  # nothing published yet
    trainer.weights.publish()
    trainer.weights.publish()
    for key, view in views.items():
        view.sync(engines[key])
        assert engines[key].version == view.version == 2
    frozen.sync(engines["base"])
    assert engines["base"].version == 2 and not hasattr(engines["base"], "loaded")  # labelled, never loaded
    name, m = next(iter(trainer.adapters.plugin.modules.items()))
    for key in ("alpha", "beta"):
        factors = m.roles[key]
        expected = m.weight + m.scaling * (factors.lora_B @ factors.lora_A)
        torch.testing.assert_close(engines[key].loaded[f"{name}.weight"], expected)
    assert not torch.equal(engines["alpha"].loaded[f"{name}.weight"], engines["beta"].loaded[f"{name}.weight"])
    assert not any(".roles." in n or ".lora_" in n for n in engines["alpha"].loaded)  # a plain model's tensors
    trainer.pipeline.close()


def test_a_role_can_be_a_scripted_partner_in_training(tmp_path):
    trainer = _trainer(tmp_path, scenario="checked_answer", **{"roles.verifier.partner": "scripted:oracle"})
    assert trainer.role_specs["verifier"].partner is not None and not trainer.role_specs["verifier"].train
    trainer.train()  # a random policy never answers: constant rewards, nothing to train, and the run does not break
    steps = [m for kind, m in trainer.logged if kind == "step"]
    assert all(m.get("groups/zero_variance") == 1.0 for m in steps)
    assert all(m.get("episode/sessions") == 5.0 and m.get("episode/failed") == 0.0 for m in steps)  # 1 + 2 x 2


# ------------------------------------------------------------ planner, config


def test_the_planner_sizes_a_layout_and_says_what_does_not_fit():
    from palingenesis.rl.scenarios.placement import Cluster, Workload, plan, workload_of

    one = plan(Cluster(1, 1, 80), Workload(params_b=0.8), "rl.yaml")
    assert (
        one.ranks == 1 and one.mode == "single" and not one.errors and one.command.startswith("pgs rl --config rl.yaml")
    )
    assert one.trainer_gb == pytest.approx(18 * 0.8 + 8) and one.engine_weights_gb == pytest.approx(1.6)
    adapters = plan(Cluster(1, 1, 80), Workload(params_b=0.8, lora_rank=16, engines=3))
    assert adapters.trainer_gb < one.trainer_gb and adapters.overrides["rollout.backend"] == "vllm_server"
    assert not adapters.errors and 3 * adapters.gpu_memory_utilization * 80 + adapters.trainer_gb <= 80 + 1e-6
    big = plan(Cluster(1, 1, 24), Workload(params_b=4.0))
    assert big.errors and "lora.rank" in big.errors[0]
    cluster = plan(Cluster(2, 8, 80), Workload(params_b=4.0, batch_prompts=60), "rl.yaml")
    assert cluster.ranks == 16 and cluster.mode == "fsdp" and "--nnodes 2 --nproc_per_node 8" in cluster.command
    assert cluster.overrides["rollout.batch_prompts"] == 64 and not cluster.errors
    assert cluster.trainer_gb == pytest.approx(16 * 4 / 16 + 2 * 4 + 8)
    assert plan(Cluster(1, 2, 80), Workload(params_b=0.8, lora_rank=16, engines=2)).errors
    config = RLConfig()
    for key, value in {
        "scenario.type": "take_away",
        "lora.rank": 8,
        "roles.first.adapter": "p1",
        "roles.second.partner": "base",
    }.items():
        config.set(key, value)
    assert workload_of(config, 0.8).engines == 3


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"scenario.type": "no_such"}, "unknown scenario"),
        ({"roles.judge.train": False}, "are not roles of scenario"),
        ({"scenario.outcome": "judge"}, "scenario.outcome 'judge' is not a role"),
        ({"scenario.timeout": 0}, "scenario.timeout must be > 0"),
        ({"rewards.m.fn": "math"}, "the scenario's verifier sets the rewards"),
        ({"roles.first.partner": "scripted:oracle", "roles.second.partner": "scripted:oracle"}, "no role trains"),
        ({"roles.first.train": False, "roles.second.train": False}, "no role trains"),
        ({"roles.first.partner": "scripted:nobody"}, "has no such scripted partner"),
        ({"roles.first.adapter": "p1"}, "need lora.rank"),
        ({"roles.first.adapter": "p-1", "lora.rank": 4}, "letters, digits and underscores"),
        (
            {"roles.first.adapter": "p1", "roles.first.partner": "base", "lora.rank": 4},
            "partner and adapter are alternatives",
        ),
        ({"roles.first.partner": "base"}, "it needs lora.rank"),
        ({"roles.first.adapter": "p1", "lora.rank": 4, "rollout.backend": "vllm"}, "need a rollout engine each"),
        ({"roles.first.adapter": "p1", "lora.rank": 4, "lora.dora": True}, "not supported with lora.dora"),
        ({"roles.first.adapter": "p1", "lora.rank": 4, "train.fsdp": True}, "not supported with FSDP"),
        ({"roles.first.share": -1.0}, "share must be >= 0"),
        ({"roles.first.partner_args": {"model": "x"}}, "partner_args without"),
        ({"env.type": "tools", "env.tools": ["tests.test_rl:add"]}, "leave env at its default"),
        ({"loss.overlong_buffer": 2}, "is not supported"),
    ],
)
def test_configuration_errors(tmp_path, overrides, match):
    settings = {"model.policy": "p", "data.dataset": "d.jsonl", "rollout.backend": "hf", "scenario.type": "take_away"}
    config = RLConfig()
    for key, value in {**settings, **overrides}.items():
        config.set(key, value)
    with pytest.raises(RLConfigError, match=match):
        config.validate()


def test_single_agent_configs_are_untouched_and_yaml_roles_load(tmp_path):
    plain = RLConfig()
    for key, value in {"model.policy": "p", "data.dataset": "d.jsonl", "rewards.m.fn": "math"}.items():
        plain.set(key, value)
    plain.validate()
    assert plain.scenario.type == "" and plain.roles == {} and plain.loss.advantage_std == "batch"
    plain.set("loss.advantage_std", "role")
    with pytest.raises(RLConfigError, match="need a scenario"):
        plain.validate()
    (tmp_path / "rl.yaml").write_text(
        "model: {policy: p}\ndata: {dataset: d.jsonl}\nrollout: {backend: hf}\n"
        "scenario: {type: ledger_sum, args: {weights: {lead: {coverage: 0.2}}}}\n"
        "roles:\n  lead: {share: 2.0, max_turns: 6}\n  worker: {partner: 'scripted:oracle'}\n"
        "loss: {advantage_std: role, advantage_std_floor: 0.05}\n"
    )
    loaded = RLConfig.from_yaml(tmp_path / "rl.yaml")
    loaded.validate()
    assert loaded.roles["lead"].share == 2.0 and loaded.roles["worker"].partner == "scripted:oracle"
    assert build_scenario(loaded.scenario.type, loaded.scenario.args).weights["lead"] == {"total": 1.0, "coverage": 0.2}
    assert loaded.to_dict()["roles"]["lead"]["max_turns"] == 6


def test_cli_generates_audits_and_plans(tmp_path, capsys):
    from palingenesis.rl.scenarios.cli import main

    out = tmp_path / "rows.jsonl"
    assert main(["generate", "ledger_sum", "--n", "7", "--seed", "3", "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows == build_scenario("ledger_sum").generate(7, 3)
    assert main(["audit", "take_away", "--n", "40", "--json"]) == 0
    assert json.loads(capsys.readouterr().out.split("\n", 1)[1])["failures"] == []
    (tmp_path / "rl.yaml").write_text("model: {policy: p}\ndata: {dataset: d}\nscenario: {type: take_away}\n")
    assert main(["plan", "--config", str(tmp_path / "rl.yaml"), "--params-b", "0.8"]) == 0
    assert json.loads(capsys.readouterr().out)["ranks"] == 1
    assert main(["list"]) == 0 and "checked_answer" in capsys.readouterr().out


# ------------------------------------------------- several ranks, one frozen base


def test_summed_gradients_keep_absent_parameters_absent(monkeypatch):
    import torch.distributed as dist

    from palingenesis.rl.parallel import Parallel

    parallel = Parallel(fsdp=False)
    seen = []

    def all_reduce(tensor, op=None):  # two ranks with the same values: a sum doubles
        seen.append(tensor.numel())
        tensor.mul_(2)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    a, b, c = (torch.nn.Parameter(torch.ones(2, 3)) for _ in range(3))
    a.grad, c.grad = torch.full((2, 3), 0.5), None
    b.grad = torch.arange(6.0).reshape(2, 3)
    parallel.sum_gradients([a, b, c])
    assert seen == [6 + 6 + 6 + 3]  # one collective for every gradient and the presence flags
    torch.testing.assert_close(a.grad, torch.ones(2, 3))
    torch.testing.assert_close(b.grad, 2 * torch.arange(6.0).reshape(2, 3))
    assert c.grad is None  # no rank had a gradient for it: no replica may move it


def test_two_ranks_with_a_replicated_base_stay_one_policy(tmp_path):
    """train.replicate_base on two real ranks (CPU, gloo): each runs its own episodes and backward with no
    collective, the adapters' gradients are summed once per step, and both ranks end with the same adapters and
    the same optimizer state."""
    import socket
    import subprocess

    from palingenesis.rl.adapters import ADAPTER_DIR

    policy = tiny_policy(tmp_path)
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in Chat.generate(24)))
    (tmp_path / "rl.yaml").write_text(
        f"model: {{policy: {policy}, use_liger_kernel: false, chat_template_kwargs: {{enable_thinking: false}}}}\n"
        f"data: {{dataset: {tmp_path}/rows.jsonl}}\n"
        "rollout: {backend: hf, batch_prompts: 4, group_size: 3, max_new_tokens: 8}\n"
        f'scenario: {{type: "{CHAT}"}}\n'
        "lora: {rank: 4, alpha: 8}\n"
        "loss: {advantage_std: role, kl_ref: start}\n"
        f"train: {{output_dir: {tmp_path}/run, steps: 3, learning_rate: 1.0e-2, warmup_steps: 1, save_steps: 3, "
        "replicate_base: true}\n"
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    root = Path(__file__).parent.parent
    environment = {
        **__import__("os").environ,
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "WORLD_SIZE": "2",
        "PYTHONPATH": f"{root / 'src'}:{root / 'tests'}",
    }
    ranks = [
        subprocess.Popen(
            [sys.executable, "-m", "palingenesis.rl.trainer", "--config", str(tmp_path / "rl.yaml")],
            env={**environment, "RANK": str(rank), "LOCAL_RANK": str(rank)},
            stdout=open(tmp_path / f"rank{rank}.log", "w"),
            stderr=subprocess.STDOUT,
        )
        for rank in range(2)
    ]
    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline and all(process.poll() in (None, 0) for process in ranks):
            if all(process.poll() == 0 for process in ranks):
                break
            time.sleep(0.2)  # a rank that fails leaves the other waiting in a collective: stop both
    finally:
        for process in ranks:
            process.kill()
            process.wait()
        outputs = [(tmp_path / f"rank{rank}.log").read_text() for rank in range(2)]
    assert [process.returncode for process in ranks] == [0, 0], outputs[0][-2000:] + outputs[1][-2000:]
    assert "step 3 |" in outputs[0] and "role/a/reward" in outputs[0]
    step = tmp_path / "run" / "step_3"
    assert (step / ADAPTER_DIR / "adapter_model.safetensors").exists() and (tmp_path / "run" / "final").exists()
    states = [torch.load(step / f"rank{rank}.pt", weights_only=False)["optimizer"]["state"] for rank in range(2)]
    assert len(states[0]) == len(states[1]) > 0
    moved = 0
    for key in states[0]:  # the same gradients reached both replicas at every step
        torch.testing.assert_close(states[0][key]["exp_avg"], states[1][key]["exp_avg"], rtol=0, atol=0)
        moved += bool(states[0][key]["exp_avg"].abs().sum() > 0)
    assert moved > 0
