"""Asynchronous RL at high staleness: the decoupled objective (loss.type decoupled) against values computed by hand
and against the gradient of its surrogate, the per-token stale mask (rollout.stale mask), per-token policy versions
of generations that weights were loaded into (rollout.inflight tag / resume), and weights that reach the engine when
the trainer publishes them (rollout.sync publish)."""

import asyncio
import math
import sys
import threading
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from palingenesis.opd.rollout import Rollout, VersionSpans  # noqa: E402
from palingenesis.rl.config import RLConfig, RLConfigError, RLLossConfig  # noqa: E402
from palingenesis.rl.generation import GenerationClient  # noqa: E402
from palingenesis.rl.losses import decoupled_metrics, policy_loss  # noqa: E402
from palingenesis.rl.trajectory import Trajectory  # noqa: E402

# ------------------------------------------------------------ the objective


def _grad(config, lp0, behaviour, advantage, proximal=None, weight=None, fresh=None, seq=None):
    lp = torch.tensor(lp0, dtype=torch.float64, requires_grad=True)
    n = lp.numel()
    seq = torch.zeros(n, dtype=torch.long) if seq is None else seq
    n_seq = int(seq.max()) + 1
    loss, stats = policy_loss(
        lp,
        torch.tensor(behaviour, dtype=torch.float64),
        torch.tensor(advantage, dtype=torch.float64),
        torch.ones(n, dtype=torch.float64) if weight is None else weight,
        seq,
        n_seq,
        torch.bincount(seq, minlength=n_seq).double(),
        config,
        proximal=None if proximal is None else torch.tensor(proximal, dtype=torch.float64),
        fresh=fresh,
    )
    loss.backward()
    return lp.grad, stats


def test_decoupled_with_a_proximal_policy_matches_values_computed_by_hand():
    """Four tokens, π_prox given: ω = π_prox / μ weighs, u = π / π_prox is what the clip sees.
    eps_low 0.2, eps_high 0.25, is_cap 2.

      token  π      π_prox  μ      A    u      ω     clipped                coefficient
      0      0.50   0.40    0.20   +1   1.25   2.0   no (u = 1 + eps_high)  2.0 · 1.25 · 1  = 2.5
      1      0.60   0.40    0.80   +1   1.5    0.5   yes (A>0, u > 1.25)    0
      2      0.30   0.40    0.10   -2   0.75   4.0   yes (A<0, u < 0.8)     0
      3      0.36   0.40    0.10   -2   0.9    4.0   no                     min(4, 2) · 0.9 · -2 = -3.6
    The loss is -Σ coefficient · log π: its gradient with respect to log π is -coefficient."""
    config = RLLossConfig(type="decoupled", proximal="reference", eps_low=0.2, eps_high=0.25, is_cap=2.0, seq_mask=0.0)
    pi, prox, mu = [0.50, 0.60, 0.30, 0.36], [0.40] * 4, [0.20, 0.80, 0.10, 0.10]
    log = lambda xs: [math.log(x) for x in xs]  # noqa: E731
    # (token 0: u is 1.25 up to rounding; nudge the bound so the equality is not decided by the last bit)
    config.eps_high = 0.2500001
    grad, stats = _grad(config, log(pi), log(mu), [1.0, 1.0, -2.0, -2.0], log(prox))
    assert torch.allclose(grad, torch.tensor([-2.5, 0.0, 0.0, 3.6], dtype=torch.float64), atol=1e-12)
    assert stats["clipped"].item() == 2 and stats["behaviour_capped"].item() == 2
    assert stats["behaviour_weight"].item() == pytest.approx(2.0 + 0.5 + 2.0 + 2.0)
    assert stats["proximal_dev"].item() == pytest.approx(0.25 + 0.5 + 0.25 + 0.1)
    # behaviour_weight mask: a weight above the cap leaves the loss instead of being held at it
    config.behaviour_weight = "mask"
    grad, stats = _grad(config, log(pi), log(mu), [1.0, 1.0, -2.0, -2.0], log(prox))
    assert torch.allclose(grad, torch.tensor([-2.5, 0.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12)
    means = decoupled_metrics({k: float(v) for k, v in stats.items()})
    assert means["policy/behaviour_capped"] == pytest.approx(0.5)


def test_decoupled_is_the_gradient_of_its_surrogate():
    """-ω · min(u A, clip(u, 1 - eps_low, 1 + eps_high) A) with u = exp(lp - lp_prox), differentiated by autograd."""
    torch.manual_seed(0)
    config = RLLossConfig(type="decoupled", proximal="reference", eps_low=0.2, eps_high=0.28, is_cap=3.0, seq_mask=0.0)
    lp = (torch.rand(64, dtype=torch.float64) * 2 - 2.5).tolist()
    prox = [x + d for x, d in zip(lp, (torch.randn(64, dtype=torch.float64) * 0.3).tolist())]
    mu = [x + d for x, d in zip(prox, (torch.randn(64, dtype=torch.float64) * 0.8).tolist())]
    adv = torch.randn(64, dtype=torch.float64).tolist()
    grad, _ = _grad(config, lp, mu, adv, prox)
    x = torch.tensor(lp, dtype=torch.float64, requires_grad=True)
    a, p, m = (torch.tensor(v, dtype=torch.float64) for v in (adv, prox, mu))
    u = (x - p).exp()
    omega = (p - m).exp().clamp(max=config.is_cap)
    surrogate = -(omega * torch.minimum(u * a, u.clamp(1 - config.eps_low, 1 + config.eps_high) * a)).sum()
    surrogate.backward()
    assert torch.allclose(grad, x.grad, atol=1e-12)


def test_decoupled_on_the_step_s_own_policy_never_clips_and_masked_is_does():
    """loss.proximal step: π_prox is the policy the gradient is taken at, so u = 1 and only the behaviour weight is
    left: min(ρ, is_cap) · A (cispo's gradient, exactly). On the same stale tokens masked_is, whose clip is centred
    on the sampler, drops every token the policy has moved by more than the clip in the advantage's direction."""
    lp = [math.log(0.6), math.log(0.2), math.log(0.5), math.log(0.05)]
    mu = [math.log(0.3), math.log(0.4), math.log(0.5), math.log(0.005)]  # ratios 2, 0.5, 1, 10
    adv = [1.0, -1.0, 1.0, 1.0]
    decoupled, stats = _grad(RLLossConfig(type="decoupled", seq_mask=0.0), lp, mu, adv)
    assert torch.allclose(decoupled, -torch.tensor([2.0, -0.5, 1.0, 5.0], dtype=torch.float64), atol=1e-12)
    assert stats["clipped"].item() == 0 and stats["proximal_dev"].item() == 0
    cispo, _ = _grad(RLLossConfig(type="cispo", seq_mask=0.0), lp, mu, adv)
    assert torch.equal(decoupled, cispo)
    masked, stats = _grad(RLLossConfig(type="masked_is", seq_mask=0.0), lp, mu, adv)
    assert torch.allclose(masked, -torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-12)
    assert stats["clipped"].item() == 3


def test_tokens_older_than_the_staleness_bound_leave_the_loss_and_its_ratios():
    """rollout.stale mask: a token flagged as too old has no gradient, counts as on-policy in the sequence mask and
    the mismatch statistics, and is counted."""
    lp = [math.log(0.5), math.log(0.5), math.log(0.5)]
    mu = [math.log(0.5), math.log(0.01), math.log(0.25)]  # the middle token is far off-policy (ratio 50)
    fresh = torch.tensor([True, False, True])
    config = RLLossConfig(type="decoupled", seq_mask=0.5)
    grad, stats = _grad(config, lp, mu, [1.0, 1.0, 1.0], fresh=fresh)
    assert torch.allclose(grad, -torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64), atol=1e-12)
    assert stats["stale_masked"].item() == 1 and stats["seq_masked"].item() == 0
    assert stats["ratio_max"].item() == pytest.approx(2.0)
    # without the mask the sequence's mean |ratio - 1| is (0 + 49 + 1) / 3: the sequence mask drops it whole
    grad, stats = _grad(config, lp, mu, [1.0, 1.0, 1.0])
    assert stats["seq_masked"].item() == 1 and not grad.any()


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"loss.type": "decoupled", "loss.proximal": "reference"}, "loss.kl_ref"),
        ({"loss.type": "decoupled", "loss.proximal": "nearest"}, "loss.proximal must be"),
        ({"loss.behaviour_weight": "mask"}, "belong to loss.type decoupled"),
        ({"rollout.stale": "keep"}, "rollout.stale must be"),
        ({"rollout.inflight": "resume", "rollout.backend": "vllm"}, "max_staleness > 0"),
        ({"rollout.inflight": "tag", "rollout.max_staleness": 1}, "needs rollout.backend vllm or remote"),
        ({"rollout.sync": "publish", "rollout.backend": "vllm"}, "rollout.sync publish needs"),
        ({"rollout.push_link": "nccl", "rollout.backend": "vllm"}, "only used with rollout.backend remote"),
        ({"rollout.backend": "remote", "rollout.engines": ["a:1"], "rollout.min_engines": 2}, "rollout.min_engines"),
        (
            {"rollout.backend": "remote", "rollout.engines": ["a:1"], "rollout.push_link": "nccl", "lora.rank": 4},
            "sends full weights",
        ),
    ],
)
def test_configuration_errors(overrides, match):
    config = RLConfig()
    for key, value in {"model.policy": "p", "data.dataset": "d", "rollout.backend": "sglang", **overrides}.items():
        config.set(key, value)
    with pytest.raises(RLConfigError, match=match):
        config.validate(python_rewards=True)


def test_the_new_options_validate_together():
    config = RLConfig()
    for key, value in {
        "model.policy": "p",
        "data.dataset": "d",
        "loss.type": "decoupled",
        "loss.proximal": "reference",
        "loss.kl_ref": "start",
        "loss.kl_ref_reset": 4,
        "rollout.backend": "remote",
        "rollout.engines": ["a:1", "b:2"],
        "rollout.max_staleness": 8,
        "rollout.stale": "mask",
        "rollout.inflight": "resume",
        "rollout.sync": "publish",
        "rollout.push_link": "nccl",
        "rollout.push_dtype": "bfloat16",
        "rollout.min_engines": 1,
    }.items():
        config.set(key, value)
    config.validate(python_rewards=True)


# ---------------------------------------------- generations in flight at an update


class StepEngine:
    """A streaming engine a test steps by hand: every step samples one token per request in flight, the token
    100 x version + its position in the sequence (prompt included) with log-prob -(version + 1), so a completion
    says which weights sampled each of its tokens and what they had read. A request ends at its budget."""

    def __init__(self, abort: bool = True):
        self.version = 0
        self.track_tokens = False
        self.spans = VersionSpans()
        self.live: dict[str, dict] = {}
        self.added: list[tuple[str, list[int], int]] = []
        self.steps = threading.Semaphore(0)
        self.stepped = threading.Semaphore(0)
        if not abort:
            self.stream_abort = lambda request_id: None

    def stream_add(self, request_id, prompt, max_new_tokens, temperature, n, **extra):
        self.added.append((request_id, list(prompt), max_new_tokens))
        self.live[request_id] = {"at": len(prompt), "budget": max_new_tokens, "ids": [], "lp": [], "v": self.version}
        self.spans.start(request_id, 1, self.version)

    def stream_step(self):
        if not self.steps.acquire(timeout=0.002):  # (never wait inside a step: the client holds the engine's lock)
            return []
        done = []
        for request_id, state in list(self.live.items()):
            state["ids"].append(100 * self.version + state["at"] + len(state["ids"]))
            state["lp"].append(-float(self.version + 1))
            if len(state["ids"]) == state["budget"]:
                done.append((request_id, [self._rollout(request_id, "length")]))
        self.stepped.release()
        return done

    def _rollout(self, request_id, finish):
        state = self.live.pop(request_id)
        (runs,) = self.spans.finish(request_id, [len(state["ids"])])
        runs = runs if self.track_tokens else None
        return Rollout(state["ids"], state["lp"], finish, self.version, state["v"], versions=runs)

    def stream_abort(self, request_id):
        return [self._rollout(request_id, "abort")]

    def update_weights(self, named, version):
        self.spans.switch({rid: [len(s["ids"])] for rid, s in self.live.items()}, version)
        self.version = version

    def step(self):
        self.steps.release()
        assert self.stepped.acquire(timeout=10)


def _run(mode, engine, updates, budget=6):
    """One request of `budget` tokens; `updates`: {tokens sampled: version loaded then}."""
    client = GenerationClient(engine, merge=False, inflight=mode)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        future = asyncio.run_coroutine_threadsafe(client.generate([7, 8, 9], budget, 1.0), loop)
        for sampled in range(budget):
            if sampled in updates:
                with client.paused():
                    engine.update_weights((), updates[sampled])
            engine.step()
        return future.result(10), client
    finally:
        engine.steps.release()  # (the engine thread may be inside a step)
        client.close()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)


def test_version_spans_follow_each_sample_and_give_up_on_a_request_they_lose():
    spans = VersionSpans()
    spans.start("a", 2, 3)
    spans.start("b", 1, 3)
    spans.switch({"a": [2, 0], "b": [5]}, 4)  # a's second sample has nothing yet: version 4 samples all of it
    spans.start("c", 1, 4)
    spans.switch({"a": [4, 1], "c": [0]}, 5)  # b is not reported: no longer followed
    assert spans.finish("a", [6, 1]) == [[(0, 3), (2, 4), (4, 5)], [(0, 4)]]  # (the second ended before version 5)
    assert spans.finish("b", [9]) == [None]
    assert spans.finish("c", [2]) == [[(0, 5)]]


def test_a_generation_that_continues_across_updates_is_tagged_per_token():
    """rollout.inflight tag: weights loaded after 2 and after 5 of 6 tokens. Every token keeps the log-prob of the
    weights that sampled it; the trajectory's token versions are exact, where "continue" only knows the oldest."""
    out, client = _run("tag", StepEngine(), {2: 1, 5: 2})
    assert out.ids == [3, 4, 105, 106, 107, 208] and out.logprobs == [-1.0, -1.0, -2.0, -2.0, -2.0, -3.0]
    assert out.versions == [(0, 0), (2, 1), (5, 2)] and (out.first_version, out.version) == (0, 2)
    assert client.resumed == 0
    t = Trajectory({}, 0, [7, 8, 9], [])
    t.append_context([1])
    t.append_generated(out.ids, out.logprobs, out.version, out.first_version, versions=out.versions)
    t.append_context([1, 1])
    assert t.token_versions() == [-1, 0, 0, 1, 1, 1, 2, -1, -1]
    out, _ = _run("continue", StepEngine(), {2: 1, 5: 2})
    assert out.versions is None and out.ids == [3, 4, 105, 106, 107, 208]
    t = Trajectory({}, 0, [7, 8, 9], [])
    t.append_generated(out.ids, out.logprobs, out.version, out.first_version, versions=out.versions)
    assert t.token_versions() == [0] * 6


def test_a_generation_is_aborted_at_an_update_and_resumed_from_its_tokens():
    """rollout.inflight resume: at each update the request is aborted and a new one continues from the prompt plus
    what was sampled, with the budget that is left; the answer is one completion with each token's own log-prob and
    version, twice over."""
    engine = StepEngine()
    out, client = _run("resume", engine, {2: 1, 5: 2})
    assert out.ids == [3, 4, 105, 106, 107, 208] and out.logprobs == [-1.0, -1.0, -2.0, -2.0, -2.0, -3.0]
    assert out.versions == [(0, 0), (2, 1), (5, 2)]
    assert (out.first_version, out.version, out.finish) == (0, 2, "length")
    assert [(prompt, budget) for _, prompt, budget in engine.added] == [
        ([7, 8, 9], 6),
        ([7, 8, 9, 3, 4], 4),
        ([7, 8, 9, 3, 4, 105, 106, 107], 1),
    ]
    assert client.resumed == 2 and not client.inflight and not client.prefixes and not client.requests


def test_resume_on_an_engine_that_does_not_follow_a_request_continues():
    out, client = _run("resume", StepEngine(abort=False), {3: 1})
    assert out.ids == [3, 4, 5, 106, 107, 108] and client.resumed == 0


def test_stale_tokens_are_flagged_per_token_and_a_long_episode_keeps_its_group():
    """rollout.stale mask on the trainer's side: with the policy at version 5 and max_staleness 2, tokens of
    versions 1 and 2 are too old (2), version 3 and 4 stale (1), version 5 current (0)."""
    from palingenesis.rl.trainer import RLTrainer, _fresh_tokens, _sampled_by

    t = Trajectory({}, 0, [7], [])
    t.append_generated([1, 2, 3, 4, 5, 6], [-1.0] * 6, 5, 1, versions=[(0, 1), (1, 2), (2, 3), (4, 5)])
    kinds = _sampled_by(t, list(range(6)), 5, 2)
    assert kinds == [2, 2, 1, 1, 0, 0]
    assert _fresh_tokens(torch.tensor(kinds)).tolist() == [False, False, True, True, True, True]
    assert _sampled_by(t, list(range(6)), 5) == [1, 1, 1, 1, 0, 0]

    old = Trajectory({}, 1, [7], [])
    old.append_generated([1], [-1.0], 1, 1)
    other = Trajectory({}, 2, [7], [])  # another run's: a version the trainer has not published
    other.append_generated([1], [-1.0], 9, 9)

    class Weights:
        version = 5

    def fresh(mode, limit=2):
        trainer = RLTrainer.__new__(RLTrainer)
        trainer.config = RLConfig()
        trainer.config.rollout.max_staleness = limit
        trainer.mask_stale, trainer.weights, trainer.stale_groups = mode == "mask", Weights(), 0
        kept = trainer._fresh([[t], [old], [other], [t, old]])
        return [len(g) for g in kept], trainer.stale_groups

    # mask: the long episode trains (its old tokens masked), alone or next to a rollout that is stale whole
    assert fresh("mask") == ([1, 2], 2)
    assert fresh("drop") == ([1], 3)  # drop judges a rollout by the version that finished its turns


def test_the_vllm_engine_follows_token_deltas_across_an_update_and_an_abort():
    """VLLMColocateRollout with track_tokens, on a stand-in for vLLM's engine that returns token deltas as vLLM's
    DELTA output kind does (per step: the samples that got tokens, each with its index): per-sample runs of
    versions, the rollouts of a finished request, and what an aborted one had so far."""
    from types import SimpleNamespace as NS

    from palingenesis.opd.rollout import VLLMColocateRollout

    def delta(request_id, finished, *samples):
        outputs = [
            NS(index=i, token_ids=ids, logprobs=[{t: NS(logprob=-0.5 * t)} for t in ids], finish_reason=reason)
            for i, ids, reason in samples
        ]
        return NS(
            request_id=request_id, outputs=outputs, finished=finished, prompt_token_ids=[1, 2], num_cached_tokens=1
        )

    script = [
        [delta("a", False, (0, [5], None), (1, [6], None)), delta("b", False, (0, [7], None))],
        [delta("a", False, (0, [8], "stop")), delta("b", False, (0, [9], None))],  # a's first sample ends
        [delta("a", True, (1, [3, 4], "length"))],
    ]
    aborted = []
    core = NS(step=lambda: script.pop(0), abort_request=aborted.extend)
    engine = VLLMColocateRollout.__new__(VLLMColocateRollout)
    engine.llm = NS(llm_engine=core, apply_model=lambda fn: None, reset_prefix_cache=lambda: None)
    engine.version, engine.track_tokens, engine.spans = 3, True, VersionSpans()
    engine.partial, engine.samples, engine.started, engine.candidates = {}, {}, {}, {}
    engine.prompt_tokens = engine.cached_tokens = 0
    for request_id, n in (("a", 2), ("b", 1)):  # what stream_add records
        engine.spans.start(request_id, n, engine.version)
        engine.partial[request_id], engine.samples[request_id], engine.started[request_id] = {}, n, engine.version

    assert engine.stream_step() == [] and engine.stream_progress() == {"a": [1, 1], "b": [1]}
    engine.update_weights((), 4)
    assert engine.stream_step() == []
    engine.update_weights((), 5)  # a's first sample had ended: version 5 sampled none of it
    ((request_id, rollouts),) = engine.stream_step()
    assert request_id == "a" and [r.completion_ids for r in rollouts] == [[5, 8], [6, 3, 4]]
    assert [r.versions for r in rollouts] == [[(0, 3), (1, 4)], [(0, 3), (1, 5)]]
    assert [r.finish_reason for r in rollouts] == ["stop", "length"] and rollouts[1].logprobs == [-3.0, -1.5, -2.0]
    assert (rollouts[0].first_version, rollouts[0].policy_version) == (3, 5) and engine.prompt_tokens == 2
    (partial,) = engine.stream_abort("b")
    assert aborted == ["b"] and partial.finish_reason == "abort" and partial.completion_ids == [7, 9]
    assert partial.versions == [(0, 3), (1, 4)] and partial.logprobs == [-3.5, -4.5]
    assert engine.stream_abort("b") is None and not engine.partial and not engine.samples and not engine.spans.runs
