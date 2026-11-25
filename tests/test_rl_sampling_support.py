"""Truncated sampling in RL (rollout.top_p / top_k) and score centering (loss.score_centering).

The sampler draws from the policy renormalised over its support; the trainer renormalises over the same support
(losses.sampled_logprobs) from the candidates the engine returns (opd.rollout.Candidates), so that the importance
ratio of an on-policy token is 1. Score centering subtracts the expectation of the token gradient under the sampler.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from palingenesis.opd.rollout import (
    Candidates,
    Sampling,
    _RecordLogprobs,
    candidates_from_openai,
    candidates_from_steps,
    truncate_scores,
)
from palingenesis.rl.config import RLConfig, RLConfigError, sampler_candidates, training_sampling
from palingenesis.rl.losses import (
    SampledLogProbs,
    SamplerCandidates,
    centering_correction,
    policy_loss,
    sampled_logprobs,
    target_logprobs,
)
from palingenesis.rl.trajectory import Trajectory

# ------------------------------------------------------------------ candidates


def test_candidates_drop_non_finite_and_sort():
    c = Candidates.from_rows([([3, 1, 2], [-1.0, -0.5, float("-inf")]), ([7, 8], [-9999.0, -0.1]), ([], [])])
    assert c.counts.tolist() == [2, 1, 0]
    assert c.ids.tolist() == [1, 3, 8]
    np.testing.assert_allclose(c.logprobs, [-0.5, -1.0, -0.1])
    picked = c.select([2, 0])
    assert picked.counts.tolist() == [0, 2] and picked.ids.tolist() == [1, 3]
    assert c.head(1).ids.tolist() == [1, 3]
    joined = Candidates.join([c, Candidates.empty(2)])
    assert joined.counts.tolist() == [2, 1, 0, 0, 0] and len(joined) == 5
    back = Candidates.from_tensors(c.tensors())
    assert back.counts.tolist() == c.counts.tolist() and back.ids.tolist() == c.ids.tolist()
    assert Candidates.from_tensors({}) is None


def test_candidates_from_engine_outputs():
    lp = SimpleNamespace
    steps = [{5: lp(logprob=-0.2), 9: lp(logprob=-2.0), 4: lp(logprob=-5.0)}, {1: lp(logprob=float("-inf"))}]
    c = candidates_from_steps(steps, 2)  # the sampled token beyond the top 2 (id 4) is not a candidate
    assert c.counts.tolist() == [2, 0] and c.ids.tolist() == [5, 9]
    top = [{"token_id:5": -0.2, "token_id:9": -2.0, "token_id:4": -9999.0}]
    c = candidates_from_openai(top, 3)  # vLLM writes -inf as -9999
    assert c.counts.tolist() == [2] and c.ids.tolist() == [5, 9]


def test_vllm_engines_ask_for_and_parse_the_candidates():
    """The colocated engine asks vLLM for Sampling.candidates log-probs and keeps the top ones per token; the server
    engine asks /v1/completions for them keyed by token id and parses vLLM's -9999 (a -inf log-prob)."""
    from palingenesis.opd.rollout import VLLMColocateRollout, VLLMServerRollout

    engine = object.__new__(VLLMColocateRollout)
    engine.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    engine.stop_ids, engine.version = (0,), 3
    assert engine._params(1, 5, Sampling(0.7, 0.9, 4, candidates=12)).logprobs == 12
    assert engine._params(1, 5, 1.0).logprobs == 0 and engine._params(1, 5, 0.0).logprobs is None
    lp = SimpleNamespace
    output = SimpleNamespace(
        token_ids=[5, 6],
        logprobs=[{5: lp(logprob=-0.1), 7: lp(logprob=-2.4)}, {6: lp(logprob=-3.0), 2: lp(logprob=-0.05)}],
        finish_reason="stop",
    )
    (rollout,) = engine._rollouts(SimpleNamespace(outputs=[output]), candidates=1)
    assert rollout.logprobs == [-0.1, -3.0] and rollout.candidates.ids.tolist() == [5, 2]

    sent = {}

    def complete(prompts, **params):
        sent.update(params)
        top = [{"token_id:5": -0.1, "token_id:7": -2.4, "token_id:9": -9999.0}, {"token_id:6": -0.2}]
        return [
            {
                "token_ids": [5, 6],
                "finish_reason": "stop",
                "logprobs": {"token_logprobs": [-0.1, -0.2], "top_logprobs": top},
            }
        ]

    server = object.__new__(VLLMServerRollout)
    server.server, server.stop_ids, server.version = SimpleNamespace(complete=complete), (0,), 1
    (rollout,) = server.generate([[1, 2]], [4], Sampling(0.7, 0.9, 4, candidates=3))
    assert sent["logprobs"] == 3 and sent["return_tokens_as_token_ids"] and sent["top_k"] == 4
    assert rollout.candidates.counts.tolist() == [2, 1] and rollout.candidates.ids.tolist() == [5, 7, 6]


def test_trajectory_keeps_candidates_aligned_with_tokens():
    t = Trajectory(row={}, group=0, prompt_ids=[1, 2], messages=[])
    t.append_forced([50, 51], math.log(0.1))  # before any candidates
    t.append_generated([10, 11], [-0.1, -0.2], 0, candidates=Candidates.from_rows([([10, 12], [-0.1, -2.0])] * 2))
    t.append_context([60])
    t.append_generated([13], [-0.3], 1, candidates=Candidates.from_rows([([13], [-0.3])]))
    assert len(t.candidates) == len(t.tokens) == 6
    assert t.candidates.counts.tolist() == [0, 0, 2, 2, 0, 1]
    sampled = [j for j, m in enumerate(t.mask) if m]
    assert t.candidates.select(sampled).counts.tolist() == [0, 2, 2, 1]  # the forced token has none
    with pytest.raises(ValueError, match="candidates for"):
        t.append_generated([1, 2], [0.0, 0.0], 1, candidates=Candidates.empty(1))
    # a context rewrite closes a segment with its own candidates; the views carry them
    t.rewrite_context([1, 2, 3], [], [])
    t.append_generated([20], [-0.4], 2, candidates=Candidates.from_rows([([20, 21], [-0.4, -1.1])]))
    first, second = t.training_views()
    assert len(first.candidates) == len(first.tokens) and first.candidates.counts.tolist()[-1] == 1
    assert second.candidates.counts.tolist() == [2]


# ------------------------------------------------------------------ renormalisation


def _head(weight: torch.Tensor) -> nn.Linear:
    head = nn.Linear(weight.shape[1], weight.shape[0], bias=False).to(weight.dtype)
    head.weight.data.copy_(weight)
    return head


def _sampler(rows, support=True, requested=64, dtype=torch.float32):
    return SamplerCandidates.build([Candidates.from_rows(rows)], requested, support, "cpu")


def test_renormalised_logprob_matches_hand_computed_top_k_and_top_p():
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0, -3.0])
    head = _head(torch.eye(5))
    hidden = logits[None]
    # top-k 3: S = {0, 1, 2}; y = 1
    out = sampled_logprobs(hidden, head, torch.tensor([1]), _sampler([([0, 1, 2], [0.0, 0.0, 0.0])]))
    expected = 1.0 - math.log(math.exp(2) + math.exp(1) + math.exp(0))
    assert out.lp.item() == pytest.approx(expected, abs=1e-6)
    # top-p 0.8: p = .641, .236, .087, ... -> S = {0, 1}, which truncate_scores agrees with
    kept = torch.isfinite(truncate_scores(logits[None], 0, 0.8))[0].nonzero().flatten().tolist()
    assert kept == [0, 1]
    out = sampled_logprobs(hidden, head, torch.tensor([1]), _sampler([(kept, [0.0, 0.0])]), entropy=True)
    p1 = math.exp(1) / (math.exp(2) + math.exp(1))
    assert out.lp.item() == pytest.approx(math.log(p1), abs=1e-6)
    assert out.entropy.item() == pytest.approx(-(p1 * math.log(p1) + (1 - p1) * math.log(1 - p1)), abs=1e-6)
    np.testing.assert_allclose(out.candidate_lp[0].tolist(), [math.log(1 - p1), math.log(p1)], atol=1e-6)
    # temperature 0.5: z = 2 x logits
    out = sampled_logprobs(hidden, head, torch.tensor([0]), _sampler([([0, 1], [0.0, 0.0])]), scale=2.0)
    assert out.lp.item() == pytest.approx(4.0 - math.log(math.exp(4) + math.exp(2)), abs=1e-6)


def test_full_vocabulary_support_equals_the_plain_path():
    torch.manual_seed(0)
    v, h, n = 11, 6, 5
    weight = torch.randn(v, h)
    hidden = torch.randn(n, h, requires_grad=True)
    targets = torch.randint(0, v, (n,))
    rows = [(list(range(v)), [0.0] * v)] * n
    for support in (True, False):  # the whole vocabulary as support, or candidates without renormalising
        head = _head(weight)
        out = sampled_logprobs(hidden, head, targets, _sampler(rows, support, requested=v + 1), entropy=True)
        g_hidden, g_weight = torch.autograd.grad(out.lp.sum(), [hidden, head.weight])
        plain_head = _head(weight)
        lp, ent = target_logprobs(hidden, plain_head, targets, entropy=True)
        p_hidden, p_weight = torch.autograd.grad(lp.sum(), [hidden, plain_head.weight])
        torch.testing.assert_close(out.lp, lp, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out.entropy, ent, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(g_hidden, p_hidden, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(g_weight, p_weight, atol=1e-5, rtol=1e-5)
        full = torch.log_softmax(hidden.detach() @ weight.T, -1)
        torch.testing.assert_close(out.candidate_lp, full, atol=1e-5, rtol=1e-5)


def test_renormalised_gradient_gradcheck_float64():
    torch.manual_seed(1)
    v, h = 9, 4
    weight = torch.randn(v, h, dtype=torch.float64, requires_grad=True)
    hidden = torch.randn(3, h, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([2, 5, 0])
    sampler = _sampler([([2, 3, 7], [0.0] * 3), ([5, 1], [0.0] * 2), ([0, 2, 4, 6], [0.0] * 4)])
    sampler.logprobs = sampler.logprobs.double()

    def f(hidden, weight):
        head = nn.Linear(h, v, bias=False).double()
        del head.weight
        head.weight = weight  # a plain tensor attribute: the gradient reaches `weight`
        out = sampled_logprobs(hidden, head, targets, sampler)
        return out.lp, out.candidate_lp.masked_fill(torch.isneginf(out.candidate_lp), 0.0)

    assert torch.autograd.gradcheck(f, (hidden, weight))


def test_tokens_without_candidates_or_with_an_incomplete_support():
    torch.manual_seed(2)
    weight = torch.randn(7, 5)
    hidden = torch.randn(3, 5)
    targets = torch.tensor([1, 4, 2])
    # row 0: a forced token (no candidates): the whole vocabulary; row 1: as many candidates as requested (the
    # support may hold more: left out of the loss); row 2: the sampled token is not in the support (left out)
    rows = [([], []), ([4, 0], [-0.1, -2.0]), ([3, 5], [-0.2, -1.5])]
    out = sampled_logprobs(hidden, _head(weight), targets, _sampler(rows, requested=2))
    full = torch.log_softmax(hidden @ weight.T, -1)
    assert out.lp[0].item() == pytest.approx(full[0, 1].item(), abs=1e-6)
    assert out.valid.tolist() == [True, False, False]
    behaviour = out.lp.detach()
    config = RLConfig().loss
    loss, stats = policy_loss(
        out.lp,
        behaviour,
        torch.ones(3),
        torch.ones(3),
        torch.zeros(3, dtype=torch.long),
        1,
        torch.tensor([3.0]),
        config,
        None,
        out,
    )
    (grad,) = torch.autograd.grad(loss, out.lp)
    assert grad.tolist() == [-1.0, 0.0, 0.0]  # only the valid token trains
    assert stats["support_tokens"].item() == 2 and stats["support_incomplete"].item() == 2


def test_hf_recorder_candidates_give_ratio_one():
    """The candidates the hf engine records are the support it sampled from: renormalising the same logits over
    them gives back the behaviour log-prob exactly (ratio 1 on-policy); the full vocabulary would not."""
    torch.manual_seed(3)
    v, h = 13, 8
    weight = torch.randn(v, h)
    hidden = torch.randn(2, h)
    sampling = Sampling(0.7, top_p=0.9, top_k=4, candidates=4 + 8)
    recorder = _RecordLogprobs(sampling)
    scores = hidden @ weight.T
    recorder(torch.zeros(2, 1, dtype=torch.long), scores.clone())
    sampled = torch.multinomial(recorder.previous.exp(), 1)
    behaviour = recorder.finish(torch.cat([torch.zeros(2, 1, dtype=torch.long), sampled], 1))[:, 0]
    parts = [recorder.candidates(row, 1) for row in range(2)]
    assert all(1 <= p.counts[0] <= 4 for p in parts)
    sampler = SamplerCandidates.build(parts, sampling.candidates, True, "cpu")
    out = sampled_logprobs(hidden, _head(weight), sampled[:, 0], sampler, scale=1 / 0.7)
    torch.testing.assert_close(out.lp, behaviour, atol=1e-5, rtol=1e-5)
    assert out.valid.all()
    plain, _ = target_logprobs(hidden, _head(weight), sampled[:, 0], scale=1 / 0.7)
    assert (plain < behaviour - 1e-4).all()  # the bias renormalising removes


# ------------------------------------------------------------------ score centering


def _centering_config(kind="icepop", k=8, **overrides):
    config = RLConfig().loss
    config.type, config.score_centering, config.score_centering_k = kind, "topk", k
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _sampled(candidate_lp, sampler_lp):
    n = candidate_lp.shape[0]
    sampler = SamplerCandidates(
        torch.zeros(sampler_lp.shape, dtype=torch.long),
        sampler_lp,
        False,
        torch.ones(n, dtype=torch.bool),
        torch.ones(n, dtype=torch.bool),
    )
    return SampledLogProbs(candidate_lp[:, 0], torch.zeros(0), candidate_lp, torch.ones(n, dtype=torch.bool), sampler)


def test_centering_matches_trl_on_hand_values():
    """icepop is TRL's masked importance sampling (bounds C_min, C_max): the correction equals TRL compute_loss's."""
    p = torch.tensor([[0.5, 0.2, 0.1]], dtype=torch.float64)
    q = torch.tensor([[0.4, 0.3, 0.05]], dtype=torch.float64)
    low, high = 0.5, 5.0
    config = _centering_config(icepop_low=low, icepop_high=high)
    lp_v = p.log().requires_grad_(True)
    correction, stats = centering_correction(_sampled(lp_v, q.log()), torch.ones(1, dtype=torch.float64), config)
    # TRL: head = p inside [low q, high q]; tail = 1[(1 - Σp) / (1 - Σq) in bounds]; residual = head - tail x p
    head = torch.where((p >= low * q) & (p <= high * q), p, 0.0)  # p3 = 0.1 > ... 0.1 <= 0.25: inside
    tail_ratio = (1 - p.sum()) / (1 - q.sum())  # 0.2 / 0.25 = 0.8: inside
    residual = head - float(low <= tail_ratio <= high) * p
    assert torch.allclose(correction, (residual * lp_v).sum(-1))
    assert residual.abs().sum() == 0  # every ratio in bounds: nothing to center
    # out of bounds: the second candidate (0.2 / 0.03 > 5) and the tail (0.2 / 0.52 < 0.5)
    q2 = torch.tensor([[0.4, 0.03, 0.05]], dtype=torch.float64)
    correction, _ = centering_correction(_sampled(lp_v, q2.log()), torch.ones(1, dtype=torch.float64), config)
    tail_ratio = (1 - p.sum()) / (1 - q2.sum())
    head = torch.where((p >= low * q2) & (p <= high * q2), p, 0.0)
    residual = head - float(low <= tail_ratio <= high) * p
    assert torch.allclose(residual, torch.tensor([[0.5, 0.0, 0.1]], dtype=torch.float64))
    assert torch.allclose(correction, (residual * lp_v).sum(-1))
    (grad,) = torch.autograd.grad(correction.sum(), lp_v)
    assert torch.allclose(grad, residual)  # the coefficients are detached
    assert stats["sampler_mass"].item() == pytest.approx(0.75)


@pytest.mark.parametrize(
    "kind, tail",
    [("masked_is", False), ("cispo", False), ("icepop", False), ("ipo", False)]
    + [("masked_is", True), ("cispo", True), ("icepop", True)],
)
def test_centered_gradient_has_zero_mean_under_the_sampler(kind, tail):
    """With a constant advantage, the centered policy gradient averages to 0 over tokens drawn from the sampler,
    exactly when the candidates hold the whole support or the ratio is constant on the tail (ipo's tail mask is an
    approximation); uncentered it does not when ratios are clipped or capped (the pull toward the sampler)."""
    torch.manual_seed(4)
    v = 6
    logits = torch.randn(v, dtype=torch.float32, requires_grad=True)
    mu = torch.softmax(torch.randn(v, dtype=torch.float32), -1)
    k = 4 if tail else v
    if tail:  # the tail's ratio constant: π(v) / μ(v) = c on the last v - k tokens
        pi0 = torch.softmax(logits.detach(), -1)
        c = pi0[k:].sum() / mu[k:].sum()
        mu = torch.cat([mu[:k] / mu[:k].sum() * (1 - pi0[k:].sum() / c), pi0[k:] / c])
    ratios = torch.softmax(logits.detach(), -1) / mu
    assert ratios.min() < 0.8 and ratios.max() > 1.25  # some ratios are clipped / capped below
    config = _centering_config(kind, k=k, eps_high=0.25, eps_low=0.2, is_cap=1.25, ipo_eps=0.05)
    config.icepop_low, config.icepop_high, config.seq_mask = 0.8, 1.25, 0.0  # no sequence mask: every y counts

    def expected(centering):
        config.score_centering = "topk" if centering else "off"
        lp_all = torch.log_softmax(logits, -1)
        total = 0.0
        for y in range(v):  # E_{y ~ μ}[per-token loss]
            ids = torch.arange(k)
            candidate_lp = lp_all[ids][None]
            sampled = _sampled(candidate_lp, mu[ids].log()[None])
            sampled.lp = lp_all[y : y + 1]
            loss, _ = policy_loss(
                lp_all[y : y + 1],
                mu[y].log()[None],
                torch.ones(1, dtype=torch.float32),
                torch.ones(1, dtype=torch.float32),
                torch.zeros(1, dtype=torch.long),
                1,
                torch.tensor([1.0], dtype=torch.float32),
                config,
                None,
                sampled if centering else None,
            )
            total = total + mu[y] * loss
        return torch.autograd.grad(total, logits)[0]

    assert expected(True).abs().max() < 1e-6
    assert expected(False).abs().max() > 1e-3


def test_centering_off_leaves_the_loss_unchanged():
    torch.manual_seed(5)
    lp = torch.randn(4, requires_grad=True)
    behaviour = lp.detach() + 0.1 * torch.randn(4)
    args = (
        torch.tensor([1.0, -1.0, 0.5, 2.0]),
        torch.ones(4),
        torch.zeros(4, dtype=torch.long),
        1,
        torch.tensor([4.0]),
    )
    config = RLConfig().loss
    plain, _ = policy_loss(lp, behaviour, *args, config)
    candidate_lp = torch.stack([lp, lp - 1.0], 1)
    sampled = _sampled(candidate_lp, torch.stack([behaviour, behaviour - 1.0], 1))
    with_candidates, _ = policy_loss(lp, behaviour, *args, config, None, sampled)
    assert torch.equal(plain, with_candidates)
    config.score_centering = "topk"
    with pytest.raises(ValueError, match="candidates"):
        policy_loss(lp, behaviour, *args, config)


# ------------------------------------------------------------------ configuration


def _config(**options) -> RLConfig:
    config = RLConfig()
    config.set("model.policy", "x")
    config.set("data.dataset", "d.jsonl")
    config.set("rewards.m.fn", "math")
    for key, value in options.items():
        config.set(key.replace("__", "."), value)
    return config


def test_config_defaults_sample_plainly():
    config = _config()
    config.validate()
    assert sampler_candidates(config) == 0 and training_sampling(config) == 1.0


def test_config_truncated_sampling_and_centering():
    config = _config(rollout__top_p=0.95, rollout__top_k=20)
    config.validate()
    assert training_sampling(config) == Sampling(1.0, 0.95, 20, candidates=28)
    config = _config(loss__score_centering="topk", loss__score_centering_k=16)
    config.validate()
    assert training_sampling(config) == Sampling(1.0, candidates=16)
    config = _config(rollout__top_k=4, loss__score_centering="topk", loss__score_centering_k=16)
    assert sampler_candidates(config) == 16


@pytest.mark.parametrize(
    "options, message",
    [
        ({"rollout__top_p": 0.9}, "needs rollout.top_k"),
        ({"rollout__top_p": 0.0}, "rollout.top_p must be in"),
        ({"rollout__top_k": -1}, "rollout.top_k must be"),
        ({"rollout__top_k": 5000}, "at most 1024"),
        ({"loss__score_centering": "all"}, "loss.score_centering must be one of"),
        ({"loss__score_centering": "topk", "loss__score_centering_k": 0}, "score_centering_k must be"),
        ({"loss__score_centering": "topk", "loss__type": "gspo"}, "not defined for loss.type 'gspo'"),
        ({"loss__score_centering": "topk", "loss__type": "reinforce"}, "not defined for loss.type 'reinforce'"),
        ({"rollout__top_k": 20, "loss__kl_ref": "start"}, "kl_ref is not supported"),
    ],
)
def test_config_refuses(options, message):
    with pytest.raises(RLConfigError, match=message):
        _config(**options).validate()


def test_config_warns_on_large_top_k_and_running_servers():
    assert any("top_k 128" in w for w in _config(rollout__top_k=128).validate())
    warnings = _config(rollout__backend="vllm_server", rollout__url="http://x", rollout__top_k=20).validate()
    assert any("--max-logprobs >= 28" in w for w in warnings)


# ------------------------------------------------------------------ end to end (hf backend)


@pytest.mark.parametrize("centering", [False, True], ids=["top_p_top_k", "with_centering"])
def test_trainer_step_with_top_p_top_k_is_on_policy(tmp_path, centering):
    """hf backend, top-p / top-k: the policy's step-0 ratios against its own samples are 1 (renormalised over
    the support the engine returns), and the step trains."""
    from test_rl import tiny_policy

    from palingenesis.rl.trainer import RLTrainer

    rows = [{"prompt": f"What is {i} plus {i + 1}?", "answer": str(2 * i + 1)} for i in range(12)]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = RLConfig()
    for key, value in {
        "model.policy": tiny_policy(tmp_path),
        "model.use_liger_kernel": False,
        "model.chat_template_kwargs": {"enable_thinking": False},
        "model.gradient_checkpointing": "none",
        "data.dataset": str(tmp_path / "data.jsonl"),
        "rollout.backend": "hf",
        "rollout.batch_prompts": 2,
        "rollout.group_size": 3,
        "rollout.max_new_tokens": 6,
        "rollout.temperature": 0.8,
        "rollout.top_p": 0.9,
        "rollout.top_k": 5,
        "loss.score_centering": "topk" if centering else "off",
        "loss.score_centering_k": 4,
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 1,
        "train.learning_rate": 1e-3,
        "train.warmup_steps": 0,
        "train.save_final": False,
    }.items():
        config.set(key, value)
    trainer = RLTrainer(config, rewards=[lambda completion: len(completion) / 40])
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, dict(metrics)))
    trainer.train()
    metrics = next(m for kind, m in logged if "policy/mismatch_k3" in m)
    assert metrics["policy/mismatch_k3"] < 1e-6 and abs(metrics["policy/ratio"] - 1) < 1e-3
    assert metrics["sampling/support_incomplete"] == 0.0
    assert 1 <= metrics["sampling/support_size"] <= 5
    assert ("sampling/sampler_mass" in metrics) == centering


def test_a_step_with_many_incomplete_supports_warns(caplog):
    from palingenesis.rl.losses import sampling_metrics

    with caplog.at_level("WARNING", logger="palingenesis.rl.losses"):
        metrics = sampling_metrics({"support_tokens": 10.0, "support_incomplete": 0.0, "support_size": 40.0})
        assert metrics == {"sampling/support_incomplete": 0.0, "sampling/support_size": 4.0} and not caplog.records
        sampling_metrics({"support_tokens": 10.0, "support_incomplete": 9.0, "support_size": 280.0})
    assert "processed_logprobs" in caplog.text
