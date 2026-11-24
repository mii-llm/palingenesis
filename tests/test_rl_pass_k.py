"""loss.pass_k: pass@k advantages (PKPO / Pass@k Training) against hand-computed values, the informative-group test,
the outcome metric and the config's validation."""

import math

import pytest

pytest.importorskip("torch")

from palingenesis.rl.config import RLConfig, RLConfigError  # noqa: E402
from palingenesis.rl.losses import assign_advantages  # noqa: E402
from palingenesis.rl.trajectory import Trajectory, group_is_informative  # noqa: E402


def _traj(reward, scored=True):
    t = Trajectory({}, 0, [1], [], tokens=[5, 5], mask=[True, True], logprobs=[0.0, 0.0])
    t.reward, t.scored, t.trained = reward, scored, True
    return t


def test_pass_k_advantages_hand_computed():
    # n = 4 scored, c = 1 success, k = 2: success reward 1, failures 1 - C(2,1)/C(3,1) = 1/3; mean = pass@2 = 1/2
    group = [_traj(1.0), _traj(0.0), _traj(0.0), _traj(0.0), _traj(7.0, scored=False)]
    assign_advantages([group], "none", pass_k=2)
    assert [t.advantage for t in group] == pytest.approx([0.5, -1 / 6, -1 / 6, -1 / 6, 0.0])
    # pass@1 for comparison: failures pushed down three times as hard relative to the success
    assign_advantages([group], "none")
    assert [t.advantage for t in group] == pytest.approx([0.75, -0.25, -0.25, -0.25, 0.0])
    # leave-one-out baseline multiplies by n / (n - 1)
    assign_advantages([group], "none", baseline="leave_one_out", pass_k=2)
    assert [t.advantage for t in group][:4] == pytest.approx([0.5 * 4 / 3, -2 / 9, -2 / 9, -2 / 9])
    # group std: the transformed rewards' std, sqrt(mean of squares of [1/2, -1/6 x 3]) = sqrt(1/12)
    assign_advantages([group], "group", pass_k=2)
    sd = math.sqrt((0.25 + 3 / 36) / 4)
    assert group[0].advantage == pytest.approx(0.5 / (sd + 1e-6))


def test_pass_k_saturated_groups_carry_no_gradient():
    # c = 5 of n = 8 with k = 4: every 4-subset holds a success, pass@4 = 1, nothing to learn
    group = [_traj(1.0) for _ in range(5)] + [_traj(0.0) for _ in range(3)]
    assert group_is_informative(group) and not group_is_informative(group, pass_k=4)
    assign_advantages([group], "none", pass_k=4)
    assert all(t.advantage == pytest.approx(0.0) for t in group)
    # c = 2: informative under pass@4 as well
    group = [_traj(1.0), _traj(1.0)] + [_traj(0.0) for _ in range(6)]
    assert group_is_informative(group, pass_k=4)
    # k capped at the scored rollouts: 2 scored with k 4 -> k = 2 = n -> no signal
    small = [_traj(1.0), _traj(0.0), _traj(0.0, scored=False)]
    assert not group_is_informative(small, pass_k=4)


def test_pass_k_continuous_rewards_credit_the_best_of_k():
    # rewards 0.2, 0.5, 0.9 with k = 2: subsets {.2,.5}->.5, {.2,.9}->.9, {.5,.9}->.9; per sample (n/k)(1/3)(sum)
    group = [_traj(0.2), _traj(0.5), _traj(0.9)]
    assign_advantages([group], "none", pass_k=2)
    r = [1.5 / 3 * (0.5 + 0.9), 1.5 / 3 * (0.5 + 0.9), 1.5 / 3 * (0.9 + 0.9)]
    mean = sum(r) / 3
    assert [t.advantage for t in group] == pytest.approx([x - mean for x in r])
    assert mean == pytest.approx((0.5 + 0.9 + 0.9) / 3)  # E[max of 2]


def test_pass_k_config_validation(tmp_path):
    path = tmp_path / "rl.yaml"
    path.write_text("model: {policy: m}\ndata: {dataset: d.jsonl}\nrewards: {correct: math}\n")

    def errors(**options):
        config = RLConfig.from_yaml(path)
        for key, value in options.items():
            config.set(key.replace("__", "."), value)
        return config.validate()

    assert RLConfig().loss.pass_k == 1
    assert errors(loss__pass_k=4, rollout__group_size=8) == []
    with pytest.raises(RLConfigError, match="needs rollout.group_size > pass_k"):
        errors(loss__pass_k=8, rollout__group_size=8)
    with pytest.raises(RLConfigError, match="integer >= 1"):
        errors(loss__pass_k=0)


def test_outcome_stats_report_unbiased_pass_k():
    from types import SimpleNamespace

    from palingenesis.rl.pipeline import RLPipeline

    config = RLConfig()
    config.set("loss.pass_k", 2)
    config.set("rollout.group_size", 4)
    fake = SimpleNamespace(config=config, sampler=None)
    groups = [[_traj(1.0), _traj(0.0), _traj(0.0), _traj(0.0)], [_traj(0.0)] * 4]
    out = RLPipeline._outcome_stats(fake, groups)
    assert out["outcome/pass@2"] == pytest.approx((0.5 + 0.0) / 2)
    assert out["outcome/pass@1"] == pytest.approx(1 / 8) and out["outcome/pass@4"] == pytest.approx(0.5)
