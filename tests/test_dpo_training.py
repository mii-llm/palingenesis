"""DPO wired into the trainer: config validation, the preference dataloader,
held-out evaluation, and the micro-step arithmetic `train.py` uses.

The loop tests call the trainer's own helpers (`_get_hidden_states`,
`_get_lm_head`) and the same `reference_logps` / `preference_step` calls with the
same denominators as `train.py`, on a tiny GPT-2 on CPU.
"""

import copy
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from test_dpo import PAIR, QWEN35_TEMPLATE, _tiny_model, _toy_batch  # noqa: E402

from palingenesis.config import Config, ConfigError  # noqa: E402
from palingenesis.data import IGNORE_INDEX  # noqa: E402
from palingenesis.dpo import (  # noqa: E402
    PreferenceEvaluator,
    build_preference_dataloader,
    collate_preferences,
    disable_dropout,
    preference_loss,
    preference_step,
    reference_logps,
    score_weights,
    token_logps,
)
from palingenesis.loss import shift_labels  # noqa: E402
from palingenesis.train import _get_hidden_states, _get_lm_head  # noqa: E402

# ── config ──────────────────────────────────────────────────────────────────


def _dpo_config(**dpo):
    config = Config()
    config.dpo.enabled = True
    for k, v in dpo.items():
        setattr(config.dpo, k, v)
    return config


def test_valid_dpo_config_passes():
    _dpo_config(loss_type="sigmoid", beta=0.1, sft_weight=0.2, ld_alpha=0.5).validate()
    _dpo_config(loss_type="robust", label_smoothing=0.1).validate()


@pytest.mark.parametrize("dpo,match", [
    ({"loss_type": "kto"}, "loss_type"),
    ({"beta": 0.0}, "beta"),
    ({"loss_type": "robust", "label_smoothing": 0.5}, "label_smoothing"),
    ({"label_smoothing": 0.1}, "only used by loss_type=robust"),
    ({"ld_alpha": 1.5}, "ld_alpha"),
    ({"sft_weight": -1.0}, "sft_weight"),
])
def test_invalid_dpo_settings_raise(dpo, match):
    with pytest.raises(ConfigError, match=match):
        _dpo_config(**dpo).validate()


@pytest.mark.parametrize("section,field,value", [
    ("data", "packing", True),
    ("data", "sources", [{"dataset": "x"}]),
    ("data", "pretokenize", True),
    ("data", "pretrain_replay_dataset", "some/corpus"),
    ("parallel", "context_parallel", True),
    ("plugins", "deft", True),
    ("preprocess", "enabled", True),
])
def test_sft_only_features_are_rejected(section, field, value):
    config = _dpo_config()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ConfigError, match=f"does not support {section}.{field}"):
        config.validate()


def test_dpo_section_loads_from_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("dpo:\n  enabled: true\n  loss_type: ipo\n  beta: 0.5\n  sft_weight: 0.1\n")
    config = Config.from_yaml(path)
    assert (config.dpo.enabled, config.dpo.loss_type, config.dpo.beta, config.dpo.sft_weight) == (True, "ipo", 0.5, 0.1)
    config = Config.from_cli(["--config", str(path), "--dpo.beta", "0.05"])
    assert config.dpo.beta == 0.05


# ── dataloader ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qwen_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = QWEN35_TEMPLATE
    tok.pad_token = tok.eos_token
    return tok


def test_dataloader_yields_chosen_then_rejected(qwen_tokenizer):
    from datasets import Dataset

    rows = [
        {**PAIR, "chosen": [{"role": "assistant", "content": f"answer {i}", "reasoning": f"why {i}"}],
         "rejected": [{"role": "assistant", "content": f"bad {i} " * (i + 1)}],
         "chat_template_kwargs": {"enable_thinking": i % 2 == 0}}
        for i in range(6)
    ]
    config = _dpo_config()
    config.data.num_workers = 0
    loader = build_preference_dataloader(Dataset.from_list(rows), qwen_tokenizer, config.data, config.dpo,
                                         rank=0, world_size=1, batch_size=2)
    batches = list(loader)
    assert len(batches) == 3
    for batch in batches:
        assert batch["input_ids"].shape[0] == 4 and batch["input_ids"].shape[1] % 64 == 0
        assert int(batch["num_pairs"]) == 2
        chosen = qwen_tokenizer.decode(batch["input_ids"][0][batch["labels"][0] != IGNORE_INDEX])
        rejected = qwen_tokenizer.decode(batch["input_ids"][2][batch["labels"][2] != IGNORE_INDEX])
        assert "answer" in chosen and "bad" not in chosen
        assert "bad" in rejected and "answer" not in rejected


# ── the trainer's micro-step ───────────────────────────────────────────────


def _micro_step(model, ref_model, ids, mask, labels, *, world_size=1, current_ga=1, **kw):
    """The DPO branch of train.py, verbatim in its denominators."""
    num_pairs = ids.shape[0] // 2
    global_chosen = (shift_labels(labels[:num_pairs]) != IGNORE_INDEX).sum()
    ref = reference_logps(ref_model, _get_hidden_states, _get_lm_head, ids, mask, labels, 2)
    hidden = _get_hidden_states(model, ids, mask, None)
    return preference_step(hidden, labels, _get_lm_head(model), ref, num_pairs,
                           pair_denom=num_pairs * world_size * current_ga,
                           chosen_token_denom=max(global_chosen.item(), 1) * current_ga,
                           num_chunks=2, **kw)


def _grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


@pytest.mark.parametrize("loss_type", ["sigmoid", "ipo"])
def test_grad_accumulation_equals_one_big_batch(loss_type):
    """Two micro-batches with the GA denominator give the gradient of the mean
    loss over all their pairs — the same guarantee SFT's token denominator gives."""
    policy = _tiny_model()
    with torch.no_grad():                      # move off the reference so the loss is non-trivial
        for p in policy.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ref = _tiny_model()

    ids, mask, labels = _toy_batch(num_pairs=4)
    halves = [torch.cat([t[i:i + 2], t[4 + i:4 + i + 2]]) for i in (0, 2) for t in (ids, mask, labels)]
    policy.zero_grad()
    for i in range(2):
        a, m, lab = halves[3 * i: 3 * i + 3]
        _micro_step(policy, ref, a, m, lab, current_ga=2, loss_type=loss_type).loss.backward()
    accumulated = _grads(policy).clone()

    policy.zero_grad()
    _micro_step(policy, ref, ids, mask, labels, current_ga=1, loss_type=loss_type).loss.backward()
    full = _grads(policy)
    assert (accumulated - full).norm() / full.norm() < 1e-10


def test_a_few_steps_learn_the_preference():
    """End to end on CPU: frozen reference, AdamW, the trainer's micro-step."""
    policy = _tiny_model().float().train()
    ref = copy.deepcopy(policy).eval().requires_grad_(False)
    assert disable_dropout(policy) > 0          # GPT-2 has dropout; the trainer zeroes it
    ids, mask, labels = _toy_batch(num_pairs=3)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    first = None
    for _ in range(15):
        step = _micro_step(policy, ref, ids, mask, labels, beta=0.1, sft_weight=0.1)
        step.loss.backward()
        opt.step()
        opt.zero_grad()
        first = first if first is not None else step.metrics
    assert abs(first["dpo/loss"] - math.log(2)) < 1e-6     # policy == reference at step 0
    assert step.metrics["dpo/loss"] < 0.3
    assert step.metrics["rewards/accuracies"] == 1.0
    assert step.metrics["rewards/chosen"] > 0 > step.metrics["rewards/rejected"]
    # the SFT anchor holds: chosen log-likelihood improved, not just rejected dropped
    assert step.metrics["logps/chosen"] > first["logps/chosen"]


# ── evaluation ─────────────────────────────────────────────────────────────


def _evaluator(ids, mask, labels, **kw):
    batch = {"input_ids": ids, "attention_mask": mask, "labels": labels}
    return PreferenceEvaluator([batch], loss_type=kw.get("loss_type", "sigmoid"), beta=0.1,
                               label_smoothing=0.0, ld_alpha=kw.get("ld_alpha"), num_chunks_for=lambda n: 3)


def test_evaluator_at_the_reference_policy():
    model = _tiny_model()
    ids, mask, labels = _toy_batch()
    metrics = _evaluator(ids, mask, labels).evaluate(model, copy.deepcopy(model), _get_hidden_states,
                                                     _get_lm_head, torch.device("cpu"), torch.float64, False)
    assert abs(metrics["eval/loss"] - math.log(2)) < 1e-12
    assert metrics["eval/rewards/margins"] == 0.0
    assert metrics["eval/rewards/accuracies"] == 0.0


@pytest.mark.parametrize("ld_alpha", [None, 0.5])
def test_evaluator_matches_direct_computation(ld_alpha):
    policy, ref = _tiny_model(), _tiny_model()
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ids, mask, labels = _toy_batch()
    evaluator = _evaluator(ids, mask, labels, ld_alpha=ld_alpha)
    metrics = evaluator.evaluate(policy, ref, _get_hidden_states, _get_lm_head,
                                 torch.device("cpu"), torch.float64, False)

    shifted = shift_labels(labels)
    a = score_weights(shifted, 3, ld_alpha)
    with torch.no_grad():
        s_pol = (token_logps(policy.transformer(input_ids=ids)[0], shifted, policy.lm_head) * a).sum(1)
        s_ref = (token_logps(ref.transformer(input_ids=ids)[0], shifted, ref.lm_head) * a).sum(1)
    expected = preference_loss(s_pol, s_ref, a.sum(1), beta=0.1).mean()
    assert abs(metrics["eval/loss"] - float(expected)) < 1e-12
    # the reference is scored once and cached; a second call must agree exactly
    again = evaluator.evaluate(policy, ref, _get_hidden_states, _get_lm_head,
                               torch.device("cpu"), torch.float64, False)
    assert again == metrics


def test_collated_eval_batch_round_trips(qwen_tokenizer):
    from palingenesis.dpo import PreferenceDataset

    ds = PreferenceDataset([], qwen_tokenizer, max_seq_length=512)
    pairs = [ds.process(PAIR), ds.process({**PAIR, "rejected": "totally different"})]
    batch = collate_preferences(pairs, qwen_tokenizer.pad_token_id, pad_to_multiple=64)
    assert batch["input_ids"].shape[0] == 4
    assert torch.equal(batch["input_ids"][0, : pairs[0]["chosen"]["input_ids"].numel()],
                       pairs[0]["chosen"]["input_ids"])
