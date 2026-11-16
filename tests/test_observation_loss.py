"""The observation loss (after ECHO, arXiv:2605.24517): a model trained to predict what the environment returns.

SFT (data.include_observations): the tool messages' own text gets loss (never the template's wrapper), each
conversation adds observation_weight x the mean NLL of its observation tokens to its assistant loss, and the
loss's token count is the assistant's tokens only; the plain, chunked, packed and SeCO paths agree.
RL (loss.observation_weight): the environment's output in the policy's own trajectories, per trajectory summed
over its targets and divided by its observation length, averaged over the trajectories, times λ, added to the
policy loss; off, the step is bitwise the same.

Tiny randomly initialised GPT-2 models and the GPT-2 tokenizer with chat templates, CPU.
"""

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from test_agentic_masking import AGENT, TOK, needs_tok  # noqa: E402

from palingenesis.config import Config, ConfigError  # noqa: E402
from palingenesis.data import IGNORE_INDEX, OBSERVATION_FIELD, ChatDataset, PackedDataset, _collate_fn  # noqa: E402
from palingenesis.loss import (  # noqa: E402
    chunked_cross_entropy_loss,
    counted_tokens,
    cross_entropy_loss,
    shift_labels,
    shift_weights,
)
from palingenesis.rl.config import RLConfig  # noqa: E402
from palingenesis.rl.trajectory import Trajectory  # noqa: E402

LAMBDA = 0.05

# A conversation with two tool results, one of which ("user") also occurs in the template's glue
TWO_TOOLS = [
    {"role": "user", "content": "List the files, then read one."},
    {
        "role": "assistant",
        "content": "Listing.",
        "tool_calls": [{"type": "function", "function": {"name": "ls", "arguments": '{"path": "."}'}}],
    },
    {"role": "tool", "content": "user"},
    {
        "role": "assistant",
        "content": "Reading.",
        "tool_calls": [{"type": "function", "function": {"name": "cat", "arguments": '{"path": "user"}'}}],
    },
    {"role": "tool", "content": "line one\nline two of the file"},
    {"role": "assistant", "content": "The file has two lines."},
]


def _dataset(**kw):
    return ChatDataset(None, TOK, 4096, include_observations=True, **kw)


def _observed_text(sample) -> list[str]:
    """The observation tokens' text, one string per run of consecutive tokens."""
    ids, observed = sample["input_ids"].tolist(), sample[OBSERVATION_FIELD].tolist()
    runs, current = [], []
    for token, flag in zip(ids, observed):
        if flag:
            current.append(token)
        elif current:
            runs.append(TOK.decode(current))
            current = []
    return runs + ([TOK.decode(current)] if current else [])


def _gpt2(vocab: int):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(
        n_layer=2, n_head=2, n_embd=16, n_positions=512, vocab_size=vocab, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0
    )
    return GPT2LMHeadModel(cfg).double().train()


def _token_ce(model, ids: torch.Tensor) -> torch.Tensor:
    """Per position t, the CE of predicting ids[t + 1] (the last position: 0), from one row's own forward."""
    logits = model(input_ids=ids[None]).logits[0]
    ce = torch.nn.functional.cross_entropy(logits[:-1].float(), ids[1:], reduction="none")
    return torch.cat([ce, ce.new_zeros(1)])


# ----------------------------------------------------------------- SFT: which tokens


@needs_tok
def test_observation_targets_are_the_tool_text_only():
    sample = _dataset()._process({"messages": AGENT})
    assert _observed_text(sample) == ["SUNNY_24C"]
    # the assistant's tokens are those of the assistant-only masking, untouched
    plain = ChatDataset(None, TOK, 4096)._process({"messages": AGENT})
    assistant = (sample["labels"] != IGNORE_INDEX) & ~sample[OBSERVATION_FIELD]
    assert torch.equal(assistant, plain["labels"] != IGNORE_INDEX)
    observed = sample[OBSERVATION_FIELD]
    assert torch.equal(sample["labels"][observed], sample["input_ids"][observed])


@needs_tok
def test_a_tool_text_that_occurs_in_the_template_glue_is_found_in_its_message():
    sample = _dataset()._process({"messages": TWO_TOOLS})
    assert _observed_text(sample) == ["user", "line one\nline two of the file"]
    text = TOK.decode(sample["input_ids"])
    assert text.count("<|im_start|>user") == 3  # the headers stay untrained
    for i in torch.nonzero(sample[OBSERVATION_FIELD]).flatten().tolist():
        assert TOK.decode([int(sample["input_ids"][i])]).strip() not in ("<|im_start|>", "<tool_response>")


class _SlowTokenizer:
    """The GPT-2 tokenizer seen as one without offsets: ChatDataset takes the progressive masker."""

    is_fast = False

    def __init__(self, tok):
        self._tok = tok

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __call__(self, *args, **kwargs):
        return self._tok(*args, **kwargs)


@needs_tok
@pytest.mark.parametrize("messages", [AGENT, TWO_TOOLS], ids=["one-tool", "two-tools"])
def test_progressive_masker_marks_the_same_observation_tokens(messages):
    fast = _dataset()._process({"messages": messages})
    slow = ChatDataset(None, _SlowTokenizer(TOK), 4096, include_observations=True)._process({"messages": messages})
    assert torch.equal(slow["input_ids"], fast["input_ids"])
    assert _observed_text(slow) == _observed_text(fast)


# ----------------------------------------------------------------- SFT: weights


@needs_tok
def test_weights_carry_lambda_times_the_mean_observation_nll():
    sample = _dataset(observation_weight=LAMBDA)._process({"messages": TWO_TOOLS})
    observed, labelled = sample[OBSERVATION_FIELD], sample["labels"] != IGNORE_INDEX
    assistant = labelled & ~observed
    w = sample["loss_weights"]
    assert torch.equal(w[assistant], torch.ones(int(assistant.sum())))
    expected = LAMBDA * int(assistant.sum()) / int(observed.sum())
    assert torch.allclose(w[observed], torch.full((int(observed.sum()),), expected))
    assert float(w[~labelled].abs().sum()) == 0.0


@needs_tok
def test_token_normalisation_at_weight_one_is_the_old_behaviour():
    old = _dataset(observation_weight=1.0, observation_normalization="token")._process({"messages": TWO_TOOLS})
    new = _dataset()._process({"messages": TWO_TOOLS})
    assert set(old) == {"input_ids", "attention_mask", "labels"}  # no weights, no mask: plain token CE
    assert torch.equal(old["labels"], new["labels"])
    scaled = _dataset(observation_weight=0.5, observation_normalization="token")._process({"messages": TWO_TOOLS})
    assert OBSERVATION_FIELD not in scaled
    assert torch.allclose(scaled["loss_weights"][new[OBSERVATION_FIELD]], torch.tensor(0.5))


@needs_tok
def test_weight_zero_trains_no_observation_and_turn_scaling_keeps_its_mean():
    off = _dataset(observation_weight=0.0)._process({"messages": TWO_TOOLS})
    plain = ChatDataset(None, TOK, 4096)._process({"messages": TWO_TOOLS})
    assert torch.equal(off["labels"], plain["labels"]) and OBSERVATION_FIELD not in off
    scaled = _dataset(turn_scaling="progressive")._process({"messages": TWO_TOOLS})
    observed = scaled[OBSERVATION_FIELD]
    assistant = (scaled["labels"] != IGNORE_INDEX) & ~observed
    w = scaled["loss_weights"]
    assert abs(float(w[assistant].mean()) - 1.0) < 1e-6 and float(w[assistant].max()) > 1.0
    assert torch.allclose(w[observed], torch.tensor(LAMBDA * int(assistant.sum()) / int(observed.sum())))


# ----------------------------------------------------------------- SFT: the loss, by hand and on every path


def _samples():
    ds = _dataset(observation_weight=LAMBDA)
    return [ds._process({"messages": m}) for m in (AGENT, TWO_TOOLS)]


def _by_hand(model, samples) -> float:
    """sum over conversations of (assistant CE sum + λ x n_assistant x mean observation CE) / all assistant
    tokens: per conversation, its mean assistant CE + λ x its mean observation CE, weighted by its share of the
    assistant tokens."""
    numerator, assistant_total = 0.0, 0
    for s in samples:
        ce = _token_ce(model, s["input_ids"]).detach()
        target = shift_labels(s["labels"][None])[0] != IGNORE_INDEX
        obs = shift_weights(s[OBSERVATION_FIELD][None].float())[0].bool() & target
        assistant = target & ~obs
        numerator += float(ce[assistant].sum()) + LAMBDA * int(assistant.sum()) * float(ce[obs].mean())
        assistant_total += int(assistant.sum())
    return numerator / assistant_total


@needs_tok
def test_sft_loss_is_assistant_mean_plus_lambda_observation_mean():
    model = _gpt2(len(TOK))
    single = _samples()[1]
    ce = _token_ce(model, single["input_ids"]).detach()
    target = shift_labels(single["labels"][None])[0] != IGNORE_INDEX
    obs = shift_weights(single[OBSERVATION_FIELD][None].float())[0].bool() & target
    assert obs.any()
    batch = _collate_fn([single], TOK.pad_token_id)
    labels = shift_labels(batch["labels"])
    loss = cross_entropy_loss(
        model(input_ids=batch["input_ids"]).logits,
        labels,
        float(counted_tokens(labels, shift_weights(batch[OBSERVATION_FIELD].float()))),
        weights=shift_weights(batch["loss_weights"]),
    )
    expected = float(ce[target & ~obs].mean()) + LAMBDA * float(ce[obs].mean())
    assert abs(loss.item() - expected) < 1e-5 * expected


@needs_tok
@pytest.mark.parametrize("path", ["plain", "chunked"])
def test_sft_loss_over_a_batch_matches_the_hand_computation(path):
    model = _gpt2(len(TOK))
    samples = _samples()
    batch = _collate_fn(samples, TOK.pad_token_id)
    labels = shift_labels(batch["labels"])
    weights = shift_weights(batch["loss_weights"])
    denom = float(counted_tokens(labels, shift_weights(batch[OBSERVATION_FIELD].float())))
    assert denom == sum(int(((s["labels"] != IGNORE_INDEX) & ~s[OBSERVATION_FIELD]).sum()) for s in samples)
    if path == "plain":
        loss = cross_entropy_loss(model(input_ids=batch["input_ids"]).logits, labels, denom, weights=weights)
    else:
        hidden = model.transformer(input_ids=batch["input_ids"]).last_hidden_state
        loss = chunked_cross_entropy_loss(hidden, labels, model.lm_head, 3, denom, weights=weights)
    expected = _by_hand(model, samples)
    assert abs(loss.item() - expected) < 1e-5 * expected


@needs_tok
def test_packing_gives_the_same_loss():
    model = _gpt2(len(TOK))
    samples = _samples()
    (block,) = list(PackedDataset(samples, max_len=4096, sort_buffer=0))
    assert block[OBSERVATION_FIELD].sum() == sum(int(s[OBSERVATION_FIELD].sum()) for s in samples)
    # the packed row's logits: each document's own forward, side by side (what document-aware attention computes)
    logits = torch.cat([model(input_ids=s["input_ids"][None]).logits for s in samples], 1)
    position_ids = block["position_ids"][None]
    labels = shift_labels(block["labels"][None], position_ids)
    denom = float(counted_tokens(labels, shift_weights(block[OBSERVATION_FIELD][None].float())))
    loss = cross_entropy_loss(logits, labels, denom, weights=shift_weights(block["loss_weights"][None]))
    expected = _by_hand(model, samples)
    assert abs(loss.item() - expected) < 1e-5 * expected


@needs_tok
@pytest.mark.parametrize("packed", [False, True], ids=["rows", "packed"])
def test_seco_gives_the_same_loss_and_gradients(packed):
    from palingenesis.seco import chunkwise_forward_backward

    model = _gpt2(len(TOK))
    samples = _samples()
    batch = _collate_fn(samples, TOK.pad_token_id)
    labels = shift_labels(batch["labels"])
    denom = float(counted_tokens(labels, shift_weights(batch[OBSERVATION_FIELD].float())))
    model.zero_grad()
    loss = cross_entropy_loss(
        model(input_ids=batch["input_ids"]).logits, labels, denom, weights=shift_weights(batch["loss_weights"])
    )
    loss.backward()
    full = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    if packed:
        (block,) = list(PackedDataset(samples, max_len=4096, sort_buffer=0))
        ids, raw, weights, position_ids = (
            block[k][None] for k in ("input_ids", "labels", "loss_weights", "position_ids")
        )
    else:
        ids, raw, weights, position_ids = batch["input_ids"], batch["labels"], batch["loss_weights"], None
    result = chunkwise_forward_backward(
        model, ids, raw, chunk_size=16, loss_denom=denom, weights=weights, position_ids=position_ids
    )
    chunked = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert abs(result.loss - loss.item()) < 1e-6 * loss.item()
    for name, grad in full.items():
        assert torch.allclose(chunked[name], grad, rtol=1e-5, atol=1e-9), name


@needs_tok
def test_pretokenized_cache_keeps_the_weights_and_the_mask(tmp_path):
    from datasets import load_dataset

    from palingenesis.data import PRETOK_DATA, PretokenizedDataset, materialize_pretokenized, pretokenize_fingerprint

    config = Config()
    config.data.include_observations = True
    plain = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    samples = [*_samples(), _dataset()._process(plain)]
    assert OBSERVATION_FIELD not in samples[-1] and "loss_weights" not in samples[-1]
    materialize_pretokenized(lambda: iter(samples), tmp_path, "fingerprint", config, TOK)
    data = load_dataset("parquet", data_files=str(tmp_path / PRETOK_DATA), split="train")
    rows = list(PretokenizedDataset(data, shuffle=False))
    assert len(rows) == len(samples)
    for row, sample in zip(rows, samples):
        assert set(row) == set(sample)
        for key in sample:
            assert torch.allclose(row[key].to(sample[key].dtype), sample[key]), key
    before = pretokenize_fingerprint(config, TOK)
    config.data.observation_weight = 0.02
    assert pretokenize_fingerprint(config, TOK) != before  # a cache with other weights is rebuilt


def test_config_refuses_objectives_that_ignore_the_weights():
    def errors(**data):
        c = Config()
        c.plugins.pre_rl = True  # its loss ignores per-token weights
        for k, v in data.items():
            setattr(c.data, k, v)
        try:
            c.validate()
        except ConfigError as e:
            return str(e)
        return ""

    assert "include_observations" in errors(include_observations=True)
    assert "include_observations" not in errors(
        include_observations=True, observation_weight=1.0, observation_normalization="token"
    )
    assert "observation_weight" in errors(observation_weight=-1.0)
    assert "observation_normalization" in errors(observation_normalization="sequence")


# ----------------------------------------------------------------- RL: which tokens


def _chat():
    from multimodal_helpers import tiny_tokenizer

    from palingenesis.rl.chat import ChatFormat

    tok = tiny_tokenizer()
    return tok, ChatFormat(tok, tok.convert_tokens_to_ids("<|im_end|>"), {"enable_thinking": False})


def test_continuation_marks_the_environment_output_only():
    from palingenesis.rl.chat import OBSERVATION_EXCLUDED, OBSERVATION_TARGET
    from palingenesis.rl.envs.harness.terminus import TERMINUS_WARNING
    from palingenesis.rl.pipeline import RLPipeline

    tok, chat = _chat()
    long_output = chat.truncate(" ".join(f"w{i}" for i in range(400)), 40)
    messages = [
        {"role": "tool", "content": "user"},
        {"role": "tool", "content": "Previous response had warnings:\nWARNINGS: - extra text\n\nREAL OUTPUT"},
        {"role": "tool", "content": long_output},
        {"role": "tool", "content": "Error: could not parse the call."},
    ]
    pipeline = object.__new__(RLPipeline)
    pipeline.observation_exclude = RLPipeline.observation_patterns([])

    class Env:
        observation_notes = (TERMINUS_WARNING,)

    notes = pipeline._observation_notes(messages, [False, False, False, True], Env())
    ids, marked = chat.continuation(messages, None, None, notes)
    assert ids == chat.continuation_ids(messages, None, None)
    targets = [i for i, kind in marked if kind == OBSERVATION_TARGET]
    excluded = [i for i, kind in marked if kind == OBSERVATION_EXCLUDED]
    target_text = tok.decode([ids[i] for i in targets])
    assert target_text.startswith("userREAL OUTPUTw0 w1")
    assert "tokens truncated" not in target_text and "Previous response" not in target_text
    assert "Error" not in target_text and "<tool_response>" not in target_text
    excluded_text = tok.decode([ids[i] for i in excluded])
    assert "Previous response had warnings" in excluded_text and "tokens truncated" in excluded_text
    assert excluded_text.endswith("Error: could not parse the call.")
    # never the template's markup: no special or added token (<|im_start|>, <tool_response>, ...)
    added = set(tok.get_added_vocab().values())
    assert not added & {ids[i] for i, _ in marked}
    assert tok.decode([ids[i] for i, _ in marked]).replace("user", "", 1).count("user") == 0


# ----------------------------------------------------------------- RL: the loss


def _groups(seed: int, observations: bool):
    rng = random.Random(seed)
    out = []
    for g in range(3):
        prompt = [rng.randrange(1, 90) for _ in range(rng.randint(5, 20))]
        group = []
        for _ in range(4):
            n = rng.randint(12, 30)
            mask = [rng.random() < 0.6 for _ in range(n)]
            mask[0] = True
            t = Trajectory(
                {},
                g,
                prompt,
                [],
                tokens=[rng.randrange(1, 90) for _ in range(n)],
                mask=mask,
                logprobs=[-rng.random() * 3 for _ in range(n)],
            )
            t.reward = rng.random()
            group.append(t)
        out.append(group)
    if observations:  # drawn apart: the trajectories are the same with and without
        rng = random.Random(seed + 1)
        for t in (t for g in out for t in g):
            context = [j for j in range(1, len(t.tokens)) if not t.mask[j]]
            t.observation_targets = [j for j in context if rng.random() < 0.7]
            t.observation_tokens = len(context)  # the rest: harness text, counted only
    return out


def _trainer(weight: float, sharing: str = "off"):
    from test_seco import MODELS

    from palingenesis.logits import output_head
    from palingenesis.rl.losses import LowPrecisionWeight
    from palingenesis.rl.parallel import Parallel
    from palingenesis.rl.trainer import RLTrainer

    trainer = object.__new__(RLTrainer)
    trainer.config = RLConfig()
    trainer.config.set("train.prompt_sharing", sharing)
    trainer.config.set("train.micro_tokens", 64)
    trainer.config.set("loss.observation_weight", weight)
    trainer.parallel, trainer.device, trainer.pad_id = Parallel(fsdp=False), "cpu", 0
    trainer.model = MODELS["gpt2"]()
    trainer.head, trainer.head_weight = output_head(trainer.model), LowPrecisionWeight()
    return trainer


def _step(trainer, groups):
    metrics = trainer._train_step(groups)
    return metrics, {n: p.grad.clone() for n, p in trainer.model.named_parameters() if p.grad is not None}


def test_rl_off_is_bitwise_unchanged():
    plain, plain_grads = _step(_trainer(0.0), _groups(0, observations=False))
    off, off_grads = _step(_trainer(0.0), _groups(0, observations=True))
    assert plain == off
    assert plain_grads.keys() == off_grads.keys()
    assert all(torch.equal(plain_grads[n], off_grads[n]) for n in plain_grads)


def _echo_by_hand(model, groups) -> tuple[torch.Tensor, float]:
    """λ x the mean over trajectories of (summed target NLL / observation length), differentiable; and the mean
    NLL per target token."""
    terms, total, count = [], 0.0, 0
    for t in (t for g in groups for t in g):
        ids = torch.tensor(t.prompt_ids + t.tokens)
        logp = torch.log_softmax(model(input_ids=ids[None]).logits[0].float(), -1)
        nll = torch.stack([-logp[len(t.prompt_ids) + j - 1, t.tokens[j]] for j in t.observation_targets])
        terms.append(nll.sum() / t.observation_tokens)
        total, count = total + float(nll.sum()), count + nll.numel()
    return LAMBDA * torch.stack(terms).mean(), total / count


@pytest.mark.parametrize("sharing", ["off", "on"])
def test_rl_echo_term_is_lambda_times_the_mean_observation_nll(sharing):
    groups = _groups(0, observations=True)
    base, base_grads = _step(_trainer(0.0, sharing), groups)
    echo_trainer = _trainer(LAMBDA, sharing)
    echo, echo_grads = _step(echo_trainer, groups)
    model = _trainer(0.0).model  # the same initial weights
    expected, nll = _echo_by_hand(model, groups)
    assert abs((echo["loss"] - base["loss"]) - float(expected)) < 1e-5 * float(expected)
    assert abs(echo["loss/observation"] - nll) < 1e-5 * nll
    # its gradient is the hand-computed term's: the policy part is unchanged
    model.zero_grad()
    expected.backward()
    for name, grad in base_grads.items():
        hand = dict(model.named_parameters())[name].grad
        assert torch.allclose(echo_grads[name] - grad, hand, rtol=1e-4, atol=1e-7), name
    assert echo["observation_tokens"] == sum(len(t.observation_targets) for g in groups for t in g)
    for key in ("policy/abs_ratio_dev", "trained_tokens", "policy/entropy"):
        assert abs(echo[key] - base[key]) <= 1e-6 * max(1.0, abs(base[key])), key


def test_observation_targets_stay_with_their_segment():
    t = Trajectory({}, 0, [1, 2], [])
    t.append_generated([5, 6], [-1.0, -1.0], 0)
    t.append_observations(len(t.tokens), [(1, 1), (2, 2)])
    t.append_context([7, 8, 9])
    t.rewrite_context([1, 2, 5, 6, 7, 8, 9], [], [])
    t.append_generated([10], [-1.0], 0)
    first, last = t.training_views()
    assert first.tokens == [5, 6, 7, 8] and first.observation_targets == [3]
    assert last.observation_targets == [] and t.observation_tokens == 2
    assert t.observation_target_count == 1


# ----------------------------------------------------------------- RL: a whole run


class LookEnv:
    """A text-protocol environment: every reply is an action, answered with what the agent "sees"; the harness's
    note before the output (declared in observation_notes) is not a target."""

    observation_notes = (r"\A\[note\]",)

    def reset(self, **row):
        self.step = 0

    def text_actions(self, content):
        return [("look", {})]

    def look(self) -> str:
        """Look around."""
        self.step += 1
        return f"[note] the room has {self.step} doors"


@pytest.mark.parametrize("fsdp", [False, True], ids=["one-gpu", "fsdp2"])
def test_rl_run_with_a_moe_policy_routing_replay_and_fsdp(tmp_path, fsdp):
    """pgs rl with loss.observation_weight on a MoE policy (routing replayed) over a multi-turn environment, plain
    and under FSDP2: the rollouts record their observation targets and every step logs the term."""
    import json

    from test_rl import tokenizer

    from palingenesis.moe import tiny_moe
    from palingenesis.rl.trainer import RLTrainer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    policy = tiny_moe(
        "qwen3_moe",
        vocab_size=len(tok),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    policy.save_pretrained(tmp_path / "policy")
    tok.save_pretrained(tmp_path / "policy")
    rows = [{"prompt": f"Count the doors of room {i}."} for i in range(12)]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = RLConfig()
    for key, value in {
        "model.policy": str(tmp_path / "policy"),
        "model.use_liger_kernel": False,
        "model.chat_template_kwargs": {"enable_thinking": False},
        "data.dataset": str(tmp_path / "data.jsonl"),
        "data.eval_size": 0,
        "env.max_turns": 3,
        "rollout.backend": "hf",
        "rollout.batch_prompts": 2,
        "rollout.group_size": 2,
        "rollout.max_new_tokens": 6,
        "rollout.routing_replay": "auto",
        "loss.observation_weight": LAMBDA,
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 2,
        "train.warmup_steps": 1,
        "train.save_steps": 0,
        "train.fsdp": fsdp,
    }.items():
        config.set(key, value)
    trainer = RLTrainer(config, rewards=[lambda completion: len(completion) / 40], env=LookEnv)
    generate = trainer.engine.generate

    def short_turns(*args, **kwargs):  # every turn ends as if on a stop token: a random policy plays on
        rollouts = generate(*args, **kwargs)
        for rollout in rollouts:
            rollout.finish_reason = "stop"
        return rollouts

    trainer.engine.generate = short_turns
    logged, seen = [], []
    trainer._log = lambda kind, metrics, step: logged.append((kind, dict(metrics)))
    step = trainer._train_step
    trainer._train_step = lambda groups: (seen.extend(t for g in groups for t in g), step(groups))[1]
    trainer.train()
    steps = [m for kind, m in logged if kind == "step"]
    assert len(steps) == 2
    for metrics in steps:
        assert metrics["observation_tokens"] > 0 and 0 < metrics["loss/observation"] < 30
    for t in seen:
        assert t.routing is not None
        text = tok.decode([t.tokens[j] for j in t.observation_targets])
        read = [m["content"] for m in t.messages[1:] if m["role"] == "user"]  # the results, after the prompt
        assert read and text == "".join(r.removeprefix("[note]") for r in read), (text, read)
