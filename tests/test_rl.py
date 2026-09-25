"""RL: objectives (exact gradients), advantages, rewards, chat templates, sandbox, the
multi-turn rollout loop, and the trainer end to end on CPU."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, "src")

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from palingenesis.rl.config import RLConfig, RLConfigError, RLLossConfig  # noqa: E402
from palingenesis.rl.losses import assign_advantages, policy_loss, target_logprobs, token_weights  # noqa: E402
from palingenesis.rl.trajectory import Trajectory, group_is_informative  # noqa: E402


def tokenizer(name):
    try:
        return transformers.AutoTokenizer.from_pretrained(name)
    except Exception as e:  # noqa: BLE001 — offline
        pytest.skip(f"tokenizer {name} unavailable: {e}")


# ----------------------------------------------------------------- objectives


def test_target_logprobs_match_autograd():
    torch.manual_seed(0)
    head = torch.nn.Linear(16, 50, bias=False)
    hidden = torch.randn(37, 16, requires_grad=True)
    targets = torch.randint(0, 50, (37,))
    g = torch.randn(37)

    lp, ent = target_logprobs(hidden, head, targets, entropy=True)
    (lp * g).sum().backward()
    ours = hidden.grad.clone(), head.weight.grad.clone()

    hidden.grad, head.weight.grad = None, None
    logp = torch.log_softmax(head(hidden), -1)
    ref = logp.gather(1, targets[:, None]).squeeze(1)
    (ref * g).sum().backward()
    assert torch.allclose(lp, ref.detach(), atol=1e-5)
    assert torch.allclose(ent, -(logp.exp() * logp).sum(-1).detach(), atol=1e-5)
    assert torch.allclose(ours[0], hidden.grad, atol=1e-5) and torch.allclose(ours[1], head.weight.grad, atol=1e-5)


def _loss_grad(config, lp0, behaviour, advantage, seq, n_seq):
    lp = lp0.clone().requires_grad_(True)
    seq_len = torch.bincount(seq, minlength=n_seq).float()
    weight = torch.ones_like(lp)
    loss, stats = policy_loss(lp, behaviour, advantage, weight, seq, n_seq, seq_len, config)
    loss.backward()
    return lp.grad, stats


def test_masked_is_on_policy_is_reinforce_and_masks_like_ppo():
    config = RLLossConfig(seq_mask=0.0)
    lp = torch.tensor([-1.0, -2.0, -0.5, -0.3])
    seq = torch.tensor([0, 0, 1, 1])
    advantage = torch.tensor([1.0, 1.0, -2.0, -2.0])
    grad, _ = _loss_grad(config, lp, lp.clone(), advantage, seq, 2)  # on-policy: ratio 1
    assert torch.allclose(grad, -advantage)
    behaviour = lp - torch.tensor([0.5, 0.0, 0.0, -0.5])  # ratios e^0.5, 1, 1, e^-0.5
    grad, stats = _loss_grad(config, lp, behaviour, advantage, seq, 2)
    assert grad[0] == 0  # A > 0 and ratio > 1 + eps_high: masked
    assert grad[3] == 0  # A < 0 and ratio < 1 - eps_low: masked
    assert torch.allclose(grad[1:3], -advantage[1:3]) and stats["clipped"] == 2


def test_cispo_icepop_gspo_and_sequence_mask():
    lp = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    seq = torch.tensor([0, 0, 1, 1])
    advantage = torch.tensor([1.0, 1.0, 1.0, 1.0])
    far = lp - torch.tensor([2.0, 2.0, 0.0, 0.0])  # seq 0 ratio e^2 ≈ 7.4
    grad, _ = _loss_grad(RLLossConfig(type="cispo", seq_mask=0.0), lp, far, advantage, seq, 2)
    assert torch.allclose(grad[:2], torch.full((2,), -5.0))  # capped at is_cap
    grad, _ = _loss_grad(RLLossConfig(type="icepop", seq_mask=0.0), lp, far, advantage, seq, 2)
    assert (grad[:2] == 0).all() and torch.allclose(grad[2:], -advantage[2:])
    grad, _ = _loss_grad(RLLossConfig(type="gspo", seq_mask=0.0), lp, lp.clone(), advantage, seq, 2)
    assert torch.allclose(grad, torch.full((4,), -0.5))  # s · A / |y|
    grad, stats = _loss_grad(RLLossConfig(seq_mask=0.1), lp, far, advantage, seq, 2)
    assert (grad[:2] == 0).all() and stats["seq_masked"] == 1  # whole sequence dropped


def _traj(reward, tokens=3, scored=True, trained=True, group=0):
    t = Trajectory({}, group, [1], [], tokens=[5] * tokens, mask=[True] * tokens, logprobs=[0.0] * tokens)
    t.reward, t.scored, t.trained = reward, scored, trained
    return t


def test_advantages_and_aggregation_weights():
    groups = [[_traj(1.0), _traj(0.0), _traj(0.0, scored=False, trained=False)], [_traj(2.0, 5), _traj(2.0, 1)]]
    assign_advantages(groups, "none")
    assert [t.advantage for t in groups[0]] == [0.5, -0.5, 0.0] and [t.advantage for t in groups[1]] == [0, 0]
    assert group_is_informative(groups[0]) and not group_is_informative(groups[1])
    assign_advantages(groups, "batch")  # centered [0.5, -0.5, 0, 0]: batch std sqrt(0.125)
    assert abs(groups[0][0].advantage - 0.5 / 0.125**0.5) < 1e-4 and groups[1][0].advantage == 0
    w = token_weights(groups, "prompt", budget=10)
    # prompt: each group's trained tokens share 1/prompts
    assert abs(sum(w[id(t)] * t.sampled_tokens for t in groups[0] if id(t) in w) - 0.5) < 1e-9
    assert abs(sum(w[id(t)] * t.sampled_tokens for t in groups[1]) - 0.5) < 1e-9
    w = token_weights(groups, "token", budget=10)
    assert abs(sum(w[id(t)] * t.sampled_tokens for g in groups for t in g if id(t) in w) - 1.0) < 1e-9


# --------------------------------------------------------------------- config


def test_config_validation(tmp_path):
    path = tmp_path / "rl.yaml"
    path.write_text("model: {policy: m}\ndata: {dataset: d.jsonl}\nrewards: {correct: math}\n")
    config = RLConfig.from_yaml(path)
    assert config.rewards["correct"].fn == "math" and config.validate() == []
    config.set("loss.type", "gspo")
    config.set("loss.aggregation", "token")
    with pytest.raises(RLConfigError, match="gspo"):
        config.validate()
    config = RLConfig.from_yaml(path)
    config.set("rewards.tests.fn", "code")
    config.set("sandbox.backend", "subprocess")
    with pytest.raises(RLConfigError, match="allow_unsafe"):
        config.validate()
    with pytest.raises(Exception, match="did you mean"):
        config.set("rollout.group_sise", 4)


# -------------------------------------------------------------------- rewards


def test_reward_signatures():
    from palingenesis.rl.rewards import SKIPPED, SkipSample, resolve_rewards

    samples = [
        {"completion": "the answer is \\boxed{4}", "answer": "4", "prompt": [], "sandbox": None},
        {"completion": "it is 5", "answer": "4", "prompt": [], "sandbox": None},
    ]
    configured = {"math": SimpleNamespace(fn="math", weight=1.0, args={})}

    def per_sample(completion, answer):
        return None if "5" in completion else 1.0

    def batched(prompts, completions, answer):
        return [len(c) for c in completions]

    async def asynchronous(completion, **row):
        if "5" in completion:
            raise SkipSample("down")
        return True

    rewards = resolve_rewards(configured, [per_sample, (batched, 0.5), asynchronous])
    scores = {r.name: asyncio.run(r.score(samples)) for r in rewards}
    assert scores["math"] == [1.0, 0.0]
    assert scores["per_sample"] == [1.0, None]
    assert scores["batched"] == [float(len(samples[0]["completion"])), 7.0]
    assert scores["asynchronous"] == [1.0, SKIPPED]


# ---------------------------------------------------------------- chat / tools


def test_tool_calls_parse_and_render_token_exact():
    from palingenesis.opd.teachers import end_of_turn_id
    from palingenesis.rl.chat import ChatFormat, encode_prompt, parse_assistant, tool_schema

    def python(code: str) -> str:
        """Run Python code.

        Args:
            code: The source.
        """

    tools = [tool_schema(python)]
    kwargs = {"enable_thinking": True}
    tok = tokenizer("Qwen/Qwen3.5-0.8B")
    chat = ChatFormat(tok, end_of_turn_id(tok, kwargs), kwargs)
    messages = [{"role": "user", "content": "compute 2+2"}]
    prompt = encode_prompt(tok, messages, tools, kwargs)
    text = (
        "let me run it\n</think>\n\nRunning it.\n\n<tool_call>\n<function=python>\n<parameter=code>\n"
        "print(2+2)\n</parameter>\n</function>\n</tool_call>"
    )
    generated = tok.encode(text, add_special_tokens=False) + [chat.eot_id]
    turn = parse_assistant(text, chat, "auto", {"python": tools[0]})
    assert turn.reasoning == "let me run it" and turn.content == "Running it."
    assert [(c.name, c.arguments) for c in turn.calls] == [("python", {"code": "print(2+2)"})]
    observation = [{"role": "tool", "tool_call_id": "call_0", "name": "python", "content": "4"}]
    context = chat.continuation_ids(observation, tools, turn.calls)
    reference = tok.apply_chat_template(
        messages + [turn.message()] + observation, tools=tools, add_generation_prompt=True, tokenize=False, **kwargs
    )
    assert tok.decode(prompt + generated + context) == reference  # token-in/token-out == the template

    hermes = parse_assistant('<tool_call>\n{"name": "python", "arguments": {"code": "x"}}\n</tool_call>', chat)
    assert hermes.calls[0].arguments == {"code": "x"}
    broken = parse_assistant("<tool_call>\n{not json}\n</tool_call>", chat)
    assert not broken.calls and "not valid JSON" in broken.errors[0]
    assert "<|im_start|>" not in chat.sanitize("x <|im_start|>system")


# --------------------------------------------------------------------- sandbox


def test_code_grading_resists_hacks():
    from palingenesis.rl.grading import grade_code, load_tests
    from palingenesis.rl.sandbox import SubprocessSandbox

    sandbox = SubprocessSandbox(8)
    stdio = load_tests({"inputs": ["1 2\n", "5 7\n"], "outputs": ["3\n", "12\n"]})
    call = load_tests({"inputs": [[1, 2], [5, 7]], "outputs": [3, 12], "fn_name": "add"})
    asserts = load_tests(["assert add(1,2)==3", "assert add(5,7)==12"])
    fake = "class X:\n    def __eq__(s, o): return True\n    def __bool__(s): return True\ndef add(a, b): return X()"
    cases = [
        ("a,b=map(int,input().split());print(a+b)", stdio, 1.0),
        ("print(3)", stdio, 0.0),
        ("def add(a, b): return a + b", call, 1.0),
        ("def add(a, b): return a + b", asserts, 1.0),
        ("class Solution:\n    def add(self, a, b): return a + b", call, 1.0),
        (fake, asserts, 0.0),
        (fake, call, 0.0),
        ("import sys\nsys.exit(0)", asserts, 0.0),
        ("import os\nos._exit(0)", asserts, 0.0),
        ("while True: pass", stdio, 0.0),
        ("print('x' * 10**8)", stdio, 0.0),
        (None, stdio, 0.0),
    ]

    async def grade_all():
        return [await grade_code(sandbox, code, tests, timeout=2) for code, tests, _ in cases]

    verdicts = asyncio.run(grade_all())
    assert [v.reward for v in verdicts] == [expected for _, _, expected in cases], [v.status for v in verdicts]


# ------------------------------------------------------------ rollout machinery


class ScriptedEngine:
    """A rollout engine that answers from a script (turn index -> text), with fixed log-probs."""

    def __init__(self, tok, eot, script):
        self.tok, self.eot, self.script, self.version, self.calls = tok, eot, script, 0, []

    def generate(self, prompts, max_new_tokens, temperature):
        from palingenesis.opd.rollout import Rollout

        self.calls.append(len(prompts))
        out = []
        for prompt in prompts:
            turn = self.tok.decode(prompt).count("<|im_start|>assistant") - 1
            ids = self.tok.encode(self.script[min(turn, len(self.script) - 1)], add_special_tokens=False) + [self.eot]
            out.append(Rollout(ids, [-0.25] * len(ids), "stop", self.version))
        return out

    def update_weights(self, named, version):
        self.version = version

    def sleep(self):
        pass

    def wake(self):
        pass


def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: first
        b: second
    """
    return a + b


def test_multi_turn_rollout_is_token_exact_and_masked():
    from palingenesis.opd.orchestrator import PublishedWeights
    from palingenesis.opd.teachers import end_of_turn_id
    from palingenesis.rl.chat import ChatFormat
    from palingenesis.rl.data import PromptSampler
    from palingenesis.rl.env import EnvPool, ToolEnv
    from palingenesis.rl.pipeline import RLPipeline
    from palingenesis.rl.rewards import resolve_rewards

    tok = tokenizer("Qwen/Qwen3-0.6B")
    kwargs = {"enable_thinking": False}
    eot = end_of_turn_id(tok, kwargs)
    chat = ChatFormat(tok, eot, kwargs)
    script = [
        'Let me add.\n<tool_call>\n{"name": "add", "arguments": {"a": 2, "b": 3}}\n</tool_call>',
        "The sum is \\boxed{5}.",
    ]
    engine = ScriptedEngine(tok, eot, script)
    config = RLConfig()
    for key, value in {
        "model.policy": "x",
        "model.chat_template_kwargs": kwargs,
        "env.type": "tools",
        "env.tools": ["tests.test_rl:add"],
        "rollout.group_size": 2,
        "rollout.batch_prompts": 2,
        "rollout.max_new_tokens": 64,
        "rollout.max_model_len": 4096,
    }.items():
        config.set(key, value)
    rows = [{"prompt": "What is 2 + 3?", "answer": "5"}, {"prompt": "What is 2 + 3 really?", "answer": "5"}]
    pool = EnvPool(lambda: ToolEnv([add]))
    pipeline = RLPipeline(
        tok,
        chat,
        engine,
        PublishedWeights(torch.nn.Linear(1, 1)),
        config,
        PromptSampler(rows),
        resolve_rewards({"m": SimpleNamespace(fn="math", weight=1.0, args={})}),
        pool,
        None,
        (eot,),
    )
    try:
        groups = pipeline._await(pipeline._eval(rows, 1.0))
    finally:
        pipeline.close()
    t = groups[0][0]
    assert t.finish == "stop" and t.turns == 2 and t.tool_calls == 1 and t.tool_errors == 0
    assert t.rewards == {"m": 1.0} and t.reward == 1.0
    assert [m["role"] for m in t.messages] == ["user", "assistant", "tool", "assistant"]
    assert t.messages[2]["content"] == "5"
    sampled = [tok for tok, m in zip(t.tokens, t.mask) if m]
    assert tok.decode(sampled).count("<|im_end|>") == 2  # both turns, trained
    context = tok.decode([tok_ for tok_, m in zip(t.tokens, t.mask) if not m])
    assert "<tool_response>\n5\n</tool_response>" in context and "boxed" not in context
    assert all(lp == -0.25 for lp, m in zip(t.logprobs, t.mask) if m)
    assert engine.calls == [2, 2]  # both rollouts' turns batched into one engine call each


# ------------------------------------------------------------------ end to end


def tiny_policy(tmp_path):
    tok = tokenizer("Qwen/Qwen3-0.6B")
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    transformers.Qwen3ForCausalLM(config).save_pretrained(tmp_path / "policy")
    tok.save_pretrained(tmp_path / "policy")
    return str(tmp_path / "policy")


@pytest.mark.parametrize("fsdp", [False, True], ids=["single", "fsdp2"])
def test_trainer_end_to_end_and_resume(tmp_path, fsdp, monkeypatch):
    from palingenesis.rl.trainer import TRAINER_STATE_FILE, RLTrainer

    rows = [{"prompt": f"What is {i} plus {i + 1}?", "answer": str(2 * i + 1)} for i in range(30)]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def length(completion):  # dense, so a random policy's groups disagree
        return len(completion) / 40

    def make(steps, resume=""):
        config = RLConfig()
        for key, value in {
            "model.policy": tiny_policy(tmp_path),
            "model.use_liger_kernel": False,
            "model.chat_template_kwargs": {"enable_thinking": False},
            "data.dataset": str(tmp_path / "data.jsonl"),
            "data.eval_size": 3,
            "rollout.backend": "hf",
            "rollout.batch_prompts": 3,
            "rollout.group_size": 3,
            "rollout.max_new_tokens": 8,
            "rewards.m.fn": "math",
            "rewards.m.weight": 0.0,
            "train.output_dir": str(tmp_path / "run"),
            "train.steps": steps,
            "train.learning_rate": 1e-3,
            "train.warmup_steps": 1,
            "train.save_steps": 2,
            "train.resume_from": resume,
            "train.fsdp": fsdp,
            # full activation checkpointing (the cpu_offload default) recomputes every layer
            "model.gradient_checkpointing": "full" if fsdp else "none",
            "loss.overlong_buffer": 2,
        }.items():
            config.set(key, value)
        return RLTrainer(config, rewards=[length])

    def local(p):  # FSDP2 parameters are DTensors; on one rank the local shard is the whole tensor
        p = p.detach()
        return p.to_local() if hasattr(p, "to_local") else p

    trainer = make(2)
    before = [local(p).clone() for p in trainer.model.parameters()]
    trainer.train()
    after = [local(p) for p in trainer.model.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    assert (tmp_path / "run" / "step_2" / TRAINER_STATE_FILE).exists() and (tmp_path / "run" / "final").exists()

    resumed = make(3, resume="auto")
    assert resumed.start_step == 2 and resumed.weights.version == 2
    resumed.train()


@pytest.mark.parametrize("multiplier,softcap", [(1.0, None), (0.5, 3.0)])
def test_fused_linear_logprobs_match_autograd(multiplier, softcap):
    from palingenesis.logits import PostProcessedHead

    torch.manual_seed(1)
    linear = torch.nn.Linear(16, 50, bias=False)
    head = PostProcessedHead(linear, multiplier, softcap) if softcap else linear
    hidden = torch.randn(23, 16, requires_grad=True)
    targets = torch.randint(0, 50, (23,))
    g = torch.randn(23)
    lp, ent = target_logprobs(hidden, head, targets, entropy=True)  # the fused linear path
    (lp * g).sum().backward()
    ours = hidden.grad.clone(), linear.weight.grad.clone()
    hidden.grad, linear.weight.grad = None, None
    logp = torch.log_softmax(head(hidden), -1)
    ref = logp.gather(1, targets[:, None]).squeeze(1)
    (ref * g).sum().backward()
    assert torch.allclose(lp, ref.detach(), atol=1e-5)
    assert torch.allclose(ent, -(logp.exp() * logp).sum(-1).detach(), atol=1e-5)
    assert torch.allclose(ours[0], hidden.grad, atol=1e-5) and torch.allclose(ours[1], linear.weight.grad, atol=1e-5)


@pytest.mark.parametrize("arch", ["gpt2", "qwen35_hybrid"])
def test_prompt_sharing_gives_the_same_gradients(arch):
    """A group's prompt forwarded once (tree forward/backward) or once per rollout: the same
    loss statistics and the same parameter gradients."""
    import random as _random

    from test_seco import MODELS

    from palingenesis.logits import output_head
    from palingenesis.rl.losses import LowPrecisionWeight
    from palingenesis.rl.parallel import Parallel
    from palingenesis.rl.trainer import RLTrainer

    def groups(seed):
        rng = _random.Random(seed)
        out = []
        for g in range(3):
            prompt = [rng.randrange(1, 90) for _ in range(rng.randint(5, 30))]
            group = []
            for _ in range(4):
                n = rng.randint(3, 25)
                t = Trajectory(
                    {},
                    g,
                    prompt,
                    [],
                    tokens=[rng.randrange(1, 90) for _ in range(n)],
                    mask=[rng.random() < 0.8 for _ in range(n)],
                    logprobs=[-rng.random() * 3 for _ in range(n)],
                )
                t.mask[0] = True
                t.reward = rng.random()
                group.append(t)
            out.append(group)
        return out

    results = []
    for sharing in ("off", "on"):
        trainer = object.__new__(RLTrainer)
        trainer.config = RLConfig()
        trainer.config.set("train.prompt_sharing", sharing)
        trainer.config.set("train.micro_tokens", 64)
        trainer.parallel, trainer.device, trainer.pad_id = Parallel(fsdp=False), "cpu", 0
        trainer.model = MODELS[arch]()
        trainer.head, trainer.head_weight = output_head(trainer.model), LowPrecisionWeight()
        metrics = trainer._train_step(groups(0))
        grads = [p.grad.clone() for p in trainer.model.parameters() if p.grad is not None]
        results.append((metrics, grads))
    (plain, plain_grads), (tree, tree_grads) = results
    for key in ("loss", "policy/abs_ratio_dev", "trained_tokens", "policy/entropy"):
        assert abs(plain[key] - tree[key]) <= 1e-6 * max(1.0, abs(plain[key])), (
            key,
            plain[key],
            tree[key],
        )  # fp32 log-probs
    assert len(plain_grads) == len(tree_grads)
    for a, b in zip(plain_grads, tree_grads):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-8), (a - b).abs().max()


def test_checkpoint_named_parameters_takes_gathered_tensors():
    """The FSDP weight push passes full tensors gathered from the shards in place of the
    model's own (sharded) parameters."""
    from palingenesis.opd.rollout import checkpoint_named_parameters

    model = torch.nn.Sequential(torch.nn.Linear(2, 3))
    full = [(name, torch.full_like(p, 7.0)) for name, p in model.named_parameters()]
    pushed = dict(checkpoint_named_parameters(model, iter(full)))
    assert sorted(pushed) == sorted(name for name, _ in full)
    assert all(bool((t == 7.0).all()) for t in pushed.values())
    own = dict(checkpoint_named_parameters(model))
    assert torch.equal(own["0.weight"], model[0].weight.detach())
