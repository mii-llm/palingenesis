"""train.packing varlen / tree for on-policy distillation (palingenesis.opd.packing over tree_packing).

The central property: with the same rollouts, the teacher's scores (hidden states, log-probs, top-k), every
loss's value and statistics, and the student's gradients are those of the padded path, on float64 tiny Qwen3
and Qwen3.5 (Gated DeltaNet) models: groups of completions, several teachers (shared vocabulary and
cross-tokenizer), privileged teacher prompts and tool views, and agent traces with onset branches.
"""

import json
import random
import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "tests")

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from test_opd_trainer import tiny_model  # noqa: E402
from test_tree_packing import _f64_delta_rule  # noqa: E402

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look a number up.",
            "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        },
    }
]


@pytest.fixture(autouse=True)
def small_trees(request, monkeypatch):
    """A row becomes a tree when that saves a tree level's cost (TREE_MIN_SAVED forwarded tokens), more than these
    tests' prompts hold: lowered, so their rows are trees."""
    if "worth_a_tree_level" not in request.node.name:
        monkeypatch.setattr("palingenesis.tree_packing.TREE_MIN_SAVED", 8)


def tiny_gdn_model(tmp_path, tokenizer_name, name, seed):
    """A tiny Qwen3.5 (Gated DeltaNet and full attention layers) on the real tokenizer."""
    try:
        tok = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:  # noqa: BLE001 — offline or not cached
        pytest.skip(f"tokenizer {tokenizer_name} unavailable: {e}")
    torch.manual_seed(seed)
    config = transformers.Qwen3_5TextConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        max_position_embeddings=1024,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    model = transformers.Qwen3_5ForCausalLM(config)
    with torch.no_grad():  # the random init makes the recurrence nearly trivial: spread it
        for key, p in model.named_parameters():
            if key.endswith(("A_log", "dt_bias")):
                p.uniform_(-1.0, 1.0)
    path = tmp_path / name
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("opd_packing_models")
    return {
        "qwen3": {
            "student": tiny_model(tmp, "Qwen/Qwen3-0.6B", "student", 0),
            "same": tiny_model(tmp, "Qwen/Qwen3-0.6B", "teacher_same", 1),
            "other": tiny_model(tmp, "Qwen/Qwen3.5-0.8B", "teacher_other", 2),
            "same2": tiny_model(tmp, "Qwen/Qwen3-0.6B", "teacher_same2", 6),
        },
        "qwen35": {
            "student": tiny_gdn_model(tmp, "Qwen/Qwen3.5-0.8B", "gdn_student", 3),
            "same": tiny_gdn_model(tmp, "Qwen/Qwen3.5-0.8B", "gdn_teacher", 4),
            "other": tiny_model(tmp, "Qwen/Qwen3-0.6B", "gdn_teacher_other", 5),
            "same2": tiny_gdn_model(tmp, "Qwen/Qwen3.5-0.8B", "gdn_teacher2", 7),
        },
    }


def write_prompts(path, n=16):
    rows = [
        {
            "messages": [
                {"role": "system", "content": "You add numbers. Answer with the number only, nothing else."},
                {"role": "user", "content": f"What is {i} plus {i + 1}? " + "Think. " * (i % 4)},
            ]
        }
        for i in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def make_config(tmp_path, paths, packing="tree", **overrides):
    from palingenesis.opd.config import OPDConfig

    config = OPDConfig()
    prompts = write_prompts(tmp_path / "prompts.jsonl")
    settings = {
        "model.student": paths["student"],
        "model.chat_template_kwargs": {"enable_thinking": False},
        "model.use_liger_kernel": False,
        "teachers.same.model": paths["same"],
        "teachers.other.model": paths["other"],
        "sources.math.path": prompts,
        "sources.math.teacher": "same",
        "sources.math.max_new_tokens": 10,
        "sources.math.dev_size": 4,
        "sources.chat.path": prompts,
        "sources.chat.teacher": "other",
        "sources.chat.max_new_tokens": 10,
        "sources.chat.dev_size": 4,
        "rollout.batch_prompts": 4,
        "rollout.group_size": 4,
        "loss.xtok_dense_weight": 0.5,
        "loss.top_k": 6,
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 2,
        "train.learning_rate": 1e-3,
        "train.warmup_steps": 1,
        "train.eval_every": 1,
        "train.eval_samples": 4,
        "train.score_micro_seqs": 4,
        "train.packing": packing,
    }
    settings.update(overrides)
    for key, value in settings.items():
        config.set(key, value, "test")
    return config


def f64(trainer):
    """The student and the hf teachers in float64, the Gated DeltaNet rule too (transformers' torch kernel computes
    in float32), so packed and padded agree to float64 rounding."""
    for model in [trainer.student] + [route.teacher.model for route in trainer.routes.values()]:
        model.double()
        for module in model.modules():
            # transformers' RMSNorm computes in float32 whatever the model's dtype: a hidden state then sits on a
            # float32 grid, and a 1e-16 difference before the norm is now and then a whole float32 step after it
            if type(module).__name__ == "Qwen3RMSNorm":
                module.forward = lambda x, m=module: m.weight * (
                    x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + m.variance_epsilon)
                )
            elif type(module).__name__ == "Qwen3_5RMSNorm":
                module.forward = lambda x, m=module: m._norm(x) * (1.0 + m.weight)
            elif type(module).__name__.endswith("RotaryEmbedding"):
                # (float32 cos / sin tables: a vectorized float32 cos is not the same to the last bit at every
                # offset of a row, and a packed row puts a position elsewhere than a padded one)
                def rotary(x, position_ids, m=module):
                    if position_ids.ndim == 3:  # Qwen3.5: one row of position ids per axis, all the same for text
                        position_ids = position_ids[0]
                    freqs = position_ids[:, :, None].double() * m.inv_freq.double()[None, None, :]
                    emb = torch.cat((freqs, freqs), dim=-1)
                    return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

                module.forward = rotary
            route = module.__dict__.get("_pgs_packed_route")
            if route is not None:
                route.delta_rule, route.varlen_kernel = _f64_delta_rule, False
            elif "chunk_gated_delta_rule" in module.__dict__:
                module.chunk_gated_delta_rule = _f64_delta_rule
    trainer.fused = set()  # the fused kernel works in float32: the float64 reference path (its own test below)
    return trainer


def set_mode(trainer, mode):
    trainer.config.train.packing = mode
    for route in trainer.routes.values():
        route.teacher.packing = trainer._packing()


def requests_for(trainer, metas=None, group=4, prompts=4):
    """`prompts` prompts per teacher, `group` rollouts each; `metas`: per teacher, the rows' meta."""
    out = []
    for name in trainer.routes:
        for i in range(prompts):
            messages = [
                {"role": "system", "content": "You add numbers. Answer with the number only."},
                {"role": "user", "content": f"What is {i} plus {3 * i + 1}? " + "Slowly. " * i},
            ]
            meta = dict((metas or {}).get(name, {}))
            if "teacher_messages" in meta:
                meta["teacher_messages"] = messages[:1] + [
                    {"role": "user", "content": messages[1]["content"] + f"\n\nThe answer is {4 * i + 1}."}
                ]
            source = next(s for s, c in trainer.config.sources.items() if c.teacher == name)
            out += [trainer.pipeline.request(messages, 6 + 3 * i, meta, source, name)] * group
    return out


def scores_of(trainer, batch):
    return {
        name: route.teacher.score(
            [s.view for s in batch.samples if s.teacher == name],
            top_k=route.top_k,
            keep_hidden=route.keep_hidden,
            micro_seqs=3,
            sample_rounds=route.sample_rounds,
        )
        for name, route in trainer.routes.items()
    }


def step(trainer, batch, train=True):
    trainer.student.zero_grad()
    stats = dict(trainer._scores(batch, train=train))
    grads = {n: p.grad.clone() for n, p in trainer.student.named_parameters() if p.grad is not None}
    return stats, grads


def assert_close_stats(stats, ref, tol, where):
    assert stats.keys() == ref.keys()
    for key, want in ref.items():
        assert abs(stats[key] - want) <= tol * max(1.0, abs(want)), (where, key, stats[key], want)


def assert_close_grads(grads, ref, tol, where):
    assert grads.keys() == ref.keys()
    for key, want in ref.items():
        error = (grads[key] - want).abs().max() / want.abs().max().clamp_min(1e-30)
        assert error <= tol, (where, key, float(error))


def assert_same_step(trainer, batch, tol=1e-10, grad_tol=1e-6, teacher_tol=1e-10):
    """Teacher scores, loss statistics and gradients of `batch` in every mode against padded."""
    set_mode(trainer, "padded")
    ref_scores = scores_of(trainer, batch)
    for name, scores in ref_scores.items():
        for s, sc in zip([s for s in batch.samples if s.teacher == name], scores):
            s.scores = sc
    ref_stats, ref_grads = step(trainer, batch)
    assert ref_grads and all(v == v for v in ref_stats.values())
    for mode in ("varlen", "tree"):
        set_mode(trainer, mode)
        for name, scores in scores_of(trainer, batch).items():
            for got, want in zip(scores, ref_scores[name]):
                for field in ("hidden", "token_lp", "topk_lp"):
                    a, b = getattr(got, field), getattr(want, field)
                    assert (a is None) == (b is None)
                    if a is not None:  # (log-probs go through the teacher's float32 softmax)
                        limit = teacher_tol if field == "hidden" else max(teacher_tol, 2e-6)
                        assert torch.allclose(a, b, rtol=limit, atol=limit), (mode, name, field, (a - b).abs().max())
                if want.topk_ids is not None:
                    assert torch.equal(got.topk_ids, want.topk_ids)
        stats, grads = step(trainer, batch)
        assert_close_stats(stats, ref_stats, tol, mode)
        assert_close_grads(grads, ref_grads, grad_tol, mode)
        dev, _ = step(trainer, batch, train=False)  # the no-grad path of dev_kl
        assert_close_stats(dev, ref_stats, tol, mode)
    set_mode(trainer, "tree")


def shares_prompts(trainer, batch):
    """The tree rows of the student and of every teacher hold fewer tokens than their sequences."""
    packing = trainer._packing()
    for name in trainer.routes:
        samples = [s for s in batch.samples if s.teacher == name]
        for seqs in (
            [s.request.prompt_ids + s.completion[:-1] for s in samples],
            [s.view.input_ids[:-1] for s in samples],
        ):
            row = packing.row(seqs)
            assert not row.flat and row.num_tokens < 0.7 * sum(map(len, seqs)), (name, row)


LOSSES = [
    {},  # full_rkl (same vocabulary) and xtok with the dense term (other tokenizer)
    {"teachers.same.loss": "topk_kl"},
    {"teachers.same.loss": "sampled_rkl", "teachers.other.loss": "xtok", "loss.xtok_dense_weight": 0.0},
    {"loss.token_weighting": "sure", "loss.length_norm": True},
    # entropy keeps the highest-entropy tokens of each micro-batch: the same tokens when the micro-batches are the
    # same (here one per teacher in every mode)
    {"loss.token_weighting": "entropy", "train.score_micro_seqs": 64},
]
LOSS_IDS = ["full_rkl+xtok", "topk_kl", "sampled+xtok", "sure+length_norm", "entropy"]


@pytest.mark.parametrize("arch", ["qwen3", "qwen35"])
@pytest.mark.parametrize("losses", LOSSES, ids=LOSS_IDS)
def test_every_loss_is_the_same_padded_varlen_and_tree(tmp_path, models, arch, losses):
    from palingenesis.opd.trainer import OPDTrainer

    trainer = f64(OPDTrainer(make_config(tmp_path, models[arch], **losses)))
    set_mode(trainer, "padded")
    batch = trainer.pipeline.run(requests_for(trainer), 1.0)
    assert {s.teacher for s in batch.samples} == {"same", "other"} and len(batch.samples) == 32
    set_mode(trainer, "tree")
    shares_prompts(trainer, batch)
    assert_same_step(trainer, batch)


@pytest.mark.parametrize("arch", ["qwen3", "qwen35"])
def test_privileged_and_tools_views_share_the_teachers_own_prompt(tmp_path, models, arch):
    """The teacher reads another prompt than the student (privileged context; tool schemas only the student, or
    only the teacher, sees): each model's trie is built from its own sequences."""
    from palingenesis.opd.trainer import OPDTrainer

    trainer = f64(OPDTrainer(make_config(tmp_path, models[arch])))
    metas = {
        "same": {"teacher_messages": True, "tools": TOOLS, "teacher_tools": None},
        "other": {"teacher_messages": True, "teacher_tools": TOOLS},
    }
    set_mode(trainer, "padded")
    batch = trainer.pipeline.run(requests_for(trainer, metas), 1.0)
    for s in batch.samples:
        assert s.request.teacher_prompt_ids != s.request.prompt_ids
    same = next(s for s in batch.samples if s.teacher == "same")
    assert len(same.request.prompt_ids) > len(same.request.teacher_prompt_ids)  # the student's prompt lists the tool
    set_mode(trainer, "tree")
    shares_prompts(trainer, batch)
    assert_same_step(trainer, batch)


@pytest.mark.parametrize("arch", ["qwen3", "qwen35"])
def test_gradient_checkpointing_recomputes_the_packed_layers(tmp_path, models, arch):
    """model.gradient_checkpointing: the checkpointed layers get the packed layout again when they are recomputed."""
    from palingenesis.opd.trainer import OPDTrainer

    trainer = f64(OPDTrainer(make_config(tmp_path, models[arch])))
    set_mode(trainer, "padded")
    batch = trainer.pipeline.run(requests_for(trainer, prompts=2), 1.0)
    ref_stats, ref_grads = step(trainer, batch)
    trainer.student.gradient_checkpointing_enable()
    trainer.student.train()
    for mode in ("varlen", "tree"):
        set_mode(trainer, mode)
        stats, grads = step(trainer, batch)
        assert_close_stats(stats, ref_stats, 1e-10, mode)
        assert_close_grads(grads, ref_grads, 1e-6, mode)


def test_fused_kernel_reads_the_packed_rows(tmp_path, models):
    """full_rkl through the fused head kernel (float32 inside): the same loss and gradients from packed rows."""
    from palingenesis.opd.trainer import OPDTrainer

    trainer = OPDTrainer(make_config(tmp_path, models["qwen3"]))
    fused = set(trainer.fused)
    assert fused == {"same"}
    for route in trainer.routes.values():  # the kernel takes float32 hidden states of float32 heads
        route.teacher.model.float()
    set_mode(trainer, "padded")
    batch = trainer.pipeline.run(requests_for(trainer), 1.0)
    assert_same_step(trainer, batch, tol=2e-5, grad_tol=2e-3, teacher_tol=2e-5)
    assert trainer.fused == fused


def test_packed_micro_batches_keep_groups_and_the_budget():
    from palingenesis.opd.packing import Packing

    rng = random.Random(0)
    prompts = [[1000 + g] + [rng.randrange(5, 90) for _ in range(39)] for g in range(6)]
    seqs = [p + [rng.randrange(100, 190) for _ in range(rng.randint(5, 20))] for p in prompts for _ in range(4)]
    scored = [len(s) - 40 for s in seqs]
    budget = 4 * max(map(len, seqs))
    tree = Packing("tree").micro_batches(seqs, scored, micro_seqs=4)
    assert sorted(i for part in tree for i in part) == list(range(len(seqs)))
    for part in tree:  # the exact trie fits the budget
        assert Packing("tree").row([seqs[i] for i in part]).num_tokens <= budget
    varlen = Packing("varlen").micro_batches(seqs, scored, micro_seqs=4)
    assert all(sum(len(seqs[i]) for i in part) <= budget for part in varlen)
    assert len(tree) < len(varlen)  # shared prompts: more completions per row
    assert len(Packing("tree", tokens=10_000).micro_batches(seqs, scored, 4)) == 1
    # the gathered K/V rows are bounded too: the completions are spread when their paths exceed them
    assert len(Packing("tree", tokens=10_000, gather_tokens=600).micro_batches(seqs, scored, 4)) > 1


def test_a_row_is_a_tree_only_when_it_is_worth_a_tree_level():
    """Real TREE_MIN_SAVED: prompts too short to pay for a level by themselves are shared when the row's groups
    together do, and the few tokens completions start alike never make a level."""
    from palingenesis.opd.packing import Packing
    from palingenesis.tree_packing import TREE_MIN_SAVED, pack

    rng = random.Random(1)

    def groups(n, prompt_len, k=8):
        out = []
        for g in range(n):
            prompt = [1000 + g] + [rng.randrange(5, 90) for _ in range(prompt_len - 1)]
            out += [prompt + [7, 8] + [rng.randrange(100, 900) for _ in range(30)] for _ in range(k)]
        return out

    few = groups(1, 100)  # saves about 700 tokens: less than a level costs
    assert Packing("tree").row(few).flat
    many = groups(6, 100)  # six prompts on one level
    row = Packing("tree").row(many)
    assert len(row.levels) == 2 and row.num_tokens == sum(map(len, many)) - 6 * 7 * 102
    assert 7 * 102 < TREE_MIN_SAVED and pack(many, share=True).flat  # (the RL rule asks each run to pay by itself)
    for i, s in enumerate(many):
        assert row.tokens[row.positions(i)].tolist() == s
    assert Packing("varlen").row(many).flat


@pytest.mark.parametrize("arch,mode", [("qwen3", "tree"), ("qwen35", "tree"), ("qwen3", "varlen")])
def test_training_runs_end_to_end_with_packing(tmp_path, models, arch, mode):
    from palingenesis.opd.trainer import OPDTrainer

    trainer = OPDTrainer(make_config(tmp_path, models[arch], packing=mode))
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, step, metrics))
    trainer.train()
    steps = [m for kind, _, m in logged if kind == "step"]
    assert len(steps) == 2 and all(m["loss"] == m["loss"] and m["grad_norm"] > 0 for m in steps)
    evals = [m for kind, _, m in logged if kind == "eval"]
    assert evals and all(any(k.startswith("dev_kl") for k in m) for m in evals)


def test_packing_options_are_validated(tmp_path, models):
    from palingenesis.opd.config import OPDConfigError

    def errors(**overrides):
        with pytest.raises(OPDConfigError) as e:
            make_config(tmp_path, models["qwen3"], **overrides).validate()
        return str(e.value)

    assert "train.packing must be one of" in errors(**{"train.packing": "trie"})
    assert "only used with train.packing" in errors(**{"train.packing": "padded", "train.packing_tokens": 4096})
    assert "only used with train.packing tree" in errors(
        **{"train.packing": "varlen", "train.packing_gather_tokens": 4096}
    )
    assert "must be >= train.packing_tokens" in errors(
        **{"train.packing_tokens": 8192, "train.packing_gather_tokens": 4096}
    )
    assert "integer >= 0" in errors(**{"train.packing_tokens": -1})
    make_config(tmp_path, models["qwen3"], **{"train.packing_tokens": 4096}).validate()


def test_a_model_that_cannot_be_packed_is_a_config_error(tmp_path, models, monkeypatch):
    from palingenesis.opd import trainer as module
    from palingenesis.opd.config import OPDConfigError
    from palingenesis.tree_packing import PackingError

    def refuse(model, device):
        raise PackingError("the model uses sliding-window attention")

    monkeypatch.setattr(module, "support_for", refuse)
    with pytest.raises(OPDConfigError, match="train.packing tree: the student: .*sliding-window"):
        module.OPDTrainer(make_config(tmp_path, models["qwen3"]))


# ── agent traces: the trunk once under every regenerated turn and onset branch ──


def trace_config(tmp_path, paths, packing="tree", **overrides):
    from test_opd_trace_trainer import write_traces

    from palingenesis.opd.config import OPDConfig

    tmp_path.mkdir(parents=True, exist_ok=True)
    config = OPDConfig()
    settings = {
        "model.student": paths["student"],
        "model.chat_template_kwargs": {"enable_thinking": True},
        "model.use_liger_kernel": False,
        "teachers.coder.model": paths["same"],
        "teachers.general.model": paths["same2"],
        "sources.traces.format": "agent_traces",
        "sources.traces.path": write_traces(tmp_path / "traces.jsonl"),
        "sources.traces.dev_size": 3,
        "sources.traces.max_new_tokens": 6,
        "sources.traces.branches_per_trace": 3,
        "sources.traces.teacher": "general",
        "sources.traces.topic_field": "domain",
        "sources.traces.topic_teachers": {"coder": ["code", "search"]},
        "sources.loops.format": "agent_traces",
        "sources.loops.path": write_traces(tmp_path / "loops.jsonl", looping=True),
        "sources.loops.dev_size": 3,
        "sources.loops.max_new_tokens": 6,
        "sources.loops.branches_per_trace": 1,
        "sources.loops.onset_branches": 2,
        "sources.loops.onset_kd_tokens": 4,
        "sources.loops.teacher": "general",
        "rollout.batch_prompts": 6,
        "rollout.group_size": 2,
        "loss.trace_kd_weight": 0.5,
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 2,
        "train.learning_rate": 1e-3,
        "train.warmup_steps": 1,
        "train.eval_every": 1,
        "train.eval_samples": 3,
        "train.tree_chunk_size": 64,
        "train.packing": packing,
    }
    settings.update(overrides)
    for key, value in settings.items():
        config.set(key, value, "test")
    return config


def set_trace_mode(trainer, mode):
    trainer.config.train.packing = mode
    trainer.pipeline.packing = trainer._packing()


def naive_trace_step(trainer, samples):
    """The reference: every regenerated turn (its whole context and completion) and every trunk forwarded alone,
    the trainer's own loss terms on their hidden states; the statistics and the student's gradients."""
    from collections import defaultdict

    from palingenesis.logits import final_hidden_states
    from palingenesis.opd.trace_trainer import trace_sequences

    loss = trainer.config.loss
    turn_tokens = sum(len(c) * b.weight for s in samples for b, c in zip(s.plan.branches, s.completions))
    kd_tokens = sum(sum(s.plan.kd_weights()) for s in samples)
    stats = defaultdict(float)
    trainer.student.zero_grad()
    for s in samples:
        trace_stats, turn_loss, recorded_loss, kd_positions = trainer._trace_terms(
            s, loss.trace_branch_weight / turn_tokens, loss.trace_kd_weight / kd_tokens
        )
        seqs, rows = trace_sequences(s.plan, s.completions, kd_positions)
        for i, (ids, at) in enumerate(zip(seqs, rows)):
            h = final_hidden_states(trainer.student, torch.tensor([ids]), None)[0, torch.from_numpy(at)]
            (turn_loss(i, h) if i < len(s.completions) else recorded_loss(kd_positions, h)).backward()
        for k, v in trace_stats.items():
            stats[k] += v
    return dict(stats), {n: p.grad.clone() for n, p in trainer.student.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("arch", ["qwen3", "qwen35"])
def test_traces_with_onset_branches_are_the_same_as_unpacked(tmp_path, models, arch):
    """A batch of traces (regenerated turns, onset branches inside recorded turns, recorded-turn distillation, two
    teachers by topic, two rollouts per turn): the teachers' hidden states, every statistic and the student's
    gradients from packed rows against every sequence forwarded alone (float64), and against seco_tree's
    chunk-wise pass (which keeps float32 state between chunks: to that precision)."""
    from palingenesis.logits import final_hidden_states
    from palingenesis.opd.trace_trainer import TraceTrainer, trace_sequences

    trainer = f64(TraceTrainer(trace_config(tmp_path, models[arch])))
    set_trace_mode(trainer, "padded")
    batch = trainer.pipeline.run(trainer._draw(), 1.0)
    samples = batch.samples
    assert {s.teacher for s in samples} == {"coder", "general"}
    assert any(b.onset for s in samples for b in s.plan.branches) and any(s.plan.kd_spans for s in samples)
    chunked_stats, chunked_grads = step(trainer, batch)
    ref_stats, ref_grads = naive_trace_step(trainer, samples)
    assert any(k.startswith("onset_kl/") for k in ref_stats) and any(k.startswith("kd_kl/") for k in ref_stats)

    set_trace_mode(trainer, "tree")
    packing = trainer._packing()
    for s in samples:  # a trace's sequences fold into far fewer tokens: its trunk is forwarded once
        seqs, _ = trace_sequences(s.plan, s.completions, s.plan.kd_positions())
        row = packing.row(seqs)
        assert not row.flat and row.num_tokens < 0.6 * sum(map(len, seqs))
    packed = trainer.pipeline._packed_hidden(samples)
    assert len(packed) == len(samples)
    for s in samples:
        turns, trunk = packed[id(s)]
        teacher = trainer.routes[s.teacher].teacher.model
        seqs, rows = trace_sequences(s.plan, s.completions, s.plan.kd_positions())
        for got, ids, at in zip([*turns, trunk], seqs, rows):
            with torch.no_grad():
                want = final_hidden_states(teacher, torch.tensor([ids]), None)[0, torch.from_numpy(at)]
            assert torch.allclose(got, want, rtol=1e-10, atol=1e-10), (got - want).abs().max()
        for got, want in zip([*turns, trunk], [*s.teacher_hidden, s.teacher_kd_hidden]):
            assert torch.allclose(got, want, rtol=1e-6, atol=1e-6), (got - want).abs().max()
    for tokens in (0, 400):  # one row, then rows of a few traces each (a small token budget)
        trainer.config.train.packing_tokens = tokens
        assert (len(trainer._packing().micro_batches(*_trace_rows(samples), 16)) > 1) == (tokens > 0)
        stats, grads = step(trainer, batch)
        assert_close_stats(stats, ref_stats, 1e-10, tokens)
        assert_close_grads(grads, ref_grads, 1e-6, tokens)
        assert_close_stats(stats, chunked_stats, 1e-5, tokens)
        assert_close_grads(grads, chunked_grads, 1e-4, tokens)
        dev, _ = step(trainer, batch, train=False)
        assert_close_stats(dev, ref_stats, 1e-10, tokens)


def _trace_rows(samples):
    from palingenesis.opd.trace_trainer import trace_sequences

    sequences, scored = [], []
    for s in samples:
        seqs, rows = trace_sequences(s.plan, s.completions, s.plan.kd_positions())
        sequences += seqs
        scored += [len(r) for r in rows]
    return sequences, scored


def test_trace_teacher_cutoff_reads_the_packed_trunk(tmp_path, models):
    """TeacherCutoff reads the teacher's log-probs of the recorded turns from the packed pass: the same cut."""
    from palingenesis.opd.trace_trainer import TraceTrainer

    overrides = {
        "sources.traces.cutoff_logp": -2.0,
        "sources.traces.cutoff_window": 2,
        "sources.traces.cutoff_tokens": 8,
        "sources.traces.cutoff_weight": 0.5,
    }
    trainer = f64(TraceTrainer(trace_config(tmp_path, models["qwen3"], **overrides)))
    requests = [r for r in trainer._draw() if r.source == "traces"]
    assert requests
    torch.manual_seed(0)
    set_trace_mode(trainer, "padded")
    ref = trainer.pipeline.run(requests, 1.0)
    torch.manual_seed(0)
    set_trace_mode(trainer, "tree")
    got = trainer.pipeline.run(requests, 1.0)
    assert any(s.cutoff for s in ref.samples)
    for a, b in zip(got.samples, ref.samples):
        assert a.completions == b.completions and a.plan.turn_weights == b.plan.turn_weights
        assert a.cutoff["cut"] == b.cutoff["cut"] and abs(a.cutoff["logp_sum"] - b.cutoff["logp_sum"]) < 1e-5
        for x, y in zip(a.teacher_hidden, b.teacher_hidden):  # (seco_tree keeps float32 state between chunks)
            assert torch.allclose(x, y, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("arch", ["qwen3", "qwen35"])
def test_trace_training_runs_end_to_end_with_packing(tmp_path, models, arch):
    from palingenesis.opd.trace_trainer import TraceTrainer

    trainer = TraceTrainer(trace_config(tmp_path, models[arch]))
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, step, metrics))
    trainer.train()
    steps = [m for kind, _, m in logged if kind == "step"]
    evals = [m for kind, _, m in logged if kind == "eval"]
    assert len(steps) == 2 and all(m["grad_norm"] > 0 and m["loss"] == m["loss"] for m in steps)
    assert any(k.startswith("kd_kl/") for m in steps for k in m)
    assert evals and all("dev_kl/traces" in m for m in evals)


def test_traces_refuse_varlen(tmp_path, models):
    from palingenesis.opd.config import OPDConfigError

    with pytest.raises(OPDConfigError, match="train.packing varlen with agent_traces"):
        trace_config(tmp_path, models["qwen3"], packing="varlen").validate()
