"""On-policy distillation on agent traces, end to end on CPU with tiny models.

The trainer's per-trace loss (one tree forward/backward) must equal the naive one:
every regenerated turn as its own sequence (context + completion) through the
student, fused reverse KL against the teacher's hidden states of the same sequence,
plus distillation on the recorded turns from one full forward of the trunk.
"""

import json
import random
import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "tests")

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from test_opd_traces import TOOLS, agent_trace, looping_trace  # noqa: E402
from test_opd_trainer import tiny_model  # noqa: E402


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("trace_models")
    return {
        "student": tiny_model(tmp, "Qwen/Qwen3-0.6B", "student", 0),
        "coder": tiny_model(tmp, "Qwen/Qwen3-0.6B", "coder", 1),
        "general": tiny_model(tmp, "Qwen/Qwen3-0.6B", "general", 2),
    }


def write_traces(path, n=12, looping=False):
    rows = []
    for i in range(n):
        messages = looping_trace() if looping else agent_trace(2 + i % 3)
        if i % 4 == 3 and not looping:  # a second user query: the template rewrites the earlier turns
            messages += [
                {"role": "user", "content": "And now?"},
                {"role": "assistant", "reasoning_content": "Think again.", "content": "Fine."},
            ]
        rows.append(
            {
                "messages": json.dumps(messages),
                "tools": json.dumps(TOOLS),
                "domain": ["code", "search", "office"][i % 3],
                "uuid": f"t{i}",
            }
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def make_config(tmp_path, models, **overrides):
    from palingenesis.opd.config import OPDConfig

    config = OPDConfig()
    settings = {
        "model.student": models["student"],
        "model.chat_template_kwargs": {"enable_thinking": True},
        "teachers.coder.model": models["coder"],
        "teachers.general.model": models["general"],
        "sources.traces.format": "agent_traces",
        "sources.traces.path": write_traces(tmp_path / "traces.jsonl"),
        "sources.traces.dev_size": 3,
        "sources.traces.max_new_tokens": 6,
        "sources.traces.branches_per_trace": 3,
        "sources.traces.teacher": "general",
        "sources.traces.topic_field": "domain",
        "sources.traces.topic_teachers": {"coder": ["code", "search"]},
        "rollout.batch_prompts": 3,
        "rollout.group_size": 2,
        "loss.trace_kd_weight": 0.5,
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 2,
        "train.learning_rate": 1e-3,
        "train.warmup_steps": 1,
        "train.eval_every": 1,
        "train.eval_samples": 3,
        "train.tree_chunk_size": 64,
        "model.use_liger_kernel": False,
    }
    settings.update(overrides)
    for key, value in settings.items():
        config.set(key, value, "test")
    return config


@pytest.mark.parametrize("branch, kd", [(1.0, 0.5), (0.0, 0.5), (1.0, 0.0)])  # the ablation arms
def test_training_run(tmp_path, models, branch, kd):
    from palingenesis.opd.trace_trainer import TraceTrainer

    trainer = TraceTrainer(
        make_config(tmp_path, models, **{"loss.trace_branch_weight": branch, "loss.trace_kd_weight": kd})
    )
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, step, metrics))
    trainer.train()
    steps = [m for kind, _, m in logged if kind == "step"]
    evals = [m for kind, _, m in logged if kind == "eval"]
    assert len(steps) == 2 and len(evals) == 3
    for m in steps:
        assert m["grad_norm"] > 0 and m["staleness"] == 0
        assert (m["branches"] > 0) == (branch > 0)  # no rollouts in the recorded-only arm
        assert any(k.startswith("kd_kl/") for k in m) == (kd > 0)  # recorded turns distilled
    routed = {k.split("/")[1] for m in steps for k in m if k.startswith(("tokens/", "kd_tokens/"))}
    assert routed <= {"coder", "general"} and routed  # topic routing picked the teachers
    # every arm is evaluated the same way: the student regenerating held-out turns
    assert {"dev_kl/traces", "dev_kl_full/traces", "dev_stop_rate/traces"} <= set(evals[0])
    assert evals[0]["dev_len/traces"] > 0
    assert any(k.startswith("dev_kl_full/traces/") for k in evals[0])  # per topic
    assert (tmp_path / "run" / "final" / "config.json").exists()


def test_trace_loss_equals_every_turn_as_its_own_sequence(tmp_path, models, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # an fp32 CPU reference (GPU path: test_opd_gpu.py)
    from palingenesis.logits import output_head
    from palingenesis.opd.fused_rkl import fused_full_rkl
    from palingenesis.opd.trace_trainer import TraceTrainer
    from palingenesis.opd.traces import TraceSample, branch_inputs, completion_rows

    trainer = TraceTrainer(make_config(tmp_path, models, **{"rollout.group_size": 1}))
    student, teacher = trainer.student, trainer.routes["general"].teacher
    messages = agent_trace(4) + [
        {"role": "user", "content": "And now?"},
        {"role": "assistant", "reasoning_content": "Hm.", "content": "Ok."},
    ]
    plan = trainer.planners["traces"].plan(messages, TOOLS, random.Random(0))
    assert len(plan.branches) >= 3 and plan.kd_spans
    g = torch.Generator().manual_seed(0)
    completions = [torch.randint(10, 1000, (n,), generator=g).tolist() for n in (3, 5, 2, 4)][: len(plan.branches)]
    plan.branches = plan.branches[: len(completions)]
    sample = TraceSample(plan, completions, [[]] * len(completions), ["stop"] * len(completions), "general", {})
    trainer.pipeline._score(sample)
    kd_positions = plan.kd_positions()
    total = sum(len(c) for c in completions)
    kd_w = 0.5 / len(kd_positions)

    # The teacher's tree pass equals its naive forwards (bf16 weights: to bf16 rounding).
    for b, c, h in zip(plan.branches, completions, sample.teacher_hidden):
        with torch.no_grad():
            want = teacher.model.base_model(input_ids=torch.tensor([b.context + c[:-1]])).last_hidden_state[
                0, -len(c) :
            ]
        torch.testing.assert_close(h.float(), want.float(), rtol=2e-2, atol=3e-2)
    trunk = torch.tensor([plan.trunk])
    with torch.no_grad():
        want = teacher.model.base_model(input_ids=trunk).last_hidden_state[0, kd_positions]
    torch.testing.assert_close(sample.teacher_kd_hidden.float(), want.float(), rtol=2e-2, atol=3e-2)

    # The student side, against the SAME teacher states: each turn as its own sequence, and the
    # recorded turns from one full forward of the trunk (fp32: exact to rounding).
    student.zero_grad()
    head, t_head = output_head(student), teacher.head
    for b, c, t_hidden in zip(plan.branches, completions, sample.teacher_hidden):
        hidden = student.base_model(input_ids=torch.tensor([b.context + c[:-1]])).last_hidden_state[0, -len(c) :]
        loss, _ = fused_full_rkl(
            hidden,
            head,
            t_hidden,
            t_head,
            torch.tensor(c),
            torch.full((len(c),), 1 / total),
            trainer.routes["general"].aligner.bridge.shared_vocab_size,
        )
        loss.backward()
    s_trunk = student.base_model(input_ids=trunk).last_hidden_state[0, kd_positions]
    loss, _ = fused_full_rkl(
        s_trunk,
        head,
        sample.teacher_kd_hidden,
        t_head,
        trunk[0, [p + 1 for p in kd_positions]],
        torch.full((len(kd_positions),), kd_w),
        trainer.routes["general"].aligner.bridge.shared_vocab_size,
    )
    loss.backward()
    want = {n: p.grad.clone() for n, p in student.named_parameters() if p.grad is not None}

    student.zero_grad()
    trainer._score_trace(sample, True, 1 / total, kd_w)
    got = {n: p.grad.clone() for n, p in student.named_parameters() if p.grad is not None}
    assert got.keys() == want.keys()
    for n in want:
        err = float((got[n] - want[n]).norm() / want[n].norm().clamp(min=1e-30))
        assert err < 1e-4, (n, err)
    # the rows that predict each completion: its last len(completion) hidden states
    assert all(r.stop - r.start == len(c) for r, c in zip(completion_rows(plan, completions), completions))
    assert all(
        len(ids) == len(b.prefix) + len(c) - 1
        for ids, b, c in zip(branch_inputs(plan, completions), plan.branches, completions)
    )


def test_traces_mixed_with_prompt_source(tmp_path, models):
    """A messages source joins an agent_traces run as one-turn traces, with its own template arguments."""
    from palingenesis.opd.trace_trainer import TraceTrainer

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(
            json.dumps({"messages": [{"role": "user", "content": f"Count to {i}."}], "tools": TOOLS if i % 2 else None})
            + "\n"
            for i in range(10)
        )
    )
    trainer = TraceTrainer(
        make_config(
            tmp_path,
            models,
            **{
                "sources.chat.format": "messages",
                "sources.chat.path": str(prompts),
                "sources.chat.dev_size": 2,
                "sources.chat.max_new_tokens": 6,
                "sources.chat.teacher": "coder",
                "sources.chat.weight": 1.0,
                "sources.chat.chat_template_kwargs": {"enable_thinking": False},
            },
        )
    )
    assert trainer.planners["chat"].kwargs["enable_thinking"] is False
    assert trainer.planners["traces"].kwargs["enable_thinking"] is True
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, step, metrics))
    trainer.train()
    evals = [m for kind, _, m in logged if kind == "eval"]
    assert "dev_kl_full/chat" in evals[-1] and "dev_kl_full/traces" in evals[-1]


def test_stop_at_chunks_follow_one_schedule(tmp_path, models):
    """steps 4 with stop_at 2 then a resume: the LR follows the 4-step schedule across the chunks, step_2 is
    resumable, final only at the end (a chunked run equals one run)."""
    from palingenesis.opd.trace_trainer import TraceTrainer

    base = {
        "train.steps": 4,
        "train.warmup_steps": 1,
        "train.save_steps": 2,
        "train.eval_every": 2,
        "train.resume_from": "auto",
        "loss.trace_kd_weight": 0.0,
    }
    lrs = {}

    def run(**extra):
        t = TraceTrainer(make_config(tmp_path, models, **{**base, **extra}))
        t._log = lambda kind, m, step: lrs.__setitem__(step, m["lr"]) if kind == "step" else None
        t.train()

    run(**{"train.stop_at": 2})
    out = tmp_path / "run"
    assert (out / "step_2" / "trainer_state.pt").exists() and not (out / "final").exists()
    run()
    assert (out / "final" / "config.json").exists()
    ref = TraceTrainer(make_config(tmp_path, models, **base))
    assert lrs == {k + 1: pytest.approx(ref._lr_at(k)) for k in range(4)}  # one cosine over 4 steps, no restart


def test_cmr_holds_the_common_margin_in_a_trace_run(tmp_path, models):
    """cmr.enabled in an agent_traces run: metrics logged, and the twins' common margin moves far less than without."""
    from palingenesis.opd.trace_trainer import TraceTrainer

    tw = tmp_path / "twins.jsonl"
    rows = []
    for i in range(6):
        q = [{"role": "user", "content": f"Run the command number {i}."}]
        rows.append({"pos": {"messages": q, "tools": TOOLS}, "neg": {"messages": q, "tools": []}})
    tw.write_text("".join(json.dumps(r) + "\n" for r in rows))
    common = {}
    for arm, on in (("off", False), ("on", True)):
        extra = {
            "train.output_dir": str(tmp_path / arm),
            "train.steps": 3,
            "train.eval_every": 0,
            "loss.trace_kd_weight": 0.0,
            "train.learning_rate": 3e-3,
        }
        if on:
            extra.update({"cmr.enabled": True, "cmr.families": {"t": str(tw)}, "cmr.pairs_per_refresh": 6})
        t = TraceTrainer(make_config(tmp_path, models, **extra))
        from palingenesis.cmr import CMRConfig, CommonModeRejection

        probe = CommonModeRejection(
            CMRConfig(enabled=True, families={"t": str(tw)}), t.student, t.tok, list(t.student.parameters()), t.device
        )
        before = probe.evaluate(probe.twins["t"])["common"]
        logged = []
        t._log = lambda kind, m, step: logged.append((kind, m))
        t.train()
        common[arm] = abs(probe.evaluate(probe.twins["t"])["common"] - before)
        steps = [m for kind, m in logged if kind == "step"]
        assert any("cmr/component/t" in m for m in steps) == on
    assert common["on"] < 0.5 * common["off"] + 1e-4


def test_onset_branches_and_kd_cut_in_a_training_run(tmp_path, models):
    """Student-recorded traces that repeat themselves: regenerated turns start at the repetition (reported apart as
    kl_onset in training and dev_kl_onset in evaluation), recorded-turn distillation keeps only a run's first tokens."""
    from palingenesis.opd.trace_trainer import TraceTrainer

    overrides = {
        "sources.traces.path": write_traces(tmp_path / "loops.jsonl", looping=True),
        "sources.traces.branches_per_trace": 1,
        "sources.traces.onset_branches": 2,
        "sources.traces.onset_kd_tokens": 4,
        "rollout.group_size": 1,
    }
    trainer = TraceTrainer(make_config(tmp_path, models, **overrides))
    plans = [
        trainer.planners["traces"].plan(r["messages"], r.get("tools"), random.Random(0))
        for r in trainer.source.train["traces"]
    ]
    assert any(b.onset for p in plans for b in p.branches)
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, step, metrics))
    trainer.train()
    steps = [m for kind, _, m in logged if kind == "step"]
    evals = [m for kind, _, m in logged if kind == "eval"]
    assert sum(m["onset_branches"] for m in steps) > 0
    assert any(k.startswith("kl_onset/") for m in steps for k in m)
    assert any("dev_kl_onset/traces" in m for m in evals)
    assert all(m["grad_norm"] > 0 for m in steps)


def test_trace_config_validation(tmp_path, models):
    from palingenesis.opd.config import OPDConfigError

    def invalid(match, **overrides):
        with pytest.raises(OPDConfigError, match=match):
            make_config(tmp_path, models, **overrides).validate()

    invalid("assigned to both", **{"sources.traces.topic_teachers": {"coder": ["code"], "general": ["code"]}})
    invalid("not one of the teachers", **{"sources.traces.topic_teachers": {"nobody": ["code"]}})
    invalid("needs topic_field", **{"sources.traces.topic_field": ""})
    invalid("full_rkl from an hf teacher", **{"teachers.coder.loss": "sampled_rkl"})
    invalid("branches_per_trace", **{"sources.traces.branches_per_trace": -1})
    invalid("onset_branches needs", **{"sources.traces.onset_branches": 2, "loss.trace_branch_weight": 0.0})
    invalid("onset_kd_tokens shapes", **{"sources.traces.onset_kd_tokens": 8, "loss.trace_kd_weight": 0.0})
    invalid("onset_threshold", **{"sources.traces.onset_threshold": 0.0})
    invalid("onset_common", **{"sources.traces.onset_common": 1.5})
    invalid("tree_chunk_size bounds", **{"model.gradient_checkpointing": True})
    invalid("cmr.families", **{"cmr.enabled": True})
    invalid("mcqa sources need a plain OPD run", **{"sources.pool.format": "mcqa", "sources.pool.path": "x.jsonl"})
    invalid(
        "without sampling_mix",
        **{
            "sources.chat.format": "messages",
            "sources.chat.path": "x.jsonl",
            "sources.chat.sampling_mix": [{"weight": 1}],
        },
    )


# ---------------------------------------------------------------- failed turns, teacher cutoff, vLLM teacher

ERRORS = ["^Error:", "command not found"]


def failing_trace():
    from test_opd_traces import failing_trace as trace

    return trace()


def write_failing_traces(path, n=12):
    rows = [
        {
            "messages": json.dumps(failing_trace() if i % 2 == 0 else agent_trace(3 + i % 2)),
            "tools": json.dumps(TOOLS),
            "domain": ["code", "office"][i % 2],
            "solved": i % 4 == 1,
        }
        for i in range(n)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def naive_teacher_logp(model, ids: list[int]) -> torch.Tensor:
    """log p(ids[t] | ids[:t]) for t >= 1 from one full forward."""
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0, :-1].float()
    return torch.log_softmax(logits, -1).gather(1, torch.tensor(ids[1:])[:, None]).squeeze(1)


def test_weighted_turns_loss_equals_the_weighted_naive_loss(tmp_path, models, monkeypatch):
    """Failed-turn weights scale each regenerated and recorded token's KL; the result is the naive per-turn loss
    with the same per-token weights."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from palingenesis.logits import output_head
    from palingenesis.opd.fused_rkl import fused_full_rkl
    from palingenesis.opd.trace_trainer import TraceBatch, TraceTrainer
    from palingenesis.opd.traces import TraceSample

    overrides = {
        "rollout.group_size": 1,
        "sources.traces.error_patterns": ERRORS,
        "sources.traces.error_weight": 2.0,
        "sources.traces.first_error_weight": 5.0,
        "sources.traces.error_branches": 2,
    }
    trainer = TraceTrainer(make_config(tmp_path, models, **overrides))
    student, teacher = trainer.student, trainer.routes["general"].teacher
    plan = trainer.planners["traces"].plan(failing_trace(), TOOLS, random.Random(0), {})
    assert {4, 8} <= {b.turn for b in plan.branches} and plan.turn_weights == {4: 5.0, 8: 2.0}
    g = torch.Generator().manual_seed(1)
    completions = [torch.randint(10, 1000, (2 + i,), generator=g).tolist() for i in range(len(plan.branches))]
    sample = TraceSample(plan, completions, [[]] * len(completions), ["stop"] * len(completions), "general", {})
    trainer.pipeline._score(sample)
    kd_positions, kd_weights = plan.kd_positions(), plan.kd_weights()
    branch_total = sum(len(c) * b.weight for b, c in zip(plan.branches, completions))
    size = trainer.routes["general"].aligner.bridge.shared_vocab_size

    student.zero_grad()
    head = output_head(student)
    for b, c, t_hidden in zip(plan.branches, completions, sample.teacher_hidden):
        hidden = student.base_model(input_ids=torch.tensor([b.context + c[:-1]])).last_hidden_state[0, -len(c) :]
        w = torch.full((len(c),), b.weight / branch_total)
        fused_full_rkl(hidden, head, t_hidden, teacher.head, torch.tensor(c), w, size)[0].backward()
    trunk = torch.tensor([plan.trunk])
    s_trunk = student.base_model(input_ids=trunk).last_hidden_state[0, kd_positions]
    w = 0.5 * torch.tensor(kd_weights) / sum(kd_weights)
    targets = trunk[0, [p + 1 for p in kd_positions]]
    fused_full_rkl(s_trunk, head, sample.teacher_kd_hidden, teacher.head, targets, w, size)[0].backward()
    want = {n: p.grad.clone() for n, p in student.named_parameters() if p.grad is not None}

    student.zero_grad()
    stats = trainer._scores(TraceBatch([sample], 0, {}), train=True)  # the batch normalization included
    got = {n: p.grad.clone() for n, p in student.named_parameters() if p.grad is not None}
    for n in want:
        err = float((got[n] - want[n]).norm() / want[n].norm().clamp(min=1e-30))
        assert err < 1e-4, (n, err)
    assert stats["error_tokens/general"] == sum(len(c) for b, c in zip(plan.branches, completions) if b.error)


def test_teacher_cutoff_drops_the_turns_after_the_teacher_stops_endorsing(tmp_path, models, monkeypatch):
    """The per-turn means are the teacher's own log-probs of the recorded tokens; the plan, the completions and the
    teacher's states lose everything after the cutoff turn, and the remaining kd states are the right rows."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from palingenesis.opd.trace_trainer import TraceTrainer
    from palingenesis.opd.traces import TraceSample

    def build(logp):
        overrides = {
            "rollout.group_size": 1,
            "sources.traces.cutoff_logp": logp,
            "sources.traces.branches_per_trace": 0,
        }
        return TraceTrainer(make_config(tmp_path, models, **overrides))

    trainer = build(-1e-9)
    teacher = trainer.routes["general"].teacher
    plan = trainer.planners["traces"].plan(agent_trace(5), TOOLS, random.Random(0), {})
    assert len(plan.turns) >= 4 and plan.kd_spans
    lp = naive_teacher_logp(teacher.model, plan.trunk)  # lp[p] = log p(trunk[p + 1])
    means = [float(lp[a - 1 : b - 1].mean()) for _, a, b in plan.turns]
    # a threshold between the running sums after the second and third turns: the cut falls on the third turn
    sums = torch.tensor(means).cumsum(0).tolist()
    trainer = build((sums[1] + sums[2]) / 2)
    cut_turn = plan.turns[2][0]
    completions = [[11, 12, 13]] * len(plan.branches)
    sample = TraceSample(plan, completions, [[]] * len(completions), ["stop"] * len(completions), "general", {})
    sample.meta["_src"], sample.meta["_cutoff"] = "traces", True
    trainer.pipeline._score(sample)
    assert sample.cutoff["cut"] == 1 and sample.cutoff["logp_sum"] == pytest.approx(sum(means), abs=2e-2)
    assert all(b.turn <= cut_turn for b in sample.plan.branches)
    assert len(sample.completions) == len(sample.teacher_hidden) == len(sample.plan.branches) < len(plan.branches)
    assert sample.plan.kd_spans and max(b for _, b in sample.plan.kd_spans) <= plan.turns[2][2]
    with torch.no_grad():
        want = teacher.model.base_model(input_ids=torch.tensor([plan.trunk])).last_hidden_state[0]
    rows = sample.plan.kd_positions()
    torch.testing.assert_close(sample.teacher_kd_hidden.float(), want[rows].float(), rtol=2e-2, atol=3e-2)
    stats = trainer.pipeline._cut_stats([sample])
    later = sum(b - a for k, a, b in plan.turns if k > cut_turn)
    assert stats["cut_token_share"] == pytest.approx(later / sum(b - a for _, a, b in plan.turns))
    assert stats["cut_trace_share"] == 1 and 0 < stats["cut_branch_share"] < 1


def test_failed_turns_and_cutoff_in_a_training_run(tmp_path, models):
    from palingenesis.opd.trace_trainer import TraceTrainer

    overrides = {
        "sources.traces.path": write_failing_traces(tmp_path / "fail.jsonl"),
        "sources.traces.topic_teachers": {"coder": ["code"]},
        "sources.traces.branches_per_trace": 1,
        "sources.traces.error_patterns": ERRORS,
        "sources.traces.error_weight": 2.0,
        "sources.traces.first_error_weight": 3.0,
        "sources.traces.success_field": "solved",
        "sources.traces.error_branches": 1,
        "sources.traces.cutoff_logp": -2.0,
        "sources.traces.cutoff_window": 2,
        "sources.traces.cutoff_tokens": 8,
        "sources.traces.cutoff_weight": 0.5,  # a random tiny teacher disowns every turn: keep them, down-weighted
        "rollout.group_size": 1,
    }
    trainer = TraceTrainer(make_config(tmp_path, models, **overrides))
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, step, metrics))
    trainer.train()
    steps = [m for kind, _, m in logged if kind == "step"]
    evals = [m for kind, _, m in logged if kind == "eval"]
    for m in steps:
        assert {"cut_trace_share", "cut_token_share", "cut_branch_share", "teacher_turn_logp"} <= set(m)
        assert m["teacher_turn_logp"] < 0 and 0 <= m["cut_token_share"] <= 1
    assert any(m.get("error_branches", 0) > 0 and 0 < m["error_token_share"] <= 1 for m in steps)
    assert any(k.startswith("kl_error/") for m in steps for k in m)
    assert any("dev_kl_error/traces" in m for m in evals)
    assert not any(k.startswith("cut_") for m in evals for k in m)  # dev: no cutoff


class FakeVLLMServer:
    """vLLM's /v1/completions with prompt_logprobs, computed by an HF model: the top-k and the actual token at
    every prompt position (None at position 0), as the server returns them."""

    def __init__(self, path):
        self.model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
        self.requests = []

    def complete(self, prompts, parallel=4, max_tokens=1, temperature=1.0, prompt_logprobs=1):
        self.requests.append([len(p) for p in prompts])
        choices = []
        for i, ids in enumerate(prompts):
            with torch.no_grad():
                logp = torch.log_softmax(self.model(input_ids=torch.tensor([ids])).logits[0].float(), -1)
            entries = [None]
            for t in range(1, len(ids)):
                row = logp[t - 1]
                top = row.topk(prompt_logprobs)
                entry = {str(int(j)): {"logprob": float(v), "rank": r + 1} for r, (v, j) in enumerate(zip(*top))}
                rank = int((row > row[ids[t]]).sum()) + 1
                entry.setdefault(str(ids[t]), {"logprob": float(row[ids[t]]), "rank": rank})
                entries.append(entry)
            choices.append({"index": i, "prompt_logprobs": entries})
        return choices


@pytest.fixture
def vllm_teacher(monkeypatch, models):
    from palingenesis.opd.teachers import VLLMTeacher
    from palingenesis.opd.trace_trainer import TraceTrainer

    servers = {}

    def fake(self, name, teacher):
        servers[name] = FakeVLLMServer(teacher.model)
        return VLLMTeacher(servers[name])

    monkeypatch.setattr(TraceTrainer, "_vllm_teacher", fake)
    return servers


def test_vllm_teacher_trace_loss_equals_naive_topk_kl(tmp_path, models, vllm_teacher, monkeypatch):
    """A vLLM teacher: sparse targets from prompt log-probs of each regenerated turn (its full context) and of the
    trunk; the trace loss equals topk_kl turn by turn against the same server's answers."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from palingenesis.logits import output_head
    from palingenesis.opd import losses
    from palingenesis.opd.trace_trainer import TraceBatch, TraceTrainer
    from palingenesis.opd.traces import TraceSample

    trainer = TraceTrainer(
        make_config(tmp_path, models, **{"rollout.group_size": 1, "teachers.general.backend": "vllm", "loss.top_k": 4})
    )
    student, server = trainer.student, vllm_teacher["general"]
    plan = trainer.planners["traces"].plan(agent_trace(4), TOOLS, random.Random(0), {})
    completions = [[21 + i, 400, 7 + i] for i in range(len(plan.branches))]
    sample = TraceSample(plan, completions, [[]] * len(completions), ["stop"] * len(completions), "general", {})
    trainer.pipeline._score(sample)
    # one request per regenerated turn (context + completion) and one for the trunk
    assert server.requests[-1] == [len(b.context) + len(c) for b, c in zip(plan.branches, completions)] + [
        len(plan.trunk)
    ]
    kd_positions = plan.kd_positions()
    assert sample.teacher_kd_hidden.topk_ids.shape == (len(kd_positions), 4)
    size = trainer.routes["general"].aligner.bridge.shared_vocab_size
    vocab = losses.SharedVocab(size, {})
    head = output_head(student)

    def topk(hidden, targets, lp_rows, weights):
        """topk_kl against the teacher's full log-probs restricted to its top-4 plus the realized token."""
        top = lp_rows.topk(4)
        token_lp = lp_rows.gather(1, targets[:, None]).squeeze(1)
        in_top = (top.indices == targets[:, None]).any(1, keepdim=True)
        support = torch.cat([top.indices, targets[:, None]], 1)
        valid = torch.cat([torch.ones_like(top.values, dtype=torch.bool), ~in_top], 1)
        lp = torch.cat([top.values, token_lp[:, None]], 1)
        return losses.topk_kl(hidden, head, targets, weights, vocab, support, lp, valid, token_lp)[0]

    student.zero_grad()
    total = sum(len(c) for c in completions)
    for b, c in zip(plan.branches, completions):
        ids = b.context + c
        with torch.no_grad():
            lp = torch.log_softmax(server.model(input_ids=torch.tensor([ids])).logits[0, -len(c) - 1 : -1].float(), -1)
        hidden = student.base_model(input_ids=torch.tensor([ids[:-1]])).last_hidden_state[0, -len(c) :]
        topk(hidden, torch.tensor(c), lp, torch.full((len(c),), 1 / total)).backward()
    with torch.no_grad():
        lp = torch.log_softmax(server.model(input_ids=torch.tensor([plan.trunk])).logits[0].float(), -1)
    trunk_hidden = student.base_model(input_ids=torch.tensor([plan.trunk])).last_hidden_state[0, kd_positions]
    targets = torch.tensor([plan.trunk[p + 1] for p in kd_positions])
    w = torch.full((len(kd_positions),), 0.5 / len(kd_positions))
    topk(trunk_hidden, targets, lp[kd_positions], w).backward()
    want = {n: p.grad.clone() for n, p in student.named_parameters() if p.grad is not None}

    student.zero_grad()
    trainer._scores(TraceBatch([sample], 0, {}), train=True)
    got = {n: p.grad.clone() for n, p in student.named_parameters() if p.grad is not None}
    assert got.keys() == want.keys()
    for n in want:
        err = float((got[n] - want[n]).norm() / want[n].norm().clamp(min=1e-30))
        assert err < 1e-4, (n, err)


def test_vllm_teacher_in_a_training_run_with_cutoff(tmp_path, models, vllm_teacher):
    """One vLLM teacher next to an hf one, the cutoff reading the vLLM teacher's prompt log-probs."""
    from palingenesis.opd.trace_trainer import TraceTrainer

    overrides = {
        "teachers.coder.backend": "vllm",
        "loss.top_k": 3,
        "sources.traces.cutoff_logp": -3.0,
        "rollout.group_size": 1,
    }
    trainer = TraceTrainer(make_config(tmp_path, models, **overrides))
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, step, metrics))
    trainer.train()
    steps = [m for kind, _, m in logged if kind == "step"]
    assert all(m["grad_norm"] > 0 and "teacher_turn_logp" in m for m in steps)
    assert any(k.startswith("kl/coder") for m in steps for k in m) and vllm_teacher["coder"].requests


def test_failed_turn_and_cutoff_config_validation(tmp_path, models):
    from palingenesis.opd.config import OPDConfigError

    def invalid(match, **overrides):
        with pytest.raises(OPDConfigError, match=match):
            make_config(tmp_path, models, **overrides).validate()

    invalid("need a failure signal", **{"sources.traces.error_weight": 2.0})
    invalid("not a regular expression", **{"sources.traces.error_patterns": ["("]})
    invalid("must be >= 0", **{"sources.traces.error_patterns": ["x"], "sources.traces.error_weight": -1.0})
    invalid(
        "error_branches needs",
        **{
            "sources.traces.error_patterns": ["x"],
            "sources.traces.error_branches": 1,
            "loss.trace_branch_weight": 0.0,
        },
    )
    invalid("success_field only selects", **{"sources.traces.success_field": "solved"})
    invalid("cutoff_logp is a log-prob", **{"sources.traces.cutoff_logp": 1.0})
    invalid("cutoff_window and cutoff_tokens", **{"sources.traces.cutoff_window": -1})
    invalid("cutoff_weight must be", **{"sources.traces.cutoff_weight": 1.0})
    invalid("topk_kl from a vllm teacher", **{"teachers.coder.backend": "vllm", "teachers.coder.loss": "full_rkl"})
    invalid("loss.top_k >= 1", **{"teachers.coder.backend": "vllm", "loss.top_k": 0})
    make_config(tmp_path, models, **{"teachers.coder.backend": "vllm"}).validate()  # a vllm teacher is allowed
