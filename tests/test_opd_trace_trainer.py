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

from test_opd_traces import TOOLS, agent_trace  # noqa: E402
from test_opd_trainer import tiny_model  # noqa: E402


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("trace_models")
    return {
        "student": tiny_model(tmp, "Qwen/Qwen3-0.6B", "student", 0),
        "coder": tiny_model(tmp, "Qwen/Qwen3-0.6B", "coder", 1),
        "general": tiny_model(tmp, "Qwen/Qwen3-0.6B", "general", 2),
    }


def write_traces(path, n=12):
    rows = []
    for i in range(n):
        messages = agent_trace(2 + i % 3)
        if i % 4 == 3:  # a second user query: the template rewrites the earlier turns
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


def test_trace_loss_equals_every_turn_as_its_own_sequence(tmp_path, models):
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
