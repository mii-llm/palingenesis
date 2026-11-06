"""Off-policy distillation in the trainer, end to end on the CPU (the loop of `pgs train` with tiny
Llama students and teachers): online teachers, teacher views (context distillation), feature
matching with projections, routing by data source, SeCO, packing, config validation and the
command-line overrides of named teachers."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402

from palingenesis.config import Config, ConfigError  # noqa: E402

IGNORE = -100


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("kd")
    tok = H.tokenizer()
    H.save_model(tmp / "student", tok, seed=0)
    H.save_model(tmp / "teacher", tok, seed=1, hidden=48)
    data = H.write_rows(tmp / "rows.jsonl")
    return tmp, tok, data


def _config(setup, name, **distill):
    tmp, _, data = setup
    c = H.cpu_config(tmp / name, tmp / "student", data)
    c.distill.enabled = True
    c.distill.teachers = {"t": {"model": str(tmp / "teacher"), "dtype": "float32"}}
    for k, v in distill.items():
        setattr(c.distill, k, v)
    return c


def test_online_teacher_trains_and_exports(setup):
    from transformers import AutoModelForCausalLM

    c = _config(setup, "online", ce_weight=0.5, features=[{"student": -1, "teacher": -1, "weight": 0.1}])
    c.train.max_steps = 6
    result = H.run_cpu(c)
    m = result.metrics
    assert result.global_step == 6
    assert m["train/kd_coverage"] == 1.0 and m["train/kd_loss"] > 0 and "train/feature/f0" in m
    # the projection (student 32 -> teacher 48) trained with the student but is not exported
    final = AutoModelForCausalLM.from_pretrained(Path(c.train.output_dir) / "final")
    assert not any("distill_aux" in k for k in final.state_dict())


def _kl_to_teacher(student, teacher, tok, data, max_len=128) -> float:
    """Mean forward KL(teacher || student) over the trained tokens of the data."""
    from palingenesis.data import _load_dataset_source, source_stage

    c = Config()
    c.data.max_seq_length = max_len
    total, n = 0.0, 0
    stage = source_stage(_load_dataset_source(data, "train", False), tok, c.data, None, 0, 1)
    with torch.no_grad():
        for sample in stage:
            ids = sample["input_ids"][None]
            rows = (sample["labels"][1:] != IGNORE).nonzero().flatten()
            lp = torch.log_softmax(teacher(input_ids=ids).logits[0, rows].float(), -1)
            lq = torch.log_softmax(student(input_ids=ids).logits[0, rows].float(), -1)
            total += float((lp.exp() * (lp - lq)).sum())
            n += rows.numel()
    return total / n


def test_kd_pulls_the_student_towards_the_teacher(setup):
    from transformers import AutoModelForCausalLM

    tmp, tok, data = setup
    c = _config(setup, "decrease")
    c.train.epochs, c.train.max_steps = 3, -1
    c.train.learning_rate, c.train.lr_scheduler = 3e-3, "constant"
    teacher = AutoModelForCausalLM.from_pretrained(tmp / "teacher").eval()
    before = _kl_to_teacher(AutoModelForCausalLM.from_pretrained(tmp / "student").eval(), teacher, tok, data)
    H.run_cpu(c)
    after = _kl_to_teacher(
        AutoModelForCausalLM.from_pretrained(Path(c.train.output_dir) / "final").eval(), teacher, tok, data
    )
    assert after < 0.7 * before


@pytest.mark.parametrize("divergence", ["forward_kl", "jsd", "skew_reverse_kl", "tvd"])
def test_every_divergence_runs(setup, divergence):
    c = _config(setup, f"div_{divergence}", divergence=divergence, beta=0.2)
    c.train.max_steps = 2
    assert H.run_cpu(c).metrics["train/kd_loss"] >= 0


def test_teacher_view_context_distillation(setup):
    c = _config(setup, "view")
    c.distill.teachers["t"].update(privileged_field="hint", privileged_role="system")
    c.train.max_steps = 3
    result = H.run_cpu(c)
    assert result.metrics["train/kd_coverage"] == 1.0


def test_view_alignment_pairs_positions_that_predict_the_same_token(setup):
    from palingenesis.data import _load_dataset_source, source_stage
    from palingenesis.distill.config import parse_teachers
    from palingenesis.distill.views import TeacherView

    _, tok, data = setup
    c = _config(setup, "align")
    for options in (
        {"privileged_field": "hint"},
        {"system_prompt": "You are a careful calculator."},
        {"prefix_messages": [{"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"}]},
    ):
        c.distill.teachers["t"] = {"model": "x", **options}
        spec = parse_teachers(c.distill.teachers)[0]
        stage = source_stage(
            _load_dataset_source(data, "train", False),
            tok,
            c.data,
            None,
            0,
            1,
            teacher_views=[(TeacherView(spec), tok, 512)],
        )
        sample = next(iter(stage))
        s_ids, t_ids, pos = sample["input_ids"], sample["teacher_input_ids.t"], sample["teacher_positions.t"]
        trained = (sample["labels"][1:] != IGNORE).nonzero().flatten()
        assert (pos[trained] >= 0).all() and (pos >= 0).sum() == trained.numel()
        # student position t and teacher position pos[t] predict the same next token
        assert torch.equal(s_ids[trained + 1], t_ids[pos[trained] + 1])
        assert t_ids.numel() > s_ids.numel()


def test_view_that_changes_the_answer_drops_the_row(setup):
    from palingenesis.data import _load_dataset_source, source_stage
    from palingenesis.distill.config import TeacherSpec
    from palingenesis.distill.views import TeacherView

    _, tok, data = setup
    c = _config(setup, "mismatch")

    class Rewrite(TeacherView):
        def messages(self, row, messages):
            out = [dict(m) for m in messages]
            out[-1]["content"] = "something else entirely"
            return out

    stage = source_stage(
        _load_dataset_source(data, "train", False),
        tok,
        c.data,
        None,
        0,
        1,
        teacher_views=[(Rewrite(TeacherSpec("t", model="x", system_prompt="s")), tok, 512)],
    )
    assert list(stage) == []
    assert stage.stats["kd_view_mismatch"] == len(H.ROWS)


def test_routing_by_source_and_packing(setup):
    tmp, _, data = setup
    c = _config(setup, "route", ce_weight=0.2, missing="skip")
    c.data.sources = [{"dataset": data, "name": "a", "weight": 1.0}, {"dataset": data, "name": "b", "weight": 1.0}]
    c.distill.teachers = {
        "t": {"model": str(tmp / "teacher"), "dtype": "float32", "sources": ["a"]},
        "u": {"model": str(tmp / "student"), "dtype": "float32", "sources": ["b"], "weight": 2.0},
    }
    c.data.packing = True
    c.train.max_steps = 3
    m = H.run_cpu(c).metrics
    assert m["train/kd_coverage"] == pytest.approx(1.0)
    assert "train/kd/t" in m and "train/kd/u" in m


def test_seco_distillation_matches_the_full_backward(setup):
    """Chunk-wise forward/backward with the distillation loss gives the full-sequence gradients."""
    from palingenesis.distill.distiller import build_distiller
    from palingenesis.loss import shift_labels
    from palingenesis.seco import chunkwise_forward_backward, use_chunk_attention
    from palingenesis.train import _get_hidden_states, _get_lm_head

    tmp, tok, _ = setup
    c = _config(setup, "seco", ce_weight=0.3)
    student = H.tiny_llama(len(tok), seed=0)
    use_chunk_attention(student)
    distiller = build_distiller(c, tok, torch.device("cpu"))
    torch.manual_seed(0)
    ids = torch.randint(0, len(tok), (2, 40))
    raw = ids.clone()
    raw[:, :10] = IGNORE
    labels = shift_labels(raw)
    denom = float((labels != IGNORE).sum())
    targets = distiller.targets({}, ids, labels, {"attention_mask": None}, compute_dtype=torch.float32)
    head = _get_lm_head(student)

    loss, _ = distiller.loss(_get_hidden_states(student, ids), targets, head, denom)
    loss.backward()
    ref = {n: p.grad.clone() for n, p in student.named_parameters()}
    student.zero_grad()
    result = chunkwise_forward_backward(
        student, ids, raw, chunk_size=16, loss_denom=denom, loss_fn=distiller.seco_loss_fn(targets, head)
    )
    assert result.loss == pytest.approx(float(loss.detach()), rel=1e-4)
    for n, p in student.named_parameters():
        torch.testing.assert_close(p.grad, ref[n], rtol=1e-3, atol=1e-5, msg=n)


# ── configuration ───────────────────────────────────────────────────────────


def _validated(**distill):
    c = Config()
    c.distill.enabled = True
    c.distill.teachers = {"t": {"model": "m"}}
    for k, v in distill.items():
        setattr(c.distill, k, v)
    return c.validate()


@pytest.mark.parametrize(
    "distill,match",
    [
        ({"divergence": "kl"}, "divergence"),
        ({"divergence": "jsd", "beta": 1.0}, "beta"),
        ({"temperature": 0.0}, "temperature"),
        ({"teachers": {}}, "at least one teacher"),
        ({"teachers": {"t": {"modle": "m"}}}, "did you mean model"),
        ({"teachers": {"t": {}}}, "needs `model`"),
        ({"teachers": {"t": {"model": "m", "placement": "everywhere"}}}, "placement"),
        ({"teachers": {"t": {"model": "m", "sources": ["nope"]}}}, "not data source names"),
        ({"topk_mode": "full"}, "topk_mode"),
        ({"missing": "ignore"}, "missing"),
        ({"features": [{"student": 1}]}, "student` and `teacher"),
        ({"features": [{"student": 1, "teacher": 1, "loss": "l1"}]}, "loss must be"),
        ({"kd_weight": 0.0, "ce_weight": 0.0}, "nothing to train"),
    ],
)
def test_invalid_distill_settings_raise(distill, match):
    with pytest.raises(ConfigError, match=match):
        _validated(**distill)


def test_view_with_packing_is_refused_for_online_teachers():
    c = Config()
    c.distill.enabled = True
    c.distill.teachers = {"t": {"model": "m", "system_prompt": "x"}}
    c.data.packing = True
    with pytest.raises(ConfigError, match="needs data.packing: false"):
        c.validate()
    c.distill.teachers = {"t": {"cache": "dir", "system_prompt": "x"}}
    c.validate()  # cached: the view was applied when the cache was built


def test_distill_is_exclusive_with_dpo_and_gated_objectives():
    c = Config()
    c.distill.enabled, c.distill.teachers, c.plugins.deft = True, {"t": "m"}, True
    with pytest.raises(ConfigError, match="replaces the SFT objective"):
        c.validate()


def test_yaml_and_nested_command_line_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "distill:\n  enabled: true\n  divergence: jsd\n  teachers:\n    big: {model: org/big, placement: shard}\n"
    )
    c = Config.from_cli(
        [
            "--config",
            str(path),
            "--distill.teachers.big.weight",
            "2",
            "--distill.teachers.small.cache",
            "c/s",
            "--distill.beta",
            "0.3",
        ]
    )
    assert c.distill.teachers == {
        "big": {"model": "org/big", "placement": "shard", "weight": 2},
        "small": {"cache": "c/s"},
    }
    assert c.distill.beta == 0.3
    c.validate()
    with pytest.raises(ConfigError, match="only mapping options"):
        Config.from_cli(["--distill.beta.x", "1"])


def test_logit_lens_target_trains_the_layers_not_the_head(setup):
    """lens_kl reads an intermediate layer through each model's own final norm and head: forward KL between
    the two readings; the head and the final norm get no gradient from it, the layers below do."""
    import torch.nn.functional as F

    from palingenesis.distill.distiller import build_distiller
    from palingenesis.loss import shift_labels
    from palingenesis.train import _get_hidden_states, _get_lm_head

    _, tok, _ = setup
    c = _config(setup, "lens", kd_weight=0.0, features=[{"student": 0, "teacher": 1, "loss": "lens_kl"}])
    c.validate()
    student = H.tiny_llama(len(tok), seed=0)
    distiller = build_distiller(c, tok, torch.device("cpu"))
    distiller.attach_features(student, torch.device("cpu"))
    torch.manual_seed(0)
    ids = torch.randint(0, len(tok), (2, 12))
    labels = shift_labels(ids.clone())
    denom = float((labels != IGNORE).sum())
    targets = distiller.targets({}, ids, labels, {"attention_mask": None}, compute_dtype=torch.float32)
    with distiller.capturing():
        hidden = _get_hidden_states(student, ids)
    rows = distiller.student_feature_rows(targets)
    s_rows, t_rows = distiller.feature_pairs(targets, rows)["f0"]
    loss, stats = distiller.loss(hidden, targets, _get_lm_head(student), denom, rows)
    loss.backward()
    assert student.lm_head.weight.grad is None or not student.lm_head.weight.grad.any()
    assert student.model.norm.weight.grad is None or not student.model.norm.weight.grad.any()
    assert student.model.layers[0].mlp.up_proj.weight.grad.abs().sum() > 0
    above = student.model.layers[1].mlp.up_proj.weight.grad  # above the matched layer (kd_weight 0)
    assert above is None or not above.any()
    teacher = distiller.online["t"].model
    with torch.no_grad():
        lq = F.log_softmax(student.lm_head(student.model.norm(s_rows)), -1)
        lp = F.log_softmax(teacher.lm_head(teacher.model.norm(t_rows)), -1)
        expected = float((lp.exp() * (lp - lq)).sum(-1).mean())
    assert stats["feature_sum/f0"] / stats["feature_tokens/f0"] == pytest.approx(expected, rel=1e-4)
