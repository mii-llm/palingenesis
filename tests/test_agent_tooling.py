"""agent_tooling: the diagnostics must read what the trainer really logs and
consumes, and report exact numbers where they can."""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palingenesis.agent_tooling._logparse import parse_steps  # noqa: E402
from palingenesis.agent_tooling.check_loss import parse_losses_from_text  # noqa: E402
from palingenesis.agent_tooling.monitor_run import analyze_run, parse_training_log, print_brief  # noqa: E402
from palingenesis.config import Config  # noqa: E402

# Lines exactly as palingenesis.train logs them (SFT, DEFT, DPO) plus the
# best-model line, which also contains "step=" and a loss.
LOG = """\
[rank 0] 22:41:32 | INFO |   Preference eval set: 64 pairs
[rank 0] 22:52:04 | INFO | step=9 loss=0.2697 acc=1.000 margin=3.106 lr=3.30e-07 tok/s=957 grad_norm=59.264 dt=85.35s
[rank 0] 22:52:04 | INFO | Best model updated: step=10, eval_loss=0.3454 -> /ckpt/best
[rank 0] 22:52:04 | INFO | step=10 loss=0.2917 acc=1.000 margin=2.750 lr=2.94e-07 tok/s=958 grad_norm=58.951 dt=68.30s eval=0.3454
[rank 0] 22:53:04 | INFO | step=11 loss=0.3121 ce=1.2044 lr=1.00e-05 tok/s=5000 grad_norm=1.2 dt=1.10s
[rank 0] 22:54:04 | INFO | step=12 loss=0.9027 lr=5.00e-07 tok/s=897 grad_norm=283.619 dt=54.94s entropy=2.10
"""


# ── log parsing ─────────────────────────────────────────────────────────────


def test_parse_steps_reads_every_objective_and_skips_eval_lines():
    steps = parse_steps(LOG)
    assert [s["step"] for s in steps] == [9, 10, 11, 12]
    assert [s["loss"] for s in steps] == [0.2697, 0.2917, 0.3121, 0.9027]  # never eval_loss=0.3454
    assert steps[1]["acc"] == 1.0 and steps[1]["eval"] == 0.3454 and steps[1]["dt"] == 68.30
    assert steps[2]["ce"] == 1.2044 and steps[3]["grad_norm"] == 283.619


def test_resumed_run_keeps_the_last_occurrence_of_a_step():
    text = "step=5 loss=2.0 lr=1e-5 tok/s=1 grad_norm=1 dt=1s\nstep=5 loss=1.5 lr=1e-5 tok/s=1 grad_norm=1 dt=1s\n"
    assert [(s["step"], s["loss"]) for s in parse_steps(text)] == [(5, 1.5)]


def test_check_loss_uses_training_loss_only():
    assert [loss for _, loss in parse_losses_from_text(LOG)] == [0.2697, 0.2917, 0.3121, 0.9027]


def test_monitor_parses_dpo_and_deft_lines():
    steps = parse_training_log(LOG)
    assert len(steps) == 4 and steps[0].step_time == 85.35 and steps[-1].tokens_per_sec == 897
    result = analyze_run(steps, max_steps=20)
    assert result["current_step"] == 12 and result["eta_steps"] == 8


def test_monitor_without_step_lines_reports_no_data(capsys):
    result = analyze_run(parse_training_log("nothing useful here\n"))
    assert result["status"] == "NO_DATA"
    print_brief(result)  # used to raise KeyError
    assert "NO_DATA" in capsys.readouterr().out


def test_monitor_grad_norm_is_judged_against_the_run_itself():
    lines = [f"step={i} loss=1.0 lr=1e-5 tok/s=100 grad_norm=15.0 dt=1s" for i in range(1, 30)]
    calm = analyze_run(parse_training_log("\n".join(lines)))
    assert not any("Gradient norm" in i for i in calm["issues"])  # 15 is normal for this run
    spiked = analyze_run(
        parse_training_log("\n".join(lines + ["step=30 loss=1.0 lr=1e-5 tok/s=100 grad_norm=120 dt=1s"]))
    )
    assert any("spiked" in i for i in spiked["issues"])


# ── the trainer's data pipeline ─────────────────────────────────────────────


TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>"
    "{% if m['role'] == 'assistant' %}{% generation %}{{ m['content'] }}<|end|>{% endgeneration %}"
    "{% else %}{{ m['content'] }}<|end|>{% endif %}{% endfor %}"
)


@pytest.fixture
def model_dir(tmp_path):
    """A tiny local model + tokenizer (gpt2 tokenizer, pad == eos, chat template)."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = TEMPLATE
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tok),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    path = tmp_path / "model"
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path


def _config(tmp_path, model_dir, rows, **data):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = Config()
    config.model.name_or_path = str(model_dir)
    config.data.dataset = str(data_file)  # a local file: the trainer's common case
    config.data.streaming = False
    config.data.max_seq_length = 256
    config.data.length_group_buffer = 0
    for k, v in data.items():
        setattr(config.data, k, v)
    return config


CHATS = [
    {"messages": [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": f"answer number {i}"}]}
    for i in range(6)
]


def test_training_samples_are_the_trainers_stream(tmp_path, model_dir):
    from palingenesis.agent_tooling._pipeline import load_tokenizer, training_samples
    from palingenesis.data import IGNORE_INDEX, _load_dataset_source, build_dataset

    config = _config(tmp_path, model_dir, CHATS)
    tok = load_tokenizer(config)
    ours = list(training_samples(config, tok))
    raw = _load_dataset_source(config.data.dataset, config.data.dataset_split, False)
    theirs = list(build_dataset(raw, tok, config.data, 0, 1, config.train.per_device_batch_size))
    assert len(ours) == len(theirs) == 6
    for a, b in zip(ours, theirs):
        assert torch.equal(a["input_ids"], b["input_ids"]) and torch.equal(a["labels"], b["labels"])
    scored = tok.decode(ours[0]["input_ids"][ours[0]["labels"] != IGNORE_INDEX])
    assert "answer number" in scored and "question" not in scored


def test_training_samples_follow_dpo(tmp_path, model_dir):
    from palingenesis.agent_tooling._pipeline import load_tokenizer, training_samples

    pairs = [
        {
            "prompt": [{"role": "user", "content": "q"}],
            "chosen": [{"role": "assistant", "content": "good"}],
            "rejected": [{"role": "assistant", "content": "bad bad"}],
        }
    ] * 2
    config = _config(tmp_path, model_dir, pairs)
    config.dpo.enabled = True
    samples = list(training_samples(config, load_tokenizer(config)))
    assert [s["side"] for s in samples] == ["chosen", "rejected"] * 2


def test_validate_masking_runs_on_local_data_without_false_pad_bug(tmp_path, model_dir):
    from palingenesis.agent_tooling.validate_masking import validate

    config = _config(tmp_path, model_dir, CHATS)  # gpt2: pad token == eos token
    report = validate(config, num_samples=6)
    assert report["total_samples"] == 6
    assert report["pad_tokens_trained"] == 0
    assert not any("BUG" in i or "CRITICAL" in i for i in report["issues"])


# ── memory profile ──────────────────────────────────────────────────────────


def test_profile_counts_parameters_exactly(tmp_path, model_dir):
    from transformers import AutoModelForCausalLM

    from palingenesis.agent_tooling.profile_memory import estimate_memory

    config = _config(tmp_path, model_dir, CHATS)
    config.model.torch_dtype = "float32"
    est = estimate_memory(config)
    real = AutoModelForCausalLM.from_pretrained(model_dir)
    n = sum(p.numel() for p in real.parameters())
    assert est["total_params_B"] * 1e9 == pytest.approx(n)
    assert est["params_memory_gb"] == pytest.approx(n * 4 / 2**30)
    assert est["optimizer_memory_gb"] == pytest.approx(2 * n * 4 / 2**30)  # AdamW: 2 states, fp32 weights
    config.model.torch_dtype = "bfloat16"
    config.train.optimizer = "lion8bit"
    assert estimate_memory(config)["optimizer_memory_gb"] == pytest.approx(n / 2**30)


def test_profile_knows_activation_offload_and_kahan_state(tmp_path, model_dir):
    from palingenesis.agent_tooling.profile_memory import estimate_memory

    config = _config(tmp_path, model_dir, CHATS)
    config.data.max_seq_length = 4096
    for mode in ("none", "full"):
        config.train.gradient_checkpointing = mode
        config.memory.activation_offload = False
        before = estimate_memory(config)
        config.memory.activation_offload = True
        after = estimate_memory(config)
        assert before["activation_offload_host_gb"] == 0.0
        assert after["activation_offload_host_gb"] > 0
        assert after["activation_memory_gb"] <= before["activation_memory_gb"]
    config.memory.activation_offload = False
    config.model.torch_dtype = "bfloat16"
    config.train.optimizer = "adamw_bf16"
    stochastic = estimate_memory(config)["optimizer_memory_gb"]
    config.train.bf16_rounding = "kahan"
    assert estimate_memory(config)["optimizer_memory_gb"] == pytest.approx(stochastic * 1.5)  # 6 B/param, not 4


def test_profile_refuses_to_guess_an_unknown_model(tmp_path, model_dir):
    from palingenesis.agent_tooling.profile_memory import estimate_memory

    config = _config(tmp_path, model_dir, CHATS)
    config.model.name_or_path = str(tmp_path / "does-not-exist")
    with pytest.raises(SystemExit, match="Cannot load the model config"):
        estimate_memory(config)


def test_profile_accepts_gpu_flag(tmp_path, model_dir, monkeypatch, capsys):
    import yaml

    from palingenesis.agent_tooling import profile_memory

    config_file = tmp_path / "c.yaml"
    config_file.write_text(
        yaml.safe_dump({"model": {"name_or_path": str(model_dir)}, "data": {"dataset": "x", "max_seq_length": 128}})
    )
    monkeypatch.setattr(sys, "argv", ["pgs", "--config", str(config_file), "--gpu", "40"])
    with pytest.raises(SystemExit) as exit_info:
        profile_memory.main()
    assert exit_info.value.code == 0
    assert "of 40 GiB" in capsys.readouterr().out


def test_monitor_throughput_flags_sustained_drops_not_batch_variation():
    varied = [f"step={i} loss=1.0 lr=1e-5 tok/s={300 if i % 2 else 900} grad_norm=1 dt=1s" for i in range(1, 31)]
    assert not any("Throughput" in i for i in analyze_run(parse_training_log("\n".join(varied)))["issues"])
    slowed = [f"step={i} loss=1.0 lr=1e-5 tok/s={900 if i <= 20 else 200} grad_norm=1 dt=1s" for i in range(1, 31)]
    assert any("Throughput dropped" in i for i in analyze_run(parse_training_log("\n".join(slowed)))["issues"])


# ── diagnostics: cost, machine-readable output, exit codes ───────────────────


def test_training_samples_do_not_buffer_a_length_group(tmp_path, model_dir, monkeypatch):
    """Length grouping only reorders rows; with it, showing 3 samples tokenised a whole 512-row buffer."""
    from palingenesis import data
    from palingenesis.agent_tooling._pipeline import load_tokenizer, training_samples

    rows = [
        {"messages": [{"role": "user", "content": f"q {i}"}, {"role": "assistant", "content": f"a {i}"}]}
        for i in range(600)
    ]
    config = _config(tmp_path, model_dir, rows, length_group_buffer=512)
    config.train.per_device_batch_size = 4
    calls = []
    process = data.ChatDataset._process
    monkeypatch.setattr(data.ChatDataset, "_process", lambda self, ex: calls.append(1) or process(self, ex))
    stream = training_samples(config, load_tokenizer(config))
    first = [next(stream) for _ in range(3)]
    assert len(calls) == 3
    assert [load_tokenizer(config).decode(s["input_ids"]).count("q ") for s in first] == [1, 1, 1]
    assert config.data.length_group_buffer == 512  # the caller's config is left alone


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pgs", *argv])
    with pytest.raises(SystemExit) as exit_info:
        module.main()
    return exit_info.value.code


def _yaml(tmp_path, config, **data):
    import yaml

    path = tmp_path / "c.yaml"
    section = {"dataset": config.data.dataset, "streaming": False, "max_seq_length": 256, **data}
    path.write_text(
        yaml.safe_dump(
            {"model": {"name_or_path": config.model.name_or_path, "torch_dtype": "float32"}, "data": section}
        )
    )
    return str(path)


def test_diagnose_json_prints_only_json_on_stdout(tmp_path, model_dir, monkeypatch, capsys):
    from palingenesis.agent_tooling import diagnose

    path = _yaml(tmp_path, _config(tmp_path, model_dir, CHATS))
    code = _run(diagnose, ["--config", path, "--mode", "pre", "--json", "--gpu_memory_gb", "80"], monkeypatch)
    report = json.loads(capsys.readouterr().out)  # progress lines go to stderr
    assert report["mode"] == "pre" and report["checks"]["masking"]["samples_checked"] == 6
    assert code == (0 if report["overall"] == "HEALTHY" else 1)
    log = tmp_path / "train.log"
    log.write_text(LOG)
    _run(diagnose, ["--config", path, "--mode", "post", "--log_file", str(log), "--json"], monkeypatch)
    assert json.loads(capsys.readouterr().out)["checks"]["loss"]["steps"] == 4


USER_ONLY = [{"messages": [{"role": "user", "content": f"question {i}"}]} for i in range(4)]  # nothing to train on


def test_inspect_exits_1_when_nothing_is_trained(tmp_path, model_dir, monkeypatch, capsys):
    from palingenesis.agent_tooling import inspect_batch

    bad = _yaml(tmp_path, _config(tmp_path, model_dir, USER_ONLY))
    assert _run(inspect_batch, ["--config", bad], monkeypatch) == 1
    out = capsys.readouterr().out
    assert "CRITICAL" in out and "ALL tokens" not in out
    good = _yaml(tmp_path, _config(tmp_path, model_dir, CHATS))
    assert _run(inspect_batch, ["--config", good, "--num_samples", "2"], monkeypatch) == 0


def test_check_gradients_reports_an_empty_dataset(tmp_path, model_dir, monkeypatch, capsys):
    from palingenesis.agent_tooling import check_gradients

    bad = _yaml(tmp_path, _config(tmp_path, model_dir, USER_ONLY))
    assert _run(check_gradients, ["--config", bad], monkeypatch) == 1  # used to die on a KeyError in the report
    assert "CRITICAL: No samples" in capsys.readouterr().out


def test_check_gradients_gives_the_model_only_its_inputs(tmp_path, model_dir, monkeypatch):
    """With data.turn_scaling the collated batch also holds loss_weights, which is not a model input."""
    from transformers import LlamaForCausalLM

    from palingenesis.agent_tooling.check_gradients import check_gradients

    two_turns = [
        {
            "messages": c["messages"]
            + [{"role": "user", "content": "more"}, {"role": "assistant", "content": "a second answer"}]
        }
        for c in CHATS
    ]
    config = _config(tmp_path, model_dir, two_turns, turn_scaling="progressive")
    config.model.torch_dtype = "float32"
    config.train.bf16 = False
    seen = {}
    forward = LlamaForCausalLM.forward

    def spy(self, *args, **kwargs):
        seen["keys"] = sorted(kwargs)
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(LlamaForCausalLM, "forward", spy)
    result = check_gradients(config)
    assert seen["keys"] == ["attention_mask", "input_ids", "labels"]
    assert result["loss"] > 0 and result["total_grad_norm"] > 0


def test_check_loss_reports_logged_step_numbers():
    from palingenesis.agent_tooling.check_loss import analyze_losses, parse_losses_from_text

    lines = [f"step={10 * i} loss={'nan' if i == 7 else 1.0} lr=1e-5 tok/s=1 grad_norm=1 dt=1s" for i in range(1, 9)]
    pairs = sorted(parse_losses_from_text("\n".join(lines)))
    analysis = analyze_losses([v for _, v in pairs], steps=[s for s, _ in pairs])
    assert analysis.nan_steps == [70]  # the logged step, not its position in the list (6)
    assert "steps: [70]" in analysis.issues[0]
    flat = [2.0] * 30 + [50.0] + [2.0] * 200
    spiky = analyze_losses(flat, steps=[5 * (i + 1) for i in range(len(flat))])
    assert spiky.spike_steps == [155]
    assert spiky.plateau_ranges and spiky.plateau_ranges[-1][1] == 5 * len(flat)
    assert analyze_losses(flat).spike_steps == [30]  # without step numbers: positions, as before
