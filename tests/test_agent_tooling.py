"""agent_tooling: the diagnostics must read what the trainer really logs and
consumes, and report exact numbers where they can."""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_tooling._logparse import parse_steps  # noqa: E402
from agent_tooling.check_loss import parse_losses_from_text  # noqa: E402
from agent_tooling.monitor_run import analyze_run, parse_training_log, print_brief  # noqa: E402
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
    assert [s["loss"] for s in steps] == [0.2697, 0.2917, 0.3121, 0.9027]   # never eval_loss=0.3454
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
    print_brief(result)                                   # used to raise KeyError
    assert "NO_DATA" in capsys.readouterr().out


def test_monitor_grad_norm_is_judged_against_the_run_itself():
    lines = [f"step={i} loss=1.0 lr=1e-5 tok/s=100 grad_norm=15.0 dt=1s" for i in range(1, 30)]
    calm = analyze_run(parse_training_log("\n".join(lines)))
    assert not any("Gradient norm" in i for i in calm["issues"])      # 15 is normal for this run
    spiked = analyze_run(parse_training_log("\n".join(lines + ["step=30 loss=1.0 lr=1e-5 tok/s=100 grad_norm=120 dt=1s"])))
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
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2))
    path = tmp_path / "model"
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path


def _config(tmp_path, model_dir, rows, **data):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = Config()
    config.model.name_or_path = str(model_dir)
    config.data.dataset = str(data_file)          # a local file: the trainer's common case
    config.data.streaming = False
    config.data.max_seq_length = 256
    config.data.length_group_buffer = 0
    for k, v in data.items():
        setattr(config.data, k, v)
    return config


CHATS = [{"messages": [{"role": "user", "content": f"question {i}"},
                       {"role": "assistant", "content": f"answer number {i}"}]} for i in range(6)]


def test_training_samples_are_the_trainers_stream(tmp_path, model_dir):
    from agent_tooling._pipeline import load_tokenizer, training_samples
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
    from agent_tooling._pipeline import load_tokenizer, training_samples

    pairs = [{"prompt": [{"role": "user", "content": "q"}],
              "chosen": [{"role": "assistant", "content": "good"}],
              "rejected": [{"role": "assistant", "content": "bad bad"}]}] * 2
    config = _config(tmp_path, model_dir, pairs)
    config.dpo.enabled = True
    samples = list(training_samples(config, load_tokenizer(config)))
    assert [s["side"] for s in samples] == ["chosen", "rejected"] * 2


def test_validate_masking_runs_on_local_data_without_false_pad_bug(tmp_path, model_dir):
    from agent_tooling.validate_masking import validate

    config = _config(tmp_path, model_dir, CHATS)       # gpt2: pad token == eos token
    report = validate(config, num_samples=6)
    assert report["total_samples"] == 6
    assert report["pad_tokens_trained"] == 0
    assert not any("BUG" in i or "CRITICAL" in i for i in report["issues"])


# ── memory profile ──────────────────────────────────────────────────────────


def test_profile_counts_parameters_exactly(tmp_path, model_dir):
    from transformers import AutoModelForCausalLM

    from agent_tooling.profile_memory import estimate_memory

    config = _config(tmp_path, model_dir, CHATS)
    config.model.torch_dtype = "float32"
    est = estimate_memory(config)
    real = AutoModelForCausalLM.from_pretrained(model_dir)
    n = sum(p.numel() for p in real.parameters())
    assert est["total_params_B"] * 1e9 == pytest.approx(n)
    assert est["params_memory_gb"] == pytest.approx(n * 4 / 1e9)
    assert est["optimizer_memory_gb"] == pytest.approx(2 * n * 4 / 1e9)      # AdamW: 2 states, fp32 weights
    config.model.torch_dtype = "bfloat16"
    config.train.optimizer = "lion8bit"
    assert estimate_memory(config)["optimizer_memory_gb"] == pytest.approx(n / 1e9)


def test_profile_refuses_to_guess_an_unknown_model(tmp_path, model_dir):
    from agent_tooling.profile_memory import estimate_memory

    config = _config(tmp_path, model_dir, CHATS)
    config.model.name_or_path = str(tmp_path / "does-not-exist")
    with pytest.raises(SystemExit, match="Cannot load the model config"):
        estimate_memory(config)


def test_profile_accepts_gpu_flag(tmp_path, model_dir, monkeypatch, capsys):
    import yaml

    from agent_tooling import profile_memory

    config_file = tmp_path / "c.yaml"
    config_file.write_text(yaml.safe_dump({"model": {"name_or_path": str(model_dir)},
                                           "data": {"dataset": "x", "max_seq_length": 128}}))
    monkeypatch.setattr(sys, "argv", ["pgs", "--config", str(config_file), "--gpu", "40"])
    with pytest.raises(SystemExit) as exit_info:
        profile_memory.main()
    assert exit_info.value.code == 0
    assert "of 40 GB" in capsys.readouterr().out


def test_monitor_throughput_flags_sustained_drops_not_batch_variation():
    varied = [f"step={i} loss=1.0 lr=1e-5 tok/s={300 if i % 2 else 900} grad_norm=1 dt=1s" for i in range(1, 31)]
    assert not any("Throughput" in i for i in analyze_run(parse_training_log("\n".join(varied)))["issues"])
    slowed = [f"step={i} loss=1.0 lr=1e-5 tok/s={900 if i <= 20 else 200} grad_norm=1 dt=1s" for i in range(1, 31)]
    assert any("Throughput dropped" in i for i in analyze_run(parse_training_log("\n".join(slowed)))["issues"])
