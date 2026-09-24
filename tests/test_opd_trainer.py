"""OPD trainer end to end on CPU: tiny random models on the real Qwen3 / Qwen3.5 tokenizers.

One run covers the wiring the unit tests cannot: prompt rendering per model,
HF rollouts with behaviour log-probs, routing each source to its teacher, a
shared-vocabulary teacher (full_rkl) and a cross-tokenizer one (xtok with the
dense term) in the same step, evaluation, and the saved checkpoint.
"""

import json
import sys

import pytest

sys.path.insert(0, "src")

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def tiny_model(tmp_path, tokenizer_name, name, seed):
    try:
        tok = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:  # noqa: BLE001 — offline or not cached
        pytest.skip(f"tokenizer {tokenizer_name} unavailable: {e}")
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                      max_position_embeddings=1024, tie_word_embeddings=True,
                                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    path = tmp_path / name
    transformers.Qwen3ForCausalLM(config).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("opd_models")
    return {"student": tiny_model(tmp, "Qwen/Qwen3-0.6B", "student", 0),
            "same": tiny_model(tmp, "Qwen/Qwen3-0.6B", "teacher_same", 1),
            "other": tiny_model(tmp, "Qwen/Qwen3.5-0.8B", "teacher_other", 2)}


def write_prompts(path, n=24):
    rows = [{"messages": [{"role": "user", "content": f"What is {i} plus {i + 1}?"}], "answer": str(2 * i + 1)}
            for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def make_config(tmp_path, models, **teacher_losses):
    from palingenesis.opd.config import OPDConfig

    config = OPDConfig()
    prompts = write_prompts(tmp_path / "prompts.jsonl")
    settings = {
        "model.student": models["student"], "model.chat_template_kwargs": {"enable_thinking": False},
        # Liger patches the model classes process-wide; later tests here run Qwen3 on the CPU
        "model.use_liger_kernel": False,
        "teachers.same.model": models["same"], "teachers.other.model": models["other"],
        "sources.math.path": prompts, "sources.math.teacher": "same", "sources.math.max_new_tokens": 8,
        "sources.math.dev_size": 4,
        "sources.chat.path": prompts, "sources.chat.teacher": "other", "sources.chat.max_new_tokens": 8,
        "sources.chat.dev_size": 4,
        "rollout.batch_prompts": 6, "rollout.group_size": 2, "loss.xtok_dense_weight": 0.5,
        "train.output_dir": str(tmp_path / "run"), "train.steps": 2, "train.learning_rate": 1e-3,
        "train.warmup_steps": 1, "train.eval_every": 1, "train.eval_samples": 4, "train.score_micro_seqs": 4,
    }
    for name, loss in teacher_losses.items():
        settings[f"teachers.{name}.loss"] = loss
    for key, value in settings.items():
        config.set(key, value, "test")
    return config


def test_multi_teacher_training_run(tmp_path, models, caplog):
    from palingenesis.opd.trainer import OPDTrainer

    trainer = OPDTrainer(make_config(tmp_path, models))
    assert trainer.kinds == {"same": "full_rkl", "other": "xtok"}
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, step, metrics))
    trainer.train()

    steps = [m for kind, _, m in logged if kind == "step"]
    evals = [m for kind, _, m in logged if kind == "eval"]
    assert len(steps) == 2 and len(evals) == 3                  # before, after step 1, final
    for metrics in steps:
        assert metrics["staleness"] == 0 and metrics["dropped_samples"] == 0
        assert metrics["rollout_tokens"] > 0 and metrics["grad_norm"] > 0
        routed = [k for k in metrics if k.startswith("tokens/")]
        assert routed and set(routed) <= {"tokens/same", "tokens/other"}
    both = {k.split("/")[1] for m in steps for k in m if k.startswith("kl/")}
    assert both == {"same", "other"}                            # each source's prompts went to its teacher
    assert any("dense_kl/other" in m for m in steps)            # xtok's dense term (xtok_dense_weight 0.5)
    assert {"dev_kl/math", "dev_kl_full/math", "dev_kl/chat", "dev_acc/math"} <= set(evals[0])
    assert "dev_kl_full/chat" not in evals[0]                   # xtok has no exact KL

    saved = tmp_path / "run" / "final"
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(saved)
    for (name, p), (_, q) in zip(trainer.student.named_parameters(), reloaded.named_parameters()):
        torch.testing.assert_close(p.cpu(), q.cpu(), msg=name)
    assert json.loads((saved / "opd_config.json").read_text())["teachers"]["other"]["model"] == models["other"]


@pytest.mark.parametrize("loss", ["sampled_rkl", "xtok", "rs_kd"])
def test_shared_vocabulary_teacher_with_other_losses(tmp_path, models, loss):
    from palingenesis.opd.trainer import OPDTrainer

    config = make_config(tmp_path, models, same=loss)
    config.sources.pop("chat")
    config.teachers.pop("other")
    config.train.eval_every = 0
    trainer = OPDTrainer(config)
    assert trainer.kinds == {"same": loss}
    trainer.train()


def test_shared_vocabulary_loss_with_another_tokenizer_is_rejected(tmp_path, models):
    from palingenesis.opd.config import OPDConfigError
    from palingenesis.opd.trainer import OPDTrainer

    with pytest.raises(OPDConfigError, match="shares the student's vocabulary.*Use loss xtok"):
        OPDTrainer(make_config(tmp_path, models, other="full_rkl"))


def test_score_pool_reads_the_option_letter_logits(models):
    from palingenesis.opd.formatting import build_messages, encode_prompt, letter_token_ids
    from palingenesis.opd.score_pool import score_rows

    tok = transformers.AutoTokenizer.from_pretrained(models["same"])
    model = transformers.AutoModelForCausalLM.from_pretrained(models["same"]).eval()
    rows = [{"question": f"Q{i}?", "options": [("A", "x"), ("B", "y"), ("C", "z")], "answer": "B",
             "category": "c"} for i in range(5)]
    letters = letter_token_ids(tok)
    scored = list(score_rows(model, tok, rows, [], letters, batch_size=2, device="cpu"))
    for row, out in zip(rows, scored):
        ids = torch.tensor([encode_prompt(tok, build_messages(row))])
        with torch.no_grad():
            last = model(ids).logits[0, -1]
        want = max("ABC", key=lambda letter: last[letters[letter]].item())
        assert out["teacher_answer"] == want and out["teacher_correct"] == (want == "B")


def test_hf_rollout_records_the_sampling_log_probs(models):
    """Behaviour log-probs are those of the temperature-scaled distribution each token was drawn from."""
    from palingenesis.opd.rollout import HFRollout

    model = transformers.AutoModelForCausalLM.from_pretrained(models["student"]).eval()
    engine = HFRollout(model, stop_ids=(151645,), pad_id=151643, micro_seqs=4)
    prompts = [[9707, 11, 1879], [3838, 374, 220, 17, 10, 17, 30]]
    rollouts = engine.generate(prompts, [6, 6], temperature=0.7)
    for prompt, rollout in zip(prompts, rollouts):
        ids = torch.tensor([prompt + rollout.completion_ids])
        with torch.no_grad():
            logits = model(ids).logits[0, len(prompt) - 1: -1] / 0.7
        want = torch.log_softmax(logits.float(), -1).gather(1, torch.tensor(rollout.completion_ids)[:, None])
        torch.testing.assert_close(torch.tensor(rollout.logprobs), want.squeeze(1), atol=1e-4, rtol=1e-4)
    assert all(r.finish_reason in ("stop", "length") for r in rollouts)
    assert [len(r.completion_ids) for r in engine.generate(prompts, [3, 5], temperature=0.0)] == [3, 5]


def test_resume_continues_from_the_checkpoint(tmp_path, models):
    from palingenesis.opd.trainer import TRAINER_STATE_FILE, OPDTrainer

    def config(steps, **extra):
        config = make_config(tmp_path, models)
        config.sources.pop("chat")
        config.teachers.pop("other")
        config.train.eval_every = 0
        config.train.steps = steps
        config.train.save_steps = 2
        for key, value in extra.items():
            setattr(config.train, key, value)
        return config

    first = OPDTrainer(config(3))
    first.train()
    run = tmp_path / "run"
    assert (run / "step_2" / TRAINER_STATE_FILE).exists() and not (run / "final" / TRAINER_STATE_FILE).exists()
    saved = transformers.AutoModelForCausalLM.from_pretrained(run / "step_2")
    saved_state = torch.load(run / "step_2" / TRAINER_STATE_FILE, weights_only=False)
    (run / "step_9").mkdir()                                    # a save interrupted before its trainer state

    resumed = OPDTrainer(config(4, resume_from="auto"))
    assert resumed.resume_path == str(run / "step_2") and resumed.start_step == 2
    assert resumed.weights.version == 2
    for (name, p), (_, q) in zip(resumed.student.named_parameters(), saved.named_parameters()):
        torch.testing.assert_close(p.cpu(), q.cpu(), msg=name)
    for got, want in zip(resumed.opt.state_dict()["state"].values(), saved_state["optimizer"]["state"].values()):
        torch.testing.assert_close(got["exp_avg_sq"].cpu(), want["exp_avg_sq"].cpu())
    logged = []
    resumed._log = lambda kind, metrics, step: logged.append((kind, step, metrics))
    resumed.train()
    steps = [(step, m) for kind, step, m in logged if kind == "step"]
    assert [step for step, _ in steps] == [3, 4]
    assert all(m["staleness"] == 0 and m["dropped_samples"] == 0 for _, m in steps)   # no batch generated stale
    assert steps[0][1]["lr"] == pytest.approx(resumed._lr_at(2)) != resumed._lr_at(0)   # the schedule continues
    assert (run / "step_4" / TRAINER_STATE_FILE).exists()


def test_resume_from_path_needs_a_complete_checkpoint(tmp_path):
    from palingenesis.opd.config import OPDConfigError
    from palingenesis.opd.trainer import TRAINER_STATE_FILE, checkpoint_steps, resolve_resume

    assert resolve_resume("", str(tmp_path)) is None
    assert resolve_resume("auto", str(tmp_path / "missing")) is None                   # fresh start
    for step, complete in [(2, True), (10, True), (12, False)]:
        (tmp_path / f"step_{step}").mkdir()
        if complete:
            (tmp_path / f"step_{step}" / TRAINER_STATE_FILE).write_bytes(b"")
    (tmp_path / "final").mkdir()
    assert resolve_resume("auto", str(tmp_path)) == str(tmp_path / "step_10")          # numeric, complete only
    assert checkpoint_steps(str(tmp_path), complete=False)[-1] == str(tmp_path / "step_12")
    with pytest.raises(OPDConfigError, match="not a complete OPD checkpoint"):
        resolve_resume(str(tmp_path / "final"), str(tmp_path))


@pytest.mark.parametrize("weighting", ["sure", "entropy"])
def test_token_weighting_run(tmp_path, models, weighting):
    from palingenesis.opd.trainer import OPDTrainer

    config = make_config(tmp_path, models)
    config.sources.pop("chat")
    config.teachers.pop("other")
    config.train.eval_every = 0
    config.loss.token_weighting = weighting
    trainer = OPDTrainer(config)
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append(metrics)
    trainer.train()
    assert all(m["grad_norm"] > 0 for m in logged)


def test_token_weighting_rules():
    from types import SimpleNamespace

    from palingenesis.opd.config import OPDLossConfig
    from palingenesis.opd.trainer import OPDTrainer

    torch.manual_seed(0)
    head = torch.nn.Linear(4, 7, bias=False)
    hidden = torch.randn(10, 4)
    weights = torch.full((10,), 0.1)
    behaviour = torch.log(torch.linspace(0.05, 1.0, 10))

    def run(**options):
        fake = SimpleNamespace(config=SimpleNamespace(loss=OPDLossConfig(**options)), head=head)
        return OPDTrainer._token_weights(fake, weights, hidden, behaviour)

    torch.testing.assert_close(run(), weights)
    torch.testing.assert_close(run(token_weighting="sure", sure_alpha=0.5), weights * (1 + 0.5 * (1 - behaviour.exp())))
    kept = run(token_weighting="entropy", entropy_keep=0.3)
    from palingenesis.opd import losses
    entropy = losses.token_entropy(hidden, head)
    assert (kept > 0).sum() == 3 and set((kept > 0).nonzero().flatten().tolist()) == set(entropy.topk(3).indices.tolist())
