"""pgs screen: RL pool difficulty screening through the RL pipeline, with a scripted engine (no GPU)."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from palingenesis.rl.screen import band, group_stats, keep, main, summarize  # noqa: E402
from palingenesis.rl.trajectory import Trajectory  # noqa: E402


class CountingEngine:
    """Answers \\boxed{<answer>} for the first `solved[question]` samples of each question and \\boxed{0} after."""

    def __init__(self, tok, eot, solved):
        self.tok, self.eot, self.solved, self.seen, self.version = tok, eot, solved, {}, 0

    def generate(self, prompts, max_new_tokens, temperature):
        from palingenesis.opd.rollout import Rollout

        out = []
        for prompt in prompts:
            text = self.tok.decode(prompt)
            question = next(q for q in self.solved if q in text)
            n = self.seen.get(question, 0)
            self.seen[question] = n + 1
            answer = "7" if n < self.solved[question] else "0"
            ids = self.tok.encode(f"The answer is \\boxed{{{answer}}}.", add_special_tokens=False) + [self.eot]
            out.append(Rollout(ids, [-0.5] * len(ids), "stop", self.version))
        return out

    def update_weights(self, named, version):
        self.version = version

    def sleep(self):
        pass

    def wake(self):
        pass


def _t(reward, scored=True, finish="stop"):
    t = Trajectory({}, 0, [1], [], tokens=[5, 5, 5], mask=[True] * 3, logprobs=[0.0] * 3, finish=finish)
    t.reward, t.scored = reward, scored
    return t


def test_group_stats_bands_and_keep():
    record = group_stats([_t(1.0), _t(0.0), _t(0.2), _t(0.0, scored=False, finish="error")], threshold=0.5)
    assert record["samples"] == 4 and record["scored"] == 3 and record["successes"] == 1
    assert record["pass_rate"] == pytest.approx(1 / 3) and record["mean_reward"] == pytest.approx(0.4)
    assert record["finishes"] == {"stop": 3, "error": 1} and record["completion_tokens"] == 3
    assert record["wilson_lo"] < 1 / 3 < record["wilson_hi"]
    assert band(record) == "partial" and keep(record, 0.0, 1.0) and not keep(record, 0.5, 1.0)
    failed = group_stats([_t(0.0, scored=False)], 0.5)
    assert failed["pass_rate"] is None and band(failed) == "failed" and not keep(failed, 0.0, 1.0)
    solved = group_stats([_t(1.0), _t(1.0)], 0.5)
    assert band(solved) == "solved" and not keep(solved, 0.0, 1.0) and keep(solved, 0.0, 1.01)
    s = summarize([({"task": "a"}, {**record, "kept": True}), ({"task": "b"}, {**solved, "kept": False})], ["task"])
    assert s["rows"] == 2 and s["partial"] == 1 and s["solved"] == 1 and s["kept"] == 1
    assert s["histogram"]["(0.25, 0.375]"] == 1 and s["histogram"]["(0.875, 1]"] == 1
    assert s["families"]["a"]["kept_share"] == 1.0 and s["families"]["b"]["kept"] == 0


def test_screen_cli_end_to_end_and_resume(tmp_path, monkeypatch, capsys):
    from palingenesis.opd.teachers import end_of_turn_id
    from tests.test_rl import tokenizer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    rows = [
        {"prompt": "Question alpha: what is 3 + 4?", "answer": "7", "task": "easy"},
        {"prompt": "Question beta: what is 3 + 4?", "answer": "7", "task": "mid"},
        {"prompt": "Question gamma: what is 3 + 4?", "answer": "7", "task": "hard"},
    ]
    data = tmp_path / "pool.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = tmp_path / "rl.yaml"
    config.write_text(
        f"model: {{policy: Qwen/Qwen3-0.6B, chat_template_kwargs: {{enable_thinking: false}}}}\n"
        f"data: {{dataset: {data}, metrics_key: [task]}}\n"
        "rewards: {correct: math}\nrollout: {backend: hf, max_new_tokens: 32, max_model_len: 1024}\n"
    )
    solved = {"alpha": 4, "beta": 2, "gamma": 0}
    engines = []

    def fake_engine(cfg, model, stop_ids, pad_id):
        engines.append(CountingEngine(tok, end_of_turn_id(tok, {"enable_thinking": False}), solved))
        return engines[-1]

    monkeypatch.setattr("palingenesis.rl.screen._engine", fake_engine)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: tok)
    monkeypatch.setattr(
        "palingenesis.opd.trainer.student_stop_ids", lambda *a, **k: (tok.convert_tokens_to_ids("<|im_end|>"),)
    )
    out = tmp_path / "screened.jsonl"
    code = main(["--config", str(config), "--k", "4", "--output", str(out), "--json", "--chunk", "2"])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["solved"], summary["partial"], summary["unsolved"], summary["kept"]) == (1, 1, 1, 1)
    assert summary["families"]["mid"]["mean_pass_rate"] == 0.5
    screened = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["screen"]["successes"] for r in screened] == [4, 2, 0]
    kept = [json.loads(line) for line in (tmp_path / "screened.kept.jsonl").read_text().splitlines()]
    assert kept == [rows[1]]  # the row as it was, no screen field
    # resume: nothing left to screen, no engine built; new bands apply to the stored records
    code = main(["--config", str(config), "--k", "4", "--output", str(out), "--json", "--max_rate", "1.5"])
    assert code == 0 and len(engines) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["kept"] == 2


def test_screen_refuses_an_empty_band(tmp_path):
    config = tmp_path / "rl.yaml"
    config.write_text("model: {policy: m}\ndata: {dataset: d.jsonl}\nrewards: {correct: math}\n")
    with pytest.raises(SystemExit):
        main(["--config", str(config), "--output", str(tmp_path / "o.jsonl"), "--min_rate", "0.5", "--max_rate", "0.5"])


def test_screen_rows_go_through_the_pipeline_rewards():
    """screen() itself on an RLPipeline: k rollouts per row, the sink sees every row in order."""
    import torch

    from palingenesis.opd.orchestrator import PublishedWeights
    from palingenesis.opd.teachers import end_of_turn_id
    from palingenesis.rl.chat import ChatFormat
    from palingenesis.rl.config import RLConfig
    from palingenesis.rl.data import PromptSampler
    from palingenesis.rl.pipeline import RLPipeline
    from palingenesis.rl.rewards import resolve_rewards
    from palingenesis.rl.screen import screen
    from tests.test_rl import tokenizer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    kwargs = {"enable_thinking": False}
    eot = end_of_turn_id(tok, kwargs)
    config = RLConfig()
    for key, value in {"model.policy": "x", "model.chat_template_kwargs": kwargs, "rollout.max_new_tokens": 32}.items():
        config.set(key, value)
    rows = [{"prompt": "delta: 3 + 4?", "answer": "7"}, {"prompt": "omega: 3 + 4?", "answer": "7"}]
    engine = CountingEngine(tok, eot, {"delta": 1, "omega": 3})
    pipeline = RLPipeline(
        tok,
        ChatFormat(tok, eot, kwargs),
        engine,
        PublishedWeights(torch.nn.Linear(1, 1)),
        config,
        PromptSampler(rows),
        resolve_rewards({"m": SimpleNamespace(fn="math", weight=1.0, args={})}),
        None,
        None,
        (eot,),
    )
    seen = []
    try:
        out = screen(pipeline, rows, 3, 1.0, 0.5, chunk=1, sink=lambda row, record: seen.append(row["prompt"]))
    finally:
        pipeline.close()
    assert [r["successes"] for _, r in out] == [1, 3] and [r["kept"] for _, r in out] == [True, False]
    assert seen == [r["prompt"] for r in rows]
