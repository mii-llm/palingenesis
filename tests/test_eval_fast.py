"""pgs eval's single-turn executor and what surrounds it: the same records and scores as the RL pipeline, one
request per greedy item, prompts in prefix order, the generation cache, scorer processes, the result stream,
categories and metrics in the summary, the report, contamination, suites, the engines' layout (no GPU)."""

import json
import sys

import pytest

pytest.importorskip("torch")

from palingenesis.eval.config import EvalConfig, EvalConfigError, SamplingConfig, load_suite, load_task  # noqa: E402
from tests.test_eval import ScriptedEngine, _sums, _write, qwen  # noqa: E402, F401

RIGHT, WRONG = "3 + 4 = 7.\nAnswer: 7", "Answer: 8"


def _evaluator(tok, engine, **kwargs):
    from palingenesis.eval.runner import Evaluator

    return Evaluator(tok, engine, "Qwen/Qwen3-0.6B", engine_options={"max_model_len": 1024}, resamples=200, **kwargs)


def test_direct_executor_gives_the_pipeline_s_records(tmp_path, qwen, monkeypatch):  # noqa: F811
    from palingenesis.eval.runner import Evaluator

    tok, eot = qwen
    plain = {"alpha": [RIGHT], "beta": [RIGHT, WRONG], "gamma": [WRONG, WRONG, RIGHT], "delta": [WRONG]}
    task = load_task({**_sums(tmp_path), "system_prompt": "Be brief."})
    sampling = SamplingConfig(temperature=0.7)
    outputs = {}
    for name in ("direct", "pipeline"):
        engine = ScriptedEngine(tok, eot, plain, truncate=("delta",))
        with _evaluator(tok, engine) as evaluator:
            if name == "pipeline":
                monkeypatch.setattr(Evaluator, "_direct", lambda self, *a: False)
            for mode in ("plain", "think"):
                summary = evaluator.run(task, mode, k=3, out_dir=tmp_path / name, sampling=sampling)
                rows = [json.loads(x) for x in (tmp_path / name / f"sums.{mode}.jsonl").read_text().splitlines()]
                samples = [json.loads(x) for x in (tmp_path / name / f"sums.{mode}.samples.jsonl").read_text().splitlines()]
                outputs[name, mode] = (summary, rows, sorted(samples, key=lambda s: (s["id"], s["sample"])))
    for mode in ("plain", "think"):
        (s_direct, rows_direct, samples_direct), (s_pipe, rows_pipe, samples_pipe) = outputs["direct", mode], outputs["pipeline", mode]
        assert [{k: v for k, v in r.items() if k != "key"} for r in rows_direct] == rows_pipe
        assert samples_direct == samples_pipe  # prompt, completion, reasoning, score, rewards, finish, tokens, loop
        for key in ("mean_score", "pass@1", "pass@2", "truncated_share", "loop_share", "completion_tokens", "finish"):
            assert s_direct[key] == s_pipe[key], key
    assert outputs["direct", "plain"][2][0]["prompt"][0] == {"role": "system", "content": "Be brief."}


def test_everything_is_submitted_at_once_in_prefix_order_and_greedy_is_one_request(tmp_path, qwen):  # noqa: F811
    tok, eot = qwen
    task = load_task(_sums(tmp_path, n=5))
    seen = []

    class Engine(ScriptedEngine):
        def generate(self, prompts, max_new_tokens, temperature):
            seen.append([list(p) for p in prompts])
            return super().generate(prompts, max_new_tokens, temperature)

    script = {name: [RIGHT] for name in ("alpha", "beta", "gamma", "delta", "epsilon")}
    with _evaluator(tok, Engine(tok, eot, script)) as evaluator:
        summary = evaluator.run(task, "plain", k=4, sampling=SamplingConfig(temperature=0.0))
        # greedy: the 4 samples of an item are one request (they are the same completion); one engine call for all
        assert len(seen) == 1 and len(seen[0]) == 5 and summary["samples"] == 20 and summary["engine_requests"] == 5
        assert seen[0] == sorted(seen[0])  # lexicographic: prompts sharing a prefix reach the prefix cache together
        assert summary["mean_score"]["estimate"] == 1.0 and summary["pass@4"]["estimate"] == 1.0
        assert summary["time_to_first_result"] is not None and summary["items_per_second"] > 0
        assert summary["generated_tokens"] == 5 * summary["completion_tokens"]["mean"]
        seen.clear()
        evaluator.run(task, "plain", k=4, sampling=SamplingConfig(temperature=0.8))
        assert len(seen) == 1 and len(seen[0]) == 20  # sampled: k requests per item, merged to n = k by the engine


def test_generation_cache_and_rescoring(tmp_path, qwen):  # noqa: F811
    tok, eot = qwen
    task = load_task(_sums(tmp_path))
    script = {"alpha": [RIGHT], "beta": [WRONG], "gamma": [RIGHT], "delta": [RIGHT]}
    engine = ScriptedEngine(tok, eot, script)
    cache = tmp_path / "cache"
    with _evaluator(tok, engine, cache_dir=str(cache)) as evaluator:
        first = evaluator.run(task, "plain", out_dir=tmp_path / "a")
    assert first["cache_hits"] == 0 and first["engine_requests"] == 4 and len(list(cache.glob("*.jsonl"))) == 1
    calls = len(engine.samplings)
    with _evaluator(tok, engine, cache_dir=str(cache)) as evaluator:  # another process, the same checkpoint
        again = evaluator.run(task, "plain", out_dir=tmp_path / "b")
        assert len(engine.samplings) == calls and again["cache_hits"] == 4 and again["engine_requests"] == 0
        assert again["mean_score"] == first["mean_score"]
        # a changed scorer scores the cached completions again: nothing is generated
        strict = load_task({**_sums(tmp_path), "scorer": "exact_match", "scorer_args": {"field": "answer"}})
        rescored = evaluator.run(strict, "plain", out_dir=tmp_path / "c")
        assert len(engine.samplings) == calls and rescored["mean_score"]["estimate"] == 0.0
        # another sampling, another mode or another k is another request
        evaluator.run(task, "think", out_dir=tmp_path / "d")
        evaluator.run(task, "plain", k=2, sampling=SamplingConfig(temperature=0.5), out_dir=tmp_path / "e")
        assert len(engine.samplings) == calls + 2
    lines = next(cache.glob("*.jsonl")).read_text().splitlines()
    assert len(lines) == 12 and set(json.loads(lines[0])) == {"key", "value"}
    # a cut last line (an interrupted run) is skipped
    with next(cache.glob("*.jsonl")).open("a") as f:
        f.write('{"key": "abc", "val')
    with _evaluator(tok, engine, cache_dir=str(cache)) as evaluator:
        assert evaluator.run(task, "plain")["cache_hits"] == 4


def test_model_fingerprint_follows_the_weights(tmp_path):
    from palingenesis.eval.direct import model_fingerprint, request_key

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"a" * 10)
    before = model_fingerprint(str(tmp_path))
    assert before == model_fingerprint(str(tmp_path))
    (tmp_path / "model.safetensors").write_bytes(b"b" * 11)
    assert model_fingerprint(str(tmp_path)) != before
    assert model_fingerprint("no/such-model-anywhere") == model_fingerprint("no/such-model-anywhere")
    assert request_key("f", [1, 2, 3], 16, "s", 1) != request_key("f", [1, 2, 4], 16, "s", 1)
    assert request_key("f", [1, 2, 3], 16, "s", 1) != request_key("g", [1, 2, 3], 16, "s", 1)
    assert request_key("f", [1, 2, 3], 16, "s", 1) != request_key("f", [1, 2, 3], 16, "s", 2)


def test_scorer_processes(tmp_path, qwen):  # noqa: F811
    """Plain scorers run in spawned processes; their scores are the in-thread ones."""
    from palingenesis.eval import direct

    tok, eot = qwen
    task = load_task(_sums(tmp_path))
    script = {"alpha": [RIGHT], "beta": [WRONG], "gamma": [RIGHT], "delta": [WRONG, RIGHT]}
    with _evaluator(tok, ScriptedEngine(tok, eot, script), score_workers=1) as evaluator:
        summary = evaluator.run(task, "plain", k=2, sampling=SamplingConfig(temperature=0.7), out_dir=tmp_path / "out")
        assert evaluator._pool is not None and evaluator._pool.workers == 1
    assert summary["mean_score"]["estimate"] == pytest.approx((1 + 0 + 1 + 0.5) / 4) and summary["unscored_samples"] == 0
    assert direct.auto_score_workers(-1, 100) == 0 and direct.auto_score_workers(3, 1) == 3
    assert direct.combine({"a": 1.0, "b": None, "c": 0.5}, {"a": 2.0}) == (2.5, {"a": 1.0, "b": None, "c": 0.5})
    assert direct.combine({"a": 1.0, "b": direct.SKIPPED_MARK}, {})[0] is None
    assert direct.combine({"a": None}, {})[0] is None


def test_a_failing_scorer_is_recorded_and_the_run_goes_on(tmp_path, qwen, monkeypatch):  # noqa: F811
    tok, eot = qwen

    def brittle(completion, **row):
        if "8" in completion:
            raise ValueError("cannot grade this")
        return 1.0

    monkeypatch.setitem(sys.modules, "brittle_scorer", type(sys)("brittle_scorer"))
    sys.modules["brittle_scorer"].score = brittle
    task = load_task({**_sums(tmp_path), "scorer": "brittle_scorer:score", "scorer_args": {}})
    script = {"alpha": [RIGHT], "beta": [WRONG], "gamma": [RIGHT], "delta": [RIGHT]}
    with _evaluator(tok, ScriptedEngine(tok, eot, script)) as evaluator:
        summary = evaluator.run(task, "plain", out_dir=tmp_path / "out")
    assert summary["items"] == 3 and summary["unscored_items"] == 1 and summary["error_share"] == 0.25
    samples = [json.loads(x) for x in (tmp_path / "out" / "sums.plain.samples.jsonl").read_text().splitlines()]
    assert [s["error"] for s in samples if s["finish"] == "error"] == ["ValueError: cannot grade this"]


def test_summary_metrics_categories_and_stderr():
    from palingenesis.eval.runner import summarize

    rows = [
        {"id": "t/0", "scores": [1.0], "finish": ["stop"], "tokens": [5], "loop": [False], "category": "a",
         "metrics": {"strict": [1.0], "flexible": [1.0]}},
        {"id": "t/1", "scores": [0.0], "finish": ["stop"], "tokens": [5], "loop": [False], "category": "a",
         "metrics": {"strict": [0.0], "flexible": [1.0]}},
        {"id": "t/2", "scores": [1.0], "finish": ["length"], "tokens": [9], "loop": [True], "category": "b",
         "metrics": {"strict": [1.0], "flexible": [1.0]}},
        {"id": "t/3", "scores": [], "finish": ["error"], "tokens": [0], "loop": [False], "category": "b"},
    ]  # fmt: skip
    s = summarize(rows, resamples=200)
    assert s["items"] == 3 and s["stderr"] == pytest.approx((2 / 9 / 2 * 3 / 3) ** 0.5, rel=1e-6)
    assert s["metrics"]["strict"]["mean"] == pytest.approx(2 / 3) and s["metrics"]["flexible"] == {"mean": 1.0, "stderr": 0.0, "items": 3}
    assert s["categories"] == {"a": {"items": 2, "score": 0.5, "stderr": pytest.approx(0.5)}, "b": {"items": 1, "score": 1.0, "stderr": 0.0}}


def test_report_table_and_markdown(tmp_path, qwen):  # noqa: F811
    from palingenesis.eval import report

    tok, eot = qwen
    task = load_task({**_sums(tmp_path), "category_field": "qid"})
    script = {"alpha": [RIGHT], "beta": [WRONG], "gamma": [RIGHT], "delta": [RIGHT, WRONG]}
    with _evaluator(tok, ScriptedEngine(tok, eot, script)) as evaluator:
        results = [evaluator.run(task, mode, k=2, sampling=SamplingConfig(temperature=0.7)) for mode in ("plain", "think")]
    text = report.table(results, categories=True)
    lines = text.splitlines()
    assert lines[0].startswith("┌") and "task" in lines[1] and "pass@k" in lines[1]
    assert len({len(line) for line in lines[:-1]}) == 1  # aligned
    assert "0.6250 ± " in text and "(k=2)" in text and "mean of tasks" in text and "  alpha" in text
    assert lines[-1].startswith("8 items in ") and "first result after" in lines[-1]
    total = report.overall(results)
    assert total["macro_mean"]["plain"] == pytest.approx(0.625) and total["items"] == 8
    md = report.markdown({"model": "m", "results": results})
    assert md.startswith("# Evaluation of `m`") and "| sums | plain | 4 |" in md and "| category | items | score | stderr |" in md
    assert report.table([]).splitlines()[-1].startswith("0 items")


def test_contamination(tmp_path):
    from palingenesis.eval import contamination

    leaked = "Natalia sold clips to 48 of her friends in April and then she sold half as many clips in May"
    clean = "A farmer has seventeen sheep and buys four more every week for the next twelve weeks of the season"
    train = tmp_path / "train"
    train.mkdir()
    _write(train / "sft.jsonl", [{"messages": [{"role": "user", "content": f"Solve: {leaked}. How many?"}]}, {"text": "other"}])
    (train / "notes.txt").write_text("nothing to see here " * 20)
    report = contamination.check([leaked + " altogether?", clean, "too short"], [train], n=13)
    assert report["flagged"] == 1 and report["overlap"][0] > 0.5 and report["overlap"][1:] == [0.0, 0.0]
    assert report["too_short"] == 1 and report["files"] == 2
    assert contamination.check([leaked], [train / "notes.txt"], n=13)["flagged"] == 0
    assert contamination.words("L'été, 3 pommes_rouges!") == ["l", "été", "3", "pommes", "rouges"]
    with pytest.raises(FileNotFoundError):
        contamination.check(["x"], [tmp_path / "missing.jsonl"])

    class Item:
        def __init__(self, i, text):
            self.id, self.text = f"t/{i}", text

    items = [Item(0, leaked), Item(1, clean)]
    rows = [{"id": "t/0", "scores": [1.0]}, {"id": "t/1", "scores": [0.0]}]
    summary = contamination.task_report(items, rows, [train], 13, lambda item: item.text)
    assert summary["flagged_ids"] == ["t/0"] and (summary["flagged_score"], summary["clean_score"]) == (1.0, 0.0)


def test_suites_and_new_field_sources(tmp_path):
    from palingenesis.eval.config import builtin_suites
    from palingenesis.eval.tasks import build_items

    assert {"core", "quick", "multilingual", "loglikelihood"} <= set(builtin_suites())
    core = load_suite("core")
    assert {"gsm8k", "math500", "mmlu_pro", "ifeval", "bbh", "humaneval", "mbpp", "bfcl_simple", "belebele_it"} <= {t.name for t in core}
    assert next(t for t in core if t.name == "mmlu_pro").limit == 2000
    with pytest.raises(EvalConfigError, match="unknown suite 'cor'.*did you mean core"):
        load_suite("cor")
    rows = [{"Q": "q", "Correct Answer": "right", "Incorrect Answer 1": "w1", "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3"}]
    task = load_task({"task": "gpqa_diamond", "category_field": ""})
    (item,) = build_items(task, rows=[{**rows[0], "Question": "q"}])
    options = item.row["options"]
    assert sorted(options) == ["right", "w1", "w2", "w3"] and "ABCD"[options.index("right")] == item.row["answer"]
    assert build_items(task, rows=[{**rows[0], "Question": "q"}])[0].row["options"] == options  # a fixed order
    bbh = load_task("bbh")  # several dataset configurations as one task
    assert len(bbh.subsets) == 27 and bbh.category_field == "subset"


def test_engine_layout_and_worker_command(monkeypatch):
    from palingenesis.eval import engines
    from palingenesis.eval.runner import rl_config

    options, layout = engines.engine_layout({"max_model_len": 4096}, gpus=[0])
    assert (layout.devices, layout.replicas, layout.engines, layout.pooled) == ([], 1, 1, False)
    assert options["prefix_caching"] is True and options["max_num_seqs"] == 512
    assert options["vllm_args"] == {"max_num_batched_tokens": 16384, "max_logprobs": 32} and layout.max_logprobs == 32
    options, layout = engines.engine_layout(
        {"replicas": 2, "vllm_args": {"max_num_batched_tokens": 4096}, "speculative_config": {"method": "ngram"}}, gpus=[0, 1, 2]
    )
    assert layout.devices == [0, 1, 2] and layout.engines == 6 and layout.pooled
    assert options["vllm_args"]["max_num_batched_tokens"] == 4096 and options["vllm_args"]["speculative_config"] == {"method": "ngram"}
    assert "replicas" not in options and "speculative_config" not in options
    options, layout = engines.engine_layout({"tensor_parallel_size": 2}, gpus=[0, 1])
    assert layout.devices == [] and options["vllm_args"]["tensor_parallel_size"] == 2  # one engine over both GPUs
    assert engines.engine_layout({"backend": "hf"}, gpus=[0, 1])[1].devices == []
    assert engines.engine_layout({"devices": "1,3"}, gpus=[0, 1, 2, 3])[1].devices == [1, 3]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    assert engines.visible_gpus() == [0, 1] and engines._device_name(1) == "7"
    # the engine section's evaluation keys never reach the RL config; the rest is checked by it
    options, layout = engines.engine_layout({"replicas": 2, "max_model_len": 2048, "gpu_memory_utilization": 0.8})
    config = rl_config("some/model", load_task("gsm8k"), {"enable_thinking": False}, SamplingConfig(temperature=0.0, max_tokens=8), options)
    command = engines.worker_command("some/model", config, 7001, 0.4, {"enable_thinking": False}, [])
    assert command[1:3] == ["-m", "palingenesis.rl.pool.worker"] and "--port" in command and "7001" in command
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.4000" and command[command.index("--max-num-seqs") + 1] == "512"
    assert json.loads(command[command.index("--vllm-args") + 1]) == {"max_num_batched_tokens": 16384}
    assert command[command.index("--max-logprobs") + 1] == "32"
    with pytest.raises(EvalConfigError, match="replicas .* and tensor_parallel_size .* exclude each other"):
        config = EvalConfig()
        config.model.path, config.tasks = "m", [load_task("gsm8k")]
        config.engine = {"replicas": 2, "tensor_parallel_size": 2}
        config.validate()
    with engines.GpuMonitor(interval=0.01) as gpu:
        pass
    assert isinstance(gpu.stats(), dict)


def test_cli_suite_glob_and_report_files(tmp_path, qwen, monkeypatch, capsys):  # noqa: F811
    from palingenesis.eval import cli

    tok, eot = qwen
    script = {"alpha": [RIGHT], "beta": [WRONG], "gamma": [RIGHT], "delta": [RIGHT]}
    monkeypatch.setattr("palingenesis.eval.runner.build_engine", lambda *a, **k: ScriptedEngine(tok, eot, script))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: tok)
    monkeypatch.setenv("PGS_EVAL_CACHE", str(tmp_path / "cache"))
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for name in ("one", "two"):
        import yaml

        (tasks / f"{name}.yaml").write_text(yaml.safe_dump({**_sums(tmp_path), "name": name}))
    (tmp_path / "suite.yaml").write_text(f"tasks:\n  - {tasks / 'one.yaml'}\n  - {{task: {tasks / 'two.yaml'}, limit: 2}}\n")
    train = _write(tmp_path / "train.jsonl", [{"text": "Question alpha: what is 3 + 4? " * 3}])
    out = tmp_path / "out"
    argv = ["--model", "Qwen/Qwen3-0.6B", "--suite", str(tmp_path / "suite.yaml"), "--modes", "plain", "--out", str(out),
            "--backend", "hf", "--run.resamples", "100", "--contamination", str(train), "--run.contamination_ngram", "5"]  # fmt: skip
    assert cli.main(argv) == 0
    printed = capsys.readouterr().out
    assert "┌" in printed and "mean of tasks" in printed and "report.md" in printed
    report = json.loads((out / "summary.json").read_text())
    assert [(r["task"], r["items"]) for r in report["results"]] == [("one", 4), ("two", 2)]
    assert report["overall"]["items"] == 6 and report["engine"]["engines"] == 1 and report["wall_seconds"] > 0
    assert report["results"][0]["contamination"]["flagged_ids"] == ["one/alpha"]
    assert json.loads((out / "one.plain.summary.json").read_text())["contamination"]["flagged"] == 1
    assert (out / "report.md").read_text().startswith("# Evaluation of") and (tmp_path / "cache").is_dir()
    # a glob of task files; the second run of the same checkpoint is served by the cache
    assert cli.main(["--model", "Qwen/Qwen3-0.6B", "--tasks", str(tasks / "*.yaml"), "--out", str(out), "--backend", "hf", "--json",
                     "--modes", "plain", "--run.resamples", "100"]) == 0  # fmt: skip
    again = json.loads(capsys.readouterr().out)
    assert [r["task"] for r in again["results"]] == ["one", "two"] and again["results"][0]["cache_hits"] == 4
    assert cli.main(["--model", "m", "--tasks", str(tasks / "none*.yaml")]) == 1
    assert cli.main(["--list"]) == 0 and "core" in capsys.readouterr().out


def test_engine_pool_spreads_requests_over_worker_processes():
    """Two engine workers (palingenesis.rl.pool) behind one client: loglikelihood requests, with their candidates
    and allowed tokens, cross the wire and come back as the model's log-probs."""
    import asyncio
    import socket

    from palingenesis.eval import engines
    from palingenesis.eval import loglikelihood as ll
    from tests.test_eval_loglikelihood import ToyLM

    ports = []
    for _ in range(2):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            ports.append(s.getsockname()[1])
    commands = [
        [sys.executable, "-m", "palingenesis.rl.pool.worker", "--engine", "tests.test_eval_loglikelihood:ToyLM", "--port", str(port)]
        for port in ports
    ]
    import os
    from pathlib import Path

    root = str(Path(__file__).resolve().parent.parent)  # the workers import the toy engine from the tests package
    environment = {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.pathsep.join([root, os.environ.get("PYTHONPATH", "")])}
    pool = engines.EnginePool(commands, ports, [environment] * 2, timeout=120)
    try:
        client = pool.make_client()
        lm = ToyLM()
        context = [3, 1, 4, 1, 5, 9]
        items = [[[7], [8], [9, 2], [9, 3]], [[1, 2, 3], [1, 2, 4], [30]], [[5], [6]]]

        async def run():
            return await asyncio.gather(
                *(ll.trie_loglikelihoods(ll.client_peek(client, 6), context + [i], choices, 6) for i, choices in enumerate(items))
            )

        scored = asyncio.run(run())
        for i, (choices, scores) in enumerate(zip(items, scored)):
            assert scores.loglikelihoods == pytest.approx([lm.loglikelihood(context + [i], c) for c in choices], abs=1e-4)
        assert client.calls == sum(s.requests for s in scored)
    finally:
        pool.close()
    assert all(process.poll() is not None for process in pool.processes) or pool.processes == []
