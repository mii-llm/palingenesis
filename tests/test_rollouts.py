"""Rollout dumps (palingenesis.rollouts) and their reader (pgs rollouts): the writer never blocks a
trainer and flushes on shutdown, the records carry what an agent needs to debug a run, and the
reader turns them into shares, tables and alarms."""

import gzip
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palingenesis.agent_tooling.rollouts import analyze  # noqa: E402
from palingenesis.rollouts import (  # noqa: E402
    SCHEMA_VERSION,
    RolloutSink,
    chat_turns,
    completion_record,
    dump_files,
    pick,
    read_records,
    rl_record,
)

# ----------------------------------------------------------------- the writer


def test_sink_writes_records_and_timeline_to_step_range_shards(tmp_path):
    sink = RolloutSink(tmp_path, "rl", rank=1, world=2, shard_steps=10)
    assert sink.submit(3, [{"finish": "stop"}, {"finish": "length"}], timeline=[(0.0, 2, 0)])
    assert sink.submit(12, [{"finish": "turns"}])
    assert sink.submit(13, [])  # nothing to write: no file, no thread work
    sink.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "steps_00000000-00000009.rank1.jsonl.gz",
        "steps_00000010-00000019.rank1.jsonl.gz",
    ]
    records = list(read_records(tmp_path))
    assert [(r["step"], r["finish"]) for r in records] == [(3, "stop"), (3, "length"), (12, "turns")]
    first = records[0]
    assert first["schema_version"] == SCHEMA_VERSION and first["trainer"] == "rl" and first["rank"] == 1
    assert first["kind"] == "trajectory" and first["trajectory_id"] == "3-1-0"
    (timeline,) = read_records(tmp_path, kind="timeline")
    assert timeline["step"] == 3 and timeline["events"] == [[0.0, 2, 0]]
    assert list(read_records(tmp_path, steps=[12]))[0]["finish"] == "turns"
    assert sink.written == 3 and sink.dropped == 0


def test_sink_drops_and_counts_when_the_writer_is_behind_and_never_blocks(tmp_path):
    release = threading.Event()

    def slow(item):  # a disk that does not keep up: the writer is stuck on the first batch
        release.wait(10)
        return {"value": item}

    sink = RolloutSink(tmp_path, "rl", encode=slow, queue_size=2)
    assert sink.submit(1, [1, 1])
    deadline = time.time() + 5
    while sink._queue.qsize() and time.time() < deadline:  # the writer took batch 1 and is stuck on it
        time.sleep(0.01)
    start = time.perf_counter()
    results = [sink.submit(step, [step, step]) for step in range(2, 8)]
    assert time.perf_counter() - start < 1.0  # submit() returned at once every time
    assert results == [True, True, False, False, False, False] and sink.dropped == 8  # 2 batches wait, 4 dropped
    results = [True, *results]
    release.set()
    sink.close()
    steps = sorted({r["step"] for r in read_records(tmp_path)})
    assert steps == [s for s, ok in zip(range(1, 8), results) if ok]
    assert sink.written + sink.dropped == 14


def test_sink_close_flushes_queued_batches_and_is_idempotent(tmp_path):
    sink = RolloutSink(tmp_path, "opd", encode=lambda item: (time.sleep(0.05), {"i": item})[1], queue_size=8)
    for step in range(1, 6):
        assert sink.submit(step, [step])
    sink.close()
    sink.close()
    assert [r["i"] for r in read_records(tmp_path)] == [1, 2, 3, 4, 5]
    assert not sink.submit(6, [6])  # closed: refused, not raised


def test_sink_survives_a_write_failure(tmp_path):
    blocker = tmp_path / "rollouts"
    blocker.write_text("not a directory")
    sink = RolloutSink(blocker / "sub", "rl")  # the directory cannot be made: counted, never raised
    assert sink.submit(1, [{"a": 1}])
    sink.close()
    assert sink.dropped == 1
    sink = RolloutSink(tmp_path / "ok", "rl", encode=lambda item: 1 / 0)  # an encoder that raises
    assert sink.submit(1, [1, 2])
    sink.close()
    assert sink.dropped == 2 and sink.written == 0
    assert not list((tmp_path / "ok").glob("*.gz"))  # a batch that fails to encode leaves no partial file


def test_sigterm_flushes_the_queue_before_the_process_ends(tmp_path):
    script = textwrap.dedent(
        f"""
        import os, signal, sys, time
        sys.path.insert(0, {str(ROOT / "src")!r})
        from palingenesis.rollouts import RolloutSink, flush_on_signal

        sink = RolloutSink({str(tmp_path)!r}, "opd", encode=lambda i: (time.sleep(0.1), {{"i": i}})[1], queue_size=8)
        with flush_on_signal(sink):
            for step in range(1, 6):
                sink.submit(step, [step])
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(30)
        """
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run([sys.executable, "-c", script], env=env, timeout=60)
    assert proc.returncode == -signal.SIGTERM  # the signal still ends the process
    assert [r["i"] for r in read_records(tmp_path)] == [1, 2, 3, 4, 5]


def test_pick_draws_a_seeded_sample_or_everything():
    items = list(range(20))
    assert pick(items, 0, 3) is items and pick(items, 50, 3) is items
    drawn = pick(items, 5, 3)
    assert len(drawn) == 5 and drawn == pick(items, 5, 3) and drawn != items[:5]


# ----------------------------------------------------------------- reading


def _gz_append(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "at") as f:
        f.write("".join(json.dumps(r) + "\n" for r in records))


def test_reader_keeps_the_newest_copy_of_a_redone_step_and_survives_a_cut_batch(tmp_path):
    shard = tmp_path / "steps_00000000-00000099.jsonl.gz"
    _gz_append(shard, [{"step": 1, "rank": 0, "v": "old"}, {"step": 1, "rank": 0, "v": "old"}])
    _gz_append(shard, [{"step": 2, "rank": 0, "v": "a"}])
    _gz_append(shard, [{"step": 1, "rank": 0, "v": "redone"}])  # a resumed run redid step 1
    assert [(r["step"], r["v"]) for r in read_records(tmp_path)] == [(1, "redone"), (2, "a")]
    # a crash in the middle of a batch: the batches before it are read
    data = shard.read_bytes()
    _gz_append(shard, [{"step": 3, "rank": 0, "v": "cut"}] * 50)
    shard.write_bytes(shard.read_bytes()[: len(data) + 20])
    assert [r["v"] for r in read_records(tmp_path)] == ["redone", "a"]


def test_reader_reads_the_unversioned_format(tmp_path):
    (tmp_path / "step_4.jsonl").write_text(json.dumps({"finish": "stop", "reward": 1.0}) + "\nnot json\n")
    (tmp_path / "step_4.timeline.json").write_text("[]")
    (record,) = read_records(tmp_path)
    assert record["step"] == 4 and record["kind"] == "trajectory" and record["reward"] == 1.0
    assert [f[:2] for f in dump_files(tmp_path)] == [(4, 4)]


# ----------------------------------------------------------------- records


def test_chat_turns_match_results_to_calls_and_keep_text_protocol_calls():
    messages = [
        {
            "role": "assistant",
            "content": "let me look",
            "tool_calls": [
                {"id": "c0", "function": {"name": "search", "arguments": '{"q": "x"}'}},
                {"id": "c1", "function": {"name": "open", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "opened"},
        {"role": "tool", "tool_call_id": "c0", "content": "Error: boom"},
        {"role": "tool", "content": "Error: malformed call"},
        {"role": "assistant", "content": "ACTION: click 3"},  # a text-protocol action, parsed by the environment
        {"role": "user", "content": "clicked"},
    ]
    timings = [
        {
            "generate_s": 0.5,
            "tokens": 7,
            "context": 100,
            "finish": "stop",
            "tools_s": 1.5,
            "tools": [{"name": "search", "s": 1.5, "failed": True}, {"name": "open", "s": 0.2, "failed": False}],
        },
        {"generate_s": 0.25, "tokens": 4, "context": 140, "finish": "stop", "tools": [{"name": "click", "s": 0.1}]},
    ]
    first, second = chat_turns(messages, timings, [-0.5, None], content=True)
    assert first["gen_ai.usage.input_tokens"] == 100 and first["gen_ai.response.finish_reasons"] == ["stop"]
    search, opened = first["tool_calls"]
    assert search["gen_ai.tool.name"] == "search" and search["error"] and search["duration_s"] == 1.5
    assert search["gen_ai.tool.call.result"] == "Error: boom" and opened["gen_ai.tool.call.result"] == "opened"
    assert first["observations"] == [{"role": "tool", "content": "Error: malformed call"}]
    (click,) = second["tool_calls"]
    assert click["gen_ai.tool.name"] == "click" and click["gen_ai.tool.call.id"] is None and not click["error"]
    assert second["observations"][0]["content"] == "clicked" and second["logprob_mean"] is None
    first, _ = chat_turns(messages, timings, [], content=False)
    assert "content" not in first and "gen_ai.tool.call.arguments" not in first["tool_calls"][0]
    assert "gen_ai.tool.call.result" not in first["tool_calls"][0] and first["observations"] == [{"role": "tool"}]


class _Tools:
    """A lookup tool that fails, and a submit tool that ends the episode."""

    def reset(self, **row):
        self.done, self.answer = False, None

    def tool_schemas(self):
        return [
            {"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}},
            {
                "type": "function",
                "name": "submit",
                "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}},
            },
        ]

    def call_tool(self, name, arguments):
        if name == "lookup":
            raise RuntimeError("the index is down")
        self.answer, self.done = arguments["answer"], True
        return "submitted"

    def get_reward(self):
        return float(self.answer == "5")


def test_rl_record_of_a_real_multi_turn_rollout():
    import torch

    from palingenesis.opd.orchestrator import PublishedWeights
    from palingenesis.opd.teachers import end_of_turn_id
    from palingenesis.rl.chat import ChatFormat
    from palingenesis.rl.config import RLConfig
    from palingenesis.rl.data import PromptSampler
    from palingenesis.rl.env import EnvPool
    from palingenesis.rl.pipeline import RLPipeline
    from tests.test_rl import tokenizer
    from tests.test_rl_envs import _SequenceEngine

    tok = tokenizer("Qwen/Qwen3-0.6B")
    kwargs = {"enable_thinking": False}
    eot = end_of_turn_id(tok, kwargs)
    texts = [
        '<tool_call>\n{"name": "lookup", "arguments": {}}\n</tool_call>',
        '<tool_call>\n{"name": "submit", "arguments": {"answer": "5"}}\n</tool_call>',
    ]
    config = RLConfig()
    for key, value in {
        "model.policy": "x",
        "model.chat_template_kwargs": kwargs,
        "env.max_turns": 4,
        "rollout.max_new_tokens": 64,
        "rollout.max_model_len": 4096,
    }.items():
        config.set(key, value)
    rows = [{"prompt": "What is 2 + 3? Submit it."}]
    pipeline = RLPipeline(
        tok,
        ChatFormat(tok, eot, kwargs),
        _SequenceEngine(tok, eot, texts),
        PublishedWeights(torch.nn.Linear(1, 1)),
        config,
        PromptSampler(rows),
        [],
        EnvPool(_Tools),
        None,
        (eot,),
    )
    try:
        t, env = pipeline._await(pipeline._trajectory(rows[0], 0, 1.0, True))
        pipeline._await(pipeline.env_pool.release(env))
    finally:
        pipeline.close()
    record = rl_record((True, t))
    assert record["finish"] == "env_done" and record["turn_count"] == 2 and record["kept"]
    assert record["tool_calls"] == 2 and record["tool_errors"] == 1
    assert record["prompt"][-1]["content"].startswith("What is 2 + 3")
    lookup, submit = (turn["tool_calls"][0] for turn in record["turns"])
    assert lookup["gen_ai.tool.name"] == "lookup" and lookup["error"]
    assert "the index is down" in lookup["gen_ai.tool.call.result"] and lookup["duration_s"] >= 0
    assert submit["gen_ai.tool.name"] == "submit" and not submit["error"]
    assert submit["gen_ai.tool.call.result"] == "submitted" and submit["gen_ai.tool.call.arguments"]
    assert abs(record["logprob_mean"] + 0.25) < 1e-9 and abs(record["turns"][0]["logprob_mean"] + 0.25) < 1e-9
    assert record["gen_ai.usage.output_tokens"] == t.sampled_tokens and not record["loop"]
    assert "timing" not in record["info"] and "prompt" not in record["info"]
    json.dumps(record, default=str)
    bare = rl_record((False, t), content=False)
    assert "prompt" not in bare and "content" not in bare["turns"][0] and not bare["kept"]
    assert all(isinstance(v, bool | int | float) for v in bare["info"].values())


def test_completion_record_flags_loops():
    looping = "The answer is " + "and then we repeat the same sentence again, " * 40
    record = completion_record(looping, "length", 12, [-0.5] * 30, [{"role": "user", "content": "hi"}], group=0)
    assert record["loop"] and record["finish"] == "length" and record["group"] == 0
    assert record["gen_ai.usage.output_tokens"] == 30 and record["turns"][0]["content"] == looping
    record = completion_record("Fine.", "stop", 12, [-0.1], None, content=False)
    assert not record["loop"] and "content" not in record["turns"][0] and "prompt" not in record


def test_opd_dump_items_and_records_carry_no_tensors():
    import torch

    from palingenesis.opd.orchestrator import Request, Sample
    from palingenesis.opd.teachers import TeacherScores
    from palingenesis.opd.trainer import OPDTrainer, _dump_record

    request = Request(
        [{"role": "user", "content": "2+2?"}], 16, {"tools": [{}], "difficulty": 3}, "math", "big", [1, 2, 3], [1, 2, 3]
    )
    scores = TeacherScores(token_lp=torch.tensor([-0.5, -1.5]))
    samples = [Sample(request, [7, 8], [-1.0, -2.0], 4, None, scores, finish="stop") for _ in range(2)]
    other = Request([{"role": "user", "content": "3+3?"}], 16, {}, "math", "big", [1, 2], [1, 2])
    samples.append(Sample(other, [9], [-0.3], 4, None, None, finish="length"))
    trainer = SimpleNamespace(config=SimpleNamespace(logging=SimpleNamespace(dump_groups=0)))
    items = OPDTrainer._dump_items(trainer, SimpleNamespace(samples=samples), 5)
    assert len(items) == 3 and items[0]["fields"]["group"] == items[1]["fields"]["group"] != items[2]["fields"]["group"]
    assert abs(items[0]["fields"]["k1_mean"] - (-1.5 + 1.0)) < 1e-6 and "k1_mean" not in items[2]["fields"]
    trainer.config.logging.dump_groups = 1
    assert len(OPDTrainer._dump_items(trainer, SimpleNamespace(samples=samples), 5)) in (1, 2)

    tok = SimpleNamespace(decode=lambda ids, skip_special_tokens: " ".join(map(str, ids)))
    record = _dump_record(tok, True, items[0])
    assert record["finish"] == "stop" and record["turns"][0]["content"] == "7 8" and record["source"] == "math"
    assert record["meta"] == {"difficulty": 3} and record["prompt"] == request.messages
    json.dumps(record)


# ----------------------------------------------------------------- the reader


def _synthetic(tmp_path: Path) -> Path:
    def trajectory(i, finish="stop", reward=1.0, loop=False, kept=True, errors=0, seconds=1.0):
        calls = [{"gen_ai.tool.name": "bash", "error": k < errors, "duration_s": seconds} for k in range(2)]
        return {
            "group": i // 2,
            "kept": kept,
            "finish": finish,
            "reward": reward,
            "rewards": {"correct": reward, "format": 0.5},
            "tool_calls": 2,
            "tool_errors": errors,
            "loop": loop,
            "gen_ai.usage.output_tokens": 100 * (i + 1),
            "turns": [
                {"duration_s": seconds, "gen_ai.usage.output_tokens": 50, "tool_calls": calls, "content": f"answer {i}"}
            ],
        }

    sink = RolloutSink(tmp_path, "rl")
    sink.submit(10, [trajectory(0), trajectory(1, reward=0.0, errors=2, seconds=9.0), trajectory(2), trajectory(3)])
    sink.submit(
        20,
        [
            trajectory(4, finish="length", reward=0.0, loop=True),
            trajectory(5, finish="turns", reward=0.0),
            trajectory(6, kept=False),
            trajectory(7, kept=False),
        ],
    )
    sink.close()
    return tmp_path


def test_reader_reports_shares_tools_worst_and_alarms(tmp_path):
    report = analyze(read_records(_synthetic(tmp_path)), top=2)
    assert report["trajectories"] == 8 and report["steps"] == [10, 20] and report["dumped_steps"] == 2
    assert report["finish"] == {"stop": 0.75, "length": 0.125, "turns": 0.125}
    assert report["loop_share"] == 0.125 and report["zero_variance_share"] == 0.25 and report["groups"] == 4
    assert report["reward"]["mean"] == 0.625 and report["reward_components"] == {"correct": 0.625, "format": 0.5}
    assert report["tool_calls"] == 16 and report["tool_error_rate"] == 0.125
    assert report["tools"]["bash"]["calls"] == 16 and report["tools"]["bash"]["seconds"]["max"] == 9.0
    assert report["slowest_turns"][0]["seconds"] == 9.0 and report["slowest_turns"][0]["trajectory_id"] == "10-0-1"
    assert report["slowest_tool_calls"][0]["tool"] == "bash"
    assert [w["reward"] for w in report["worst"]["lowest_reward"]] == [0.0, 0.0]
    assert report["worst"]["longest"][0]["tokens"] == 800 and report["worst"]["longest"][0]["text"] == "answer 7"
    assert report["status"] == "ISSUES" and len(report["alarms"]) == 1  # 1 loop in 8 > 0.1
    assert report["alarms"][0].startswith("loop_share 0.125 > 0.1")
    strict = analyze(read_records(tmp_path), thresholds={"loop_share": 0.2, "length_share": 0.1})
    assert strict["status"] == "ISSUES" and len(strict["alarms"]) == 1
    assert strict["alarms"][0].startswith("length_share 0.125 > 0.1")
    assert analyze(read_records(tmp_path), thresholds={"loop_share": 0.2})["status"] == "HEALTHY"
    assert analyze(read_records(tmp_path, steps=[20]))["trajectories"] == 4


def test_reader_cli_exit_codes_and_json(tmp_path):
    _synthetic(tmp_path / "rollouts")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT / "src")}
    cmd = [sys.executable, "-m", "palingenesis.agent_tooling.rollouts", "--dir", str(tmp_path), "--json"]
    relaxed = ["--threshold", "loop_share=0.5"]
    proc = subprocess.run(cmd + relaxed, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["trajectories"] == 8
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1 and json.loads(proc.stdout)["alarms"][0].startswith("loop_share")
    relaxed += ["--threshold", "length_share=0.5", "--threshold", "turns_share=0.5"]  # step 20: 1 of 4 each
    proc = subprocess.run(cmd[:-1] + ["--since", "15"] + relaxed, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "4 trajectories" in proc.stdout and "Slowest turns" in proc.stdout
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = subprocess.run(cmd[:4] + [str(empty)], env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1 and "no trajectory records" in proc.stdout
