"""RL data formats and environment standards: Nemotron-RL (NeMo Gym) and UltraData rows,
Responses <-> chat conversion, the NeMo Gym adapter (against a local server), dynamic tools,
environment-ended episodes, and the OpenEnv adapter (when openenv is installed)."""

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, "src")

from palingenesis.rl.formats import convert, responses_to_chat  # noqa: E402

ROWS = Path(__file__).parent / "fixtures" / "rl_rows"


def load(name):
    return [json.loads(line) for line in (ROWS / name).read_text().splitlines()]


def test_nemotron_rows_convert_render_and_score():
    from transformers import AutoTokenizer

    from palingenesis.rl.chat import encode_prompt
    from palingenesis.rl.env import row_tools
    from palingenesis.rl.rewards import boxed_choice_reward

    workplace = convert(load("nemotron_workplace.jsonl"))
    row = workplace[0]
    assert [m["role"] for m in row["messages"]] == ["system", "user"]
    tools = row_tools(row, "tools")
    assert len(tools) == 27 and all(t["type"] == "function" and "name" in t["function"] for t in tools)
    assert "responses_create_params" in row and row["ground_truth"]  # the verifier's fields stay
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    except Exception as e:  # noqa: BLE001 — offline
        pytest.skip(f"tokenizer unavailable: {e}")
    text = tok.decode(encode_prompt(tok, row["messages"], tools, {"enable_thinking": False}))
    assert "company_directory_find_email_address" in text and row["messages"][1]["content"][:40] in text

    mcqa = convert(load("nemotron_mcqa.jsonl"))[0]
    assert [m["role"] for m in mcqa["messages"]] == ["system", "user"]
    assert {t["function"]["name"] for t in row_tools(mcqa, "tools")} == {"search", "browse"}
    answer = mcqa["expected_answer"]
    assert boxed_choice_reward(f"I checked. \\boxed{{{answer}}}. Done", field="expected_answer", **mcqa) == 1.0
    wrong = "A" if answer != "A" else "B"
    assert boxed_choice_reward(f"The answer is {answer} ... \\boxed{{{wrong}}}", field="expected_answer", **mcqa) == 0.0


def test_ultradata_rows_work_as_is():
    from palingenesis.rl.data import prompt_messages
    from palingenesis.rl.rewards import math_reward

    row = convert(load("ultradata_math.jsonl"))[0]  # plain rows: no conversion
    assert prompt_messages(row, "query")[0]["content"] == row["query"]
    assert math_reward(f"so \\boxed{{{row['ground_truth']}}}", field="ground_truth", **row) == 1.0


def test_responses_items_round_trip():
    from palingenesis.rl.envs.nemo_gym import chat_to_responses

    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"x": 1}'},
        {"type": "function_call", "call_id": "c2", "name": "g", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "one"},
        {"type": "function_call_output", "call_id": "c2", "output": "two"},
    ]
    chat = responses_to_chat(items)
    assert [m["role"] for m in chat] == ["user", "assistant", "tool", "tool"]
    assert chat[1]["reasoning_content"] == "think" and [c["function"]["name"] for c in chat[1]["tool_calls"]] == [
        "f",
        "g",
    ]
    back = chat_to_responses(chat[1:])
    assert [i["type"] for i in back] == [
        "reasoning",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]
    assert json.loads(back[1]["arguments"]) == {"x": 1}


class _NemoGymServer(BaseHTTPRequestHandler):
    sessions: dict = {}

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cookie = self.headers.get("Cookie", "")
        if self.path == "/seed_session":
            sid = str(len(self.sessions))
            self.sessions[sid] = body
            return self._reply({}, {"Set-Cookie": f"session={sid}"})
        session = self.sessions[cookie.split("=")[1]]
        if self.path == "/echo":
            return self._reply(f"{session['word']}:{body['text']}")
        if self.path == "/verify":
            calls = [i for i in body["response"]["output"] if i["type"] == "function_call"]
            return self._reply({"reward": float(any(c["name"] == "echo" for c in calls)), "mask_sample": False})

    def _reply(self, payload, headers=None):
        data = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
        self.send_response(200)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_nemo_gym_adapter_against_a_server():
    from palingenesis.rl.env import EnvPool, run_tool
    from palingenesis.rl.envs.nemo_gym import NemoGymAdapter

    server = ThreadingHTTPServer(("127.0.0.1", 0), _NemoGymServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    tools = [
        {
            "type": "function",
            "name": "echo",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
        }
    ]
    row = {"word": "w", "messages": [{"role": "user", "content": "say"}], "tools": tools}

    async def episode():
        pool = EnvPool(NemoGymAdapter, {"base_url": f"http://127.0.0.1:{server.server_port}"}, max_concurrent=2)
        env = await pool.acquire()
        await env.reset(**row)
        assert [s["function"]["name"] for s in await pool.episode_schemas(env)] == ["echo"]
        output, failed = await run_tool(env, "echo", {"text": "hi"}, 10)
        transcript = row["messages"] + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c", "function": {"name": "echo", "arguments": {"text": "hi"}}}],
            },
            {"role": "tool", "tool_call_id": "c", "content": output},
        ]
        return output, failed, await env.get_reward(transcript)

    try:
        assert asyncio.run(episode()) == ("w:hi", False, 1.0)
    finally:
        server.shutdown()


class Submission:
    """A dynamic-tool environment (tool_schemas / call_tool) that ends its episode on submit."""

    def reset(self, **row):
        self.done, self.answer = False, None

    def tool_schemas(self):
        return [
            {
                "type": "function",
                "name": "submit",
                "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}},
            }
        ]

    def call_tool(self, name, arguments):
        self.answer, self.done = arguments["answer"], True
        return "submitted"

    def get_reward(self):
        return float(self.answer == "5")


def test_dynamic_tools_and_environment_ended_episode():
    import torch

    from palingenesis.opd.orchestrator import PublishedWeights
    from palingenesis.opd.teachers import end_of_turn_id
    from palingenesis.rl.chat import ChatFormat
    from palingenesis.rl.config import RLConfig
    from palingenesis.rl.data import PromptSampler
    from palingenesis.rl.env import EnvPool
    from palingenesis.rl.pipeline import RLPipeline
    from tests.test_rl import ScriptedEngine, tokenizer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    kwargs = {"enable_thinking": False}
    eot = end_of_turn_id(tok, kwargs)
    engine = ScriptedEngine(
        tok, eot, ['<tool_call>\n{"name": "submit", "arguments": {"answer": "5"}}\n</tool_call>', "never"]
    )
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
        engine,
        PublishedWeights(torch.nn.Linear(1, 1)),
        config,
        PromptSampler(rows),
        [],
        EnvPool(Submission),
        None,
        (eot,),
    )
    try:
        (group,) = pipeline._await(pipeline._eval(rows, 1.0))
    finally:
        pipeline.close()
    t = group[0]
    assert t.finish == "env_done" and t.turns == 1 and t.rewards == {"env": 1.0} and t.reward == 1.0
    assert "submit" in tok.decode(t.prompt_ids)  # the dynamic schema reached the prompt


def test_openenv_adapter_in_process():
    pytest.importorskip("openenv")
    echo = pytest.importorskip("echo_env.server.echo_environment")
    from palingenesis.rl.env import EnvPool, run_tool
    from palingenesis.rl.envs.openenv import OpenEnvAdapter

    async def episode():
        pool = EnvPool(OpenEnvAdapter, {"env_class": f"{echo.__name__}:EchoEnvironment"})
        env = await pool.acquire()
        await env.reset()
        names = [s["function"]["name"] for s in await pool.episode_schemas(env)]
        return names, await run_tool(env, "echo_message", {"message": "hi"}, 30)

    names, result = asyncio.run(episode())
    assert "echo_message" in names and result == ("hi", False)


def test_verifier_column_routes_rewards():
    from palingenesis.rl.rewards import BUILTINS, Reward

    samples = [
        {"completion": "\\boxed{4}", "answer": "4", "verifier": "math"},
        {"completion": "\\boxed{B}", "answer": "B", "verifier": ["boxed_choice"]},
        {"completion": "\\boxed{4}", "answer": "4"},
    ]
    math = asyncio.run(Reward("correct", BUILTINS["math"]).score(samples))
    choice = asyncio.run(Reward("mc", BUILTINS["boxed_choice"]).score(samples))
    assert math == [1.0, None, 1.0] and choice == [None, 1.0, 0.0]


class _Leaky:
    """A public helper that must not become a tool once the class pins its tools."""

    tools = ("answer",)

    def reset(self, **row):
        self.done = False

    def answer(self, value: str) -> str:
        """Answer.

        Args:
            value: The answer.
        """
        return "ok"

    def grade(self) -> str:
        """Reveal the solution.

        Returns:
            The solution.
        """
        return "SECRET"


def test_pinned_and_allowed_tools_are_the_only_callable_ones():
    from palingenesis.rl.env import EnvPool, run_tool

    pool = EnvPool(_Leaky)
    assert [s["function"]["name"] for s in pool.schemas] == ["answer"]
    exposed = {s["function"]["name"] for s in pool.schemas}
    env = pool.free[0]
    text, failed = asyncio.run(run_tool(env, "grade", {}, 5, exposed))
    assert failed and "unknown tool" in text and "SECRET" not in text  # refused, though the method exists
    assert asyncio.run(run_tool(env, "answer", {"value": "1"}, 5, exposed)) == ("ok", False)

    class Unpinned(_Leaky):
        tools = None

    assert sorted(s["function"]["name"] for s in EnvPool(Unpinned).schemas) == ["answer", "grade"]
    assert [s["function"]["name"] for s in EnvPool(Unpinned, allowed=["ans*"]).schemas] == ["answer"]
    with pytest.raises(ValueError, match="match no tool"):
        EnvPool(Unpinned, allowed=["answr"])

    class Typo(_Leaky):
        tools = ("answr",)

    with pytest.raises(ValueError, match="not methods"):
        EnvPool(Typo)


def test_allowed_tools_filter_dynamic_environments():
    from palingenesis.rl.env import EnvPool

    class Remote:
        def tool_schemas(self):
            return [{"type": "function", "name": n, "parameters": {"type": "object"}} for n in ("a__x", "a__y", "b__x")]

        def call_tool(self, name, arguments):
            return name

    pool = EnvPool(Remote, allowed=["a__*"])
    names = [s["function"]["name"] for s in asyncio.run(pool.episode_schemas(pool.free[0]))]
    assert names == ["a__x", "a__y"]
    bad = EnvPool(Remote, allowed=["c__*"])
    with pytest.raises(ValueError, match="match no tool"):
        asyncio.run(bad.episode_schemas(bad.free[0]))
