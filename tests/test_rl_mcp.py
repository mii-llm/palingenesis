"""MCP servers as RL environments: in-process, stdio and Streamable HTTP, with hidden state
handles and a server-side grader."""

import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from palingenesis.rl.env import EnvPool, run_tool  # noqa: E402
from palingenesis.rl.envs.mcp import MCPEnv  # noqa: E402

SERVER = Path(__file__).parent / "fixtures" / "mcp_basket_server.py"
STATEFUL = dict(state_tool="create_basket", state_arg="basket_id", reward_tool="grade")


async def _episode(env: MCPEnv, pool: EnvPool | None = None) -> dict:
    await env.reset(prompt="Buy fruit worth 5.")
    schemas = await (pool.episode_schemas(env) if pool else env.tool_schemas())
    names = [s["function"]["name"] for s in schemas]
    add = next(s for s in schemas if s["function"]["name"] == "add_item")["function"]["parameters"]
    outputs = [await run_tool(env, "add_item", {"sku": sku}, 10) for sku in ("apple", "pear", "kiwi")]
    total = await run_tool(env, "total", {}, 10)
    reward = await env.get_reward([{"role": "assistant", "content": "The total is 5."}])
    return dict(names=names, add=add, outputs=outputs, total=total, reward=reward)


def _check(result: dict) -> None:
    # the handle and the hidden tools never reach the policy
    assert sorted(result["names"]) == ["add_item", "total"]
    assert "basket_id" not in result["add"]["properties"] and result["add"]["required"] == ["sku"]
    (a, a_failed), (p, p_failed), (k, k_failed) = result["outputs"]
    assert "added apple" in a and not a_failed and "2 item(s)" in p and not p_failed
    assert k_failed and k.startswith("Error:") and "unknown sku" in k  # isError: an observation
    assert result["total"] == ("5", False)
    assert result["reward"] == 1.0


def test_mcp_env_in_process():
    async def main():
        env = MCPEnv(server=f"{SERVER}:server", **STATEFUL)
        try:
            return await _episode(env)
        finally:
            await env.aclose()

    _check(asyncio.run(main()))


def test_mcp_env_stdio_concurrent_episodes():
    """One stdio server process serves concurrent episodes; handles keep them apart."""

    async def main():
        pool = EnvPool(MCPEnv, {"server": [sys.executable, str(SERVER)], **STATEFUL}, max_concurrent=4)

        async def one():
            env = await pool.acquire()
            try:
                return await _episode(env, pool)
            finally:
                await pool.release(env)

        try:
            return await asyncio.gather(*(one() for _ in range(6)))
        finally:
            await pool.shutdown()

    for result in asyncio.run(main()):
        _check(result)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mcp_env_streamable_http():
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, str(SERVER), "http", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.2)

        async def main():
            env = MCPEnv(server=f"http://127.0.0.1:{port}/mcp", **STATEFUL)
            try:
                return await _episode(env)
            finally:
                await env.aclose()

        _check(asyncio.run(main()))
    finally:
        process.terminate()
        process.wait(10)


def test_mcp_env_multiple_servers_are_prefixed_and_allowlisted():
    async def main():
        env = MCPEnv(servers={"a": f"{SERVER}:server", "b": f"{SERVER}:server"}, tools=["total", "create_basket"])
        try:
            names = [s["function"]["name"] for s in await env.tool_schemas()]
            handle = await run_tool(env, "a__create_basket", {}, 10)
            return names, handle
        finally:
            await env.aclose()

    names, (text, failed) = asyncio.run(main())
    assert sorted(names) == ["a__create_basket", "a__total", "b__create_basket", "b__total"]
    assert "bsk_" in text and not failed


def test_mcp_env_through_the_rollout_pipeline():
    """A scripted policy calls the MCP tools in a real multi-turn rollout; the server grades it."""
    import torch

    from palingenesis.opd.orchestrator import PublishedWeights
    from palingenesis.opd.teachers import end_of_turn_id
    from palingenesis.rl.chat import ChatFormat
    from palingenesis.rl.config import RLConfig
    from palingenesis.rl.data import PromptSampler
    from palingenesis.rl.pipeline import RLPipeline
    from tests.test_rl import ScriptedEngine, tokenizer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    kwargs = {"enable_thinking": False}
    eot = end_of_turn_id(tok, kwargs)
    call = '<tool_call>\n{{"name": "add_item", "arguments": {{"sku": "{}"}}}}\n</tool_call>'
    engine = ScriptedEngine(tok, eot, [call.format("apple"), call.format("pear"), "The total is 5."])
    config = RLConfig()
    for key, value in {
        "model.policy": "x",
        "model.chat_template_kwargs": kwargs,
        "env.max_turns": 4,
        "rollout.max_new_tokens": 64,
        "rollout.max_model_len": 4096,
    }.items():
        config.set(key, value)
    rows = [{"prompt": "Buy fruit worth exactly 5, then say the total."}]
    pool = EnvPool(MCPEnv, {"server": [sys.executable, str(SERVER)], **STATEFUL})
    pipeline = RLPipeline(
        tok,
        ChatFormat(tok, eot, kwargs),
        engine,
        PublishedWeights(torch.nn.Linear(1, 1)),
        config,
        PromptSampler(rows),
        [],
        pool,
        None,
        (eot,),
    )
    try:
        (group,) = pipeline._await(pipeline._eval(rows, 1.0))
    finally:
        pipeline.close()
    t = group[0]
    assert t.turns == 3 and t.tool_calls == 2 and t.tool_errors == 0
    assert t.rewards == {"env": 1.0} and t.reward == 1.0
    prompt = tok.decode(t.prompt_ids)
    assert "add_item" in prompt and "basket_id" not in prompt and "grade" not in prompt
