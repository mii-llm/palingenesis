"""Harness profiles, phase 2: unknown-tool wording and aliases, per-profile chat template kwargs, the recording stub
(pgs harness serve / capture), conformance with recorded requests, the new profiles (swe-agent, qwen-code, goose,
hermes, kimi-cli, terminus-2), run_episodes. Docker tests run Scrivania tasks per new profile (terminus-2 needs the
harbor3-base:3 image with tmux)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import urllib.request
from pathlib import Path

import pytest

from palingenesis.rl.env import run_tool
from palingenesis.rl.envs.harbor import HarborEnv
from palingenesis.rl.envs.harness import FORMAT_ERROR, get_profile
from palingenesis.rl.envs.harness.capture import Stub, serve
from palingenesis.rl.envs.harness.terminus import TerminusJSON, limit_output, parse_response
from tests.test_harness_profiles import (
    ARCHIVE,
    DOTENV,
    ENV_EDIT,
    SED,
    TAR,
    TASKS,
    LocalRuntime,
    _docker_tasks_ok,
    _pipeline_episode,
    _row,
    make_runtime,
    play,
    session,
    write,
)


@pytest.fixture
def rt(tmp_path):
    return make_runtime(tmp_path)


@pytest.fixture
def local_harbor(rt, monkeypatch):
    from palingenesis.rl.envs import harbor

    monkeypatch.setattr(harbor, "DockerRuntime", LocalRuntime)
    LocalRuntime.verified = []
    return rt


FIXTURES = Path(__file__).parent.parent / "research" / "tiny-reliable-agent" / "harness" / "fixtures"


# ================================================================== unknown tools and aliases


def test_unknown_tool_wording_and_aliases_per_harness(rt):
    oc = session("opencode", rt)
    assert oc.unknown_tool("Bash", {"command": "echo hi"}) == "hi\n"  # repairToolCall lowercases
    assert oc.unknown_tool("todo_x", {}) == ("The arguments provided to the tool are invalid: Model tried to call "
                                             "unavailable tool 'todo_x'. Available tools: bash, edit, glob, grep, "
                                             "invalid, read, write.")
    oh = session("openhands-sdk", rt)
    assert "[The command completed with exit code 0.]" in oh.unknown_tool("bash", {"command": "true"})
    assert "[Command finished with exit code 0]" in oh.unknown_tool("git", {"command": "--version || true"})
    assert oh.unknown_tool("str_replace</parameter", {"command": "view", "path": "/nope"}).endswith(
        "does not exist. Please provide a valid path.")
    assert oh.unknown_tool("python", {}) == ("Tool 'python' not found. Available: ['terminal', 'file_editor', "
                                             "'finish', 'think']")
    mini = session("mini-swe-agent", rt)
    assert "Unknown tool 'str_replace'.Missing 'command' argument in bash tool call." in \
        mini.unknown_tool("str_replace", {"path": "x"})
    assert session("pi", rt).unknown_tool("grep", {}) == "Tool grep not found"
    assert session("qwen-code", rt).unknown_tool("bash", {}).startswith(
        'Tool "bash" not found in registry. Tools must use the exact names that are registered. Did you mean one of: ')
    assert session("kimi-cli", rt).unknown_tool("bash", {}) == "<system>ERROR: Tool `bash` not found</system>"
    assert session("swe-agent", rt).unknown_tool("ls", {}) == ("Your action could not be parsed properly: Command 'ls' "
                                                               "not found in list of available commands..\n")


def test_pipeline_asks_the_environment_for_the_unknown_tool_answer(local_harbor):
    env = HarborEnv(profile="opencode")
    env.reset(**_row())
    text, failed = asyncio.run(run_tool(env, "Read", {"filePath": "/nope"}, 30, {"bash", "read"}))
    assert failed and text.startswith("File not found: /nope")
    base = HarborEnv()
    base.reset(**_row())
    text, _ = asyncio.run(run_tool(base, "nope", {}, 30, {"bash", "submit"}))
    assert text == "Error: unknown tool 'nope'. Available tools: bash, submit."  # the trainer's own wording


# ================================================================== chat template kwargs per profile


def test_profile_chat_template_kwargs_reach_the_prompt(local_harbor):
    call = json.dumps({"name": "write", "arguments": {"filePath": "hello.txt", "content": "ciao\n"}})
    script = [f"<tool_call>\n{call}\n</tool_call>", "Done.", "never"]
    off = _pipeline_episode("opencode", script)
    on = _pipeline_episode("opencode", script, {"profile_chat_kwargs": {"opencode": {"enable_thinking": True}}})
    assert "chat_template_kwargs" not in off.info and on.info["chat_template_kwargs"] == {"enable_thinking": True}
    from tests.test_rl import tokenizer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    assert tok.decode(off.prompt_ids).endswith("<think>\n\n</think>\n\n")  # the model's kwargs: no thinking
    assert not tok.decode(on.prompt_ids).endswith("</think>\n\n")  # the profile's: thinking on


# ================================================================== the recording stub


def _post(url, body, stream=False):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read().decode()


def test_recording_stub_replays_the_script_and_records_requests(tmp_path):
    script = [{"tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]}, {"content": "All done."}]
    stub = Stub(script, tmp_path / "req.jsonl", model="m")
    server = serve(stub)
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        convo = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        tools = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]
        first = json.loads(_post(f"{base}/chat/completions", {"model": "m", "messages": convo, "tools": tools}))
        call = first["choices"][0]["message"]["tool_calls"][0]
        assert first["choices"][0]["finish_reason"] == "tool_calls" and call["function"]["name"] == "bash"
        assert json.loads(call["function"]["arguments"]) == {"command": "ls"}
        side = json.loads(_post(f"{base}/chat/completions", {"model": "m", "messages": [{"role": "user", "content":
                                                                                             "title?"}]}))
        assert side["choices"][0]["message"]["content"] == "Capture"  # a side request without tools
        sse = _post(f"{base}/chat/completions", {"model": "m", "messages": convo, "tools": tools, "stream": True,
                                                 "stream_options": {"include_usage": True}})
        chunks = [json.loads(line[6:]) for line in sse.splitlines() if line.startswith("data: {")]
        text = "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c["choices"])
        assert text == "All done." and sse.rstrip().endswith("data: [DONE]") and "usage" in chunks[-1]
        assert json.loads(urllib.request.urlopen(f"{base}/models").read())["data"][0]["id"] == "m"
        with pytest.raises(urllib.error.HTTPError):
            _post(f"{base}/responses", {"model": "m", "input": "x"})
    finally:
        server.shutdown()
    rows = [json.loads(line) for line in open(tmp_path / "req.jsonl")]
    assert [r["path"] for r in rows] == ["/v1/chat/completions"] * 3 + ["/v1/responses"]
    assert rows[0]["body"]["tools"] == tools and rows[2]["body"]["stream"] is True


def test_capture_command_line_for_the_harbor_cli():
    from palingenesis.rl.envs.harness.capture import harbor_agent_args

    args, env = harbor_agent_args("opencode", "qwen", "http://h:1/v1")
    config = json.loads(args[args.index("--ak") + 1].split("=", 1)[1])
    assert args[:2] == ["-m", "vllm/qwen"] and config["provider"]["vllm"]["options"]["baseURL"] == "http://h:1/v1"
    args, env = harbor_agent_args("mini-swe-agent", "qwen", "http://h:1/v1")
    assert args == ["-m", "openai/qwen"] and env["OPENAI_BASE_URL"] == "http://h:1/v1"


# ================================================================== conformance with recorded requests


def _content(m) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def _recorded(agent: str):
    path = FIXTURES / agent / "requests.jsonl"
    if not path.exists():
        pytest.skip(f"no recorded requests for {agent} (pgs harness capture / serve)")
    rows = [json.loads(line) for line in open(path)]
    meta = json.loads((FIXTURES / agent / "meta.json").read_text()) if (FIXTURES / agent / "meta.json").exists() \
        else {}
    main = [r for r in rows if r["body"].get("tools")]
    first = main[0]["body"]
    last = max(main, key=lambda r: len(r["body"]["messages"]))["body"]
    pairs, pending = [], {}
    for m in last["messages"]:
        for c in m.get("tool_calls") or []:
            pending[c["id"]] = (c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
        if m.get("role") == "tool" and m.get("tool_call_id") in pending:
            pairs.append((*pending.pop(m["tool_call_id"]), _content(m)))
    return first, pairs, meta


def _replay(agent: str, rt, pairs, meta, setup):
    s = session(agent, rt)
    setup(Path(rt.workdir))
    old = meta.get("workdir", "")
    out = []
    for name, args, expected in pairs:
        args = json.loads(json.dumps(args).replace(old, rt.workdir))
        got = s.call_tool(name, args)
        out.append((name, args, expected.replace(old, rt.workdir), got))
    return out


@pytest.mark.parametrize("agent", ["qwen-code", "kimi-cli"])
def test_tool_schemas_match_the_recorded_request(agent):
    first, _, _ = _recorded(agent)
    recorded = {t["function"]["name"]: t["function"]["parameters"] for t in first["tools"]}
    for t in get_profile(agent).tools:
        assert t.schema["function"]["parameters"] == recorded[t.name], t.name


def test_qwen_code_results_match_the_recording(rt):
    _, pairs, meta = _recorded("qwen-code")
    rows = _replay("qwen-code", rt, pairs, meta, lambda w: (w / "notes.txt").write_text("alpha\nbeta\nbeta\n"))
    for name, args, expected, got in rows:
        if name not in {t.name for t in get_profile("qwen-code").tools}:  # the real registry has more tools to
            assert got.split("Did you mean")[0] == expected.split("Did you mean")[0]  # suggest
            continue
        assert got == expected, (name, args)


def test_kimi_cli_results_match_the_recording(rt):
    _, pairs, meta = _recorded("kimi-cli")
    rows = _replay("kimi-cli", rt, pairs, meta, lambda w: (w / "notes.txt").write_text("alpha\nbeta\nbeta\n"))
    for name, args, expected, got in rows:
        assert got == expected, (name, args)


@pytest.mark.parametrize("agent", ["opencode", "mini-swe-agent"])
def test_harbor_cli_capture_matches_the_profile(agent):
    """Recorded on the box with `pgs harness capture <agent>` (Harbor 0.24 CLI, the real agent installed in the task
    container): tool schemas equal; system prompt / first user turn compared loosely (profiles paraphrase them)."""
    first, pairs, _ = _recorded(agent)
    profile = get_profile(agent)
    recorded = {t["function"]["name"]: t["function"] for t in first["tools"]}
    for t in profile.tools:
        assert json.dumps(recorded[t.name]["parameters"]) == json.dumps(t.schema["function"]["parameters"]), t.name
    assert [n for n in recorded if n in profile.by_name] == [t.name for t in profile.tools]  # the same order
    if agent == "mini-swe-agent":
        from palingenesis.rl.envs.harness.profiles import mini_format_error, mini_unknown_tool

        assert _content(first["messages"][0]) == profile.system
        assert recorded["bash"] == profile.tools[0].schema["function"]
        rows = [json.loads(line) for line in open(FIXTURES / agent / "requests.jsonl")]
        users = [_content(m) for m in rows[-1]["body"]["messages"][2:] if m["role"] == "user"]
        assert users[0] == mini_unknown_tool("str_replace", ["bash"], {"path": "/workdir/Projects/sito/.env"})
        assert users[1] == mini_format_error("No tool calls found in the response. Every response MUST include at "
                                             "least one tool call.", 0)
    if agent == "opencode":
        assert _content(first["messages"][1]).startswith('"') and _content(first["messages"][1]).endswith('\n"')
        assert "Workspace root folder:" in _content(first["messages"][0])


def _docker_replay(agent: str, task: str):
    """Replay a Harbor CLI capture's tool calls through the profile on the same task (Docker): the results to compare."""
    _, pairs, _ = _recorded(agent)
    env = HarborEnv(profile=agent)
    try:
        env.reset(task_dir=str(TASKS / task))
        return [(name, args, expected, env.call_tool(name, args)) for name, args, expected in pairs]
    finally:
        env.close()


@pytest.mark.skipif(not _docker_tasks_ok(), reason="docker or the Scrivania tasks unavailable")
def test_opencode_results_match_the_harbor_cli_recording():
    rows = _docker_replay("opencode", ENV_EDIT)
    for name, args, expected, got in rows:
        if name == "grep" or (name == "read" and "Did you mean" in expected):  # ripgrep / readdir order
            assert sorted(got.split("\n")) == sorted(expected.split("\n")), (name, args)
        elif name == "invalid":  # the real registry also lists skill, task, todowrite, webfetch
            continue
        else:
            assert got == expected, (name, args)


@pytest.mark.skipif(not _docker_tasks_ok(), reason="docker or the Scrivania tasks unavailable")
def test_mini_swe_agent_results_match_the_harbor_cli_recording():
    rows = _docker_replay("mini-swe-agent", ENV_EDIT)
    assert rows and all(got == expected for _, _, expected, got in rows), rows


# ================================================================== new profiles


def test_swe_agent_editor_bash_and_submit_twice(rt):
    s = session("swe-agent", rt)
    p = f"{rt.workdir}/a.py"
    ed = lambda **a: s.call_tool("str_replace_editor", a)  # noqa: E731
    assert ed(command="create", path=p, file_text="x = 1\ny = 2\n") == f"OBSERVATION:\nFile created successfully at: {p}"
    assert ed(command="view", path=p) == (f"OBSERVATION:\nHere's the result of running `cat -n` on {p}:\n     1\tx = 1\n"
                                          "     2\ty = 2\n     3\t")
    assert ed(command="str_replace", path=p, old_str="= ", new_str="=") == (
        "OBSERVATION:\nNo replacement was performed. Multiple occurrences of old_str `= ` in lines [1, 2]. Please "
        "ensure it is unique")
    assert ed(command="view", path="a.py").startswith("OBSERVATION:\nThe path a.py is not an absolute path")
    assert s.call_tool("bash", {"command": "cd /tmp"}) == "Your command ran successfully and did not produce any output."
    assert s.call_tool("bash", {"command": "pwd"}).endswith("tmp")  # persistent shell
    first = s.call_tool("submit", {})
    assert first.startswith("OBSERVATION:\nThank you for your work on this issue.") and not s.done
    s.call_tool("submit", {})
    assert s.done
    [(name, args)] = s.text_actions("I am done")
    assert name == FORMAT_ERROR and args["message"].startswith("Your last output did not use any tool calls!")


def test_goose_developer_tools(rt):
    s = session("goose", rt)
    write(rt, "src/a.py", "a = 1\na = 1\n")
    assert s.call_tool("write", {"path": "new.txt", "content": "x\ny\n"}) == "Created new.txt (2 lines)"
    assert s.call_tool("write", {"path": "new.txt", "content": "z\n"}) == "Wrote new.txt (1 lines)"
    multi = s.call_tool("edit", {"path": "src/a.py", "before": "a = 1", "after": "b"})
    assert multi.startswith("Found 2 matches. Please provide more context to identify a unique match:\n\nMatch 1 "
                            "(line 1):\n```\na = 1\na = 1\n```")
    none = s.call_tool("edit", {"path": "src/a.py", "before": "zz", "after": "b"})
    assert none == "No match found for the specified text.\n\nFile preview:\n```\n   1: a = 1\n   2: a = 1\n```"
    assert s.call_tool("edit", {"path": "new.txt", "before": "z", "after": "w\nv"}) == "Edited new.txt (1 lines -> 2 lines)"
    assert s.call_tool("shell", {"command": "echo hi; exit 2"}) == "hi\n\nCommand exited with code 2"
    assert s.call_tool("shell", {"command": "true"}) == "(no output)"
    tree = s.call_tool("tree", {"path": "."})
    assert "src/  [2]\n  a.py  [2]\n" in tree and "new.txt  [2]" in tree


def test_hermes_json_results(rt):
    s = session("hermes", rt)
    write(rt, "f.txt", "one\ntwo\n")
    assert json.loads(s.call_tool("terminal", {"command": "echo hi; exit 3"})) == {"output": "hi", "exit_code": 3,
                                                                                    "error": None}
    assert json.loads(s.call_tool("terminal", {"command": "mkdir -p d && cd d"}))["cwd"].endswith("/d")
    blocked = json.loads(s.call_tool("write_file", {"path": f"{rt.workdir}/f.txt", "content": "x"}))
    assert "read_file" in blocked["error"]  # an existing file must be read first
    r = json.loads(s.call_tool("read_file", {"path": f"{rt.workdir}/f.txt"}))
    assert r["content"] == "1|one\n2|two" and r["total_lines"] == 2
    p = json.loads(s.call_tool("patch", {"path": f"{rt.workdir}/f.txt", "old_string": "two", "new_string": "2"}))
    assert p["success"] and "-two\n+2\n" in p["diff"]
    hits = json.loads(s.call_tool("search_files", {"pattern": "one", "path": rt.workdir}))
    assert hits["total_count"] == 1 and hits["matches"][0]["line"] == 1


def test_qwen_code_prompt_states_the_destructive_action_scope():
    system = get_profile("qwen-code").system
    assert "otherwise stop and report the blocker" in system and "explicitly requested is authorized" in system


def test_terminus_parser_matches_harbor_behaviour():
    ok = parse_response('{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\\n", "duration": 0.1}], '
                        '"task_complete": false}')
    assert not ok.error and ok.commands[0].keystrokes == "ls\n" and not ok.warning
    extra = parse_response('Sure!\n{"analysis": "a", "plan": "p", "commands": []}')
    assert not extra.error and extra.warning == "- Extra text detected before JSON object"
    fixed = parse_response('{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\\n", "duration": 1}]')
    assert not fixed.error and fixed.warning.startswith("- AUTO-CORRECTED: Fixed incomplete JSON by adding missing "
                                                       "closing brace")
    nodur = parse_response('{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls"}, {"keystrokes": "pwd\\n", '
                           '"duration": 1}], "task_complete": "true"}')
    assert nodur.is_task_complete and "Command 1: Missing duration field, using default 1.0" in nodur.warning
    assert "Command 1 should end with newline" in nodur.warning
    bad = parse_response('{"analysis": "a", "commands": []}')
    assert bad.error == "Missing required fields: plan"
    assert parse_response("no json here").error == "No valid JSON found in response"
    proto = TerminusJSON()
    with pytest.raises(Exception, match="Previous response had parsing errors:\nERROR: Missing required fields: plan"):
        proto.parse('{"analysis": "a", "commands": []}')
    [(name, args)] = proto.parse(proto.render([("x", {"commands": [{"keystrokes": "ls\n", "duration": 0.1}],
                                                      "task_complete": True})]))
    assert args["task_complete"] is True and args["commands"] == [{"keystrokes": "ls\n", "duration": 0.1}]
    assert limit_output("é" * 6000).count("output limited to 10000 bytes") == 1


def test_run_episodes_uses_the_pipeline(local_harbor):
    import torch  # noqa: F401

    from palingenesis.opd.teachers import end_of_turn_id
    from palingenesis.rl.episodes import run_episodes
    from tests.test_rl import ScriptedEngine, tokenizer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    kwargs = {"enable_thinking": False}
    call = json.dumps({"name": "write", "arguments": {"filePath": "hello.txt", "content": "ciao\n"}})
    engine = ScriptedEngine(tok, end_of_turn_id(tok, kwargs), [f"<tool_call>\n{call}\n</tool_call>", "Done.", "x"])
    groups = run_episodes(engine, tok, [_row()], HarborEnv, {"profiles": {"opencode": 1, "pi": 1}}, k=2,
                          temperature=1.0, chat_template_kwargs=kwargs, max_turns=4, max_new_tokens=64)
    [group] = groups
    assert len(group) == 2 and len({t.info["harness_profile"] for t in group}) == 1  # one profile per group
    name = group[0].info["harness_profile"]
    assert all(t.rewards[f"env/metric/profile/{name.split('@')[0]}"] == 1.0 for t in group)



def test_run_episodes_budget_covers_every_turn(local_harbor):
    """The default episode budget is max_new_tokens per turn (the pipeline's own default, one turn's worth, cut
    multi-turn episodes as "length" after about one turn of tokens)."""
    from palingenesis.opd.teachers import end_of_turn_id
    from palingenesis.rl.episodes import run_episodes
    from tests.test_rl import ScriptedEngine, tokenizer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    kwargs = {"enable_thinking": False}
    call = json.dumps({"name": "write", "arguments": {"filePath": "hello.txt", "content": "ciao\n"}})
    engine = ScriptedEngine(tok, end_of_turn_id(tok, kwargs), [f"<tool_call>\n{call}\n</tool_call>"] * 3 + ["Done."])
    n = len(tok.encode(f"<tool_call>\n{call}\n</tool_call>", add_special_tokens=False))
    [[t]] = run_episodes(engine, tok, [_row()], HarborEnv, {"profile": "opencode"}, k=1, temperature=1.0,
                         chat_template_kwargs=kwargs, max_turns=6, max_new_tokens=n + 8)
    assert t.finish != "length" and t.turns == 4


# ================================================================== Docker: Scrivania tasks per new profile

TMUX_IMAGE = os.environ.get("PGS_TMUX_IMAGE", "harbor3-base:3")


def _image_ok(tag):
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def _terminus(cmds, complete=False):
    return json.dumps({"analysis": "", "plan": "", "commands": [{"keystrokes": c, "duration": 1.0} for c in cmds],
                       "task_complete": complete})


SCRIPTS = {
    "swe-agent": {
        ARCHIVE: [[("bash", {"command": "cd /workdir/Projects"})],
                  [("bash", {"command": "tar -czf ../website.tar.gz website"})], [("submit", {})], [("submit", {})]],
        ENV_EDIT: [[("str_replace_editor", {"command": "view", "path": DOTENV})],
                   [("str_replace_editor", {"command": "str_replace", "path": DOTENV, "old_str": "DEBUG=true",
                                            "new_str": "DEBUG=false"})], [("submit", {})], [("submit", {})]],
        "noop": [[("submit", {})], [("submit", {})]]},
    "qwen-code": {
        ARCHIVE: [[("run_shell_command", {"command": "tar -czf website.tar.gz -C Projects website"})], "Done."],
        ENV_EDIT: [[("read_file", {"file_path": DOTENV})],
                   [("edit", {"file_path": DOTENV, "old_string": "DEBUG=true", "new_string": "DEBUG=false"})], "Done."],
        "noop": ["Done."]},
    "goose": {
        ARCHIVE: [[("tree", {"path": "."})], [("shell", {"command": TAR})], "Done."],
        ENV_EDIT: [[("edit", {"path": DOTENV, "before": "DEBUG=true", "after": "DEBUG=false"})], "Done."],
        "noop": ["Done."]},
    "hermes": {
        ARCHIVE: [[("terminal", {"command": "cd Projects"})], [("terminal", {"command":
                                                                             "tar -czf ../website.tar.gz website"})],
                  "Done."],
        ENV_EDIT: [[("read_file", {"path": "Projects/sito/.env"})],
                   [("patch", {"path": "Projects/sito/.env", "old_string": "DEBUG=true", "new_string": "DEBUG=false"})],
                   "Done."],
        "noop": ["Done."]},
    "kimi-cli": {
        ARCHIVE: [[("Shell", {"command": TAR})], "Done."],
        ENV_EDIT: [[("ReadFile", {"path": "Projects/sito/.env"})],
                   [("StrReplaceFile", {"path": "Projects/sito/.env", "edit": {"old": "DEBUG=true",
                                                                               "new": "DEBUG=false"}})], "Done."],
        "noop": ["Done."]},
    "terminus-2": {
        ARCHIVE: [_terminus(["cd /workdir/Projects\n", "tar -czf ../website.tar.gz website\n"]),
                  _terminus([], True), _terminus([], True)],
        ENV_EDIT: [_terminus([SED + "\n"]), "oops, not json", _terminus(["cat " + DOTENV + "\n"], True),
                   _terminus([], True)],
        "noop": [_terminus([], True), _terminus([], True)]},
}


@pytest.mark.skipif(not _docker_tasks_ok(), reason="docker or the Scrivania tasks unavailable")
@pytest.mark.parametrize("profile", sorted(SCRIPTS))
def test_scrivania_oracle_and_noop_per_new_profile(profile):
    images = {}
    if profile == "terminus-2":
        if not _image_ok(TMUX_IMAGE):
            pytest.skip(f"{TMUX_IMAGE} (tmux) not built")
        images = {"harbor3-base:1": TMUX_IMAGE}
    for task in (ARCHIVE, ENV_EDIT):
        for mode in (task, "noop"):
            env = HarborEnv(profile=profile, images=images)
            try:
                conv = env.reset(task_dir=str(TASKS / task))
                if profile == "terminus-2":
                    assert conv[0]["role"] == "user" and "Current Terminal Screen:" in conv[0]["content"]
                messages, seen = play(env, SCRIPTS[profile][mode])
                r = env.get_reward(messages)
                assert r["reward"] == (0.0 if mode == "noop" else 1.0), (profile, task, mode, r, seen)
                if profile in ("swe-agent", "terminus-2"):
                    assert env.done, (profile, mode, seen)
                if profile == "terminus-2" and mode == ENV_EDIT:
                    assert any(o.startswith("Previous response had parsing errors:") for o in seen)
                    assert any("Are you sure you want to mark the task as complete?" in o for o in seen)
            finally:
                env.close()
