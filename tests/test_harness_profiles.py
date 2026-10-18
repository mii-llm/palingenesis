"""Harness profiles for HarborEnv (palingenesis.rl.envs.harness). The ops run against a local stand-in for the
episode container (bash in a temporary directory); the Docker test runs Scrivania tasks per profile and is skipped
unless docker and the task directory (PGS_SCRIVANIA_TASKS) are available."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from palingenesis.rl.envs import harbor
from palingenesis.rl.envs.harbor import ExecResult, HarborEnv
from palingenesis.rl.envs.harness import FORMAT_ERROR, HarnessMix, Session, available, get_profile, ops
from palingenesis.rl.envs.harness.ops import OpError
from palingenesis.rl.envs.harness.profiles import mini_observation, mini_text_observation
from palingenesis.rl.envs.harness.protocols import FencedCommand, FormatError, JsonBlock, ReAct, Tags, XmlTool
from palingenesis.rl.envs.harness.rules import FLAG, Flag, LastText, Submit

FIXTURE = Path(__file__).parent / "fixtures" / "harbor_task"


class LocalRuntime:
    """DockerRuntime's contract on the local machine: bash in a working directory, files at absolute paths."""

    root: Path | None = None  # set by the fixture (HarborEnv constructs runtimes itself)
    verified: list = []

    count = 0

    def __init__(self, task=None, workdir=None, max_output=8_000_000):
        self.task = task
        LocalRuntime.count += 1
        work = self.root / f"rt{LocalRuntime.count}"  # each runtime its own "container"
        work.mkdir(parents=True, exist_ok=True)
        self.workdir = str(work)
        self.scratch = str(self.root / "scratch")  # the container's /tmp
        Path(self.scratch).mkdir(exist_ok=True)
        self.max_output = max_output
        self.started = False

    def start(self, provision=True, allow_internet=None):
        self.started = True

    def stop(self):
        self.started = False

    def exec(self, command, timeout=60, env=None, workdir=None):
        try:
            r = subprocess.run(
                ["bash", "-c", command],
                cwd=workdir or self.workdir,
                capture_output=True,
                timeout=timeout,
                env={**os.environ, **(env or {})},
            )
        except subprocess.TimeoutExpired:
            return ExecResult(124, "", f"timed out after {timeout:g} s", True)
        out, err = r.stdout.decode(errors="replace"), r.stderr.decode(errors="replace")
        cut = self.max_output
        if len(out) > cut:
            out = out[: cut // 2] + f"\n... [{len(out) - cut} characters cut] ...\n" + out[-cut // 2 :]
        return ExecResult(r.returncode, out, err[-cut:])

    def put_files(self, files, dest="/", as_user=False):
        for path, data in files.items():
            p = Path(path if path.startswith("/") else os.path.join(dest, path))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data.encode() if isinstance(data, str) else data)

    def read_file(self, path, limit=200_000):
        r = self.exec(f"head -c {limit} '{path}'")
        return r.stdout if r.exit_code == 0 else None

    def verify(self, isolated=True, final_message=None, capture=None):
        LocalRuntime.verified.append(final_message)
        return {"reward": 1.0 if (Path(self.workdir) / "hello.txt").exists() else 0.0, "extra": 0.5}


def make_runtime(tmp_path) -> LocalRuntime:
    root = tmp_path / "work"
    root.mkdir()
    LocalRuntime.root = root.resolve()
    return LocalRuntime()


@pytest.fixture
def rt(tmp_path):
    return make_runtime(tmp_path)


@pytest.fixture
def ws(rt):
    return ops.Workspace(rt)


def session(name, rt):
    return Session(get_profile(name), rt)


def write(rt, rel, text):
    p = Path(rt.workdir) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return str(p)


# ================================================================== ops


def test_exec_stateless_merges_stderr_and_reports_the_exit_code(ws):
    r = ops.exec_(ws, "echo out; echo err >&2; exit 3")
    assert r.exit_code == 3 and "out" in r.output and "err" in r.output
    ops.exec_(ws, "cd /tmp; export FOO=1")
    r = ops.exec_(ws, 'pwd; echo "foo=$FOO"')
    assert r.output.splitlines() == [ws.workdir, "foo="]  # nothing carries over
    assert ops.exec_(ws, "pwd", workdir="/tmp").output.strip().endswith("tmp")
    assert ops.exec_(ws, "sleep 5", timeout=0.5).timed_out


def test_exec_persistent_keeps_cwd_and_exported_variables(ws):
    r = ops.exec_(ws, "mkdir -p sub && cd sub && export FOO=bar", persistent=True)
    assert r.exit_code == 0 and r.cwd.endswith("/sub")
    r = ops.exec_(ws, 'echo "$FOO"; pwd', persistent=True)
    assert r.output.split() == ["bar", ws.workdir + "/sub"] and "__PGS_CWD__" not in r.output
    r = ops.exec_(ws, "false", persistent=True)
    assert r.exit_code == 1
    ops.reset_shell(ws)
    assert ops.exec_(ws, "pwd", persistent=True).cwd == ws.workdir


def test_read_write_keep_mode_and_report_creation(ws, rt):
    p = write(rt, "run.sh", "echo hi\n")
    os.chmod(p, 0o755)
    assert ops.write(ws, p, "echo bye\n") is False
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o755 and ops.read(ws, p) == "echo bye\n"
    assert ops.write(ws, ws.abspath("new/deep/file.txt"), "x") is True
    assert ops.read(ws, ws.abspath("new/deep/file.txt")) == "x"
    with pytest.raises(OpError) as e:
        ops.read(ws, ws.abspath("missing.txt"))
    assert e.value.kind == "not_found"
    with pytest.raises(OpError) as e:
        ops.read(ws, ws.workdir)
    assert e.value.kind == "is_dir"
    assert ops.lines_of("a\nb\n") == ["a", "b"] and ops.lines_of("a\nb") == ["a", "b"] and ops.lines_of("") == []


def test_replace_variants_and_errors(ws, rt):
    p = write(rt, "f.txt", "alpha\nbeta\ngamma\nbeta\n")
    for args, kind in (
        ((p, "beta", "B"), "multiple"),
        ((p, "zzz", "B"), "no_match"),
        ((p, "beta", "beta"), "identical"),
        ((p, "", "B"), "empty_old"),
        ((rt.workdir, "a", "b"), "is_dir"),
    ):
        with pytest.raises(OpError) as e:
            ops.replace(ws, *args)
        assert e.value.kind == kind
    with pytest.raises(OpError) as e:
        ops.replace(ws, p, "beta", "B")
    assert e.value.data["lines"] == [2, 4]
    ed = ops.replace(ws, p, "gamma", "G")
    assert ed.line == 3 and ops.read(ws, p) == "alpha\nbeta\nG\nbeta\n"
    ed = ops.replace(ws, p, "beta", "B", replace_all=True)
    assert ed.count == 2 and ops.read(ws, p) == "alpha\nB\nG\nB\n"


def test_multi_replace_matches_the_original_text(ws, rt):
    p = write(rt, "f.txt", "one two three\n")
    ed = ops.multi_replace(ws, p, [("three", "3"), ("one", "two")])  # "two" from edit 2 is not re-matched
    assert ops.read(ws, p) == "two two 3\n" and ed.count == 2
    for edits, kind in (
        ([("two", "x"), ("two two", "y")], "multiple"),
        ([("nope", "x")], "no_match"),
        ([("", "x")], "empty_old"),
        ([("two 3", "a"), ("3\n", "b")], "overlap"),
        ([("3", "3")], "identical"),
    ):
        with pytest.raises(OpError) as e:
            ops.multi_replace(ws, p, edits)
        assert e.value.kind == kind
    with pytest.raises(OpError) as e:
        ops.multi_replace(ws, p, [("3", "x"), ("zz", "y")])
    assert e.value.data["index"] == 1


def test_insert_and_single_level_undo(ws, rt):
    p = write(rt, "f.txt", "a\nb\n")
    ops.insert(ws, p, 0, "top")
    ops.insert(ws, p, 3, "end")
    assert ops.read(ws, p) == "top\na\nb\nend\n"
    with pytest.raises(OpError) as e:
        ops.insert(ws, p, 9, "x")
    assert e.value.kind == "bad_line" and e.value.data["n_lines"] == 4
    ops.undo_edit(ws, p)
    assert ops.read(ws, p) == "top\na\nb\n"
    with pytest.raises(OpError) as e:  # single level
        ops.undo_edit(ws, p)
    assert e.value.kind == "no_history"
    q = ws.abspath("created.txt")
    ops.write(ws, q, "new")
    ops.undo_edit(ws, q)
    assert ws.kind(q) is None


def test_list_dir_glob_and_grep(ws, rt):
    write(rt, "src/a.py", "import os\nprint('x')\n")
    write(rt, "src/deep/b.ts", "let x = 1\n")
    write(rt, "src/deep/c.tsx", "x\n")
    write(rt, ".env", "DEBUG=true\n")
    write(rt, ".git/config", "DEBUG=true\n")
    entries, hidden = ops.list_dir(ws, rt.workdir, 2)
    assert "src/" in entries and "src/a.py" in entries and "src/deep/" in entries and ".env" not in entries
    assert hidden == 2  # .env and .git
    assert ".env" in ops.list_dir(ws, rt.workdir, 1, hidden=True)[0]
    g = lambda pat, path=None: sorted(os.path.relpath(f, rt.workdir) for f in ops.glob(ws, pat, path))  # noqa: E731
    assert g("**/*.py") == ["src/a.py"] and g("*.py") == ["src/a.py"]
    assert g("src/**/*.{ts,tsx}") == ["src/deep/b.ts", "src/deep/c.tsx"]
    assert g("src/*.ts") == [] and g(".env") == [".env"] and g("config") == []  # .git left out
    rows = ops.grep(ws, "DEBUG=(true|false)")
    assert [os.path.relpath(r[0], rt.workdir) for r in rows] == [".env"] and rows[0][1] == 1
    assert sorted(os.path.basename(r[0]) for r in ops.grep(ws, "x", include="*.{ts,tsx}")) == ["b.ts", "c.tsx"]
    with pytest.raises(OpError):
        ops.glob(ws, "*", ws.abspath("src/a.py"))


def test_apply_patch_is_a_stub(ws):
    with pytest.raises(NotImplementedError, match="apply_patch grammar"):
        ops.apply_patch(ws, "*** Begin Patch")


# ================================================================== wording per profile


def test_openhands_file_editor_wording_matches_the_sdk(rt):
    """Strings as openhands-tools 1.55.1's FileEditor produces them (recorded from the package)."""
    s = session("openhands-sdk", rt)
    W = rt.workdir
    p = f"{W}/a.txt"
    fe = lambda **a: s.call_tool("file_editor", a)  # noqa: E731
    assert fe(command="create", path=p, file_text="alpha\nbeta\ngamma\nbeta\n") == f"File created successfully at: {p}"
    assert fe(command="create", path=p, file_text="x") == (
        f"[An error occurred during execution.]\nInvalid `path` parameter: {p}. File already exists at: {p}. Cannot "
        "overwrite files using command `create`."
    )
    assert fe(command="view", path=p) == (
        f"Here's the result of running `cat -n` on {p}:\n     1\talpha\n     2\tbeta\n     3\tgamma\n     4\tbeta\n"
    )
    assert fe(command="view", path=p, view_range=[2, 3]) == (
        f"Here's the result of running `cat -n` on {p}:\n     2\tbeta\n     3\tgamma\n"
    )
    assert fe(command="view", path=W) == (
        f"Here's the files and directories up to 2 levels deep in {W}, excluding hidden items:\n{W}/\n{W}/a.txt"
    )
    assert fe(command="str_replace", path=p, old_str="beta", new_str="B") == (
        "[An error occurred during execution.]\nNo replacement was performed. Multiple occurrences of old_str `beta` in "
        "lines [2, 4]. Please ensure it is unique."
    )
    assert fe(command="str_replace", path=p, old_str="zzz", new_str="B") == (
        f"[An error occurred during execution.]\nNo replacement was performed, old_str `zzz` did not appear verbatim "
        f"in {p}."
    )
    assert fe(command="str_replace", path=p, old_str="alpha", new_str="ALPHA") == (
        f"The file {p} has been edited. Here's the result of running `cat -n` on a snippet of {p}:\n     1\tALPHA\n"
        "     2\tbeta\n     3\tgamma\n     4\tbeta\nReview the changes and make sure they are as expected. Edit the "
        "file again if necessary."
    )
    assert fe(command="insert", path=p, insert_line=1, new_str="ins") == (
        f"The file {p} has been edited. Here's the result of running `cat -n` on a snippet of the edited file:\n"
        "     1\tALPHA\n     2\tins\n     3\tbeta\n     4\tgamma\n     5\tbeta\nReview the changes and make sure they "
        "are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
    )
    assert fe(command="undo_edit", path=p) == (
        f"Last edit to {p} undone successfully. Here's the result of running "
        f"`cat -n` on {p}:\n     1\tALPHA\n     2\tbeta\n     3\tgamma\n"
        "     4\tbeta\n"
    )
    assert fe(command="view", path=f"{W}/nope.txt") == (
        f"[An error occurred during execution.]\nInvalid `path` parameter: {W}/nope.txt. The path {W}/nope.txt does "
        "not exist. Please provide a valid path."
    )
    assert fe(command="str_replace", path=p, old_str="gamma", new_str="gamma") == (
        "[An error occurred during execution.]\nInvalid `new_str` parameter: gamma. No replacement was performed. "
        "`new_str` and `old_str` must be different."
    )
    assert fe(command="create", path="rel.txt", file_text="x") == (
        "[An error occurred during execution.]\nInvalid `path` parameter: rel.txt. The path should be an absolute path."
    )


def test_openhands_terminal_is_persistent_and_reports_exit_and_cwd(rt):
    s = session("openhands-sdk", rt)
    out = s.call_tool("terminal", {"command": "mkdir -p d && cd d && echo hi"})
    assert out == (
        f"hi\n[The command completed with exit code 0.]\n[Current working directory: {rt.workdir}/d]\n"
        "[Command finished with exit code 0]"
    )
    assert s.call_tool("terminal", {"command": "pwd"}).startswith(f"{rt.workdir}/d\n")
    assert s.call_tool("terminal", {"command": "", "is_input": True}).endswith(
        "No previous running command to interact with."
    )
    out = s.call_tool("terminal", {"command": "sleep 3", "timeout": 0.5})
    assert "[The command timed out after 0.5 seconds." in out and "finished with exit code" not in out
    assert s.call_tool("think", {"thought": "x"}) == "Your thought has been logged."


def test_opencode_wording(rt):
    s = session("opencode", rt)
    p = write(rt, "f.txt", "alpha\nbeta\nbeta\n")
    call = s.call_tool
    assert call("read", {"filePath": p}) == (
        f"<path>{p}</path>\n<type>file</type>\n<content>\n1: alpha\n2: beta\n"
        "3: beta\n\n(End of file - total 3 lines)\n</content>"
    )
    assert call("read", {"filePath": p, "offset": 2, "limit": 1}).endswith(
        "2: beta\n\n(Showing lines 2-2 of 3. Use offset=3 to continue.)\n</content>"
    )
    assert call("read", {"filePath": p, "offset": 9}) == "Offset 9 is out of range for this file (3 lines)"
    assert call("read", {"filePath": f"{rt.workdir}/f.tx"}) == (
        f"File not found: {rt.workdir}/f.tx\n\nDid you mean one of these?\n{rt.workdir}/f.txt"
    )
    assert call("edit", {"filePath": p, "oldString": "beta", "newString": "B"}) == (
        "Found multiple matches for oldString. Provide more surrounding context to make the match unique."
    )
    assert call("edit", {"filePath": p, "oldString": "zz", "newString": "B"}).startswith("Could not find oldString")
    assert call("edit", {"filePath": p, "oldString": "beta", "newString": "beta"}) == (
        "No changes to apply: oldString and newString are identical."
    )
    assert call("edit", {"filePath": p, "oldString": "beta", "newString": "B", "replaceAll": True}) == (
        "Edit applied successfully."
    )
    assert Path(p).read_text() == "alpha\nB\nB\n"
    assert call("write", {"filePath": f"{rt.workdir}/n/x.txt", "content": "x"}) == "Wrote file successfully."
    assert call("bash", {"command": "true"}) == "(no output)"
    assert call("bash", {"command": "echo hi; exit 2"}) == "hi\n"  # opencode shows no exit code
    assert "<shell_metadata>\nshell tool terminated command after exceeding timeout 300 ms" in call(
        "bash", {"command": "sleep 3", "timeout": 300}
    )
    big = call("bash", {"command": "seq 1 3000"})
    assert big.startswith(f"...output truncated...\n\nFull output saved to: {rt.scratch}/opencode/tool-output/tool_")
    kept = big.split("\n\n", 2)[2]  # the last 2000 lines ("\n"-split, the trailing empty line included)
    assert kept.startswith("1002\n") and kept.endswith("\n3000\n")
    assert sorted(call("glob", {"pattern": "**/*.txt"}).splitlines()) == [
        f"{rt.workdir}/f.txt",
        f"{rt.workdir}/n/x.txt",
    ]
    assert call("grep", {"pattern": "alpha"}) == f"Found 1 matches\n{p}:\n  Line 1: alpha\n"
    assert call("grep", {"pattern": "nothing-here"}) == "No files found"
    assert call("todowrite", {}) == (
        "The arguments provided to the tool are invalid: Model tried to call unavailable "
        "tool 'todowrite'. Available tools: bash, edit, glob, grep, invalid, read, write."
    )


def test_pi_wording(rt):
    s = session("pi", rt)
    write(rt, "f.txt", "one\ntwo\ntwo\n")
    call = s.call_tool
    assert call("read", {"path": "f.txt"}) == "one\ntwo\ntwo\n"
    assert call("read", {"path": "f.txt", "limit": 1}) == "one\n\n[3 more lines in file. Use offset=2 to continue.]"
    assert call("read", {"path": "nope"}) == f"ENOENT: no such file or directory, access '{rt.workdir}/nope'"
    edit = lambda *e: call("edit", {"path": "f.txt", "edits": [{"oldText": o, "newText": n} for o, n in e]})  # noqa
    assert edit(("two", "2")) == (
        "Found 2 occurrences of the text in f.txt. The text must be unique. Please provide "
        "more context to make it unique."
    )
    assert edit(("one", "1"), ("two", "2")) == (
        "Found 2 occurrences of edits[1] in f.txt. Each oldText must be "
        "unique. Please provide more context to make it unique."
    )
    assert edit(("zz", "2")) == (
        "Could not find the exact text in f.txt. The old text must match exactly including all whitespace and newlines."
    )
    assert edit(("one", "1"), ("on", "2")) == (
        "edits[0] and edits[1] overlap in f.txt. Merge them into one edit or target disjoint regions."
    )
    assert edit(("one\n", "1\n"), ("two\ntwo", "2\n2")) == "Successfully replaced 2 block(s) in f.txt."
    assert (Path(rt.workdir) / "f.txt").read_text() == "1\n2\n2\n"
    assert call("edit", {"path": "nope", "edits": [{"oldText": "a", "newText": "b"}]}) == (
        "Could not edit file: nope. Error code: ENOENT."
    )
    assert call("write", {"path": "w/x.txt", "content": "x"}) == "Successfully wrote to w/x.txt"
    assert call("bash", {"command": "true"}) == "(no output)"
    assert call("bash", {"command": "echo hi; exit 4"}) == "hi\n\n\nCommand exited with code 4"
    assert call("bash", {"command": "sleep 3", "timeout": 0.5}) == "Command timed out after 0.5 seconds"


def test_mini_observations_match_the_upstream_jinja_templates(rt):
    jinja2 = pytest.importorskip("jinja2")
    fc = (
        '{%- if output.output | length < 10000 -%}\n{\n  "returncode": {{ output.returncode }},\n  "output": '
        '{{ output.output | tojson }}\n  {%- if output.exception_info %}, "exception_info": {{ output.exception_info '
        '| tojson }}{% endif %}\n}\n{%- else -%}\n{\n  "returncode": {{ output.returncode }},\n  "output_head": '
        '{{ output.output[:5000] | tojson }},\n  "output_tail": {{ output.output[-5000:] | tojson }},\n  '
        '"elided_chars": {{ output.output | length - 10000 }},\n  "warning": "Output too long."\n  {%- if '
        'output.exception_info %}, "exception_info": {{ output.exception_info | tojson }}{% endif %}\n}\n'
        "{%- endif -%}\n"
    )
    text = (
        "{% if output.exception_info -%}\n<exception>{{output.exception_info}}</exception>\n{% endif -%}\n"
        "<returncode>{{output.returncode}}</returncode>\n{% if output.output | length < 10000 -%}\n<output>\n"
        "{{ output.output -}}\n</output>\n{%- else -%}\n<warning>\nThe output of your last command was too long.\n"
        "Please try a different command that produces less output.\nIf you're looking at a file you can try use "
        "head, tail or sed to view a smaller number of lines selectively.\nIf you're using grep or find and it "
        "produced too much output, you can use a more selective search pattern.\nIf you really need to see "
        "something from the full command's output, you can redirect output to a file and then search in that "
        "file.\n</warning>\n{%- set elided_chars = output.output | length - 10000 -%}\n<output_head>\n"
        "{{ output.output[:5000] }}\n</output_head>\n<elided_chars>\n{{ elided_chars }} characters elided\n"
        "</elided_chars>\n<output_tail>\n{{ output.output[-5000:] }}\n</output_tail>\n{%- endif -%}\n"
    )
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    for out, rc, exc in (
        ("hi <b> & 'q' è\n", 0, ""),
        ("", -1, "An error occurred: timed out"),
        ("x" * 12000, 1, ""),
        ("y" * 10500, 2, "boom"),
    ):
        o = {"output": out, "returncode": rc, "exception_info": exc}
        assert mini_observation(out, rc, exc) == env.from_string(fc).render(output=o)
        assert mini_text_observation(out, rc, exc) == env.from_string(text).render(output=o)


def test_mini_bash_runs_under_sh_with_the_harness_environment(rt):
    s = session("mini-swe-agent", rt)
    out = json.loads(s.call_tool("bash", {"command": "echo $PAGER; exit 3"}))
    assert out == {"returncode": 3, "output": "cat\n"}


# ================================================================== rendering through the chat template


def _template():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", local_files_only=True)
    except Exception:  # noqa: BLE001 — no local tokenizer: a stub template with the same inputs
        import jinja2

        class Stub:
            def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True, **kw):
                return jinja2.Template(
                    "{% if tools %}<tools>{% for t in tools %}{{ t | tojson }}\n{% endfor %}</tools>{% endif %}"
                    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
                ).render(messages=messages, tools=tools)

        return Stub()


@pytest.mark.parametrize("name", ["mini-swe-agent", "mini-swe-agent-text", "opencode", "pi", "openhands-sdk"])
def test_profiles_render_through_the_chat_template(name, rt):
    tok = _template()
    s = session(name, rt)
    msgs = s.conversation("Rename notes.txt to todo.txt.")
    schemas = s.tool_schemas()
    text = tok.apply_chat_template(msgs, tools=schemas or None, tokenize=False, add_generation_prompt=True)
    assert msgs[-1]["role"] == "user" and "Rename notes.txt to todo.txt." in msgs[-1]["content"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"].split("\n")[0][:40] in text
    assert "{workdir}" not in text and "{instruction}" not in text and "{git_repo}" not in text
    if s.profile.protocol.native:
        for t in schemas:
            f = t["function"]
            assert f'"name": "{f["name"]}"' in text or f'"name":"{f["name"]}"' in text
            for param in f["parameters"]["properties"]:
                assert f'"{param}"' in text
    else:
        assert schemas == [] and "mswea_bash_command" in text and '"name": "bash"' not in text
    if name == "openhands-sdk":
        assert f"Your current working directory is: {rt.workdir}" in json.dumps(schemas)
    if name == "opencode":
        assert f"Working directory: {rt.workdir}" in msgs[0]["content"] and "git repo: no" in msgs[0]["content"]


def test_tool_contracts_are_exact():
    oc = {t.name: t.schema["function"]["parameters"] for t in get_profile("opencode").tools}
    assert oc["edit"]["required"] == ["filePath", "oldString", "newString"]
    assert list(oc["edit"]["properties"]) == ["filePath", "oldString", "newString", "replaceAll"]
    assert list(oc["bash"]["properties"]) == ["command", "timeout", "workdir"]
    pi = {t.name: t.schema["function"]["parameters"] for t in get_profile("pi").tools}
    assert pi["edit"]["properties"]["edits"]["items"]["required"] == ["oldText", "newText"]
    oh = {t.name: t.schema["function"]["parameters"] for t in get_profile("openhands-sdk").tools}
    assert list(oh["terminal"]["properties"]) == ["security_risk", "summary", "command", "is_input", "timeout", "reset"]
    assert oh["finish"]["required"] == ["message"] and oh["file_editor"]["required"] == ["command", "path"]
    mini = get_profile("mini-swe-agent").tools[0].schema
    assert mini == {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
                "required": ["command"],
            },
        },
    }
    for key in available():
        p = get_profile(key)
        assert p.id == key and p.version and (p.emulates or p.legacy)


# ================================================================== protocols


@pytest.mark.parametrize("proto", [XmlTool(), JsonBlock(), Tags(["bash", "read"])])
def test_text_protocols_round_trip(proto):
    calls = [("bash", {"command": 'ls -la "x y"'}), ("read", {"path": "a.txt", "n": 2})]
    assert proto.parse("Let me look.\n" + proto.render(calls) + "\nthen answer") == calls
    assert proto.parse("no action here") == []
    assert proto.observation([("bash", "out")])


def test_react_and_fenced_round_trip_and_errors():
    r = ReAct()
    one = [("bash", {"command": "ls"})]
    assert r.parse("Thought: x\n" + r.render(one)) == one
    with pytest.raises(FormatError):
        r.parse(r.render(one) + "\n" + r.render(one))
    f = FencedCommand()
    assert f.parse("THOUGHT: list\n\n" + f.render(one)) == one
    with pytest.raises(FormatError, match="found 2"):
        f.parse(f.render(one) + "\n" + f.render(one))
    with pytest.raises(FormatError, match="found 0"):
        f.parse("I am done.")
    assert "bash(command: string" in XmlTool().instructions(get_profile("pi").schemas())


# ================================================================== stop and answer rules


def test_mini_sentinel_submits_and_a_turn_without_a_call_is_a_format_error(rt):
    s = session("mini-swe-agent", rt)
    [(name, args)] = s.text_actions("I think I am done.")
    assert name == FORMAT_ERROR and args["message"].startswith("Tool call error:\n\n<error>\nNo tool calls found")
    s.call_tool("bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && false"})
    assert not s.done  # exit code 1
    s.call_tool("bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo extra"})
    assert s.done and s.answer == "extra\n" and s.final_message([{"role": "assistant", "content": "x"}]) == "extra\n"


def test_mini_text_parses_one_block_and_rejects_two(rt):
    s = session("mini-swe-agent-text", rt)
    assert s.text_actions("THOUGHT: x\n\n```mswea_bash_command\nls\n```") == [("bash", {"command": "ls"})]
    [(name, args)] = s.text_actions("```mswea_bash_command\nls\n```\n```mswea_bash_command\npwd\n```")
    assert name == FORMAT_ERROR and "found 2 actions" in args["message"]
    assert s.observation_text([(name, s.call_tool(name, args))]).startswith("Format error:")
    out = s.observation_text([("bash", s.call_tool("bash", {"command": "echo hi"}))])
    assert out == "<returncode>0</returncode>\n<output>\nhi\n</output>"


def test_no_call_and_finish_rules(rt):
    oc = session("opencode", rt)
    assert oc.text_actions("All done: renamed.") is None  # the episode ends
    msgs = [{"role": "assistant", "content": "", "tool_calls": []}, {"role": "assistant", "content": "Renamed."}]
    assert oc.final_message(msgs) == "Renamed."
    oh = session("openhands-sdk", rt)
    assert oh.text_actions("done") is None
    assert oh.final_message(msgs) == "Renamed."  # no finish: the last text
    oh.call_tool("finish", {"message": "The port is 8080."})
    assert oh.done and oh.final_message(msgs) == "The port is 8080."


def test_submit_confirm_and_flag_rules(rt):
    s = session("pi", rt)
    s.profile.stop = Submit("submit", confirm=True)
    rule = s.profile.stop
    assert rule.on_tool(s, "submit", {}, "ok") == rule.confirm_message and not s.done
    rule.on_tool(s, "bash", {}, "x")  # anything in between disarms
    assert rule.on_tool(s, "submit", {}, "ok") == rule.confirm_message
    assert rule.on_tool(s, "submit", {}, "ok") == "ok" and s.done
    s2 = session("pi", rt)
    s2.profile.stop = Flag(confirm=True)
    assert s2.call_tool(FLAG, {}) == s2.profile.stop.confirm_message and not s2.done
    s2.call_tool(FLAG, {})
    assert s2.done and LastText()(s2, [{"role": "assistant", "content": "a"}]) == "a"


# ================================================================== mixing and HarborEnv integration


def test_profile_draw_is_deterministic_per_group_and_varies_across_groups():
    mix = HarnessMix(profiles={"opencode": 2, "pi": 1, "mini-swe-agent": 1})
    row = {"task_dir": "/t/a", "messages": [{"role": "user", "content": "x"}]}
    assert all(mix.draw(row, 7).id == mix.draw(dict(row), 7).id for _ in range(4))
    counts = Counter(mix.draw(row, g).name for g in range(4000))
    assert set(counts) == {"opencode", "pi", "mini-swe-agent"}
    assert abs(counts["opencode"] / 4000 - 0.5) < 0.04 and abs(counts["pi"] / 4000 - 0.25) < 0.04
    assert mix.draw({**row, "harness_profile": "openhands-sdk"}, 7).name == "openhands-sdk"
    assert mix.draw(row).id == mix.draw(dict(row)).id  # no group id: the row decides
    other = HarnessMix(profiles={"opencode": 2, "pi": 1, "mini-swe-agent": 1}, seed=1)
    assert [mix.draw(row, g).name for g in range(50)] != [other.draw(row, g).name for g in range(50)]
    with pytest.raises(KeyError):
        HarnessMix(profile="claude-code")


@pytest.fixture
def local_harbor(rt, monkeypatch):
    monkeypatch.setattr(harbor, "DockerRuntime", LocalRuntime)
    LocalRuntime.verified = []
    return rt


def _row():
    return {"task_dir": str(FIXTURE), "messages": harbor.harbor_rows(FIXTURE.parent)[0]["messages"]}


def test_default_profile_is_byte_identical_to_harbor_env(local_harbor):
    base, prof = HarborEnv(), HarborEnv(profile="palingenesis-default@1")
    row = _row()
    assert base.reset(**row) is None and prof.reset(**row) is None
    assert json.dumps(base.tool_schemas()) == json.dumps(prof.tool_schemas())
    assert base.text_actions("done") is None and prof.text_actions("done") is None
    for name, args in (
        ("bash", {"command": "echo hi; echo err >&2"}),
        ("write_file", {"path": "hello.txt", "content": "cia0\n"}),
        ("str_replace", {"path": "hello.txt", "old": "cia0", "new": "ciao"}),
        ("read_file", {"path": "hello.txt"}),
        ("submit", {}),
    ):
        a, b = base.call_tool(name, args), prof.call_tool(name, args)
        assert a.replace(base.rt.workdir, "W") == b.replace(prof.rt.workdir, "W")
    assert base.done and prof.done
    msgs = [{"role": "assistant", "content": "Fatto."}]
    ra, rb = base.get_reward(msgs), prof.get_reward(msgs)
    assert LocalRuntime.verified == ["Fatto.", "Fatto."]
    assert ra == {"reward": 1.0, "metric/extra": 0.5}
    assert rb == {**ra, "metric/profile/palingenesis-default": 1.0, "metric/profile/palingenesis-default/reward": 1.0}
    tok = _template()
    render = lambda env: tok.apply_chat_template(row["messages"], tools=env.tool_schemas(), tokenize=False)  # noqa
    assert render(base) == render(prof)


def test_harbor_env_runs_a_profile_and_logs_it(local_harbor):
    env = HarborEnv(profiles={"openhands-sdk": 1, "opencode": 1})
    env.episode_group = 3
    conv = env.reset(**_row())
    name = env.session.profile.name
    assert env.episode_info == {"harness_profile": env.session.profile.id}
    assert conv[0]["role"] == "system" and conv[-1]["content"] == harbor.HarborTask.load(FIXTURE).instruction
    assert [t["function"]["name"] for t in env.tool_schemas()] == [t.name for t in env.session.profile.tools]
    if name == "openhands-sdk":
        env.call_tool("terminal", {"command": "echo ciao > hello.txt"})
        env.call_tool("finish", {"message": "Done: hello.txt."})
        assert env.done
        final = "Done: hello.txt."
    else:
        env.call_tool("write", {"filePath": f"{env.rt.workdir}/hello.txt", "content": "ciao\n"})
        assert not env.done and env.text_actions("Done.") is None
        final = "Done."
    r = env.get_reward([{"role": "assistant", "content": "Done."}])
    assert LocalRuntime.verified == [final] and r["reward"] == 1.0
    other = ({"openhands-sdk", "opencode"} - {name}).pop()
    assert r[f"metric/profile/{name}"] == 1.0 and r[f"metric/profile/{other}"] == 0.0
    assert r[f"metric/profile/{name}/reward"] == 1.0
    env.close()
    assert env.session is None


def _pipeline_episode(profile, script, env_args=None, kwargs=None):
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
    kwargs = kwargs or {"enable_thinking": False}
    eot = end_of_turn_id(tok, kwargs)
    config = RLConfig()
    for key, value in {
        "model.policy": "x",
        "model.chat_template_kwargs": kwargs,
        "env.max_turns": 6,
        "rollout.max_new_tokens": 128,
        "rollout.max_model_len": 16384,
    }.items():
        config.set(key, value)
    rows = [_row()]
    pipeline = RLPipeline(
        tok,
        ChatFormat(tok, eot, kwargs),
        ScriptedEngine(tok, eot, script),
        PublishedWeights(torch.nn.Linear(1, 1)),
        config,
        PromptSampler(rows),
        [],
        EnvPool(HarborEnv, {"profile": profile, **(env_args or {})}),
        None,
        (eot,),
    )
    try:
        (group,) = pipeline._await(pipeline._eval(rows, 1.0))
    finally:
        pipeline.close()
    return group[0]


def test_trainer_rollout_with_a_text_profile(local_harbor):
    t = _pipeline_episode(
        "mini-swe-agent-text",
        [
            "THOUGHT: write it.\n\n```mswea_bash_command\necho ciao > hello.txt\n```",
            "I am done.",  # no block: mini answers with its format error
            "THOUGHT: submit.\n\n```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
            "never",
        ],
    )
    assert t.finish == "env_done" and t.reward == 1.0 and t.info["harness_profile"] == "mini-swe-agent-text@2.4.6"
    assert t.info["prompt"][0]["content"].startswith("You are a helpful assistant that can interact with a computer.")
    users = [m["content"] for m in t.messages if m["role"] == "user"]
    assert users[1].startswith("<returncode>0</returncode>") and users[2].startswith("Format error:")
    assert t.rewards["env/metric/profile/mini-swe-agent-text"] == 1.0
    assert LocalRuntime.verified == [""]  # mini has no answer channel: the (empty) submission


def test_trainer_rollout_with_a_native_profile(local_harbor):
    call = json.dumps({"name": "write", "arguments": {"filePath": "hello.txt", "content": "ciao\n"}})
    t = _pipeline_episode("opencode", [f"<tool_call>\n{call}\n</tool_call>", "Wrote hello.txt.", "never"])
    assert t.finish == "stop" and t.reward == 1.0 and t.tool_calls == 1
    assert [m["content"] for m in t.messages if m["role"] == "tool"] == ["Wrote file successfully."]
    assert t.info["harness_profile"] == "opencode@1.19.2" and LocalRuntime.verified == ["Wrote hello.txt."]


# ================================================================== Docker: Scrivania tasks per profile

TASKS = Path(os.environ.get("PGS_SCRIVANIA_TASKS", str(Path.home() / "pgs-run/data/harbor3/scrivania/tasks")))
ARCHIVE, ENV_EDIT = "scr_archive_create_l1_en_0000", "scr_env_edit_l1_en_0003"
TAR = "cd /workdir && tar -czf website.tar.gz -C Projects website"
SED = "sed -i 's/^DEBUG=true$/DEBUG=false/' /workdir/Projects/sito/.env"
DOTENV = "/workdir/Projects/sito/.env"
SENTINEL = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def _docker_tasks_ok():
    if not shutil.which("docker") or not (TASKS / ARCHIVE).exists() or not (TASKS / ENV_EDIT).exists():
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def _fenced(cmd):
    return f"THOUGHT: next step.\n\n```mswea_bash_command\n{cmd}\n```"


# scripted episodes: per turn, native calls [(name, args)] or the assistant's text
SCRIPTS = {
    "palingenesis-default": {
        ARCHIVE: [[("bash", {"cmd": TAR})], "Created website.tar.gz."],
        ENV_EDIT: [
            [("glob", {"pat": "**/.env*"})],
            [("read", {"path": "Projects/sito/.env"})],
            [("edit", {"path": "Projects/sito/.env", "old": "DEBUG=true", "new": "DEBUG=false"})],
            "Done.",
        ],
        "noop": ["Done."],
    },
    "palingenesis-default@1": {
        ARCHIVE: [[("bash", {"command": TAR})], [("submit", {})]],
        ENV_EDIT: [
            [("read_file", {"path": "Projects/sito/.env"})],
            [("str_replace", {"path": "Projects/sito/.env", "old": "DEBUG=true", "new": "DEBUG=false"})],
            [("submit", {})],
        ],
        "noop": [[("submit", {})]],
    },
    "mini-swe-agent": {
        ARCHIVE: [[("bash", {"command": TAR})], [("bash", {"command": SENTINEL})]],
        ENV_EDIT: [
            [("bash", {"command": f"nl -ba {DOTENV}"})],
            [("bash", {"command": SED})],
            [("bash", {"command": SENTINEL})],
        ],
        "noop": [[("bash", {"command": SENTINEL})]],
    },
    "mini-swe-agent-text": {
        ARCHIVE: [_fenced(TAR), _fenced(SENTINEL)],
        ENV_EDIT: [_fenced(f"cat {DOTENV}"), _fenced(SED), _fenced(SENTINEL)],
        "noop": [_fenced(SENTINEL)],
    },
    "opencode": {
        ARCHIVE: [
            [("bash", {"command": "tar -czf website.tar.gz -C Projects website", "workdir": "/workdir"})],
            "Created website.tar.gz.",
        ],
        ENV_EDIT: [
            [("glob", {"pattern": "**/.env"})],
            [("read", {"filePath": DOTENV})],
            [("edit", {"filePath": DOTENV, "oldString": "DEBUG=true", "newString": "DEBUG=false"})],
            "Done.",
        ],
        "noop": ["Done."],
    },
    "pi": {
        ARCHIVE: [[("bash", {"command": TAR})], "Created website.tar.gz."],
        ENV_EDIT: [
            [("read", {"path": "Projects/sito/.env"})],
            [("edit", {"path": "Projects/sito/.env", "edits": [{"oldText": "DEBUG=true", "newText": "DEBUG=false"}]})],
            "Done.",
        ],
        "noop": ["Done."],
    },
    "openhands-sdk": {
        ARCHIVE: [
            [("terminal", {"command": "cd /workdir/Projects", "summary": "enter", "security_risk": "LOW"})],
            [("terminal", {"command": "tar -czf ../website.tar.gz website", "security_risk": "LOW"})],
            [("finish", {"message": "Created website.tar.gz."})],
        ],
        ENV_EDIT: [
            [("file_editor", {"command": "view", "path": "/workdir/Projects/sito"})],
            [
                (
                    "file_editor",
                    {
                        "command": "str_replace",
                        "path": DOTENV,
                        "old_str": "DEBUG=true",
                        "new_str": "DEBUG=false",
                        "security_risk": "MEDIUM",
                    },
                )
            ],
            [("finish", {"message": "DEBUG is now false."})],
        ],
        "noop": [[("finish", {"message": "Done."})]],
    },
}


def play(env: HarborEnv, turns) -> tuple[list[dict], list[str]]:
    """Run a scripted episode the way the trainer's rollout loop does; returns (messages, observations)."""
    messages, seen = [], []
    for turn in turns:
        if isinstance(turn, str):  # text: the environment parses (text protocols) or the episode ends
            messages.append({"role": "assistant", "content": turn})
            actions = env.text_actions(turn) or []
            if not actions:
                break
            outs = [env.call_tool(n, a) for n, a in actions]
            obs = env.observation_text(list(zip([n for n, _ in actions], outs)))
            messages.append({"role": "user", "content": obs})
            seen.append(obs)
        else:
            messages.append(
                {"role": "assistant", "content": "", "tool_calls": [{"name": n, "arguments": a} for n, a in turn]}
            )
            for n, a in turn:
                out = env.call_tool(n, a)
                messages.append({"role": "tool", "name": n, "content": out})
                seen.append(out)
        if env.done:
            break
    return messages, seen


@pytest.mark.skipif(not _docker_tasks_ok(), reason="docker or the Scrivania tasks unavailable")
@pytest.mark.parametrize("profile", sorted(SCRIPTS))
def test_scrivania_oracle_and_noop_per_profile(profile):
    for task in (ARCHIVE, ENV_EDIT):
        for mode in (task, "noop"):
            env = HarborEnv(profile=profile)
            try:
                env.reset(task_dir=str(TASKS / task))
                messages, seen = play(env, SCRIPTS[profile][mode])
                r = env.get_reward(messages)
                expected = 1.0 if mode != "noop" else 0.0
                assert r["reward"] == expected, (profile, task, mode, r, seen)
                assert r[f"metric/profile/{env.session.profile.name}"] == 1.0
                if profile.startswith("mini") and mode != "noop":
                    assert env.done and env.session.answer == ""
            finally:
                env.close()


# ================================================================== palingenesis-default@2 (nanocode)


def test_nanocode_default_schemas_prompt_and_stop(rt):
    p = get_profile("palingenesis-default")
    assert p.id == "palingenesis-default@2" and get_profile("palingenesis-default@1").legacy
    tools = {t.name: t.schema["function"] for t in p.tools}
    assert list(tools) == ["read", "write", "edit", "glob", "grep", "bash"]
    assert tools["read"]["parameters"] == {
        "type": "object",
        "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
        "required": ["path"],
    }
    assert tools["edit"]["parameters"]["required"] == ["path", "old", "new"]
    assert tools["edit"]["parameters"]["properties"]["all"] == {"type": "boolean"}
    assert tools["bash"] == {
        "name": "bash",
        "description": "Run shell command",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
    }
    s = Session(p, rt)
    assert s.conversation("ciao") == [
        {"role": "system", "content": f"Concise coding assistant. cwd: {rt.workdir}"},
        {"role": "user", "content": "ciao"},
    ]
    assert s.text_actions("Fatto.") is None


def test_nanocode_default_tools_word_results_like_nanocode(rt):
    s = Session(get_profile("palingenesis-default"), rt)
    call = s.call_tool
    assert call("write", {"path": "a.txt", "content": "uno\ndue\ndue\n"}) == "ok"
    assert call("read", {"path": "a.txt"}) == "   1| uno\n   2| due\n   3| due\n"
    assert call("read", {"path": "a.txt", "offset": 1, "limit": 1}) == "   2| due\n"
    assert call("read", {"path": "nope.txt"}) == "error: [Errno 2] No such file or directory: 'nope.txt'"
    assert call("read", {}) == "error: 'path'"
    assert call("edit", {"path": "a.txt", "old": "tre", "new": "x"}) == "error: old_string not found"
    assert (
        call("edit", {"path": "a.txt", "old": "due", "new": "x"})
        == "error: old_string appears 2 times, must be unique (use all=true)"
    )
    assert call("edit", {"path": "a.txt", "old": "due", "new": "tre", "all": True}) == "ok"
    assert (Path(rt.workdir) / "a.txt").read_text() == "uno\ntre\ntre\n"
    assert call("write", {"path": "sub/b.py", "content": "x"}).startswith("error: [Errno 2] No such file")
    (Path(rt.workdir) / "sub").mkdir()
    assert call("write", {"path": "sub/b.py", "content": "import os\n"}) == "ok"
    assert call("glob", {"pat": "**/*.py"}) == "./sub/b.py"
    assert call("glob", {"pat": "*.md"}) == "none"
    assert call("grep", {"pat": "^tre"}) == "./a.txt:2:tre\n./a.txt:3:tre"
    assert call("grep", {"pat": "import", "path": "sub"}) == "sub/b.py:1:import os"
    assert call("bash", {"cmd": "echo out; echo err >&2"}) == "out\nerr"
    assert call("bash", {"cmd": "true"}) == "(empty)"
    assert call("bash", {}) == "error: 'cmd'"
    assert call("str_replace", {"path": "a.txt"}) == "error: 'str_replace'"
    assert not s.done
