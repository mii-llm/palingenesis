"""Harbor task environments (palingenesis.rl.envs.harbor). Docker tests are skipped when docker is unavailable."""

import shutil
import subprocess
from pathlib import Path

import pytest

from palingenesis.rl.envs.harbor import HarborEnv, HarborTask, check_task, harbor_rows

FIXTURE = Path(__file__).parent / "fixtures" / "harbor_task"


def _docker_ok():
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def test_task_loads_and_rows():
    t = HarborTask.load(FIXTURE)
    assert t.name == "fixture/hello" and "hello.txt" in t.instruction
    assert t.environment["memory_mb"] == 512 and t.verifier["env"]["EXPECTED"] == "ciao"
    assert t.image().startswith("pgs-harbor:")
    rows = harbor_rows(FIXTURE.parent)
    assert rows and rows[0]["task_dir"].endswith("harbor_task") and rows[0]["messages"][-1]["role"] == "user"


@pytest.mark.skipif(not _docker_ok(), reason="docker unavailable")
def test_oracle_passes_and_nop_fails():
    r = check_task(FIXTURE)
    assert r["oracle"] == 1.0 and r["nop"] == 0.0 and r["valid"], r


@pytest.mark.skipif(not _docker_ok(), reason="docker unavailable")
def test_episode_with_tools_and_verifier_reward():
    def episode():
        env = HarborEnv()
        env.reset(task_dir=str(FIXTURE))
        try:
            out = env.call_tool("bash", {"command": "ls /workdir; echo hi"})
            assert "[exit code 0]" in out and "hi" in out
            env.call_tool("write_file", {"path": "hello.txt", "content": "cia0\n"})
            assert env.get_reward()["reward"] == 0.0
            assert (env.call_tool("str_replace", {"path": "hello.txt", "old": "cia0", "new": "ciao"})).startswith(
                "Edited"
            )
            env.call_tool("submit", {})
            assert env.done
            assert env.get_reward()["reward"] == 1.0
            assert (
                "network" in (env.call_tool("bash", {"command": "getent hosts example.com || echo no network"})).lower()
            )
        finally:
            env.close()

    episode()


@pytest.mark.skipif(not _docker_ok(), reason="docker unavailable")
def test_setup_runs_and_verifier_is_isolated_from_the_agent(tmp_path):
    task = tmp_path / "task"
    shutil.copytree(FIXTURE, task)
    (task / "environment" / "setup.sh").write_text("set -e\ncd /workdir\necho seeded > seeded.txt\n")

    def episode():
        env = HarborEnv()
        env.reset(task_dir=str(task))
        try:
            assert "seeded" in env.call_tool("bash", {"command": "cat seeded.txt"})
            # an agent that forges the reward from a background process must not reach the verifier
            env.call_tool(
                "bash",
                {
                    "command": "nohup bash -c 'while true; do echo 1 > /logs/verifier/reward.txt; "
                    "done' >/dev/null 2>&1 &"
                },
            )
            assert env.get_reward()["reward"] == 0.0
        finally:
            env.close()

    episode()


def test_task_workdir_and_verifier_network_settings(tmp_path):
    task = tmp_path / "task"
    shutil.copytree(FIXTURE, task)
    toml = (
        (task / "task.toml")
        .read_text()
        .replace("allow_internet = false", 'allow_internet = false\nworkdir = "/work/ws"')
    )
    (task / "task.toml").write_text(toml.replace("[verifier]\n", "[verifier]\nallow_internet = true\n"))
    t = HarborTask.load(task)
    assert t.workdir == "/work/ws" and t.verifier["allow_internet"] is True
    assert HarborTask.load(FIXTURE).workdir is None


def _variant(tmp_path, workdir=None, user=False, test_body=None):
    """The fixture with an optional task working directory, an unprivileged image user and another test.sh."""
    task = tmp_path / "task"
    shutil.copytree(FIXTURE, task)
    wd = workdir or "/workdir"
    if workdir:
        toml = (task / "task.toml").read_text()
        (task / "task.toml").write_text(
            toml.replace("allow_internet = false", f'allow_internet = false\nworkdir = "{wd}"')
        )
    if user:
        (task / "environment" / "Dockerfile").write_text(
            f"FROM python:3.11-slim\nRUN useradd -m -u 1234 agent && mkdir -p {wd} /logs && chown agent {wd} /logs\n"
            f"USER agent\nWORKDIR {wd}\n"
        )
    body = test_body or (
        f'if [ "$(cat {wd}/hello.txt 2>/dev/null | tr -d \'[:space:]\')" = "$EXPECTED" ]; '
        "then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi"
    )
    (task / "tests" / "test.sh").write_text("#!/usr/bin/env bash\nmkdir -p /logs/verifier\n" + body + "\n")
    (task / "solution" / "solve.sh").write_text(f"echo ciao > {wd}/hello.txt\n")
    return task


@pytest.mark.skipif(not _docker_ok(), reason="docker unavailable")
def test_task_workdir_is_used_by_tools_verifier_and_check(tmp_path):
    task = _variant(tmp_path, workdir="/work/ws")
    r = check_task(task)
    assert r["oracle"] == 1.0 and r["nop"] == 0.0 and r["valid"], r
    env = HarborEnv()
    env.reset(task_dir=str(task))
    try:
        assert "/work/ws" in env.call_tool("bash", {"command": "pwd"})
        assert env.call_tool("write_file", {"path": "hello.txt", "content": "ciao\n"}) == "Wrote /work/ws/hello.txt."
        assert env.get_reward()["reward"] == 1.0
    finally:
        env.close()


@pytest.mark.skipif(not _docker_ok(), reason="docker unavailable")
def test_tool_writes_belong_to_an_unprivileged_image_user(tmp_path):
    task = _variant(tmp_path, user=True)
    env = HarborEnv()
    env.reset(task_dir=str(task))
    try:
        env.call_tool("write_file", {"path": "hello.txt", "content": "cia0\n"})
        assert "agent" in env.call_tool("bash", {"command": "stat -c %U hello.txt"})
        # the agent's own shell can still change a file the tools wrote
        assert "[exit code 0]" in env.call_tool("bash", {"command": "echo ciao > hello.txt"})
        assert env.call_tool("str_replace", {"path": "hello.txt", "old": "ciao", "new": "cia0"}).startswith("Edited")
        assert env.call_tool("str_replace", {"path": "hello.txt", "old": "cia0", "new": "ciao"}).startswith("Edited")
        # a no-op edit is an error and writes nothing
        assert env.call_tool("str_replace", {"path": "hello.txt", "old": "ciao", "new": "ciao"}).startswith("Error:")
        assert env.get_reward()["reward"] == 1.0
    finally:
        env.close()


@pytest.mark.skipif(not _docker_ok(), reason="docker unavailable")
def test_final_message_reaches_the_verifier(tmp_path):
    body = (
        'grep -q "fatto" /logs/agent/final_message.md 2>/dev/null && echo 1 > /logs/verifier/reward.txt '
        "|| echo 0 > /logs/verifier/reward.txt"
    )
    task = _variant(tmp_path, test_body=body)
    env = HarborEnv()
    env.reset(task_dir=str(task))
    try:
        assert env.get_reward()["reward"] == 0.0  # no conversation: no final message
        msgs = [{"role": "user", "content": "fai"}, {"role": "assistant", "content": "Ho fatto."}]
        assert env.get_reward(msgs)["reward"] == 1.0
    finally:
        env.close()


@pytest.mark.skipif(not _docker_ok(), reason="docker unavailable")
def test_verifier_network_is_its_own_setting(tmp_path):
    body = (
        "if getent hosts example.com >/dev/null; then echo 1 > /logs/verifier/reward.txt; "
        "else echo 0 > /logs/verifier/reward.txt; fi"
    )
    task = _variant(tmp_path, test_body=body)
    toml = (task / "task.toml").read_text()
    (task / "task.toml").write_text(toml.replace("[verifier]\n", "[verifier]\nallow_internet = true\n"))
    env = HarborEnv()
    env.reset(task_dir=str(task))
    try:
        assert "no network" in env.call_tool("bash", {"command": "getent hosts example.com || echo no network"})
        assert env.get_reward()["reward"] == 1.0  # the verifier resolves names, the agent cannot
    finally:
        env.close()


@pytest.mark.skipif(not _docker_ok(), reason="docker unavailable")
def test_containers_of_exited_runs_are_swept_and_failed_starts_leave_none(tmp_path):
    import sys

    from palingenesis.rl.envs import harbor

    def ours():
        r = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label={harbor._LABEL}"], capture_output=True, text=True
        )
        return set(r.stdout.split())

    before = ours()
    # a run killed mid-episode: its container outlives it
    code = (
        "import os\nfrom palingenesis.rl.envs.harbor import HarborEnv\n"
        f"env = HarborEnv(); env.reset(task_dir={str(FIXTURE)!r}); os._exit(0)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    left = ours() - before
    assert len(left) == 1
    assert harbor.sweep_orphans() >= 1 and not (ours() & left)

    # a setup that fails removes the container it started
    task = tmp_path / "task"
    shutil.copytree(FIXTURE, task)
    (task / "environment" / "setup.sh").write_text("exit 3\n")
    env = HarborEnv()
    with pytest.raises(RuntimeError, match="setup failed"):
        env.reset(task_dir=str(task))
    env.close()
    assert ours() == before and not harbor._live


class _FakeDocker:
    """Records docker calls; answers exec / cp like a container whose capture prints CAPTURE."""

    CAPTURE = b'{"procs": [{"tag": "s1"}]}'

    def __init__(self):
        self.calls, self.copied = [], {}

    def __call__(self, *args, input=None, timeout=None, check=True, limit=None):
        import io
        import tarfile

        self.calls.append(args)
        out = b""
        if args[0] == "exec":
            cmd = args[-1]
            if "-u" in args and "capture" in cmd:
                out = self.CAPTURE
            elif "reward.json" in cmd:
                out = b'{"reward": 1.0}'
        elif args[0] == "cp" and input:  # (the working-directory archive from the fake container is empty)
            with tarfile.open(fileobj=io.BytesIO(input)) as tar:
                for m in tar.getmembers():
                    if m.isfile():
                        self.copied["/" + m.name] = tar.extractfile(m).read()
        return subprocess.CompletedProcess(args, 0, out, b"")

    def pipe(self, src, dst, limit, timeout, what="the stream"):  # harbor._pipe_capped of two docker commands
        out = self(*src[1:]).stdout
        if len(out) > limit:
            raise RuntimeError(f"{what} is over {limit >> 20} MiB")
        self(*dst[1:], input=out)
        return len(out)

    def install(self, monkeypatch, harbor):
        monkeypatch.setattr(harbor, "_docker", self)
        monkeypatch.setattr(harbor, "_pipe_capped", self.pipe)
        return self


def _capture_task(tmp_path, capture=True):
    task = tmp_path / "task"
    shutil.copytree(FIXTURE, task)
    if capture:
        toml = (task / "task.toml").read_text()
        (task / "task.toml").write_text(
            toml.replace("[verifier]\n", '[verifier]\ncapture = "python3 /opt/capture.py"\n')
        )
    return task


def test_verifier_capture_runs_as_root_in_the_agent_container_and_reaches_the_tests(tmp_path, monkeypatch):
    from palingenesis.rl.envs import harbor

    fake = _FakeDocker().install(monkeypatch, harbor)
    monkeypatch.setattr(harbor, "_swept", True)
    rt = harbor.DockerRuntime(HarborTask.load(_capture_task(tmp_path)))
    rt.started = True
    assert rt.verify(final_message="fatto") == {"reward": 1.0}
    caps = [c for c in fake.calls if c[0] == "exec" and "-u" in c]
    assert len(caps) == 1 and caps[0][caps[0].index("-u") + 1] == "0" and caps[0][-1] == "python3 /opt/capture.py"
    assert caps[0][caps[0].index("-w") + 1] == "/" and rt.name in caps[0]  # the agent's container, not the clean one
    first_cp = next(i for i, c in enumerate(fake.calls) if c[0] == "cp")
    assert fake.calls.index(caps[0]) < first_cp  # captured before the working directory is copied out
    assert fake.copied["/logs/agent/capture.json"] == _FakeDocker.CAPTURE
    assert fake.copied["/logs/agent/final_message.md"] == b"fatto"


def test_no_capture_keeps_the_default_verification(tmp_path, monkeypatch):
    from palingenesis.rl.envs import harbor

    fake = _FakeDocker().install(monkeypatch, harbor)
    monkeypatch.setattr(harbor, "_swept", True)
    rt = harbor.DockerRuntime(HarborTask.load(_capture_task(tmp_path, capture=False)))
    rt.started = True
    assert rt.verify() == {"reward": 1.0}
    assert not any("-u" in c for c in fake.calls if c[0] == "exec")
    assert "/logs/agent/capture.json" not in fake.copied


def test_a_failed_capture_scores_zero(tmp_path, monkeypatch):
    from palingenesis.rl.envs import harbor

    class Dead(_FakeDocker):
        def __call__(self, *args, **kw):
            if args[0] == "exec" and "-u" in args:
                return subprocess.CompletedProcess(args, 137, b"", b"container not running")
            return super().__call__(*args, **kw)

    Dead().install(monkeypatch, harbor)
    monkeypatch.setattr(harbor, "_swept", True)
    env = HarborEnv()
    env.task = HarborTask.load(_capture_task(tmp_path))
    env.rt = harbor.DockerRuntime(env.task)
    env.rt.started = True
    r = env.get_reward([{"role": "assistant", "content": "fatto"}])
    assert r["reward"] == 0.0 and r["metric/verifier_failed"] == 1.0


class _FakeRuntime:
    """DockerRuntime stand-in for check_task: records exec / verify; the tests grade the final message."""

    runs: list = []
    passes_empty = False

    def __init__(self, task, workdir=None):
        self.task, self.execs, self.final = task, [], "unset"
        _FakeRuntime.runs.append(self)

    def start(self):
        pass

    def stop(self):
        pass

    def put_dir(self, src, dst):
        pass

    def exec(self, cmd, timeout=None):
        from types import SimpleNamespace

        self.execs.append(cmd)
        return SimpleNamespace(exit_code=0, output="")

    def verify(self, final_message=None):
        self.final = final_message
        ok = final_message is not None and ("è fatto" in final_message or (self.passes_empty and final_message == ""))
        return {"reward": float(ok)}


def _reply_task(tmp_path, solve=False):
    task = tmp_path / "reply_task"
    shutil.copytree(FIXTURE, task)
    if not solve:
        (task / "solution" / "solve.sh").unlink()
    (task / "solution" / "final_message.md").write_bytes("Ecco: è fatto.\n".encode("utf-8"))
    return task


def test_check_task_reply_only_oracle_and_empty_reply_nop(tmp_path, monkeypatch):
    from palingenesis.rl.envs import harbor

    monkeypatch.setattr(harbor, "DockerRuntime", _FakeRuntime)
    _FakeRuntime.runs, _FakeRuntime.passes_empty = [], False
    r = check_task(_reply_task(tmp_path))
    assert r["oracle"] == 1.0 and r["nop"] == 0.0 and r["nop_reply"] == 0.0 and r["valid"], r
    nop, nop_reply, oracle = _FakeRuntime.runs
    assert nop.final is None and nop_reply.final == "" and oracle.final == "Ecco: è fatto." and not oracle.execs
    assert "oracle_exit" not in r  # no solve.sh: the oracle is the reply alone

    _FakeRuntime.passes_empty = True  # tests that accept an empty reply: invalid
    assert not check_task(_reply_task(tmp_path / "b"))["valid"]


def test_check_task_runs_solve_and_reply_together_and_plain_tasks_unchanged(tmp_path, monkeypatch):
    from palingenesis.rl.envs import harbor

    monkeypatch.setattr(harbor, "DockerRuntime", _FakeRuntime)
    _FakeRuntime.runs, _FakeRuntime.passes_empty = [], False
    r = check_task(_reply_task(tmp_path, solve=True))
    assert r["valid"] and r["oracle_exit"] == 0 and _FakeRuntime.runs[-1].execs == ["bash /solution/solve.sh"]
    _FakeRuntime.runs = []
    r = check_task(FIXTURE)  # no reply: nop and oracle only (the fake tests need a reply, so the oracle scores 0)
    assert "nop_reply" not in r and len(_FakeRuntime.runs) == 2 and r["oracle"] == 0.0 and not r["valid"]


def test_run_capped_keeps_head_and_tail_with_bounded_memory():
    """A command printing without end must not be buffered whole (an agent's `yes` OOM-killed a trainer)."""
    import subprocess

    from palingenesis.rl.envs.harbor import _run_capped

    r = _run_capped(
        ["bash", "-c", "printf 'HEAD'; head -c 50000000 /dev/zero | tr '\\0' x; printf 'TAIL'"], None, 60, False, 1000
    )
    assert r.stdout.startswith(b"HEAD") and r.stdout.endswith(b"TAIL") and b"bytes cut" in r.stdout
    assert len(r.stdout) < 1100
    small = _run_capped(["bash", "-c", "echo out; echo err >&2; exit 3"], None, 10, False, 1000)
    assert (small.returncode, small.stdout, small.stderr) == (3, b"out\n", b"err\n")
    assert _run_capped(["cat"], b"abc", 10, False, 1000).stdout == b"abc"
    with pytest.raises(subprocess.TimeoutExpired):
        _run_capped(["bash", "-c", "yes"], None, 1, False, 1000)
    with pytest.raises(subprocess.CalledProcessError):
        _run_capped(["false"], None, 10, True, 1000)


def test_exec_capture_is_capped_whatever_the_runtime_max_output(tmp_path, monkeypatch):
    """Harness profiles allow 8M characters (32 MB per stream); the raw capture stops at EXEC_LIMIT, with the same
    character-cut marker (the file ops read through base64 and refuse an output carrying it)."""
    from palingenesis.rl.envs import harbor
    from palingenesis.rl.envs.harness.ops import CUT_MARK

    script = {"cmd": ""}
    limits = []

    def docker(*args, input=None, timeout=None, check=True, limit=None):
        limits.append(limit)
        return harbor._run_capped(["bash", "-c", script["cmd"]], input, timeout, check, limit)

    monkeypatch.setattr(harbor, "_docker", docker)
    rt = harbor.DockerRuntime(HarborTask.load(FIXTURE), max_output=8_000_000)
    script["cmd"] = "printf HEAD; head -c 20000000 /dev/zero | tr '\\0' x; printf TAIL"
    r = rt.exec("ignored")
    assert limits[-1] == harbor.EXEC_LIMIT
    assert r.stdout.startswith("HEADxx") and r.stdout.endswith("xxTAIL") and CUT_MARK in r.stdout
    assert len(r.stdout) < harbor.EXEC_LIMIT + 100
    n = int(r.stdout.split("... [")[1].split(" characters")[0])
    assert n + len(r.stdout.split("\n... [")[0]) + len(r.stdout.split("cut] ...\n")[1]) == 20_000_008
    script["cmd"] = "printf '%s' 0123456789abcdefghij"  # under the cap: the character cut as before
    small = harbor.DockerRuntime(HarborTask.load(FIXTURE), max_output=10).exec("ignored")
    assert limits[-1] == 4 * 10 + 4096 and small.stdout == "01234\n... [10 characters cut] ...\nfghij"


def test_pipe_capped_streams_counts_and_aborts_over_the_limit(tmp_path):
    """The verifier's working-directory copy is streamed between two docker cp, never held in memory."""
    import subprocess

    from palingenesis.rl.envs.harbor import _pipe_capped

    out = tmp_path / "out"
    src = ["bash", "-c", "head -c 3000000 /dev/zero | tr '\\0' y"]
    assert _pipe_capped(src, ["bash", "-c", f"cat > {out}"], 4 << 20, 30) == 3_000_000
    assert out.read_bytes() == b"y" * 3_000_000
    with pytest.raises(RuntimeError, match="wd is over 1 MiB"):
        _pipe_capped(src, ["bash", "-c", "cat > /dev/null"], 1 << 20, 30, "wd")
    with pytest.raises(RuntimeError, match="over 1 MiB"):  # a source that never ends
        _pipe_capped(["yes"], ["bash", "-c", "cat > /dev/null"], 1 << 20, 30)
    with pytest.raises(subprocess.CalledProcessError) as e:
        _pipe_capped(src, ["bash", "-c", "echo no space >&2; exit 2"], 4 << 20, 30)
    assert e.value.returncode == 2 and b"no space" in e.value.stderr
    with pytest.raises(subprocess.CalledProcessError):
        _pipe_capped(["bash", "-c", "exit 1"], ["bash", "-c", "cat > /dev/null"], 4 << 20, 30)
    with pytest.raises(subprocess.TimeoutExpired):
        _pipe_capped(["sleep", "30"], ["bash", "-c", "cat > /dev/null"], 4 << 20, 1)


def test_exec_caps_the_size_of_files_a_command_writes(tmp_path, monkeypatch):
    """Every command runs under ulimit -f: an agent's runaway `yes > out` must not fill the host disk."""
    from palingenesis.rl.envs import harbor

    fake = _FakeDocker().install(monkeypatch, harbor)
    rt = harbor.DockerRuntime(harbor.HarborTask.load(_capture_task(tmp_path, capture=False)))
    rt.name = "pgs-test"
    rt.exec("echo hi")
    cmd = [c for c in fake.calls if c[0] == "exec"][-1][-1]
    assert cmd.startswith(f"ulimit -f {harbor.FILE_LIMIT_KB} ") and cmd.endswith("; echo hi")
    # the prefix works in a real bash: a write past the limit is stopped (SIGXFSZ / EFBIG), the file stays at the cap
    out = tmp_path / "big"
    subprocess.run(["bash", "-lc", f"ulimit -f 4 2>/dev/null; head -c 100000 /dev/zero > {out}"], check=False)
    assert out.stat().st_size == 4096
