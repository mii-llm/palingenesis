"""Harbor task environments: one directory per task, run in a local Docker container, rewarded by the task's tests.

A task directory follows the Harbor layout (harborframework.com; the format used by e.g. SmolDataEnvs):

    task.toml            [task] name/description, [environment] resources / network / env / healthcheck / workdir
                         (default /workdir), [verifier] timeout + env + allow_internet (default: the environment's;
                         e.g. an LLM judge while the agent stays offline), [agent] timeout, [metadata] anything
    instruction.md       what the agent is asked to do (the prompt)
    environment/         Docker build context (Dockerfile + files), or [environment] docker_image to reuse a prebuilt
                         image; environment/workdir/ (optional) is copied into the working directory at start and
                         environment/setup.sh (optional) runs once after it (a database, a git repository), so many
                         small tasks can share one image
    tests/test.sh        the verifier: runs after the episode, writes /logs/verifier/reward.txt (a number) or
                         reward.json ({"name": value, ...}); anything else in tests/ is copied to /tests; the agent's
                         final message (when the trainer passes the conversation) is at /logs/agent/final_message.md
    solution/solve.sh    optional oracle solution (used by check_task: the oracle must pass, doing nothing must not)

palingenesis runs tasks itself (no Harbor dependency): images are built once per build-context hash and cached,
each episode gets a fresh container with the task's CPU / memory limits and no network unless the task allows it,
the healthcheck runs before the agent, and the tests run after it in a clean container from the same image holding
only a copy of the working directory: nothing the agent leaves running or patched outside it (a background process
forging the reward, an edited package) reaches the verifier. The agent is the policy being trained, through
the ordinary environment protocol:

    env:
      type: palingenesis.rl.envs.harbor:HarborEnv
      max_concurrent: 32                  # containers alive at once
      args: {tools: [bash, read_file, write_file, str_replace, submit]}
    data:
      dataset: tasks.jsonl                # rows {"task_dir": ".../task", "messages": [...]} (see harbor_rows)

Tools (native function calling; text protocols work through palingenesis.rl.env text_actions): bash(command,
timeout), read_file(path), write_file(path, content), str_replace(path, old, new) (exact text, must occur once),
submit() (ends the episode); files the tools write belong to the image's default user, so an unprivileged image user
can still change them from the shell. The reward is the verifier's number, or its components (a reward.json dict: every key is
logged as env/<key>, "reward" or the first key is the reward unless the caller maps them).

The same task directories also run under the Harbor CLI with external harnesses (claude-code, codex, opencode,
mini-swe-agent) against an OpenAI-compatible endpoint serving the policy, for evaluation or SFT data (not token-exact
RL trajectories); that path needs the harbor package and is not wrapped here.

    pgs harbor check <tasks_dir> [--workers 8]      oracle / nop check of every task (exit 1 if any is invalid)
    pgs harbor rows <tasks_dir> <out.jsonl>          training rows for the env above
"""

from __future__ import annotations

import atexit
import hashlib
import io
import json
import logging
import os
import shlex
import subprocess
import tarfile
import threading
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

IMAGE_PREFIX = "pgs-harbor"


# ------------------------------------------------------------------ task


@dataclass
class HarborTask:
    root: Path
    name: str
    instruction: str
    config: dict = field(default_factory=dict)

    @classmethod
    def load(cls, root: str | Path) -> "HarborTask":
        root = Path(root)
        config = tomllib.loads((root / "task.toml").read_text()) if (root / "task.toml").exists() else {}
        instruction = (root / "instruction.md").read_text() if (root / "instruction.md").exists() else \
            config.get("task", {}).get("description", "")
        name = config.get("task", {}).get("name") or root.name
        return cls(root, name, instruction.strip(), config)

    # sections with defaults
    @property
    def environment(self) -> dict:
        return self.config.get("environment", {})

    @property
    def verifier(self) -> dict:
        return self.config.get("verifier", {})

    @property
    def workdir(self) -> str | None:
        """[environment] workdir: the task's working directory, when it is not /workdir."""
        return self.environment.get("workdir")

    @property
    def agent_timeout(self) -> float:
        return float(self.config.get("agent", {}).get("timeout_sec", 900))

    @property
    def build_context(self) -> Path:
        return self.root / "environment"

    def context_hash(self) -> str:
        """Hash of the build context (file paths + bytes): tasks sharing an environment share an image."""
        h = hashlib.sha256()
        ctx = self.build_context
        if not ctx.exists():
            return "noctx"
        for p in sorted(ctx.rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(ctx)).encode())
                h.update(p.read_bytes())
        return h.hexdigest()[:16]

    def image(self) -> str:
        explicit = self.environment.get("docker_image")
        return explicit or f"{IMAGE_PREFIX}:{self.context_hash()}"


SYSTEM = ("You work in a Linux container through tools; the working directory is /workdir and there is no internet. "
          "Inspect before you change things, check your result, then call submit and tell the user briefly what you did.")


def harbor_rows(tasks_dir: str | Path, system: str | None = SYSTEM) -> list[dict]:
    """Training rows for a directory of task directories: {"task_dir", "messages": [system, instruction as the user
    turn]}; system=None leaves the system prompt out."""
    rows = []
    for d in sorted(Path(tasks_dir).iterdir()):
        if (d / "instruction.md").exists() or (d / "task.toml").exists():
            t = HarborTask.load(d)
            head = [{"role": "system", "content": system}] if system else []
            rows.append({"task_dir": str(d), "messages": head + [{"role": "user", "content": t.instruction}]})
    return rows


# ------------------------------------------------------------------ docker


def _docker(*args: str, input: bytes | None = None, timeout: float | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], input=input, capture_output=True, timeout=timeout, check=check)


# Episode containers carry the owning process in a label, so containers a dead run left behind (killed by a signal,
# OOM, a timed-out `docker rm`) are removed by the next run on the host instead of piling up: 513 leaked
# `sleep infinity` containers were found after a week of runs and evals.
_LABEL = "palingenesis.harbor.owner"
_HOST = os.uname().nodename
_live: set[str] = set()
_live_lock = threading.Lock()
_swept = False


def _remove(names: list[str], timeout: float = 60) -> bool:
    if not names:
        return True
    try:
        r = _docker("rm", "-f", *names, check=False, timeout=timeout)
        return r.returncode == 0 or b"No such container" in r.stderr
    except (subprocess.TimeoutExpired, OSError):
        return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_orphans() -> int:
    """Remove episode containers whose owning process on this host has exited; returns how many."""
    try:
        r = _docker("ps", "-a", "--filter", f"label={_LABEL}", "--format", f'{{{{.Names}}}} {{{{.Label "{_LABEL}"}}}}',
                    check=False, timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return 0
    dead = []
    for line in r.stdout.decode(errors="replace").splitlines():
        name, _, owner = line.strip().partition(" ")
        host, _, pid = owner.rpartition(":")
        if host == _HOST and pid.isdigit() and not _pid_alive(int(pid)):
            dead.append(name)
    for i in range(0, len(dead), 100):
        _remove(dead[i:i + 100], timeout=300)
    if dead:
        logger.warning("removed %d Harbor containers left behind by exited runs", len(dead))
    return len(dead)


def _sweep_once() -> None:
    global _swept
    with _live_lock:
        if _swept:
            return
        _swept = True
    sweep_orphans()


@atexit.register
def _remove_live() -> None:
    with _live_lock:
        names = sorted(_live)
        _live.clear()
    for i in range(0, len(names), 100):
        _remove(names[i:i + 100], timeout=300)


def build_image(task: HarborTask, timeout: float | None = None) -> str:
    """Build (once) the image of a task's environment; returns its tag. An explicit [environment].docker_image is
    pulled instead."""
    tag = task.image()
    if _docker("image", "inspect", tag, check=False).returncode == 0:
        return tag
    if task.environment.get("docker_image"):
        _docker("pull", tag, timeout=timeout or 1800)
        return tag
    timeout = timeout or float(task.environment.get("build_timeout_sec", 600))
    r = _docker("build", "-q", "-t", tag, str(task.build_context), timeout=timeout, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"docker build failed for {task.name}: {r.stderr.decode()[-800:]}")
    return tag


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class DockerRuntime:
    """One task episode's container: start (limits, network, env, healthcheck), exec, files, verify, stop."""

    def __init__(self, task: HarborTask, workdir: str | None = None, max_output: int = 16000):
        self.task = task
        self.workdir = workdir or task.workdir or "/workdir"  # explicit > task.toml [environment] workdir > /workdir
        self.max_output = max_output
        self.name = f"pgs-{uuid.uuid4().hex[:12]}"
        self.started = False
        self._owner: tuple[int, int] | None = None

    def start(self, provision: bool = True, allow_internet: bool | None = None) -> None:
        """provision: copy the starting files and run environment/setup.sh (off for the verifier's clean container).
        allow_internet: overrides the task's [environment] allow_internet (the verifier's container uses
        [verifier] allow_internet)."""
        env = self.task.environment
        image = build_image(self.task)
        _sweep_once()
        args = ["run", "-d", "--name", self.name, "--init", "--label", f"{_LABEL}={_HOST}:{os.getpid()}",
                "--cpus", str(env.get("cpus", 1)), "--memory", f"{int(env.get('memory_mb', 2048))}m",
                "--pids-limit", str(env.get("pids_limit", 512))]
        if not (env.get("allow_internet", False) if allow_internet is None else allow_internet):
            args += ["--network", "none"]
        for k, v in (env.get("env") or {}).items():
            args += ["-e", f"{k}={os.path.expandvars(str(v))}"]
        args += [image, "sleep", "infinity"]
        with _live_lock:
            _live.add(self.name)
        self.started = True  # before `docker run`: a run that times out or fails may still have created the container
        try:
            _docker(*args, timeout=120)
            self._provision(provision)
        except BaseException:
            self.stop()
            raise

    def _provision(self, provision: bool) -> None:
        env = self.task.environment
        self.exec(f"mkdir -p {shlex.quote(self.workdir)} /logs/verifier", timeout=30, workdir="/")  # may not exist yet
        seed = self.task.build_context / "workdir"  # starting files copied in (many tasks can share one image)
        if provision and seed.exists():
            self.put_dir(seed, self.workdir)
        setup = self.task.build_context / "setup.sh"  # one-time setup after the files (a database, a git repo)
        if provision and setup.exists():
            self.put_files({"/tmp/pgs_setup.sh": setup.read_bytes()}, "/")
            r = self.exec("bash /tmp/pgs_setup.sh && rm -f /tmp/pgs_setup.sh", timeout=300)
            if r.exit_code != 0:
                raise RuntimeError(f"setup failed for {self.task.name}: {r.stderr[-400:]}")
        hc = env.get("healthcheck") or {}
        if hc.get("command"):
            retries, interval = int(hc.get("retries", 10)), float(hc.get("interval_sec", 2))
            for _ in range(max(1, retries)):
                if self.exec(hc["command"], timeout=float(hc.get("timeout_sec", 60))).exit_code == 0:
                    break
                import time

                time.sleep(interval)
            else:
                raise RuntimeError(f"healthcheck failed for {self.task.name}")

    def exec(self, command: str, timeout: float = 60, env: dict | None = None, workdir: str | None = None) -> ExecResult:
        args = ["exec"]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        args += ["-w", workdir or self.workdir, self.name, "bash", "-lc", command]
        try:
            r = _docker(*args, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            _docker("exec", self.name, "bash", "-lc", "pkill -f -- 'bash -lc' || true", check=False, timeout=10)
            return ExecResult(124, "", f"timed out after {timeout:g} s", True)
        cut = self.max_output
        out, err = r.stdout.decode(errors="replace"), r.stderr.decode(errors="replace")
        if len(out) > cut:
            out = out[: cut // 2] + f"\n... [{len(out) - cut} characters cut] ...\n" + out[-cut // 2:]
        return ExecResult(r.returncode, out, err[-cut:])

    def owner(self) -> tuple[int, int]:
        """uid / gid of the image's default user (what the agent's shell runs as)."""
        if self._owner is None:
            r = self.exec("id -u; id -g", timeout=30, workdir="/")
            ids = r.stdout.split()
            self._owner = (int(ids[0]), int(ids[1])) if r.exit_code == 0 and len(ids) >= 2 else (0, 0)
        return self._owner

    def put_files(self, files: dict[str, bytes | str], dest: str = "/", as_user: bool = False) -> None:
        """Copy files in. as_user: owned by the image's default user (docker cp otherwise creates them as root, which
        an unprivileged image user cannot change afterwards)."""
        uid, gid = self.owner() if as_user else (0, 0)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for path, data in files.items():
                data = data.encode() if isinstance(data, str) else data
                info = tarfile.TarInfo(path.lstrip("/"))
                info.size, info.mode = len(data), 0o755 if path.endswith(".sh") else 0o644
                info.uid, info.gid = uid, gid
                tar.addfile(info, io.BytesIO(data))
        cp = ["cp", "-a"] if as_user and (uid, gid) != (0, 0) else ["cp"]  # -a keeps the tar's ownership
        _docker(*cp, "-", f"{self.name}:{dest}", input=buf.getvalue(), timeout=60)

    def put_dir(self, local: Path, dest: str) -> None:
        files = {str(Path(dest.lstrip("/")) / p.relative_to(local)): p.read_bytes() for p in local.rglob("*") if p.is_file()}
        if files:
            self.put_files(files, "/")

    def read_file(self, path: str, limit: int = 200_000) -> str | None:
        r = self.exec(f"head -c {limit} {shlex.quote(path)}", timeout=30)
        return r.stdout if r.exit_code == 0 else None

    def get_dir(self, path: str) -> bytes:
        """The tar stream of a directory in the container."""
        return _docker("cp", f"{self.name}:{path}", "-", timeout=120).stdout

    def verify(self, isolated: bool = True, final_message: str | None = None) -> dict[str, float]:
        """Run the task's tests and read the reward. isolated (default): the working directory is copied into a fresh
        container from the pristine image and tested there, so nothing the agent left behind (a background process
        rewriting the reward, a patched package, a pytest.py shadowing pytest) can reach the verifier. That container
        has network only if [verifier] allow_internet (default: the environment's) allows it. final_message: the
        agent's last reply, given to the tests as /logs/agent/final_message.md."""
        tests = self.task.root / "tests"
        if not tests.exists():
            raise FileNotFoundError(f"{self.task.name}: no tests/ directory")
        if isolated:
            archive = self.get_dir(self.workdir)
            clean = DockerRuntime(self.task, self.workdir, self.max_output)
            try:
                clean.start(provision=False, allow_internet=self.task.verifier.get("allow_internet"))
                clean.exec(f"rm -rf {shlex.quote(self.workdir)}", timeout=30, workdir="/")
                _docker("cp", "-", f"{clean.name}:{str(Path(self.workdir).parent)}", input=archive, timeout=120)
                return clean.verify(isolated=False, final_message=final_message)
            finally:
                clean.stop()
        self.exec("rm -rf /tests /logs/verifier /logs/agent && mkdir -p /tests /logs/verifier /logs/agent", timeout=30,
                  workdir="/")
        if final_message is not None:
            self.put_files({"/logs/agent/final_message.md": final_message}, "/")
        self.put_dir(tests, "/tests")
        venv = {k: os.path.expandvars(str(v)) for k, v in (self.task.verifier.get("env") or {}).items()}
        r = self.exec("bash /tests/test.sh", timeout=float(self.task.verifier.get("timeout_sec", 120)), env=venv, workdir="/")
        raw_json = self.read_file("/logs/verifier/reward.json")
        if raw_json:
            try:
                d = json.loads(raw_json)
                return {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            except ValueError:
                pass
        raw = (self.read_file("/logs/verifier/reward.txt") or "").strip().splitlines()
        try:
            if raw:
                return {"reward": float(raw[0])}
        except ValueError:
            pass
        logger.warning("%s: the verifier wrote no reward (exit %s): %s", self.task.name, r.exit_code, r.stderr[-300:])
        return {"reward": 0.0, "metric/verifier_failed": 1.0}

    def stop(self) -> None:
        """Remove the container; one that cannot be removed now (a busy daemon) stays in the process's list and is
        removed at exit, or by the next run's sweep."""
        if self.started:
            self.started = False
            if _remove([self.name]):
                with _live_lock:
                    _live.discard(self.name)
            else:
                logger.warning("could not remove container %s now; it is retried at exit", self.name)


# ------------------------------------------------------------------ environment


def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


TOOL_SCHEMAS = {
    "bash": _fn("bash", "Run a shell command in the task's container (working directory /workdir) and return its output.",
                {"command": {"type": "string"}, "timeout": {"type": "integer", "description": "Seconds (default 60)."}}, ["command"]),
    "read_file": _fn("read_file", "Read a text file.", {"path": {"type": "string"}}, ["path"]),
    "write_file": _fn("write_file", "Create or overwrite a text file.", {"path": {"type": "string"}, "content": {"type": "string"}},
                      ["path", "content"]),
    "str_replace": _fn("str_replace", "Replace an exact piece of text in a file (it must occur exactly once).",
                       {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, ["path", "old", "new"]),
    "submit": _fn("submit", "Finish the task (call it when the work is done).", {}, []),
}


class HarborEnv:
    """palingenesis environment over Harbor task directories: a fresh container per episode, coding-agent tools,
    the task's tests as the reward. Rows carry "task_dir". Synchronous (the pipeline runs sync methods in worker threads),
    so other environments can compose it; the container stops when the episode is released (close)."""

    def __init__(self, tools: list[str] | None = None, reward_key: str = "reward", workdir: str | None = None,
                 max_output: int = 16000):
        self.tool_names = list(tools or TOOL_SCHEMAS)
        self.reward_key = reward_key
        self.workdir = workdir
        self.max_output = max_output
        self.rt: DockerRuntime | None = None
        self.done = False

    def reset(self, task_dir: str, **row) -> None:
        self.close()
        self.task = HarborTask.load(task_dir)
        self.rt = DockerRuntime(self.task, self.workdir, self.max_output)  # workdir None: the task's (or /workdir)
        self.done = False
        self.rt.start()

    def tool_schemas(self) -> list[dict]:
        return [TOOL_SCHEMAS[n] for n in self.tool_names]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        rt = self.rt
        if name == "submit":
            self.done = True
            return "Submitted."
        if name == "bash":
            r = rt.exec(str(arguments.get("command", "")), float(arguments.get("timeout") or 60))
            body = r.stdout + (f"\n[stderr]\n{r.stderr}" if r.stderr.strip() else "")
            return f"[exit code {r.exit_code}]\n{body}".strip()
        if name == "read_file":
            text = rt.read_file(str(arguments.get("path", "")))
            return text if text is not None else f"Error: cannot read {arguments.get('path')!r}"
        if name == "write_file":
            path = str(arguments.get("path", ""))
            p = path if path.startswith("/") else f"{rt.workdir}/{path}"
            rt.exec(f"mkdir -p {shlex.quote(str(Path(p).parent))}", 30)
            rt.put_files({p: str(arguments.get("content", ""))}, "/", as_user=True)
            return f"Wrote {p}."
        if name == "str_replace":
            path = str(arguments.get("path", ""))
            p = path if path.startswith("/") else f"{rt.workdir}/{path}"
            text = rt.read_file(p)
            if text is None:
                return f"Error: cannot read {p}"
            old = str(arguments.get("old", ""))
            n = text.count(old)
            if n != 1:
                return f"Error: the text occurs {n} times in {p}; it must occur exactly once"
            rt.put_files({p: text.replace(old, str(arguments.get("new", "")))}, "/", as_user=True)
            return f"Edited {p}."
        return f"Error: unknown tool {name!r}"

    def get_reward(self, messages: list[dict] | None = None):
        """The verifier's reward; with the conversation (the RL pipeline passes it), the last assistant reply is given
        to the tests as /logs/agent/final_message.md."""
        rt = self.rt
        if rt is None:
            return None
        final = next((m.get("content") or "" for m in reversed(messages or []) if m.get("role") == "assistant"), None)
        try:
            comps = rt.verify(final_message=final)
        except Exception as e:  # noqa: BLE001 — a broken verifier scores 0 and is flagged
            logger.warning("verifier failed for %s: %s", self.task.name, e)
            comps = {"reward": 0.0, "metric/verifier_failed": 1.0}
        main = comps.get(self.reward_key, next(iter(comps.values()), 0.0))
        return {"reward": float(main), **{(k if k.startswith("metric/") else f"metric/{k}"): v for k, v in comps.items()
                                           if k != self.reward_key}}

    def close(self) -> None:
        if self.rt is not None:
            rt, self.rt = self.rt, None
            rt.stop()

    aclose = close


# ------------------------------------------------------------------ validation and external harnesses


def check_task(task_dir: str | Path, workdir: str | None = None) -> dict:
    """Harbor's oracle / nop check: the oracle solution (solution/solve.sh) must pass, doing nothing must not.
    workdir: overrides the task's working directory ([environment] workdir, else /workdir)."""
    task = HarborTask.load(task_dir)
    out = {"task": task.name}
    for mode in ("nop", "oracle"):
        rt = DockerRuntime(task, workdir)
        try:
            rt.start()
            if mode == "oracle":
                sol = task.root / "solution"
                if not (sol / "solve.sh").exists():
                    out["oracle"] = None
                    continue
                rt.put_dir(sol, "/solution")
                r = rt.exec("bash /solution/solve.sh", timeout=task.agent_timeout)
                out["oracle_exit"] = r.exit_code
            out[mode] = rt.verify().get("reward", 0.0)
        except Exception as e:  # noqa: BLE001
            out[mode] = None
            out[f"{mode}_error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            rt.stop()
    out["valid"] = out.get("oracle") == 1.0 and out.get("nop") is not None and out.get("nop", 1.0) < 1.0
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    from concurrent.futures import ThreadPoolExecutor

    ap = argparse.ArgumentParser(prog="pgs harbor", description="Harbor task directories for palingenesis RL")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="oracle / nop check of every task directory under a folder (or one task)")
    c.add_argument("tasks_dir")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--json", action="store_true")
    c.add_argument("--workdir", default=None, help="working directory when the tasks do not set [environment] workdir")
    r = sub.add_parser("rows", help="write training rows (task_dir + prompt) for HarborEnv")
    r.add_argument("tasks_dir")
    r.add_argument("out")
    r.add_argument("--no-system", action="store_true", help="leave the default system prompt out")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    root = Path(a.tasks_dir)
    if a.cmd == "rows":
        rows = harbor_rows(root, None if a.no_system else SYSTEM)
        with open(a.out, "w") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        print(f"{len(rows)} rows -> {a.out}")
        return 0
    dirs = [root] if (root / "task.toml").exists() else sorted(d for d in root.iterdir() if (d / "task.toml").exists())
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda d: check_task(d, a.workdir), dirs))
    if a.json:
        print(json.dumps(results, indent=1))
    else:
        for res in results:
            print(f"{'ok ' if res['valid'] else 'BAD'}  oracle={res.get('oracle')} nop={res.get('nop')}  {res['task']}"
                  + (f"  {res.get('oracle_error') or res.get('nop_error')}" if not res["valid"] else ""))
        print(f"{sum(x['valid'] for x in results)}/{len(results)} valid")
    return 0 if all(x["valid"] for x in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
