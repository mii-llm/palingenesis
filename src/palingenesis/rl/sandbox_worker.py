"""Program runner for the code sandbox. Standard library only.

Used three ways:
  - in-process by the subprocess backend: `run_job(job)`;
  - inside a locked-down container (docker backend): `python -u worker.py --slots N`
    reads length-prefixed JSON jobs on stdin and writes results on stdout, running up
    to N programs at a time;
  - in a Kubernetes Agent Sandbox pod (agent_sandbox backend), one job per command:
    `python3 worker.py --job-file job.json` prints the result.

Each program runs in a fresh temporary directory, in its own session (so the whole
process group is killed afterwards), under rlimits: address space, CPU time, file size
(which also caps stdout/stderr, written to files), open files and core dumps. Wall-clock
timeouts are enforced by the parent. Expected outputs never reach this process: the
caller compares results on its side.

Run as root (the docker backend: a root worker holding only the capabilities to switch
users, signal and manage files), each slot runs its programs as its own unprivileged user:
concurrent programs cannot read or change each other's files (0700 directories), signal each
other or the worker, and after every program all processes of its user are killed, including
any that escaped the process group (setsid, daemonized), and its /tmp leftovers removed. The
per-user process limit then caps each slot. Without root (the subprocess backend), programs
run as the caller's user.

Job:    {"id", "files": {name: text}, "argv": [...], "stdin": str, "timeout": s,
         "memory_mb": int, "max_output": bytes, "nproc": int (0 = no limit)}
Result: {"id", "status": ok|runtime_error|timeout|memory|output_limit|sandbox_error,
         "exit_code", "signal", "stdout", "stderr", "result", "wall"}
         `result` is what the program wrote to the file named by $PGS_RESULT (harnesses
         report there, and remove the variable before running untrusted code).
"""

import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time

_STDERR_KEEP = 4096
_SLOT_UID = 20000  # slot i runs as uid/gid 20000 + i (under a root worker)


def _processes_of(uid: int) -> list[int]:
    pids = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                if os.stat(f"/proc/{entry}").st_uid == uid:
                    pids.append(int(entry))
            except OSError:
                pass
    return pids


def _sweep(uid: int) -> None:
    """Kill every process of `uid` (escaped sessions, daemons) and remove its /tmp entries."""
    for _ in range(50):
        pids = _processes_of(uid)
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.01)
    tmp = tempfile.gettempdir()
    try:
        for entry in os.scandir(tmp):
            try:
                if entry.stat(follow_symlinks=False).st_uid == uid:
                    if entry.is_dir(follow_symlinks=False):
                        shutil.rmtree(entry.path, ignore_errors=True)
                    else:
                        os.unlink(entry.path)
            except OSError:
                pass
    except OSError:
        pass


def run_job(job: dict, python: str = sys.executable, root: str | None = None, uid: int | None = None) -> dict:
    """Run one program. `uid`: the unprivileged user it runs as (needs a root caller)."""
    result = {
        "id": job.get("id"),
        "status": "sandbox_error",
        "exit_code": None,
        "signal": None,
        "stdout": "",
        "stderr": "",
        "result": "",
        "wall": 0.0,
    }
    workdir = tempfile.mkdtemp(prefix="pgs_", dir=root)
    try:
        for name, text in (job.get("files") or {}).items():
            with open(os.path.join(workdir, os.path.basename(name)), "w") as f:
                f.write(text)
        if uid is not None:  # the program's user owns its directory; nobody else can enter it
            for name in os.listdir(workdir):
                os.chown(os.path.join(workdir, name), uid, uid)
            os.chown(workdir, uid, uid)
            os.chmod(workdir, 0o700)
        timeout = float(job.get("timeout", 6.0))
        memory = int(job.get("memory_mb", 1024)) * 2**20
        max_output = int(job.get("max_output", 1 << 16))
        nproc = int(job.get("nproc", 0))
        paths = {k: os.path.join(workdir, f".{k}") for k in ("stdin", "stdout", "stderr", "result")}
        with open(paths["stdin"], "w") as f:
            f.write(job.get("stdin") or "")
        argv = [python if a == "{python}" else a for a in (job.get("argv") or ["{python}", "-I", "-B", "main.py"])]

        def limits():
            import resource

            os.setsid()
            for kind, value in (
                (resource.RLIMIT_AS, memory),
                (resource.RLIMIT_CPU, int(timeout) + 1),
                (resource.RLIMIT_FSIZE, max(max_output, 1 << 20) * 4),
                (resource.RLIMIT_NOFILE, 64),
                (resource.RLIMIT_CORE, 0),
            ):
                try:
                    resource.setrlimit(kind, (value, value))
                except (ValueError, OSError):  # e.g. RLIMIT_AS on macOS
                    pass
            if nproc:
                resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
            if uid is not None:  # last: from here on no privilege remains
                os.setgroups([])
                os.setgid(uid)
                os.setuid(uid)

        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": workdir,
            "LANG": "C.UTF-8",
            "PGS_RESULT": paths["result"],
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            **(job.get("env") or {}),
        }
        open(paths["result"], "w").close()
        if uid is not None:
            for key in ("stdin", "result"):
                os.chown(paths[key], uid, uid)
        # stdout/stderr are read back through the worker's own handles: a program that deletes
        # or replaces the files cannot turn its failure into an infrastructure error (skipped)
        with (
            open(paths["stdin"]) as fin,
            open(paths["stdout"], "w+b") as fout,
            open(paths["stderr"], "w+b") as ferr,
        ):
            start = time.monotonic()
            proc = subprocess.Popen(
                argv,
                cwd=workdir,
                stdin=fin,
                stdout=fout,
                stderr=ferr,
                env=env,
                preexec_fn=limits,
                close_fds=True,
            )
            timed_out = False
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)  # the program and anything it started
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait()
            result["wall"] = time.monotonic() - start
            fout.seek(0)
            stdout = fout.read(max_output + 1)
            ferr.seek(0, 2)
            ferr.seek(max(0, ferr.tell() - _STDERR_KEEP))
            stderr = ferr.read()
        code = proc.returncode
        try:
            with open(paths["result"], "rb") as f:
                result["result"] = f.read(max_output).decode(errors="replace")
        except OSError:  # removed by the program: no result
            result["result"] = ""
        result["stdout"] = stdout[:max_output].decode(errors="replace")
        result["stderr"] = stderr.decode(errors="replace")
        result["exit_code"] = code
        result["signal"] = -code if code is not None and code < 0 else None
        if timed_out or result["signal"] == signal.SIGXCPU:
            result["status"] = "timeout"
        elif len(stdout) > max_output or result["signal"] == signal.SIGXFSZ:
            result["status"] = "output_limit"
        elif "MemoryError" in result["stderr"] or (result["signal"] == signal.SIGKILL and not timed_out):
            result["status"] = "memory"
        else:
            result["status"] = "ok" if code == 0 else "runtime_error"
    except Exception as e:  # noqa: BLE001 — reported as an infrastructure failure, never as a wrong answer
        result["status"] = "sandbox_error"
        result["stderr"] = f"{type(e).__name__}: {e}"
    finally:
        if uid is not None:
            _sweep(uid)
        shutil.rmtree(workdir, ignore_errors=True)
    return result


# ------------------------------------------------------------ worker protocol


def read_frame(stream) -> dict | None:
    header = stream.read(4)
    if len(header) < 4:
        return None
    (size,) = struct.unpack(">I", header)
    return json.loads(stream.read(size))


def write_frame(stream, obj: dict, lock: threading.Lock) -> None:
    data = json.dumps(obj).encode()
    with lock:
        stream.write(struct.pack(">I", len(data)) + data)
        stream.flush()


def serve(slots: int) -> None:
    """Run jobs from stdin until it closes, `slots` at a time; results in completion order."""
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    root = os.environ.get("PGS_SANDBOX_ROOT") or None
    # as root: one unprivileged user per slot (thread), for the life of the worker
    local, next_uid = threading.local(), iter(range(_SLOT_UID, _SLOT_UID + slots))

    def slot_uid() -> int | None:
        if os.geteuid() != 0:
            return None
        if not hasattr(local, "uid"):
            with lock:
                local.uid = next(next_uid)
        return local.uid

    def run(job: dict) -> None:
        write_frame(stdout, run_job(job, root=root, uid=slot_uid()), lock)

    with ThreadPoolExecutor(slots) as pool:
        while (job := read_frame(stdin)) is not None:
            if job.get("ping"):
                write_frame(stdout, {"id": job.get("id"), "status": "pong"}, lock)
                continue
            pool.submit(run, job)


if __name__ == "__main__":
    if "--job-file" in sys.argv:  # one job, for sandboxes reached by command (agent_sandbox)
        path = sys.argv[sys.argv.index("--job-file") + 1]
        with open(path) as f:
            job = json.load(f)
        os.remove(path)
        uid = _SLOT_UID if os.geteuid() == 0 else None
        sys.stdout.write(json.dumps(run_job(job, root=os.environ.get("PGS_SANDBOX_ROOT") or None, uid=uid)))
    else:
        serve(int(sys.argv[sys.argv.index("--slots") + 1]) if "--slots" in sys.argv else 4)
