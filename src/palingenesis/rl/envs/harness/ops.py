"""The closed op set every harness tool reduces to, implemented on a Harbor episode's container (DockerRuntime).

A harness profile's tool is a schema plus a binding that maps its arguments onto these ops and words the result (and
the errors) the way that harness does. The ops return plain data and raise OpError(kind, **data) on the policy's
mistakes; the wording is the profile's, so one op serves opencode's edit(filePath, oldString, newString), pi's
edit(path, edits[]) and OpenHands' file_editor(str_replace) with each harness's own messages.

    exec_          a shell command, stateless (a fresh shell per call) or persistent (cwd and exported variables
                   carry over between calls, as in OpenHands' terminal); stdout and stderr merged as a terminal shows
    read           a whole text file (the profile numbers / slices / truncates it)
    list_dir       a directory listing to a depth (View on a directory)
    write          create or overwrite a file (parent directories created; an existing file keeps its mode and owner)
    replace        exact text replacement, once (must be unique) or every occurrence
    multi_replace  several replacements matched against the original text (unique, non-overlapping)
    insert         text after a line (0 = at the top)
    undo_edit      restore a file's content before its last edit by these ops (single level)
    glob / grep    file search by pattern / content search by regular expression
    end            the final message (stop rules decide what ending means)
    Tmux / keys    a persistent PTY (tmux) driven by keystrokes, with deterministic waits (terminus-2)
    apply_patch    not implemented (the codex / opencode gpt-* apply_patch grammar): NotImplementedError

Runtime contract (DockerRuntime, or a test double): .workdir, .exec(command, timeout, env=None, workdir=None) ->
ExecResult(exit_code, stdout, stderr, timed_out), .put_files({path: data}, "/", as_user=True), optionally .scratch
(the temporary directory, default /tmp: the persistent shell's state and staged writes live there). Files are read
through base64 in the shell, so the runtime's max_output must exceed the largest file read (the episode's runtime
for a profile is created with HarnessProfile.runtime_max_output).
"""

from __future__ import annotations

import base64
import binascii
import posixpath
import re
import shlex
import uuid
from dataclasses import dataclass

CUT_MARK = "characters cut] ..."  # DockerRuntime's marker for an output longer than max_output


class OpError(Exception):
    """The policy's mistake, as data: kind (not_found, is_dir, not_absolute, exists, no_match, multiple, identical,
    empty_old, overlap, bad_line, bad_range, offset, no_history, not_dir, write_failed, too_large) and its fields."""

    def __init__(self, kind: str, **data):
        self.kind = kind
        self.data = data
        super().__init__(f"{kind}: {data}")


@dataclass
class ExecOut:
    output: str
    exit_code: int
    timed_out: bool = False
    cwd: str | None = None  # persistent shells: the working directory after the command


@dataclass
class Edit:
    path: str
    before: str | None  # None: the file did not exist
    after: str
    line: int = 1  # first changed line (1-based), for snippets
    count: int = 1  # replacements made


class Workspace:
    """One episode's container as the ops see it: absolute paths, whole-file reads and writes, the undo history and
    the persistent shell's state."""

    def __init__(self, rt, home: str | None = None):
        self.rt = rt
        self.workdir = rt.workdir
        self.home = home or rt.workdir
        self.history: dict[str, str | None] = {}  # path -> content before its last edit (single level)
        self.scratch = getattr(rt, "scratch", None) or "/tmp"  # the container's temporary directory
        self.state = f"{self.scratch}/.pgs_shell_{uuid.uuid4().hex[:8]}"  # persistent shell: cwd, exported variables

    def abspath(self, path: str, base: str | None = None) -> str:
        p = str(path or "")
        if not p.startswith("/"):
            p = posixpath.join(base or self.workdir, p)
        return posixpath.normpath(p)

    def sh(self, command: str, timeout: float = 30, workdir: str = "/"):
        return self.rt.exec(command, timeout, workdir=workdir)

    def kind(self, path: str) -> str | None:
        """"dir", "file" or None (missing)."""
        q = shlex.quote(path)
        r = self.sh(f"if [ -d {q} ]; then echo dir; elif [ -e {q} ]; then echo file; else echo none; fi")
        k = r.stdout.strip()
        return k if k in ("dir", "file") else None

    def fetch(self, path: str) -> str:
        """The whole text of a file (raises OpError not_found / is_dir / too_large)."""
        q = shlex.quote(path)
        r = self.sh(f"if [ -d {q} ]; then echo __PGS_DIR__; exit 3; fi; base64 < {q}")
        if r.exit_code == 3 and "__PGS_DIR__" in r.stdout:
            raise OpError("is_dir", path=path)
        if r.exit_code != 0:
            raise OpError("not_found", path=path)
        if CUT_MARK in r.stdout:
            raise OpError("too_large", path=path)
        try:
            data = base64.b64decode(r.stdout)
        except (binascii.Error, ValueError):
            raise OpError("too_large", path=path) from None
        return data.decode("utf-8", errors="replace")

    def store(self, path: str, text: str) -> None:
        """Write a file as the image's default user: staged in the scratch directory, then `cat` into place (an existing file keeps its
        inode, mode and owner; a new one gets the user's umask), parent directories created."""
        tmp = f"{self.scratch}/.pgs_put_{uuid.uuid4().hex[:12]}"
        self.rt.put_files({tmp: text}, "/", as_user=True)
        q, t = shlex.quote(path), shlex.quote(tmp)
        r = self.sh(f"mkdir -p {shlex.quote(posixpath.dirname(path) or '/')} && cat {t} > {q}; rc=$?; rm -f {t}; "
                    f"exit $rc")
        if r.exit_code != 0:
            raise OpError("write_failed", path=path, detail=(r.stderr or r.stdout).strip()[-300:])


# ------------------------------------------------------------------ shell


def exec_(ws: Workspace, command: str, timeout: float = 60, workdir: str | None = None,
          persistent: bool = False, shell: str = "bash", env: dict | None = None) -> ExecOut:
    """Run a command; stdout and stderr merged. persistent: the shell's working directory and exported variables
    carry over from the previous persistent call (a background process does not: each call is a new process).
    shell "sh": the command runs under /bin/sh -c (subprocess shell=True: dash on Debian images) instead of bash."""
    cmd = shlex.quote(str(command))
    if persistent:
        st = shlex.quote(ws.state)
        script = (f"exec 2>&1\n[ -f {st}.env ] && . {st}.env >/dev/null 2>&1\n"
                  f"cd \"$(cat {st}.cwd 2>/dev/null || echo {shlex.quote(ws.workdir)})\" 2>/dev/null\n"
                  f"__pgs_cmd={cmd}\neval \"$__pgs_cmd\"\n__pgs_rc=$?\n"
                  f"pwd > {st}.cwd; export -p > {st}.env 2>/dev/null\n"
                  f"printf '\\n__PGS_CWD__%s' \"$(pwd)\"\nexit $__pgs_rc")
    else:
        cd = f"cd {shlex.quote(workdir)} || exit 1\n" if workdir else ""
        run = f"exec sh -c {cmd}" if shell == "sh" else f"__pgs_cmd={cmd}\neval \"$__pgs_cmd\""
        script = f"exec 2>&1\n{cd}{run}"
    r = ws.rt.exec(script, timeout, env=env, workdir=ws.workdir)
    out = r.stdout + (r.stderr if r.stderr and not r.timed_out else "")
    cwd = None
    if persistent:
        head, sep, tail = out.rpartition("\n__PGS_CWD__")
        if sep:
            out, cwd = head, tail.strip()
    return ExecOut(out, r.exit_code, bool(getattr(r, "timed_out", False)), cwd)


def reset_shell(ws: Workspace) -> None:
    st = shlex.quote(ws.state)
    ws.sh(f"rm -f {st}.env {st}.cwd")


class Tmux:
    """A persistent PTY in the container: a tmux session running `bash --login` (terminus-2's TmuxSession), driven by
    keystrokes. Waits are deterministic: after the keys, poll until the pane's shell has no child process and the
    screen is unchanged between two polls, at most `duration` seconds (a real harness sleeps the whole duration;
    a command that finishes early shows the same screen either way, one still running is cut at the duration)."""

    PASTE_ABOVE = 8000  # bytes: longer literal keys go through a tmux paste buffer (send-keys has a size limit)

    def __init__(self, ws: Workspace, name: str = "pgs", width: int = 160, height: int = 40, poll: float = 0.15):
        self.ws, self.name, self.width, self.height, self.poll = ws, name, width, height, poll
        self.previous: str | None = None
        self.started = False

    def _t(self) -> str:
        return shlex.quote(self.name)

    def start(self) -> None:
        r = self.ws.sh("command -v tmux >/dev/null 2>&1 || exit 127; export TERM=xterm-256color SHELL=/bin/bash; "
                       f"tmux new-session -x {self.width} -y {self.height} -d -s {self._t()} 'bash --login' && "
                       "tmux set-option -g history-limit 10000000 >/dev/null", timeout=30, workdir=self.ws.workdir)
        if r.exit_code == 127:
            raise OpError("no_tmux", detail="tmux is not installed in the task image")
        if r.exit_code != 0:
            raise OpError("no_tmux", detail=(r.stderr or r.stdout).strip()[-300:])
        self.started = True

    def send(self, keys: str) -> None:
        if not keys:
            return
        if len(keys.encode()) > self.PASTE_ABOVE:
            path = f"{self.ws.scratch}/.pgs_paste_{uuid.uuid4().hex[:8]}"
            self.ws.rt.put_files({path: keys}, "/", as_user=True)
            q = shlex.quote(path)
            self.ws.sh(f"tmux load-buffer -b pgs {q} && tmux paste-buffer -d -b pgs -t {self._t()}; rm -f {q}")
            return
        self.ws.sh(f"tmux send-keys -t {self._t()} -- {shlex.quote(keys)}")

    def _probe(self) -> tuple[bool, str]:
        r = self.ws.sh(f"p=$(tmux display -p -t {self._t()} '#{{pane_pid}}'); "
                       f"if pgrep -P \"$p\" >/dev/null 2>&1; then echo BUSY; else echo IDLE; fi; "
                       f"tmux capture-pane -p -t {self._t()} | cksum")
        lines = r.stdout.splitlines()
        return (bool(lines) and lines[0] == "IDLE"), (lines[1] if len(lines) > 1 else "")

    def wait(self, duration: float) -> None:
        import time

        limit = max(0.0, min(float(duration), 60.0))
        start = time.monotonic()
        time.sleep(min(self.poll, limit))
        last = None
        while True:
            idle, screen = self._probe()
            if idle and screen == last:
                return
            last = screen
            if time.monotonic() - start >= limit:
                return
            time.sleep(self.poll)

    def capture(self, entire: bool = False) -> str:
        return self.ws.sh(f"tmux capture-pane -p {'-S - ' if entire else ''}-t {self._t()}").stdout

    def incremental(self) -> str:
        """New output since the last call ("New Terminal Output:"), else the visible screen ("Current Terminal
        Screen:"), as terminus-2 reports it."""
        current = self.capture(entire=True)
        if self.previous is None:
            self.previous = current
            return f"Current Terminal Screen:\n{self.capture()}"
        pb = self.previous.strip()
        self.previous = current
        if pb in current:
            idx = current.index(pb)
            if "\n" in pb:
                idx = pb.rfind("\n")
            new = current[idx:]
            if new.strip():
                return f"New Terminal Output:\n{new}"
        return f"Current Terminal Screen:\n{self.capture()}"


def keys(ws: Workspace, tmux: Tmux, keystrokes: str, duration: float = 1.0) -> None:
    """Send keystrokes to the persistent PTY and wait (deterministically) up to `duration` seconds."""
    if not tmux.started:
        tmux.start()
    tmux.send(keystrokes)
    tmux.wait(duration)


def apply_patch(ws: Workspace, patch_text: str) -> str:
    raise NotImplementedError("ApplyPatch (the codex / opencode gpt-* apply_patch grammar) is not implemented: no "
                              "built-in profile uses it yet")


# ------------------------------------------------------------------ files


def read(ws: Workspace, path: str) -> str:
    return ws.fetch(path)


def lines_of(text: str) -> list[str]:
    """Lines of a file: a terminating newline ends the last line rather than adding an empty one."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def list_dir(ws: Workspace, path: str, max_depth: int = 1, hidden: bool = False) -> tuple[list[str], int]:
    """(entries, hidden_count): paths under `path` to max_depth (directories end with "/"), sorted; hidden entries
    (a name starting with ".") left out unless hidden, and counted at the first level."""
    q = shlex.quote(path)
    r = ws.sh(f"[ -d {q} ] || exit 3; cd {q} && find . -mindepth 1 -maxdepth {int(max_depth)} "
              f"\\( -type d -exec printf '%s/\\n' {{}} \\; \\) -o -print")
    if r.exit_code == 3:
        raise OpError("not_dir", path=path)
    entries, n_hidden = [], 0
    for line in r.stdout.splitlines():
        rel = line[2:] if line.startswith("./") else line
        if not rel:
            continue
        parts = rel.rstrip("/").split("/")
        if any(p.startswith(".") for p in parts):
            n_hidden += len(parts) == 1
            if not hidden:
                continue
        entries.append(rel)
    return sorted(entries), n_hidden


def write(ws: Workspace, path: str, content: str) -> bool:
    """Create or overwrite; True if the file was created."""
    k = ws.kind(path)
    if k == "dir":
        raise OpError("is_dir", path=path)
    before = ws.fetch(path) if k == "file" else None
    ws.store(path, content)
    ws.history[path] = before
    return before is None


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _occurrences(text: str, old: str) -> list[int]:
    out, i = [], text.find(old)
    while i != -1:
        out.append(i)
        i = text.find(old, i + 1)
    return out


def replace(ws: Workspace, path: str, old: str, new: str, replace_all: bool = False) -> Edit:
    """Exact replacement: `old` must occur once (or at least once with replace_all) and differ from `new`."""
    if ws.kind(path) == "dir":
        raise OpError("is_dir", path=path)
    if old == new:
        raise OpError("identical", path=path, old=old)
    if old == "":
        raise OpError("empty_old", path=path)
    text = ws.fetch(path)
    hits = _occurrences(text, old)
    if not hits:
        raise OpError("no_match", path=path, old=old)
    if len(hits) > 1 and not replace_all:
        raise OpError("multiple", path=path, old=old, count=len(hits), lines=[_line_of(text, i) for i in hits])
    after = text.replace(old, new) if replace_all else text[:hits[0]] + new + text[hits[0] + len(old):]
    ws.store(path, after)
    ws.history[path] = text
    return Edit(path, text, after, _line_of(text, hits[0]), len(hits) if replace_all else 1)


def multi_replace(ws: Workspace, path: str, edits: list[tuple[str, str]]) -> Edit:
    """Several replacements, each matched against the original text (not after the earlier ones): each `old` unique,
    none empty, no two overlapping; errors carry the edit's index."""
    if ws.kind(path) == "dir":
        raise OpError("is_dir", path=path)
    text = ws.fetch(path)
    spans = []
    for i, (old, new) in enumerate(edits):
        if old == "":
            raise OpError("empty_old", path=path, index=i, n=len(edits))
        hits = _occurrences(text, old)
        if not hits:
            raise OpError("no_match", path=path, old=old, index=i, n=len(edits))
        if len(hits) > 1:
            raise OpError("multiple", path=path, old=old, count=len(hits), index=i, n=len(edits),
                          lines=[_line_of(text, h) for h in hits])
        spans.append((hits[0], hits[0] + len(old), new, i))
    spans.sort(key=lambda x: (x[0], x[3]))
    for a, b in zip(spans, spans[1:]):
        if b[0] < a[1]:
            raise OpError("overlap", path=path, first=a[3], second=b[3])
    out, pos = [], 0
    for start, end, new, _ in spans:
        out += [text[pos:start], new]
        pos = end
    after = "".join(out) + text[pos:]
    if after == text:
        raise OpError("identical", path=path, n=len(edits))
    ws.store(path, after)
    ws.history[path] = text
    return Edit(path, text, after, _line_of(text, spans[0][0]), len(spans))


def insert(ws: Workspace, path: str, line: int, text: str) -> Edit:
    """Insert `text` after line `line` (0: before the first line)."""
    if ws.kind(path) == "dir":
        raise OpError("is_dir", path=path)
    before = ws.fetch(path)
    lines = lines_of(before)
    if not isinstance(line, int) or line < 0 or line > len(lines):
        raise OpError("bad_line", path=path, line=line, n_lines=len(lines))
    new_lines = lines_of(text) if text else [""]
    merged = lines[:line] + new_lines + lines[line:]
    after = "\n".join(merged) + ("\n" if before.endswith("\n") or not before else "")
    ws.store(path, after)
    ws.history[path] = before
    return Edit(path, before, after, line + 1, 1)


def undo_edit(ws: Workspace, path: str) -> Edit:
    """Restore the content before the last edit of `path` by these ops (single level: a second undo has no history)."""
    if path not in ws.history:
        raise OpError("no_history", path=path)
    before = ws.history.pop(path)
    current = ws.fetch(path) if ws.kind(path) == "file" else ""
    if before is None:
        ws.sh(f"rm -f {shlex.quote(path)}")
        return Edit(path, current, "", 1, 1)
    ws.store(path, before)
    return Edit(path, current, before, 1, 1)


# ------------------------------------------------------------------ search


def _expand_braces(pattern: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    return [x for alt in m.group(1).split(",") for x in _expand_braces(pattern[:m.start()] + alt + pattern[m.end():])]


def _glob_regex(pattern: str) -> re.Pattern:
    """A glob with ** (any number of directories), * and ? (not crossing "/"), [...] classes."""
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
                i += 1
            else:
                body = pattern[i + 1:j]
                out.append("[" + ("^" + body[1:] if body.startswith("!") else body) + "]")
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z")


def glob_match(pattern: str, rel: str) -> bool:
    """ripgrep / gitignore-style: a pattern without "/" matches the file name at any depth."""
    for p in _expand_braces(pattern[2:] if pattern.startswith("./") else pattern):
        target = rel if "/" in p else rel.rsplit("/", 1)[-1]
        if _glob_regex(p).match(target):
            return True
    return False


def glob(ws: Workspace, pattern: str, path: str | None = None, limit: int = 100) -> list[str]:
    """Absolute paths of the files under `path` (default: the working directory) matching `pattern`, newest first
    (then by name), .git left out, at most `limit`."""
    root = ws.abspath(path or ws.workdir)
    k = ws.kind(root)
    if k == "file":
        raise OpError("not_dir", path=root)
    if k is None:
        raise OpError("not_found", path=root)
    q = shlex.quote(root)
    r = ws.sh(f"cd {q} && (find . -name .git -prune -o -type f -printf '%T@ %p\\n' 2>/dev/null "
              f"|| find . -name .git -prune -o -type f -print | sed 's/^/0 /')", timeout=60)
    rows = []
    for line in r.stdout.splitlines():
        stamp, _, p = line.partition(" ")
        rel = p[2:] if p.startswith("./") else p
        if rel and glob_match(pattern, rel):
            try:
                t = float(stamp)
            except ValueError:
                t = 0.0
            rows.append((-t, rel))
    rows.sort()
    return [posixpath.join(root, rel) for _, rel in rows[:limit]]


def grep(ws: Workspace, pattern: str, path: str | None = None, include: str | None = None,
         limit: int = 100) -> list[tuple[str, int, str]]:
    """(file, line number, line) for each line under `path` matching the extended regular expression `pattern`
    (binary files and .git left out), at most `limit`."""
    root = ws.abspath(path or ws.workdir)
    if ws.kind(root) is None:
        raise OpError("not_found", path=root)
    inc = " ".join(f"--include={shlex.quote(p)}" for p in _expand_braces(include)) if include else ""
    r = ws.sh(f"grep -rnHI -E --exclude-dir=.git {inc} -e {shlex.quote(pattern)} -- {shlex.quote(root)} "
              f"2>/dev/null | head -n {int(limit)}", timeout=60)
    out = []
    for line in r.stdout.splitlines():
        m = re.match(r"^(.*?):(\d+):(.*)$", line)
        if m:
            out.append((m.group(1), int(m.group(2)), m.group(3)))
    return out


def end(message: str = "") -> str:
    return str(message or "")
