"""pgs learn command line: parsing, the init / add / inbox / approve / promote / forget flow, exit codes."""

import json
import sys
import warnings

import pytest

sys.path.insert(0, "src")
warnings.simplefilter("ignore")

from palingenesis.learn.cli import build_parser, main  # noqa: E402

USER = "user:acme/support/ana"


def test_parser():
    p = build_parser()
    a = p.parse_args(["add", "s", "--scope", "org:acme", "--documents", "docs/**/*.md", "--approve", "--locked"])
    assert (a.cmd, a.documents, a.approve, a.locked) == ("add", "docs/**/*.md", True, True)
    a = p.parse_args(["approve", "s", "--scope", USER, "i_1", "i_2", "--as", "supersede"])
    assert a.ids == ["i_1", "i_2"] and a.resolution == "supersede"
    a, extra = p.parse_known_args(["run", "s", "--scope", "org:acme", "--live", "--recipe.passes", "2"])
    assert a.live and extra == ["--recipe.passes", "2"]
    with pytest.raises(SystemExit):
        p.parse_args(["add", "s", "--scope", "org:acme"])  # nothing to add
    with pytest.raises(SystemExit):
        p.parse_args(["add", "s", "--scope", "org:acme", "--note", "x", "--facts", "f.jsonl"])
    with pytest.raises(SystemExit):
        p.parse_args(["approve", "s", "--scope", USER, "i_1", "--reject", "--as", "both"])


def run(capsys, *args):
    code = main(list(args))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_flow(tmp_path, capsys):
    store = str(tmp_path / "store")
    assert run(capsys, "init", store, "--base", "tiny", "--org", "acme", "--generator", "stub")[0] == 0
    code, _, err = run(capsys, "init", store, "--base", "tiny", "--org", "acme")
    assert code == 1 and "already" in err
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# Pro\nThe price of Orion Pro is 25 euros.\n")
    code, out, _ = run(
        capsys,
        "add",
        store,
        "--scope",
        "org:acme",
        "--documents",
        "*.md",
        "--root",
        str(tmp_path / "docs"),
        "--approve",
    )
    assert code == 0 and "1 new" in out
    facts = tmp_path / "facts.jsonl"
    facts.write_text(json.dumps({"text": "my editor is vim"}) + "\n")
    code, out, _ = run(capsys, "add", store, "--scope", USER, "--facts", str(facts))
    assert code == 0 and out.startswith("held")
    code, out, _ = run(capsys, "inbox", store, "--scope", USER, "--json")
    item = json.loads(out)["inbox"][0]
    assert item["payload"]["note"]["text"] == "my editor is vim"
    code, out, _ = run(capsys, "approve", store, "--scope", USER, item["id"])
    assert code == 0 and "stored" in out
    note_id = out.split()[2]
    code, out, _ = run(capsys, "promote", store, "--scope", USER, note_id, "--to", "team:acme/support")
    queue_id = out.strip()
    code, out, _ = run(capsys, "inbox", store, "--scope", "team:acme/support")
    assert queue_id in out and "promotion" in out
    code, out, _ = run(capsys, "approve", store, "--scope", "team:acme/support", queue_id, "--reject")
    assert "rejected" in out
    code, out, _ = run(capsys, "add", store, "--scope", USER, "--instructions", str(facts))
    assert code == 1
    code, out, _ = run(capsys, "forget", store, "--scope", USER, note_id)
    assert code == 0 and "verified" in out
    assert all(b"my editor is vim" not in p.read_bytes() for p in (tmp_path / "store").rglob("*") if p.is_file())
    code, _, err = run(capsys, "forget", store, "--scope", USER, "n_missing")
    assert code == 1 and "n_missing" in err


def test_bad_inputs_exit_1(tmp_path, capsys):
    store = str(tmp_path / "store")
    run(capsys, "init", store, "--base", "tiny", "--org", "acme", "--generator", "stub")
    assert run(capsys, "add", store, "--scope", "team:other/x", "--note", "x")[0] == 1
    code, _, err = run(capsys, "run", store, "--scope", "org:acme", "--recipe.pases", "2")
    assert code == 1 and "did you mean recipe.passes" in err
    assert run(capsys, "inbox", store, "--scope", "org:acme", "--extra", "1")[0] == 1
    assert run(capsys, "approve", str(tmp_path / "missing"), "--scope", "org:acme", "i_1")[0] == 1


def test_top_level_dispatch(tmp_path, monkeypatch, capsys):
    from palingenesis import cli

    monkeypatch.setattr(sys, "argv", ["pgs", "learn", "init", str(tmp_path / "s"), "--base", "tiny", "--org", "acme"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0 and (tmp_path / "s" / "learn_state.yaml").exists()
    assert "experimental" in capsys.readouterr().err
