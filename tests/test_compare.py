"""pgs compare: result files, per-file intervals, paired tests and the --require exit code."""

import json

import pytest

from palingenesis.agent_tooling.compare import load_results, main


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_load_results_formats_and_errors(tmp_path):
    path = _write(
        tmp_path / "a.jsonl",
        [
            {"id": 1, "scores": [1, 0, True]},
            {"id": "x", "correct": False, "cluster": "c"},
            {"id": "y", "reward": 0.25},
        ],
    )
    items = load_results(path)
    assert items["1"]["scores"] == [1.0, 0.0, 1.0] and items["x"] == {"scores": [0.0], "cluster": "c"}
    assert items["y"]["scores"] == [0.25]
    for bad, message in [
        ({"scores": [1]}, "no 'id'"),
        ({"id": 1}, "no score"),
        ({"id": 1, "scores": []}, "empty"),
        ({"id": 1, "score": "yes"}, "finite number"),
        ({"id": 1, "score": float("nan")}, "finite number"),
    ]:
        with pytest.raises(ValueError, match=message):
            load_results(_write(tmp_path / "bad.jsonl", [bad]))
    with pytest.raises(ValueError, match="twice"):
        load_results(_write(tmp_path / "dup.jsonl", [{"id": 1, "score": 1}, {"id": 1, "score": 0}]))


def test_single_file_pass_at_k(tmp_path, capsys):
    rows = [{"id": i, "scores": [1, 0, 0, 0]} for i in range(10)] + [
        {"id": 10 + i, "scores": [0] * 4} for i in range(10)
    ]
    assert main([_write(tmp_path / "a.jsonl", rows), "--json", "--resamples", "500"]) == 0
    report = json.loads(capsys.readouterr().out)["a"]
    assert report["items"] == 20 and report["samples_min"] == 4
    assert report["pass@1"]["estimate"] == pytest.approx(0.125)
    assert report["pass@2"]["estimate"] == pytest.approx(0.25)  # half the items at 1 - C(3,2)/C(4,2) = 1/2
    assert report["pass@4"]["estimate"] == pytest.approx(0.5)
    assert "pass@8" not in report  # beyond the samples
    assert report["mean_score"]["lo"] < 0.125 < report["mean_score"]["hi"]


def test_paired_comparison_and_require(tmp_path, capsys):
    a = [{"id": i, "correct": i < 50} for i in range(100)]
    b = [{"id": i, "correct": i < 70} for i in range(100)] + [{"id": "extra", "correct": True}]
    pa, pb = _write(tmp_path / "a.jsonl", a), _write(tmp_path / "b.jsonl", b)
    assert main([pa, pb, "--json", "--resamples", "1000", "--require", "better"]) == 0
    report = json.loads(capsys.readouterr().out)
    c = report["comparison"]
    assert c["paired_items"] == 100 and c["only_b"] == 1 and c["only_a"] == 0
    assert c["mean_score"]["diff"] == pytest.approx(0.2) and c["mean_score"]["lo"] > 0
    assert c["mcnemar"] == {"b_only": 20.0, "a_only": 0.0, "p": pytest.approx(2 / 2**20)}
    assert c["mean_score"]["p_permutation"] < 0.001
    assert report["requirement"]["met"]
    # the other way round: B worse -> not better, and not "not worse" either
    assert main([pb, pa, "--resamples", "1000", "--require", "better"]) == 1
    assert main([pb, pa, "--resamples", "1000", "--require", "not_worse", "--margin", "0.05"]) == 1
    assert main([pa, pa, "--resamples", "200", "--require", "not_worse"]) == 0
    capsys.readouterr()
    assert main([pa, pb, "--strict"]) == 1  # an item only in B
    assert "--strict" in capsys.readouterr().err
    assert main([pa, "--require", "better"]) == 1
