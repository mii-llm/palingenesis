"""pgs pairs: contrastive twin pairs (build, check) and their edge cases."""

import json

import pytest

from palingenesis.pairs import build_twin_pairs, check_pairs, edit_stats, main, text_of

TWINS = [
    {"twin": "store", "prompt": "The store closes at 18:00. Is it open at 17:30?", "answer": "Yes", "pred": "Yes"},
    {"twin": "store", "prompt": "The store closes at 17:00. Is it open at 17:30?", "answer": "No", "pred": "Yes"},
    {"twin": "unit", "prompt": "Convert 2 km to m.", "answer": "2000 m"},
    {"twin": "unit", "prompt": "Convert 2 m to cm.", "answer": "200 cm"},
    {"twin": "same", "prompt": "Say hi.", "answer": "hi"},
    {"twin": "same", "prompt": "Say hi!", "answer": "Hi"},  # answers equal after normalisation
    {"twin": "far", "prompt": "What is the capital of France?", "answer": "Paris"},
    {"twin": "far", "prompt": "Write a haiku about winter mornings in the mountains.", "answer": "Snow..."},
    {"twin": "alone", "prompt": "x", "answer": "y"},
]
ONE_WAY_SIBLING = {"prompt": TWINS[1]["prompt"], "chosen": "No", "rejected": "Maybe"}


def test_text_and_edit_stats():
    assert text_of("a") == "a" and text_of({"role": "assistant", "content": "b", "reasoning": "r"}) == "b"
    assert text_of({"role": "assistant", "content": "b", "reasoning": "r"}, reasoning=True) == "r\nb"
    chat = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert text_of(chat) == "system: s\nuser: u"
    assert text_of([{"role": "assistant", "content": [{"type": "text", "text": "x"}]}]) == "x"
    stats = edit_stats("The store closes at 18:00.", "The store closes at 17:00.")
    assert stats["changed_words"] == 1 and stats["edit"] == pytest.approx(1 / 8)  # The store closes at 18 : 00 .
    assert edit_stats("a b c", "a b c")["changed_words"] == 0
    assert edit_stats("a", "a b c")["changed_words"] == 2


def test_build_twin_pairs_both_directions_and_drops():
    pairs, report = build_twin_pairs(TWINS, prediction_field="pred")
    assert report["pairs"] == 4 and report["twin_groups"] == 5
    assert report["dropped"] == {"same_answer": 2, "prompts_too_different": 2, "no_sibling": 1}
    store = [p for p in pairs if p["twin"] == "store"]
    assert {(p["chosen"], p["rejected"]) for p in store} == {("Yes", "No"), ("No", "Yes")}
    # the model answered "Yes" to the 17:00 twin: the sibling's answer, a misread
    misread = {p["direction"]: p["misread"] for p in store}
    assert misread == {"0>1": False, "1>0": True} and report["misread_pairs"] == 1
    only, report = build_twin_pairs(TWINS, prediction_field="pred", only_misread=True)
    assert [p["direction"] for p in only] == ["1>0"] and report["dropped"]["not_misread"] == 3
    strict, _ = build_twin_pairs(TWINS, max_edit_words=1)
    assert {p["twin"] for p in strict} == {"store"}  # the unit twins change two words
    with pytest.raises(ValueError, match="row 0: no 'answer'"):
        build_twin_pairs([{"twin": 1, "prompt": "p"}])


def test_check_finds_twins_balance_and_errors():
    pairs, _ = build_twin_pairs(TWINS)
    report = check_pairs(pairs)
    assert report["twin_pairs"] == 4 and report["unbalanced_twin_pairs"] == 0 and report["identical"] == 0
    # the store twins one way only: the sibling's chosen answer must appear somewhere for the twin to be seen; here
    # the 17:00 prompt keeps a pair with another rejected answer, so the 18:00 pair is a twin without its reverse
    one_way = [p for p in pairs if p["direction"] != "1>0"] + [ONE_WAY_SIBLING]
    report = check_pairs(one_way)
    assert report["twin_pairs"] == 3 and report["unbalanced_twin_pairs"] == 1
    assert report["examples"]["unbalanced_twin_pairs"] == [0]
    assert check_pairs([p for p in pairs if p["direction"] != "1>0"])["twin_pairs"] == 2  # invisible without it
    bad = pairs + [
        {"prompt": "q", "chosen": "a", "rejected": " A "},  # identical after normalisation
        {"prompt": "q2", "chosen": "a", "rejected": "b"},
        {"prompt": "q2", "chosen": "b", "rejected": "a"},  # contradiction
        {"prompt": "q2", "chosen": "a", "rejected": "b"},  # duplicate
    ]
    report = check_pairs(bad)
    assert report["identical"] == 1 and report["contradictions"] == 3 and report["duplicates"] == 1
    # implicit prompts: both sides are full conversations
    implicit = [
        {
            "chosen": [{"role": "user", "content": p["prompt"]}, {"role": "assistant", "content": p["chosen"]}],
            "rejected": [{"role": "user", "content": p["prompt"]}, {"role": "assistant", "content": p["rejected"]}],
        }
        for p in pairs
    ]
    assert check_pairs(implicit)["twin_pairs"] == 4


def test_cli(tmp_path, capsys):
    src = tmp_path / "twins.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in TWINS))
    out = tmp_path / "pairs.jsonl"
    assert main(["build", str(src), "-o", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["pairs"] == 4 and len(out.read_text().splitlines()) == 4
    assert main(["check", str(out), "--json", "--require_balanced"]) == 0
    capsys.readouterr()
    one_way = tmp_path / "one_way.jsonl"
    one_way.write_text(out.read_text().splitlines()[0] + "\n" + json.dumps(ONE_WAY_SIBLING) + "\n")
    assert main(["check", str(one_way)]) == 0 and main(["check", str(one_way), "--require_balanced"]) == 1
    assert main(["build", str(src), "-o", str(out), "--only_misread"]) == 1  # needs --prediction_field
    assert "--prediction_field" in capsys.readouterr().err
