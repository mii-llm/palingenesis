"""pgs eval reads lm-evaluation-harness task files: prompts, few-shot sampling, filters and metrics as the harness
builds them (golden strings), includes and !function, refusals that name the key, the conformance report, and a
generate_until task end to end in completion mode (no GPU)."""

import json
import random

import pytest

pytest.importorskip("torch")
pytest.importorskip("jinja2")

from palingenesis.eval import conformance, lmeval  # noqa: E402
from palingenesis.eval.config import EvalConfigError, SamplingConfig, load_task, task_errors  # noqa: E402
from palingenesis.eval.tasks import build_items  # noqa: E402

GSM8K = """\
tag:
  - math_word_problems
task: gsm8k
dataset_path: {data}
output_type: generate_until
training_split: train
fewshot_split: train
test_split: test
doc_to_text: "Question: {{{{question}}}}\\nAnswer:"
doc_to_target: "{{{{answer}}}}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ","
      - "\\\\$"
      - "(?s).*#### "
      - "\\\\.$"
generation_kwargs:
  until:
    - "Question:"
    - "</s>"
    - "<|im_end|>"
  do_sample: false
  temperature: 0.0
repeats: 1
num_fewshot: 2
filter_list:
  - name: "strict-match"
    filter:
      - function: "regex"
        regex_pattern: "#### (\\\\-?[0-9\\\\.\\\\,]+)"
      - function: "take_first"
  - name: "flexible-extract"
    filter:
      - function: "regex"
        group_select: -1
        regex_pattern: "(-?[$0-9.,]{{2,}})|(-?[0-9]+)"
      - function: "take_first"
metadata:
  version: 3.0
"""


def _gsm8k(tmp_path, n_test=4):
    train = [{"question": f"train {i}?", "answer": f"work {i}\n#### {i}"} for i in range(6)]
    test = [{"question": f"What is {i} + {i}?", "answer": f"{i} + {i} = {2 * i}\n#### {2 * i}"} for i in range(n_test)]
    data = tmp_path / "gsm.json"
    data.write_text(json.dumps({"train": train, "test": test}))
    path = tmp_path / "gsm8k.yaml"
    path.write_text(GSM8K.format(data=data))
    return path, train, test


def test_generate_until_task_reads_like_the_harness(tmp_path):
    path, train, test = _gsm8k(tmp_path)
    assert lmeval.is_lmeval_task(path)
    task = load_task(str(path))
    assert task_errors(task, {"default"}) == []
    assert (task.name, task.split, task.template, task.version, task.output) == ("gsm8k", "test", "completion", "3.0", "generate")
    assert task.sampling == {"max_tokens": 256, "temperature": 0.0, "stop": ["Question:", "</s>", "<|im_end|>"]}
    assert list(task.rewards) == ["exact_match,strict-match", "exact_match,flexible-extract"]
    assert [r["weight"] for r in task.rewards.values()] == [1.0, 0.0]
    items = build_items(task)
    # the harness's default sampler: random.Random(1234), num_fewshot documents per test document, in order
    rnd = random.Random(1234)
    for item, doc in zip(items, test):
        shots = rnd.sample(train, 2)
        expected = "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots)
        assert item.row["prompt_text"] == expected + f"Question: {doc['question']}\nAnswer:"
        assert item.row["target"] == doc["answer"] and item.row["messages"][0]["content"] == item.row["prompt_text"]
    # zero-shot from the command line (--num_fewshot 0); a limit samples items after the prompts were built
    assert build_items(task, num_fewshot=0)[1].row["prompt_text"] == "Question: What is 1 + 1?\nAnswer:"
    limited = build_items(task, limit=2)
    assert all(i.row["prompt_text"] == items[int(i.id.split("/")[1])].row["prompt_text"] for i in limited)


def test_filters_and_exact_match_are_the_harness_s():
    strict = [{"function": "regex", "regex_pattern": r"#### (\-?[0-9\.\,]+)"}, {"function": "take_first"}]
    flexible = [{"function": "regex", "group_select": -1, "regex_pattern": r"(-?[$0-9.,]{2,})|(-?[0-9]+)"}]
    assert lmeval.apply_filters("so 3 * 4 = 12\n#### 1,200", strict) == "1,200"
    assert lmeval.apply_filters("no marker 12", strict) == "[invalid]"
    assert lmeval.apply_filters("first 10 then $1,200.", flexible) == "$1,200."
    assert lmeval.apply_filters("the answer is 7", flexible) == "7"
    assert lmeval.apply_filters("  Yes ", [{"function": "remove_whitespace"}, {"function": "lowercase"}]) == "yes "
    options = {"ignore_case": True, "regexes_to_ignore": [",", r"\$", "(?s).*#### ", r"\.$"]}
    assert lmeval.exact_match("$1,200.", "long work\n#### 1200", **options) == 1.0
    assert lmeval.exact_match("1201", "#### 1200", **options) == 0.0
    assert lmeval.exact_match("The Answer!", "the answer", ignore_case=True, ignore_punctuation=True) == 1.0
    assert lmeval.exact_match("a1", "a2", ignore_numbers=True) == 1.0
    row = {"target": "work\n#### 18"}
    assert lmeval.generation_metric("9 + 9\n#### 18", "exact_match", options, strict, **row) == 1.0
    assert lmeval.generation_metric("it is 18 dollars", "exact_match", options, strict, **row) == 0.0
    assert lmeval.generation_metric("it is 18 dollars", "exact_match", options, flexible, **row) == 1.0
    assert lmeval.generation_metric("b", "exact_match", {}, [], target=["a", "b"]) == 1.0  # multiple targets
    assert lmeval.cut_at_stop(" 18\n\nQuestion: next", ["Question:", "</s>"]) == " 18\n\n"


def test_multiple_choice_with_include_first_n_and_group_by(tmp_path):
    dev = [{"question": f"dev {s} {i} ", "choices": ["w", "x", "y", "z"], "answer": i % 4, "subject": s}
           for s in ("high_school_math", "virology") for i in range(3)]  # fmt: skip
    test = [{"question": " What? ", "choices": ["1", "2", "3", "4"], "answer": 2, "subject": "virology"}]
    data = tmp_path / "mmlu.json"
    data.write_text(json.dumps({"dev": dev, "test": test}))
    (tmp_path / "_default_template_yaml").write_text(
        f"dataset_path: {data}\ntest_split: test\nfewshot_split: dev\nfewshot_config:\n  sampler: first_n\n"
        "  group_by: subject\noutput_type: multiple_choice\n"
        'doc_to_text: "{{question.strip()}}\\nA. {{choices[0]}}\\nB. {{choices[1]}}\\nC. {{choices[2]}}\\nD. '
        '{{choices[3]}}\\nAnswer:"\n'
        'doc_to_choice: ["A", "B", "C", "D"]\ndoc_to_target: answer\n'
        "metric_list:\n  - metric: acc\nmetadata:\n  version: 1.0\n"
    )
    path = tmp_path / "mmlu_all.yaml"
    path.write_text(
        'include: _default_template_yaml\ntask: mmlu_all\nnum_fewshot: 2\ncategory_field: subject\n'
        'description: "The following are multiple choice questions (with answers) about '
        "{{subject.replace('_', ' ')}}.\\n\\n\"\n"
    )
    task = load_task(str(path))
    assert task.output == "choices" and task.target_delimiter == " " and task.category_field == "subject"
    (item,) = build_items(task)

    def shot(d):
        return f"{d['question'].strip()}\nA. w\nB. x\nC. y\nD. z\nAnswer: {'ABCD'[d['answer']]}\n\n"

    assert item.row["prompt_text"] == (
        "The following are multiple choice questions (with answers) about virology.\n\n"
        + shot(dev[3]) + shot(dev[4])  # the first two dev rows of the question's own subject
        + "What?\nA. 1\nB. 2\nC. 3\nD. 4\nAnswer:"
    )  # fmt: skip
    assert item.row["choices"] == ["A", "B", "C", "D"] and item.row["gold"] == 2
    with pytest.raises(EvalConfigError, match="5 few-shot examples requested, 3 available"):
        build_items(task, num_fewshot=5)


def test_templated_choices_function_fields_and_process_results(tmp_path):
    data = tmp_path / "d.json"
    data.write_text(json.dumps({"validation": [
        {"q": "Sky?", "options": ["blue", "green"], "key": "A", "labels": ["A", "B"]},
        {"q": "Grass?", "options": ["blue", "green", "red"], "key": "B", "labels": ["A", "B", "C"]},
    ]}))  # fmt: skip
    (tmp_path / "utils.py").write_text(
        "def doc_to_text(doc):\n    return 'Q: ' + doc['q'] + '\\nA:'\n\n"
        "def process_results(doc, results):\n"
        "    return {'hit': float(results[0].strip() == doc['options'][0]), 'length': len(results[0])}\n"
    )
    arc = tmp_path / "arc.yaml"
    arc.write_text(
        f"task: arc-like\ndataset_path: {data}\nvalidation_split: validation\noutput_type: multiple_choice\n"
        "doc_to_text: !function utils.doc_to_text\n"
        'doc_to_target: "{{labels.index(key)}}"\ndoc_to_choice: "{{options}}"\n'
        "metric_list:\n  - metric: acc\n  - metric: acc_norm\n"
    )
    task = load_task(str(arc))
    assert task.name == "arc-like" and task.split == "validation"
    items = build_items(task)
    assert [i.row["prompt_text"] for i in items] == ["Q: Sky?\nA:", "Q: Grass?\nA:"]
    assert [(i.row["choices"], i.row["gold"]) for i in items] == [(["blue", "green"], 0), (["blue", "green", "red"], 1)]
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        f"task: custom\ndataset_path: {data}\ntest_split: validation\noutput_type: generate_until\n"
        "doc_to_text: q\ndoc_to_target: key\nprocess_results: !function utils.process_results\n"
        "metric_list:\n  - metric: hit\n  - metric: length\n"
    )
    task = load_task(str(custom))
    assert task.sampling["stop"] == ["\n\n"] and list(task.rewards) == ["hit", "length"]
    (first, _) = build_items(task)
    assert first.row["prompt_text"] == "Sky?"
    # as a Reward calls it: the row's columns under the scorer's own arguments (a column may be named `options`)
    assert lmeval.generation_metric(" blue", **{**first.row, **task.rewards["hit"]["args"]}) == 1.0
    assert lmeval.generation_metric(" blue", **{**first.row, **task.rewards["length"]["args"]}) == 5.0


def test_what_is_outside_the_subset_is_refused_by_name(tmp_path):
    def task(body):
        path = tmp_path / "t.yaml"
        path.write_text("task: t\ndataset_path: d.jsonl\ntest_split: test\ndoc_to_text: q\ndoc_to_target: a\n" + body)
        return load_task(str(path))

    with pytest.raises(EvalConfigError, match="output_type 'loglikelihood_rolling' is not supported"):
        task("output_type: loglikelihood_rolling\n")
    with pytest.raises(EvalConfigError, match="filter 'multi_choice_regex' is not supported"):
        task("filter_list:\n  - name: x\n    filter:\n      - function: multi_choice_regex\n")
    with pytest.raises(EvalConfigError, match="metric 'bleu' is not built in"):
        task("metric_list:\n  - metric: bleu\n")
    with pytest.raises(EvalConfigError, match="`doc_to_image` is outside the supported subset"):
        task("doc_to_image: image\n")
    with pytest.raises(EvalConfigError, match="multiple_choice needs doc_to_choice"):
        task("output_type: multiple_choice\n")
    group = tmp_path / "group.yaml"
    group.write_text("group: mmlu\ntask:\n  - mmlu_a\n  - mmlu_b\n")
    with pytest.raises(EvalConfigError):  # a group names tasks, it is not one
        load_task(str(group))
    ours = tmp_path / "ours.yaml"
    ours.write_text("dataset: d.jsonl\nscorer: mcq\n")
    assert not lmeval.is_lmeval_task(ours)


def test_builtin_lmeval_dialect_tasks():
    mmlu = load_task("mmlu")
    assert mmlu.output == "choices" and mmlu.lmeval["fewshot_config"]["group_by"] == "subject"
    assert mmlu.lmeval["num_fewshot"] == 5 and mmlu.category_field == "subject"
    pro = load_task("mmlu_pro_mc")
    docs = [{"question": " Pick. ", "options": ["x", "y", "z"], "answer_index": 2, "category": "law"}]
    (row,) = lmeval.build_rows(pro, docs, None)
    assert row["prompt_text"] == (
        "The following are multiple choice questions (with answers) about law.\n\nPick.\nA. x\nB. y\nC. z\nAnswer:"
    )
    assert row["choices"] == ["A", "B", "C"] and row["gold"] == 2


def test_conformance_report(tmp_path, capsys):
    ours = tmp_path / "gsm8k.default.jsonl"
    scores = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    ours.write_text("".join(
        json.dumps({"id": f"gsm8k/{i}", "scores": [s], "metrics": {"exact_match,strict-match": [s], "exact_match,flexible-extract": [1]}}) + "\n"
        for i, s in enumerate(scores)
    ))  # fmt: skip
    theirs = tmp_path / "lm" / "model"
    theirs.mkdir(parents=True)
    (theirs / "results_2026.json").write_text(json.dumps(
        {"results": {"gsm8k": {"exact_match,strict-match": 0.7, "exact_match_stderr,strict-match": 0.1, "exact_match,flexible-extract": 0.9}}}
    ))  # fmt: skip
    flipped = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]  # two documents differ, the totals agree
    (theirs / "samples_gsm8k_2026.jsonl").write_text("".join(
        json.dumps({"doc_id": i, "filter": name, "exact_match": s if name == "strict-match" else 1}) + "\n"
        for i, s in enumerate(flipped) for name in ("strict-match", "flexible-extract")
    ))  # fmt: skip
    aggregate, docs = conformance.lmeval_results(tmp_path / "lm", "gsm8k", "exact_match,strict-match")
    assert aggregate == 0.7 and len(docs) == 10
    report = conformance.compare(conformance.pgs_scores(ours, "exact_match,strict-match"), aggregate, docs)
    assert report["pgs"] == pytest.approx(0.7) and report["difference"] == pytest.approx(0.0) and report["within"]
    assert (report["paired_items"], report["same_items"]) == (10, 8)
    argv = ["--pgs", str(ours), "--lmeval", str(tmp_path / "lm"), "--task", "gsm8k", "--metric", "exact_match,flexible-extract"]
    assert conformance.main(argv) == 1 and "pgs 1.0000, lm-eval 0.9000" in capsys.readouterr().out
    assert conformance.main(["--pgs", str(ours), "--lmeval", str(tmp_path), "--task", "nope"]) == 1



def test_conformance_pairs_subtasks_and_id_columns(tmp_path):
    ours = tmp_path / "mmlu.default.jsonl"  # global row ids, subjects concatenated: a/0 a/1 b/0, listed out of order
    rows = [("mmlu/2", "b", 1), ("mmlu/0", "a", 0), ("mmlu/10", "b", 0), ("mmlu/1", "a", 1)]
    ours.write_text("".join(json.dumps({"id": i, "category": c, "scores": [s]}) + "\n" for i, c, s in rows))
    lm = tmp_path / "lm" / "model"
    lm.mkdir(parents=True)
    for name, stamp, values in (("mmlu_a", "2026-01", [1, 1]), ("mmlu_a", "2026-02", [0, 1]), ("mmlu_b", "2026-02", [1, 0])):
        samples = "".join(json.dumps({"doc_id": i, "acc": v}) + "\n" for i, v in enumerate(values))
        (lm / f"samples_{name}_{stamp}.jsonl").write_text(samples)
    _, docs = conformance.lmeval_results(tmp_path / "lm", "mmlu", "acc")
    assert docs == {"a/0": 0.0, "a/1": 1.0, "b/0": 1.0, "b/1": 0.0}  # the newest file of each subject only
    report = conformance.compare(conformance.pgs_scores(ours, group_field="category"), None, docs)
    assert (report["paired_items"], report["same_items"]) == (4, 4)
    with pytest.raises(ValueError, match="subject"):
        conformance.pgs_scores(ours, group_field="subject")
    keyed = tmp_path / "ifeval.plain.jsonl"
    keyed.write_text("".join(json.dumps({"id": f"ifeval/{k}", "scores": [s]}) + "\n" for k, s in ((1000, 1), (1005, 0))))
    (lm / "samples_ifeval_2026-02.jsonl").write_text("".join(
        json.dumps({"doc_id": i, "doc": {"key": k}, "prompt_level_strict_acc": s}) + "\n" for i, (k, s) in enumerate(((1000, 1), (1005, 1)))
    ))  # fmt: skip
    _, docs = conformance.lmeval_results(tmp_path / "lm", "ifeval", "prompt_level_strict_acc", id_field="key")
    report = conformance.compare(conformance.pgs_scores(keyed), None, docs)
    assert (report["paired_items"], report["same_items"]) == (2, 1)

def test_lmeval_task_end_to_end_in_completion_mode(tmp_path, monkeypatch):
    from palingenesis.eval.runner import Evaluator
    from palingenesis.opd.rollout import Rollout
    from palingenesis.opd.teachers import end_of_turn_id
    from tests.test_rl import tokenizer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    eot = end_of_turn_id(tok, {})
    monkeypatch.setattr("palingenesis.opd.trainer.student_stop_ids", lambda *a, **k: (eot,))
    monkeypatch.setattr("palingenesis.eval.runner.model_sampling", lambda model: SamplingConfig())
    path, _, test = _gsm8k(tmp_path, n_test=3)
    seen = []

    class Engine:
        version = 0

        def generate(self, prompts, max_new_tokens, temperature):
            out = []
            for prompt in prompts:
                text = tok.decode(prompt)
                seen.append((text, temperature))
                asked = text.rsplit("Question: What is ", 1)[1].split(" ")[0]
                answer = f" {asked} + {asked}\n#### {2 * int(asked) + (asked == '2')}\n\nQuestion: and then?\nAnswer: 99"
                out.append(Rollout(tok.encode(answer, add_special_tokens=False), [], "length", 0))
            return out

    task = load_task(str(path))
    with Evaluator(tok, Engine(), "", engine_options={"max_model_len": 2048}, resamples=100) as evaluator:
        summary = evaluator.run(task, out_dir=tmp_path / "out", items=build_items(task))
    text, sampling = seen[0]
    assert not text.startswith("<|im_start|>") and text.endswith("?\nAnswer:")  # raw text: no chat template
    assert sampling.stop == ("Question:", "</s>", "<|im_end|>") and sampling.temperature == 0.0
    assert summary["template"] == "completion" and summary["sampling"]["max_tokens"] == 256
    assert summary["mean_score"]["estimate"] == pytest.approx(2 / 3)  # the third answer is wrong
    assert summary["metrics"]["exact_match,strict-match"]["mean"] == pytest.approx(2 / 3)
    assert summary["metrics"]["exact_match,flexible-extract"]["mean"] == pytest.approx(2 / 3)
    samples = [json.loads(line) for line in (tmp_path / "out" / "gsm8k.default.samples.jsonl").read_text().splitlines()]
    assert all("99" not in s["completion"] and s["finish"] == "stop" for s in samples)  # cut at the stop string
    rows = [json.loads(line) for line in (tmp_path / "out" / "gsm8k.default.jsonl").read_text().splitlines()]
    assert [r["scores"] for r in rows] == [[1.0], [1.0], [0.0]]
    assert conformance.pgs_scores(tmp_path / "out" / "gsm8k.default.jsonl", "exact_match,strict-match") == {"0": 1.0, "1": 1.0, "2": 0.0}
