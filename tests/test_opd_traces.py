"""Agent-trace planning (palingenesis.opd.traces) on the real Qwen3.5 chat template."""

import random
import sys

import pytest

sys.path.insert(0, "src")

transformers = pytest.importorskip("transformers")

from palingenesis.opd.traces import (  # noqa: E402
    OnsetScan,
    TracePlanner,
    cut_runs,
    is_trainable_turn,
    repetition_runs,
    topic_teacher,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
]


def turn(i, tool=True):
    m = {"role": "assistant", "reasoning_content": f"Step {i}: I should look at file{i}.py.", "content": ""}
    if tool:
        m["tool_calls"] = [
            {"type": "function", "function": {"name": "shell", "arguments": {"command": f"cat file{i}.py"}}}
        ]
    else:
        m["content"] = f"Done after {i} steps."
    return m


def agent_trace(n=5):
    msgs = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "Fix the bug."}]
    for i in range(n):
        msgs.append(turn(i))
        msgs.append({"role": "tool", "content": f"print('file {i}')"})
    msgs.append(turn(n, tool=False))
    return msgs


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    except Exception as e:  # noqa: BLE001 — offline
        pytest.skip(f"tokenizer unavailable: {e}")


def planner(tok, **kw):
    options = dict(max_context=100_000, branches_per_trace=0, recorded_kd=True)
    options.update(kw)
    stop = (tok.convert_tokens_to_ids("<|im_end|>"),)
    return TracePlanner(tok, {"enable_thinking": True}, stop, **options)


def check_branches(tok, p, msgs, plan):
    for b in plan.branches:
        want = p.render(msgs, TOOLS, b.turn)
        assert tok.decode(b.context) == want  # the exact rendering, token for token in text
        if b.prefix == plan.trunk[b.attach : b.attach + 1] and b.context == plan.trunk[: b.attach + 1]:
            continue  # on the trunk: re-feeds its last context token
        assert b.context == plan.trunk[: b.attach] + b.prefix  # off the trunk: the tail is the prefix


def test_one_agentic_loop_is_one_trunk(tok):
    msgs = agent_trace()
    p = planner(tok)
    plan = p.plan(msgs, TOOLS, random.Random(0))
    turns = [k for k, m in enumerate(msgs) if is_trainable_turn(m)]
    assert [b.turn for b in plan.branches] == turns
    check_branches(tok, p, msgs, plan)
    assert all(b.context == plan.trunk[: len(b.context)] for b in plan.branches)  # every context is a prefix
    # recorded turns: every assistant message before the last one, through its <|im_end|>
    assert len(plan.kd_spans) == len(turns) - 1
    for (a, b), k in zip(plan.kd_spans, turns):
        text = tok.decode(plan.trunk[a:b])
        assert text.endswith("<|im_end|>") and f"Step {k // 2 - 1}" in text and "file" in text
        assert plan.trunk[a - 1 : a] and tok.decode(plan.trunk[:a]).endswith("<think>\n")


def test_a_new_user_message_moves_the_attach_point(tok):
    """Qwen3.5 drops earlier reasoning once a new user query arrives: the later turns leave the
    trunk where the first rewritten turn starts and carry the rest of their context."""
    msgs = (
        agent_trace(2)
        + [{"role": "user", "content": "Now add a test."}]
        + [turn(10), {"role": "tool", "content": "ok"}, turn(11, tool=False)]
    )
    p = planner(tok)
    plan = p.plan(msgs, TOOLS, random.Random(0))
    check_branches(tok, p, msgs, plan)
    early = [b for b in plan.branches if b.turn < 7]
    late = [b for b in plan.branches if b.turn > 7]
    assert early and late
    assert all(b.context == plan.trunk[: len(b.context)] for b in late)  # the trunk is the last context
    off = [b for b in early if b.context != plan.trunk[: len(b.context)]]
    assert off and all(len(b.prefix) > 1 for b in off)  # rewritten history in the branch
    # every recorded turn is a distillation target, the rewritten ones (no reasoning) included
    recorded = [k for k, m in enumerate(msgs[: max(b.turn for b in plan.branches)]) if is_trainable_turn(m)]
    assert len(plan.kd_spans) == len(recorded)
    texts = [tok.decode(plan.trunk[a:b]) for a, b in plan.kd_spans]
    assert all(t.endswith("<|im_end|>") for t in texts)
    assert "Step 0" not in texts[0] and "file0.py" in texts[0]  # rewritten: the tool call, no reasoning


def test_max_context_and_sampling(tok):
    msgs = agent_trace(12)
    full = planner(tok).plan(msgs, TOOLS, random.Random(0))
    limit = len(full.branches[5].context)
    p = planner(tok, max_context=limit, branches_per_trace=3)
    for seed in range(5):
        plan = p.plan(msgs, TOOLS, random.Random(seed))
        assert 1 <= len(plan.branches) <= 3
        assert len(plan.trunk) <= limit and all(len(b.context) <= limit for b in plan.branches)
        check_branches(tok, p, msgs, plan)
    assert planner(tok, max_context=5).plan(msgs, TOOLS, random.Random(0)) is None


def test_untrainable_turns_are_neither_branches_nor_recorded_targets(tok):
    msgs = agent_trace(3)
    msgs[2]["loss"] = False
    p = planner(tok)
    plan = p.plan(msgs, TOOLS, random.Random(0))
    assert 2 not in [b.turn for b in plan.branches]
    assert all("Step 0" not in tok.decode(plan.trunk[a:b]) for a, b in plan.kd_spans)


def test_topic_teacher():
    mapping = {"coder": ["Code_Agent", "Tool_Use"], "searcher": ["Search_Agent-en"]}
    assert topic_teacher({"domain": "Tool_Use"}, "domain", mapping, "general") == "coder"
    assert topic_teacher({"domain": "Search_Agent-en"}, "domain", mapping, "general") == "searcher"
    assert topic_teacher({"domain": "Other"}, "domain", mapping, "general") == "general"
    assert topic_teacher({"domain": "Tool_Use"}, "", mapping, "general") == "general"


def test_prompt_rows_become_one_turn_traces(tok, tmp_path):
    import json

    from palingenesis.opd.traces import load_prompt_rows_as_traces

    rows = [
        {"messages": [{"role": "user", "content": "Weather in Rome?"}], "tools": TOOLS, "id": "a"},
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},  # nothing to do
    ]
    path = tmp_path / "p.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = load_prompt_rows_as_traces(str(path), chat_template_kwargs={"enable_thinking": False})
    assert len(out) == 1 and out[0]["id"] == "a" and out[0]["tools"]
    msgs = out[0]["messages"]
    assert msgs[-1]["role"] == "assistant" and is_trainable_turn(msgs[-1])
    # the regenerated turn's context is the prompt with the generation prompt, exactly as plain OPD renders it;
    # the placeholder content never appears in any context
    stop = (tok.convert_tokens_to_ids("<|im_end|>"),)
    p = TracePlanner(tok, {"enable_thinking": False}, stop, max_context=100_000, branches_per_trace=0, recorded_kd=True)
    plan = p.plan(msgs, out[0]["tools"], random.Random(0))
    assert len(plan.branches) == 1 and not plan.kd_spans
    want = tok.apply_chat_template(
        msgs[:-1], tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    assert tok.decode(plan.branches[0].context) == want and "(regenerated)" not in want
    # tool-call arguments stored as JSON strings (wire format) are parsed, as for recorded traces
    wire = {
        "messages": [
            {"role": "user", "content": "Weather in Rome?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "shell", "arguments": json.dumps({"command": "date"})}}
                ],
            },
            {"role": "tool", "content": "Mon"},
        ],
        "tools": TOOLS,
    }
    path.write_text(json.dumps(wire) + "\n")
    row = load_prompt_rows_as_traces(str(path), chat_template_kwargs={"enable_thinking": False})[0]
    assert isinstance(row["messages"][1]["tool_calls"][0]["function"]["arguments"], dict)
    assert p.plan(row["messages"], TOOLS, random.Random(0)) is not None
    # a row whose own template arguments disagree with the source's is refused
    path.write_text(json.dumps({**rows[0], "chat_template_kwargs": {"enable_thinking": True}}) + "\n")
    with pytest.raises(ValueError, match="different sources"):
        load_prompt_rows_as_traces(str(path), chat_template_kwargs={"enable_thinking": False})


def test_source_kwargs_change_the_rendering(tok):
    stop = (tok.convert_tokens_to_ids("<|im_end|>"),)
    msgs = [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]
    think = TracePlanner(
        tok, {"enable_thinking": True}, stop, max_context=1000, branches_per_trace=0, recorded_kd=False
    )
    plain = TracePlanner(
        tok, {"enable_thinking": False}, stop, max_context=1000, branches_per_trace=0, recorded_kd=False
    )
    a, b = (tok.decode(x.plan(msgs, None, random.Random(0)).branches[0].context) for x in (think, plain))
    assert a != b and b.rstrip().endswith("</think>")  # non-thinking opens with an empty think block


# ---------------------------------------------------------------- repetition onsets (student-recorded traces)


def test_repetition_runs_on_token_ids():
    fresh = list(range(100, 200))
    # one turn, no repeats: nothing
    assert repetition_runs(fresh, [(0, 100)], ngram=4, window=8, threshold=0.75) == []
    # a turn that writes a 10-token phrase five times: the run starts where the second copy starts
    phrase = list(range(10))
    ids = fresh[:20] + phrase * 5
    runs = repetition_runs(ids, [(0, len(ids))], ngram=4, window=8, threshold=0.75)
    assert len(runs) == 1 and runs[0] == (30, len(ids))
    # across turns: the third turn copies the first (observations between them are never scanned)
    obs = list(range(500, 540))
    ids = fresh[:30] + obs + fresh[30:60] + obs + fresh[:30]
    spans = [(0, 30), (70, 100), (140, 170)]
    assert repetition_runs(ids, spans, ngram=4, window=8, threshold=0.75) == [(140, 170)]
    # the second turn copying the first is not caught: with one earlier turn, what it wrote is all "shared"
    assert repetition_runs(fresh[:30] + obs + fresh[:30], [(0, 30), (70, 100)], ngram=4, window=8) == []
    # quoting an observation is not a repeat (only assistant text counts); quoting it again is
    ids = obs[:30] + obs[:30] + obs[:30]
    assert repetition_runs(ids, [(30, 60)], ngram=4, window=8, threshold=0.75) == []
    assert repetition_runs(ids, [(30, 90)], ngram=4, window=8, threshold=0.75) == [(60, 90)]


def test_shared_scaffolding_is_not_a_repeat():
    """N-grams found in most earlier turns (call markup, say) never count; a turn that repeats one turn does."""
    scaffold = list(range(900, 912))
    turns = [scaffold + list(range(1000 + 50 * i, 1000 + 50 * i + 20)) for i in range(6)]
    ids, spans = [], []
    for t in turns:
        spans.append((len(ids), len(ids) + len(t)))
        ids += t + [7] * 5
    assert repetition_runs(ids, spans, ngram=4, window=8, threshold=0.75) == []
    # common=1: nothing is scaffolding, so every turn from the second on starts with a repeat (its markup)
    assert len(repetition_runs(ids, spans, ngram=4, window=8, threshold=0.75, common=1.0)) == 5
    looped = turns + [turns[3]]  # the 7th turn copies the 4th
    ids, spans = [], []
    for t in looped:
        spans.append((len(ids), len(ids) + len(t)))
        ids += t + [7] * 5
    # the copy's run starts after the shared markup: at its first n-gram that is not scaffolding
    assert repetition_runs(ids, spans, ngram=4, window=8, threshold=0.75) == [(spans[-1][0] + 12 - 3, spans[-1][1])]


def test_cut_runs_keeps_the_start_of_each_run():
    spans = [(0, 10), (20, 60)]
    assert cut_runs(spans, [(30, 50)], 4) == [(0, 10), (20, 34), (50, 60)]
    assert cut_runs(spans, [(30, 60)], 0) == [(0, 10), (20, 30)]
    assert cut_runs(spans, [(30, 33)], 4) == spans  # shorter than what is kept
    assert cut_runs(spans, [], 4) == spans


def looping_trace():
    """An agent trace whose last turn (cut at the token budget, say) ruminates."""
    msgs = agent_trace(4)
    loop = "Line 1 = `# ad servers`?? no. Line 2 = `ads.example.com Track.Example.NET`?? no. "
    msgs[-1]["reasoning_content"] = "The tests pass now, so I can report. Let me count the lines again. " + loop * 12
    return msgs


def test_onset_branches_start_where_a_recorded_turn_repeats_itself(tok):
    msgs = looping_trace()
    p = planner(tok, branches_per_trace=1, onsets=OnsetScan(branches=2))
    plan = p.plan(msgs, TOOLS, random.Random(0))
    onset = [b for b in plan.branches if b.onset]
    assert len(onset) == 1  # one run in this trace
    b = onset[0]
    assert b.turn == len(msgs) - 1  # the last turn: the trunk is the whole trace
    assert b.context == plan.trunk[: len(b.context)] and b.prefix == [plan.trunk[b.attach]]
    written = tok.decode(b.context).rsplit("<think>\n", 1)[1]  # what the turn wrote before the onset
    assert msgs[-1]["reasoning_content"].startswith(written)
    assert written.startswith("The tests pass now") and written.count("Line 2 =") <= 1  # within the first copy
    # regular branches are unchanged by the scan: the same turns as without it
    plain = planner(tok, branches_per_trace=1).plan(msgs, TOOLS, random.Random(0))
    assert [x.turn for x in plan.branches if not x.onset] == [x.turn for x in plain.branches]
    # a trace without repetition gets no onset branch
    assert not any(x.onset for x in p.plan(agent_trace(4), TOOLS, random.Random(0)).branches)


def test_onset_kd_tokens_cut_the_inside_of_a_run(tok):
    msgs = looping_trace()
    full = planner(tok, onsets=OnsetScan(kd_tokens=-1, branches=1)).plan(msgs, TOOLS, random.Random(0))
    cut = planner(tok, onsets=OnsetScan(kd_tokens=8)).plan(msgs, TOOLS, random.Random(0))
    assert full.trunk == cut.trunk
    last = full.kd_spans[-1]
    assert tok.decode(full.trunk[last[0] : last[1]]).count("Line 2") == 12  # the last turn is distilled whole
    onset = next(b for b in full.branches if b.onset)
    kept = {p for a, b in cut.kd_spans for p in range(a, b)}
    assert set(range(len(onset.context), len(onset.context) + 8)) <= kept  # the run's first tokens stay
    assert len(onset.context) + 8 not in kept
    assert len(kept) < sum(b - a for a, b in full.kd_spans) - 200
    # without the scan the last turn is not a recorded target (its context is the trunk)
    plain = planner(tok).plan(msgs, TOOLS, random.Random(0))
    assert len(plain.kd_spans) == len(full.kd_spans) - 1
