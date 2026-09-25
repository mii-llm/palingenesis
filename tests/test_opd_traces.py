"""Agent-trace planning (palingenesis.opd.traces) on the real Qwen3.5 chat template."""

import random
import sys

import pytest

sys.path.insert(0, "src")

transformers = pytest.importorskip("transformers")

from palingenesis.opd.traces import TracePlanner, is_trainable_turn, topic_teacher  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "shell", "description": "Run a command",
                                           "parameters": {"type": "object", "properties": {
                                               "command": {"type": "string"}}, "required": ["command"]}}}]


def turn(i, tool=True):
    m = {"role": "assistant", "reasoning_content": f"Step {i}: I should look at file{i}.py.", "content": ""}
    if tool:
        m["tool_calls"] = [{"type": "function", "function": {"name": "shell", "arguments": {"command": f"cat file{i}.py"}}}]
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
        assert tok.decode(b.context) == want                      # the exact rendering, token for token in text
        if b.prefix == plan.trunk[b.attach:b.attach + 1] and b.context == plan.trunk[:b.attach + 1]:
            continue                                              # on the trunk: re-feeds its last context token
        assert b.context == plan.trunk[:b.attach] + b.prefix      # off the trunk: the tail is the prefix


def test_one_agentic_loop_is_one_trunk(tok):
    msgs = agent_trace()
    p = planner(tok)
    plan = p.plan(msgs, TOOLS, random.Random(0))
    turns = [k for k, m in enumerate(msgs) if is_trainable_turn(m)]
    assert [b.turn for b in plan.branches] == turns
    check_branches(tok, p, msgs, plan)
    assert all(b.context == plan.trunk[:len(b.context)] for b in plan.branches)   # every context is a prefix
    # recorded turns: every assistant message before the last one, through its <|im_end|>
    assert len(plan.kd_spans) == len(turns) - 1
    for (a, b), k in zip(plan.kd_spans, turns):
        text = tok.decode(plan.trunk[a:b])
        assert text.endswith("<|im_end|>") and f"Step {k // 2 - 1}" in text and "file" in text
        assert plan.trunk[a - 1:a] and tok.decode(plan.trunk[:a]).endswith("<think>\n")


def test_a_new_user_message_moves_the_attach_point(tok):
    """Qwen3.5 drops earlier reasoning once a new user query arrives: the later turns leave the
    trunk where the first rewritten turn starts and carry the rest of their context."""
    msgs = agent_trace(2) + [{"role": "user", "content": "Now add a test."}] + [turn(10), {"role": "tool", "content": "ok"},
                                                                                  turn(11, tool=False)]
    p = planner(tok)
    plan = p.plan(msgs, TOOLS, random.Random(0))
    check_branches(tok, p, msgs, plan)
    early = [b for b in plan.branches if b.turn < 7]
    late = [b for b in plan.branches if b.turn > 7]
    assert early and late
    assert all(b.context == plan.trunk[:len(b.context)] for b in late)       # the trunk is the last context
    off = [b for b in early if b.context != plan.trunk[:len(b.context)]]
    assert off and all(len(b.prefix) > 1 for b in off)                      # rewritten history in the branch
    # every recorded turn is a distillation target, the rewritten ones (no reasoning) included
    recorded = [k for k, m in enumerate(msgs[:max(b.turn for b in plan.branches)]) if is_trainable_turn(m)]
    assert len(plan.kd_spans) == len(recorded)
    texts = [tok.decode(plan.trunk[a:b]) for a, b in plan.kd_spans]
    assert all(t.endswith("<|im_end|>") for t in texts)
    assert "Step 0" not in texts[0] and "file0.py" in texts[0]              # rewritten: the tool call, no reasoning


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
