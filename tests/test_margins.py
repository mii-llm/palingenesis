"""Decision margins (palingenesis.margins) on a tiny CPU model."""

import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "tests")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from test_cmr import TOOL, fresh  # noqa: E402
from test_opd_trainer import tiny_model  # noqa: E402

from palingenesis.margins import MarginReader, decision_ids, log_odds  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("margins")
    path = tiny_model(tmp, "Qwen/Qwen3-0.6B", "student", 0)
    return path, transformers.AutoTokenizer.from_pretrained(path)


def test_log_odds_exact():
    logits = torch.tensor([[2.0, 0.0, -1.0, 3.0]])
    p = torch.softmax(logits, -1)[0]
    single = log_odds(logits, 1)[0]
    assert torch.isclose(single, torch.log(p[1]) - torch.log(1 - p[1]), atol=1e-5)
    pair = log_odds(logits, [1, 2])[0]
    q = p[1] + p[2]
    assert torch.isclose(pair, torch.log(q) - torch.log(1 - q), atol=1e-5)
    # saturated: no clamp, finite and still differentiable
    big = torch.tensor([[60.0, 0.0, 0.0]], requires_grad=True)
    m = log_odds(big, 0)
    m.sum().backward()
    assert torch.isfinite(m).all() and big.grad.abs().sum() > 0


def test_reader_matches_full_forward_and_batches(setup):
    path, tok = setup
    model = fresh(path).eval()
    reader = MarginReader(model, tok, device="cpu")
    tid = decision_ids(tok, "<tool_call>")
    sides = [
        {"messages": [{"role": "user", "content": "Weather in Rome?"}], "tools": [TOOL]},
        {"messages": [{"role": "user", "content": "Tell me a much longer story about the weather in Rome today"}]},
        {
            "messages": [{"role": "user", "content": "Rome and Paris?"}],
            "tools": [TOOL],
            "prefix_calls": [{"name": "get_weather", "arguments": {"city": "Rome"}}],
            "prefix_content": "Checking.",
        },
    ]
    with torch.no_grad():
        batch = reader.margins(sides, tid)
        single = torch.stack([reader.margins([s], tid)[0] for s in sides])
        ids = torch.tensor([reader.prompt_ids(sides[0])])
        full = model(input_ids=ids).logits[0, -1].float()
    assert torch.allclose(batch, single, atol=1e-4)
    assert torch.isclose(batch[0], log_odds(full[None], tid)[0], atol=1e-4)
    assert tok.decode(reader.prompt_ids(sides[2])).endswith("</tool_call>")


def test_decision_ids_refuses_multi_token_and_duplicates(setup):
    _, tok = setup
    with pytest.raises(ValueError):
        decision_ids(tok, "definitely not one token")
    with pytest.raises(ValueError):
        decision_ids(tok, ["<tool_call>", "<tool_call>"])
