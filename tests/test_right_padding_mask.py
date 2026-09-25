"""Right padding needs no attention mask: outputs at the real positions are the same without it.
The SFT loop relies on this to run plain causal (flash) attention."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
torch = pytest.importorskip("torch")

from test_seco import MODELS, VOCAB  # noqa: E402


@pytest.mark.parametrize("arch", MODELS)
def test_right_padded_rows_need_no_mask(arch):
    model = MODELS[arch]().eval()
    g = torch.Generator().manual_seed(0)
    lengths = [23, 17]
    ids = torch.zeros(2, 23, dtype=torch.long)
    mask = torch.zeros(2, 23, dtype=torch.long)
    for r, n in enumerate(lengths):
        ids[r, :n] = torch.randint(1, VOCAB, (n,), generator=g)
        mask[r, :n] = 1
    with torch.no_grad():
        masked = model(input_ids=ids, attention_mask=mask).logits
        unmasked = model(input_ids=ids, attention_mask=None).logits
    for r, n in enumerate(lengths):
        torch.testing.assert_close(unmasked[r, :n], masked[r, :n], rtol=1e-6, atol=1e-6)
