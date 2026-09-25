"""The flash route for plain causal attention (seco._attention -> ChunkAttention, no prefix)
against an fp32 reference, at Qwen3.5's shapes (head_dim 256, grouped K/V heads)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (flash kernels)")


@pytest.mark.parametrize("heads, kv_heads, head_dim, length", [(8, 2, 256, 2048), (16, 16, 128, 1024)])
def test_flash_route_matches_reference(heads, kv_heads, head_dim, length):
    from palingenesis.seco_attention import chunk_attention

    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(1, heads, length, head_dim, device="cuda", generator=g, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, kv_heads, length, head_dim, device="cuda", generator=g, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, kv_heads, length, head_dim, device="cuda", generator=g, dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn(1, heads, length, head_dim, device="cuda", generator=g, dtype=torch.bfloat16)
    out = chunk_attention(q, k, v, None, 0, None, 8192)
    grads = torch.autograd.grad(out, (q, k, v), dout)

    rep = heads // kv_heads
    qf, kf, vf = (t.detach().float().requires_grad_() for t in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(
        qf, kf.repeat_interleave(rep, 1), vf.repeat_interleave(rep, 1), is_causal=True
    )
    ref_grads = torch.autograd.grad(ref, (qf, kf, vf), dout.float())
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
    for got, want in zip(grads, ref_grads):
        assert ((got.float() - want).norm() / want.norm()).item() < 2e-2
