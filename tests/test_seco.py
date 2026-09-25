"""SeCO / SpaCO (palingenesis.seco).

The central property: SeCO's chunk-wise forward/backward — one chunk's
activations at a time, gradients relayed through the cache — gives the SAME
loss and parameter gradients as one full forward/backward, for a pure attention
decoder (GPT-2) and for a Qwen3.5 hybrid (Gated DeltaNet + attention), at any
chunk size, with right-padded batches and under activation checkpointing.
"""

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from palingenesis.data import IGNORE_INDEX  # noqa: E402
from palingenesis.loss import shift_labels  # noqa: E402
from palingenesis.seco import chunkwise_forward_backward, seco_forward_backward  # noqa: E402

VOCAB = 97


def _gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(
        n_layer=3,
        n_head=2,
        n_embd=32,
        n_positions=256,
        vocab_size=VOCAB,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(cfg).double().train()


def _qwen35():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        max_position_embeddings=512,
        attention_dropout=0.0,
    )
    model = Qwen3_5ForCausalLM(cfg).double().train()
    # Random init leaves A_log/dt_bias at values that make the recurrence nearly
    # trivial; spread them so the state really carries information across chunks.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                p.uniform_(-1.0, 1.0)
    return model


MODELS = {"gpt2": _gpt2, "qwen35_hybrid": _qwen35}


def _batch(batch=1, seq=70, pad_to=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, VOCAB, (batch, seq), generator=g)
    labels = ids.clone()
    labels[:, :5] = IGNORE_INDEX  # a "prompt" that is not scored
    mask = torch.ones_like(ids)
    if pad_to:  # right padding, as the collator does
        pad = pad_to - seq
        ids = torch.cat([ids, torch.zeros(batch, pad, dtype=torch.long)], 1)
        labels = torch.cat([labels, torch.full((batch, pad), IGNORE_INDEX)], 1)
        mask = torch.cat([mask, torch.zeros(batch, pad, dtype=torch.long)], 1)
    return ids, labels, mask


def _full(model, ids, labels, mask):
    model.zero_grad()
    logits = model(input_ids=ids, attention_mask=mask).logits
    shifted = shift_labels(labels)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB).double(), shifted.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum"
    )
    loss = loss / (shifted != IGNORE_INDEX).sum()
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _chunked(model, ids, labels, **kw):
    model.zero_grad()
    result = chunkwise_forward_backward(model, ids, labels, **kw)
    return result, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _max_rel_err(a, b):
    assert a.keys() == b.keys()
    return max(float((a[n] - b[n]).norm() / b[n].norm().clamp(min=1e-30)) for n in b)


# Relative gradient tolerance. The models run in fp64, but the chunked CE works
# in fp32 and the torch Gated-DeltaNet kernel too, so the floor is fp32 noise:
# measured <= 6e-8 (GPT-2) and <= 1.7e-5 (hybrid; 6e-6 even with ONE chunk). For
# scale: dropping the cross-chunk relay (truncated backprop) gives ~0.9.
TOL = {"gpt2": 5e-7, "qwen35_hybrid": 5e-5}
LOSS_TOL = 1e-6


@pytest.mark.parametrize("arch", MODELS)
@pytest.mark.parametrize("chunk_size", [7, 16, 32, 70, 128])
def test_seco_equals_full_backprop(arch, chunk_size):
    model = MODELS[arch]()
    ids, labels, mask = _batch()
    full_loss, full_grads = _full(model, ids, labels, mask)
    result, grads = _chunked(model, ids, labels, chunk_size=chunk_size)
    assert result.num_chunks == -(-70 // chunk_size) and result.backpropagated == result.num_chunks
    assert abs(result.loss - full_loss) < LOSS_TOL * full_loss
    assert _max_rel_err(grads, full_grads) < TOL[arch]


@pytest.mark.parametrize("arch", MODELS)
def test_right_padded_batch(arch):
    model = MODELS[arch]()
    ids, labels, mask = _batch(batch=2, seq=50, pad_to=64)
    ids[1, 40:] = 0  # second row shorter: more padding
    labels[1, 40:] = IGNORE_INDEX
    mask[1, 40:] = 0
    _, full_grads = _full(model, ids, labels, mask)
    _, grads = _chunked(model, ids, labels, chunk_size=16)
    assert _max_rel_err(grads, full_grads) < TOL[arch]


@pytest.mark.parametrize("arch", MODELS)
@pytest.mark.parametrize("mode", ["full", "selective"])
def test_exact_under_activation_checkpointing(arch, mode):
    from palingenesis.kernels import apply_activation_checkpointing

    reference = MODELS[arch]()
    ids, labels, mask = _batch()
    _, full_grads = _full(reference, ids, labels, mask)
    model = MODELS[arch]()
    apply_activation_checkpointing(model, mode=mode)
    _, grads = _chunked(model, ids, labels, chunk_size=16)
    strip = lambda d: {n.replace("_checkpoint_wrapped_module.", ""): g for n, g in d.items()}  # noqa: E731
    assert _max_rel_err(strip(grads), full_grads) < TOL[arch]


def test_loss_denominator_scales_gradients():
    model = _gpt2()
    ids, labels, _ = _batch()
    _, g1 = _chunked(model, ids, labels, chunk_size=16)
    n = (shift_labels(labels) != IGNORE_INDEX).sum().item()
    _, g2 = _chunked(model, ids, labels, chunk_size=16, loss_denom=4 * n)
    assert _max_rel_err({k: 4 * v for k, v in g2.items()}, g1) < 1e-6


def test_single_chunk_is_plain_backprop():
    model = _qwen35()
    ids, labels, mask = _batch()
    _, full_grads = _full(model, ids, labels, mask)
    result, grads = _chunked(model, ids, labels, chunk_size=10_000)
    assert result.num_chunks == 1
    assert _max_rel_err(grads, full_grads) < TOL["qwen35_hybrid"]


# ── SpaCO ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("arch", MODELS)
def test_spaco_with_full_budget_is_seco(arch):
    model = MODELS[arch]()
    ids, labels, _ = _batch()
    _, seco = _chunked(model, ids, labels, chunk_size=16)
    result, spaco = _chunked(model, ids, labels, chunk_size=16, budget=5)  # k = 5
    assert result.backpropagated == 5
    assert _max_rel_err(spaco, seco) == 0.0  # identical computation


def test_spaco_single_chunk_has_no_relay():
    """t = 1: only the selected chunk's own loss is backpropagated, scaled by
    k/t (there is no relay into a chunk that is itself the last visited)."""
    model = _gpt2()
    ids, labels, _ = _batch()
    rng = random.Random(3)
    chosen = random.Random(3).sample(range(5), 1)[0]
    result, spaco = _chunked(model, ids, labels, chunk_size=16, budget=1, rng=rng)
    # Reference: the gradient of chunk `chosen`'s loss alone, with earlier
    # chunks' K/V treated as constants (no path back into them).
    model.zero_grad()
    lo, hi = 16 * chosen, min(16 * chosen + 16, 70)
    n = (shift_labels(labels) != IGNORE_INDEX).sum()
    with torch.no_grad():
        past = model(input_ids=ids[:, :lo], use_cache=True).past_key_values if lo else None
    logits = model(input_ids=ids[:, lo:hi], past_key_values=past, use_cache=True).logits
    loss = (
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB),
            shift_labels(labels)[:, lo:hi].reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        / n
    )
    loss.backward()
    expected = {k: p.grad for k, p in model.named_parameters() if p.grad is not None}
    assert _max_rel_err({n: g / 5 for n, g in spaco.items()}, expected) < TOL["gpt2"]  # k/t = 5
    # the reported loss is still the full-sequence loss (from the no-grad pass)
    full_loss, _ = _full(model, ids, labels, torch.ones_like(ids))
    assert abs(result.loss - full_loss) < LOSS_TOL * full_loss


def test_seco_wrapper_and_input_validation():
    model = _gpt2()
    ids, labels, _ = _batch()
    assert seco_forward_backward(model, ids, labels, chunk_size=32).num_chunks == 3
    with pytest.raises(ValueError, match="budget"):
        chunkwise_forward_backward(model, ids, labels, chunk_size=16, budget=0)


class _FixedSubset(random.Random):
    def __init__(self, subset):
        super().__init__(0)
        self.subset = subset

    def sample(self, population, k):
        return list(self.subset)


@pytest.mark.parametrize("arch", MODELS)
@pytest.mark.parametrize("budget,min_scale,min_cos", [(2, 0.95, 0.99), (4, 0.99, 0.9998)])
def test_spaco_expectation_approximates_the_gradient(arch, budget, min_scale, min_cos):
    """Exact expectation over every t-subset of k=5 chunks. It is not exactly the
    gradient (the paper's independence approximation), but close in scale and
    direction, and closer as t -> k."""
    import itertools

    model = MODELS[arch]()
    ids, labels, mask = _batch()
    _, true = _full(model, ids, labels, mask)
    flat = lambda g: torch.cat([g[n].flatten() for n in sorted(g)])  # noqa: E731
    subsets = list(itertools.combinations(range(5), budget))
    mean = sum(
        flat(_chunked(model, ids, labels, chunk_size=16, budget=budget, rng=_FixedSubset(s))[1]) for s in subsets
    ) / len(subsets)
    target = flat(true)
    scale = float(mean @ target / (target @ target))
    cos = float(mean @ target / (mean.norm() * target.norm()))
    assert min_scale < scale < 1.0 + 1e-6 and cos > min_cos


# ── config ─────────────────────────────────────────────────────────────────


def test_config_validation():
    from palingenesis.config import Config, ConfigError

    config = Config()
    config.memory.seco = True
    config.validate()
    for section, field, value, match in [
        ("data", "packing", True, "data.packing"),
        ("parallel", "context_parallel", True, "context_parallel"),
        ("memory", "gradient_release", True, "gradient_release"),
        ("plugins", "sym_noise", True, "sym_noise"),
        ("plugins", "deft", True, "plugins.deft"),
        ("memory", "seco_chunk_size", 0, "seco_chunk_size"),
        ("memory", "spaco_budget", -1, "spaco_budget"),
    ]:
        bad = Config()
        bad.memory.seco = True
        setattr(getattr(bad, section), field, value)
        with pytest.raises(ConfigError, match=match):
            bad.validate()


# ── every architecture: verification passes <=> gradients are exact ────────


from seco_archs import ARCHS, FP32_ONLY, build  # noqa: E402

from palingenesis.seco import verify_chunked_forward  # noqa: E402

# fp32 noise floor of the torch Gated-DeltaNet kernel (see TOL above)
_ARCH_TOL = {"qwen3_5": 5e-5, "qwen3_next": 5e-5}


def _is_cpu_kernel_gap(exc: Exception) -> bool:
    return "CPU" in str(exc) and "backend" in str(exc)


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("arch", ARCHS)
def test_architecture_exact_iff_verified(arch, attn):
    """For every architecture: if the model's chunked forward matches its full
    forward, SeCO's gradients equal full backprop; if not, the model is rejected
    (and SeCO would indeed be wrong on it)."""
    try:
        model = build(arch, attn=attn).eval()  # eval: no dropout, the relay alone is tested
    except Exception as exc:  # architecture absent from this transformers version
        pytest.skip(f"cannot build {arch}: {type(exc).__name__}")
    ids, labels, mask = _batch()
    try:
        verified = verify_chunked_forward(model) is not None
    except NotImplementedError as exc:
        if _is_cpu_kernel_gap(exc):
            pytest.skip("kernel unavailable on CPU")
        verified = False
    try:
        _, full_grads = _full(model, ids, labels, mask)
        _, grads = _chunked(model, ids, labels, chunk_size=16)
    except Exception as exc:
        if _is_cpu_kernel_gap(exc):
            pytest.skip("kernel unavailable on CPU")
        if not verified:
            return  # rejected, and indeed unable to run chunk-wise
        raise
    err = _max_rel_err(grads, full_grads)
    if verified:
        tol = _ARCH_TOL.get(arch, 1e-4 if arch in FP32_ONLY else 5e-6)
        assert err < tol, f"{arch}: verified but gradient error {err:.1e}"
    else:
        assert err > 1e-3, f"{arch}: rejected although SeCO is exact (error {err:.1e})"


def test_dropout_is_replayed_exactly():
    """With dropout active, SeCO must equal backprop through the same chunked
    stochastic forward (same masks): stage 2 replays each chunk's RNG state."""
    from transformers import DynamicCache, GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = (
        GPT2LMHeadModel(
            GPT2Config(
                n_layer=3,
                n_head=2,
                n_embd=32,
                n_positions=256,
                vocab_size=VOCAB,
                resid_pdrop=0.3,
                embd_pdrop=0.3,
                attn_pdrop=0.3,
            )
        )
        .double()
        .train()
    )
    ids, labels, _ = _batch()
    shifted = shift_labels(labels)
    n = (shifted != IGNORE_INDEX).sum()

    model.zero_grad()
    torch.manual_seed(7)
    cache, loss = DynamicCache(config=model.config), 0.0
    for lo in range(0, 70, 16):  # chunked forward WITH grad through the cache
        logits = model(input_ids=ids[:, lo : lo + 16], past_key_values=cache, use_cache=True).logits
        loss = (
            loss
            + torch.nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB),
                shifted[:, lo : lo + 16].reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            / n
        )
    loss.backward()
    expected = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}

    model.zero_grad()
    torch.manual_seed(7)
    chunkwise_forward_backward(model, ids, labels, chunk_size=16)
    got = {k: p.grad for k, p in model.named_parameters() if p.grad is not None}
    assert _max_rel_err(got, expected) < 5e-7


def test_rejects_a_model_whose_chunked_forward_differs():
    model = build("gpt2").eval()
    original = model.transformer.forward

    def drop_cache(*args, past_key_values=None, **kwargs):  # a model that ignores its cache
        return original(*args, past_key_values=None, **{**kwargs, "use_cache": False})

    model.transformer.forward = drop_cache
    with pytest.raises(NotImplementedError, match="chunked forward differs"):
        verify_chunked_forward(model)


@pytest.mark.parametrize("arch", ["cohere2", "gemma2_softcap", "granite"])
def test_output_head_reproduces_the_logit_transform(arch):
    from palingenesis.logits import output_head, verify_output_head

    model = build(arch).eval()
    head = output_head(model)
    assert head is not model.get_output_embeddings()  # a transform is applied
    verify_output_head(model, head, model.base_model)
    with pytest.raises(NotImplementedError, match="transforms the lm_head output"):
        verify_output_head(model, model.get_output_embeddings(), model.base_model)


def test_attention_path_is_correct_outside_seco():
    """Calling the model directly inside the chunk-attention context (no per-chunk
    bias registered) must still attend to the whole prefix."""
    from transformers import DynamicCache

    from palingenesis.seco import _BIASES, _chunk_attention

    model = build("llama", attn="sdpa").eval()
    ids, _, _ = _batch()
    with torch.no_grad(), _chunk_attention(model):
        _BIASES.clear()
        full = model(input_ids=ids).logits
        cache = DynamicCache(config=model.config)
        chunked = torch.cat(
            [
                model(input_ids=ids[:, lo : lo + 16], past_key_values=cache, use_cache=True).logits
                for lo in range(0, 70, 16)
            ],
            dim=1,
        )
    assert float((chunked - full).abs().max()) < 1e-10


@pytest.mark.parametrize("arch", ["llama", "gemma2_softcap", "cohere2"])
def test_scored_ce_sum_equals_cross_entropy_of_the_model_logits(arch):
    """Evaluation scores only labelled positions from hidden states; it must
    equal cross-entropy over the model's own logits (incl. logit transforms)."""
    from palingenesis.logits import scored_ce_sum

    model = build(arch).eval()
    ids, labels, mask = _batch(batch=2, seq=50, pad_to=64)
    shifted = shift_labels(labels)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB).double(), shifted.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum"
        )
    got, count = scored_ce_sum(model, ids, mask, shifted, torch.float64, chunk_bytes=VOCAB * 4 * 7)
    assert count == int((shifted != IGNORE_INDEX).sum())
    assert abs(got - float(expected)) < 1e-9 * abs(float(expected))


# ── K/V stores: no prefix copy, optional offload ─────────────────────────────


@pytest.mark.parametrize("arch", ["llama", "qwen3_5", "gemma3"])
@pytest.mark.parametrize("offload", [False, True])
def test_kv_store_path_is_used_and_exact(arch, offload, monkeypatch):
    import palingenesis.seco as seco_module

    calls = []
    original = seco_module.chunk_attention

    def counting(*args, **kwargs):
        calls.append(args[4])  # prefix length seen by the store path
        return original(*args, **kwargs)

    monkeypatch.setattr(seco_module, "chunk_attention", counting)
    monkeypatch.setattr(seco_module, "KV_BLOCK", 10)  # several prefix blocks per chunk
    model = build(arch, attn="sdpa").eval()
    ids, labels, mask = _batch()
    _, full_grads = _full(model, ids, labels, mask)
    _, grads = _chunked(model, ids, labels, chunk_size=16, kv_offload=offload)
    assert calls and max(calls) == 64  # the last chunk attended a 64-token prefix
    assert _max_rel_err(grads, full_grads) < _ARCH_TOL.get(arch, 5e-6)


def test_kv_offload_needs_sdpa():
    model = build("llama", attn="eager")
    ids, labels, _ = _batch()
    with pytest.raises(NotImplementedError, match="seco_kv_offload"):
        chunkwise_forward_backward(model, ids, labels, chunk_size=16, kv_offload=True)


def test_offload_budget_is_checked_before_allocating():
    """The estimate counts every full-attention layer's K and V for the whole
    sequence, and an impossible request is refused up front."""
    from palingenesis.seco import _offloaded_bytes
    from palingenesis.seco_attention import check_host_budget

    model = build("llama", attn="sdpa")
    cfg = model.config
    need = _offloaded_bytes(model.model, batch=2, seq_len=1000, cast=torch.bfloat16)
    expected = 2 * cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * cfg.head_dim * 1000 * 2
    assert need == expected
    check_host_budget(1 << 20)  # a megabyte is fine anywhere
    with pytest.raises(RuntimeError, match="seco_kv_offload would hold"):
        check_host_budget(1 << 50)  # a petabyte is not


def test_offload_budget_counts_only_attention_layers_of_a_hybrid():
    from palingenesis.seco import _offloaded_bytes

    model = build("qwen3_5", attn="sdpa")
    cfg = model.config
    full_layers = sum(1 for t in cfg.layer_types if t == "full_attention")
    need = _offloaded_bytes(model.model, batch=1, seq_len=512, cast=torch.bfloat16)
    assert need == 2 * full_layers * cfg.num_key_value_heads * cfg.head_dim * 512 * 2
    assert full_layers < cfg.num_hidden_layers  # the point of a hybrid
