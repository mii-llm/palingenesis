"""Tiny random models of many architectures, for SeCO exactness tests.

Each entry is (model_type, config overrides). Sizes are minimal; sliding windows
(24) are smaller than the test sequence so windows cross chunk boundaries.
"""

import torch

VOCAB = 97
COMMON = dict(
    vocab_size=VOCAB,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=512,
    attention_dropout=0.0,
    tie_word_embeddings=False,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)
WINDOW = 24

ARCHS = {
    "llama": ("llama", {}),
    "mistral": ("mistral", {"sliding_window": None}),
    "mistral_sliding": ("mistral", {"sliding_window": WINDOW}),
    "qwen2": ("qwen2", {}),
    "qwen3": ("qwen3", {}),
    "qwen3_moe": ("qwen3_moe", {"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32}),
    "qwen3_5": (
        "qwen3_5_text",
        {
            "layer_types": ["linear_attention", "linear_attention", "full_attention", "linear_attention"],
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        },
    ),
    "qwen3_next": (
        "qwen3_next",
        {
            "layer_types": ["linear_attention", "linear_attention", "full_attention", "linear_attention"],
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
        },
    ),
    "gemma2": ("gemma2", {"sliding_window": WINDOW}),
    "gemma3": (
        "gemma3_text",
        {
            "sliding_window": WINDOW,
            "sliding_window_pattern": 2,
            "layer_types": ["sliding_attention", "full_attention"] * 2,
        },
    ),
    "lfm2": (
        "lfm2",
        {
            "layer_types": ["conv", "conv", "full_attention", "conv"],
            "conv_L_cache": 3,
            "block_ff_dim": 128,
            "norm_eps": 1e-5,
        },
    ),
    "gpt_oss": (
        "gpt_oss",
        {
            "sliding_window": WINDOW,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "layer_types": ["sliding_attention", "full_attention"] * 2,
        },
    ),
    "phi3": ("phi3", {}),
    "olmo2": ("olmo2", {}),
    "smollm3": ("smollm3", {}),
    "granite": ("granite", {"logits_scaling": 4.0}),
    "cohere2": ("cohere2", {"sliding_window": WINDOW, "logit_scale": 0.25}),
    "gemma2_softcap": ("gemma2", {"sliding_window": WINDOW, "final_logit_softcapping": 5.0}),
    "mixtral": ("mixtral", {"num_local_experts": 4, "num_experts_per_tok": 2}),
    "glm4": ("glm4", {}),
    "starcoder2": ("starcoder2", {"sliding_window": WINDOW}),
    "olmo3": ("olmo3", {"sliding_window": WINDOW}),
    "exaone4": ("exaone4", {"sliding_window": WINDOW}),
    "gpt_neox": ("gpt_neox", {}),
    "bloom": ("bloom", {}),
    "gpt2": ("gpt2", {}),
    # Mamba hybrids
    "bamba": (
        "bamba",
        {
            "attn_layer_indices": [2],
            "mamba_n_heads": 8,
            "mamba_d_head": 16,
            "mamba_d_state": 16,
            "mamba_n_groups": 1,
            "mamba_chunk_size": 16,
        },
    ),
    "falcon_h1": (
        "falcon_h1",
        {
            "mamba_n_heads": 8,
            "mamba_d_head": 16,
            "mamba_d_ssm": 128,
            "mamba_d_state": 16,
            "mamba_n_groups": 1,
            "mamba_chunk_size": 16,
        },
    ),
    "jamba": (
        "jamba",
        {
            "attn_layer_period": 2,
            "attn_layer_offset": 1,
            "expert_layer_period": 4,
            "num_experts": 2,
            "mamba_d_state": 16,
            "mamba_dt_rank": 8,
        },
    ),
    "nemotron_h": (
        "nemotron_h",
        {
            "hybrid_override_pattern": "M*M-",
            "mamba_num_heads": 8,
            "mamba_head_dim": 16,
            "ssm_state_size": 16,
            "n_groups": 1,
            "chunk_size": 16,
        },
    ),
    "granitemoehybrid": (
        "granitemoehybrid",
        {
            "layer_types": ["mamba", "attention", "mamba", "mamba"],
            "mamba_n_heads": 8,
            "mamba_d_head": 16,
            "mamba_d_state": 16,
            "mamba_n_groups": 1,
            "mamba_chunk_size": 16,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
        },
    ),
}


# MoE layers use grouped matmul kernels without float64 support: build them in fp32.
FP32_ONLY = {"qwen3_moe", "qwen3_next", "gpt_oss", "mixtral", "jamba", "granitemoehybrid"}


def build(name: str, dtype=torch.float64, attn: str = "eager"):
    from transformers import AutoConfig, AutoModelForCausalLM

    model_type, extra = ARCHS[name]
    if name in FP32_ONLY and dtype == torch.float64:
        dtype = torch.float32
    cfg = AutoConfig.for_model(model_type, **{**COMMON, **extra})
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation=attn)
    with torch.no_grad():
        # Random init leaves recurrence gates near-trivial; spread them so state
        # really carries information across chunks.
        for pname, p in model.named_parameters():
            if pname.endswith(("A_log", "dt_bias")):
                p.uniform_(-1.0, 1.0)
    return model.to(dtype).train()
