"""Shared fixtures of the off-policy distillation tests: tiny Llama students / teachers saved to disk
with the GPT-2 tokenizer (cached locally by the test suite) and a Qwen3.5-style chat template."""

import json
import pathlib

import torch

TEMPLATE = (pathlib.Path(__file__).parent / "fixtures" / "qwen3_5_chat_template.jinja").read_text()


def tokenizer():
    from transformers import AutoTokenizer

    try:  # the local cache first: no Hub request (and no failure when the network is down)
        tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        tok = AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = TEMPLATE
    tok.pad_token = tok.eos_token
    return tok


def tiny_llama(vocab: int, hidden: int = 32, layers: int = 2, seed: int = 0, tie: bool = False):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=2 * hidden,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            tie_word_embeddings=tie,
        )
    )
    model.config.use_cache = False
    return model


def save_model(path, tok, **kw):
    model = tiny_llama(len(tok), **kw)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return model


ROWS = [
    {
        "messages": [
            {"role": "user", "content": f"What is {i} plus {i + 1}?"},
            {"role": "assistant", "content": f"The answer is {2 * i + 1}, since {i} + {i + 1} = {2 * i + 1}."},
        ],
        "hint": f"The sum is {2 * i + 1}.",
    }
    for i in range(24)
]


def write_rows(path, rows=ROWS):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def cpu_config(tmp_path, student_dir, data_path, steps: int = 4):
    """A Config that runs the trainer's loop on the CPU (fp32, no compile, no trackers)."""
    from palingenesis.config import Config

    c = Config()
    c.model.name_or_path = str(student_dir)
    c.model.torch_dtype = "float32"
    c.model.compile = False
    c.model.use_liger_kernel = False
    c.model.attn_implementation = "sdpa"
    c.data.dataset = data_path
    c.data.streaming = False
    c.data.max_seq_length = 128
    c.data.num_workers = 0
    c.data.length_group_buffer = 0
    c.train.output_dir = str(tmp_path / "out")
    c.train.max_steps = steps
    c.train.per_device_batch_size = 2
    c.train.gradient_accumulation_steps = 2
    c.train.learning_rate = 1e-3
    c.train.min_learning_rate = 1e-4
    c.train.warmup_ratio = 0.0
    c.train.bf16 = False
    c.train.gradient_checkpointing = "none"
    c.train.spike_detection = False
    c.train.save_steps = 0
    c.logging.use_wandb = False
    c.logging.use_trackio = False
    c.parallel.fsdp = False
    return c


def run_cpu(config, plugins=None):
    """The trainer's _train_on on the CPU (train() itself requires a GPU)."""
    from palingenesis.train import _train_on

    run = {"tracker": None}
    try:
        return _train_on(config, run, plugins, 0, 1, torch.device("cpu"))
    finally:
        if run["tracker"] is not None:
            run["tracker"].finish(exit_code=0)
