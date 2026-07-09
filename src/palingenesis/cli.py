"""CLI — the command interface for palingenesis.

Usage:
    palingenesis train --config configs/llama3_8b.yaml
    palingenesis diagnose --config configs/llama3_8b.yaml --mode pre
    palingenesis inspect --config configs/llama3_8b.yaml
    palingenesis profile --config configs/llama3_8b.yaml --gpu 80
    palingenesis version
"""

import sys

BANNER = r"""
                 _ _                            _
  _ __  __ _ _ _(_) |_ _  __ _ ___ _ _  ___ __(_)___
 | '_ \/ _` | '_| | | ' \/ _` / -_) ' \/ -_|_-< (_-<
 | .__/\__,_|_| |_|_|_||_\__, \___|_||_\___/__/_/__/
 |_|                     |___/
                                            v0.3.0
"""

HELP = f"""{BANNER}
  Fire-and-forget LLM fine-tuning with state-of-the-art defaults.

  Alias: pgs (short form)

Commands:
  train       Start training (use with torchrun for multi-GPU)
  autopilot   Autonomous training: profile, sweep LR, train, monitor, stop
  prepare     Score and filter data by difficulty (offline, uses model inference)
  prepare-multi  Prepare multiple sources with per-source scoring + MSFT allocation
  s0-tune     S0 Tuning: zero-overhead adaptation for hybrid models (Qwen3.5, FalconH1)
  diagnose    Run health diagnostics (pre/post/full)
  inspect     Visualize tokenization and masking
  validate    Validate masking across many samples
  profile     Estimate GPU memory usage
  monitor     Check status of a running training
  loss        Analyze loss curve from log file
  version     Show version info

Usage:
  pgs train --config configs/quickstart.yaml
  pgs autopilot --model Qwen/Qwen3.5-4B --dataset my_data.jsonl
  pgs prepare --model Qwen/Qwen3.5-4B --data traces.jsonl --strategy optimal

  # Preprocess driven by the SAME config as training (model/data/preprocess
  # sections are shared; output is parquet in preprocess.output_dir):
  pgs prepare --config configs/qwen35_4b/a100_80gb.yaml
  # then set preprocess.enabled=true (or pass --preprocess.enabled true) and
  # training picks up the prepared dataset automatically:
  pgs train --config configs/qwen35_4b/a100_80gb.yaml --preprocess.enabled true

For multi-GPU training, use torchrun:
  torchrun --nproc_per_node=8 -m palingenesis.train --config configs/qwen35_4b/a100_80gb.yaml
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(HELP)
        sys.exit(0)

    cmd = sys.argv[1]
    # Remove the command from argv for downstream parsers
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    match cmd:
        case "train":
            from palingenesis.train import main as train_main

            train_main()

        case "autopilot":
            from palingenesis.autopilot.run import main as autopilot_main

            autopilot_main()

        case "prepare":
            from palingenesis.prepare import main as prepare_main

            prepare_main()

        case "prepare-multi":
            _run_prepare_multi()

        case "s0-tune":
            _run_s0_tune()

        case "diagnose":
            _run_agent_tool("agent_tooling.diagnose")

        case "inspect":
            _run_agent_tool("agent_tooling.inspect_batch")

        case "validate":
            _run_agent_tool("agent_tooling.validate_masking")

        case "profile":
            # Translate --gpu to --gpu_memory_gb for the tool
            _run_agent_tool("agent_tooling.profile_memory")

        case "monitor":
            _run_agent_tool("agent_tooling.monitor_run")

        case "loss":
            _run_agent_tool("agent_tooling.check_loss")

        case "gradients":
            _run_agent_tool("agent_tooling.check_gradients")

        case "version":
            from palingenesis import __version__

            print(f"palingenesis {__version__}")
            print(f"  PyTorch: {_torch_version()}")
            print(f"  Transformers: {_transformers_version()}")
            print(f"  CUDA: {_cuda_info()}")
            print(f"  Liger Kernel: {_liger_version()}")

        case _:
            print(f"Unknown command: {cmd}")
            print("Run 'palingenesis --help' for usage.")
            sys.exit(1)


def _run_agent_tool(module_name: str):
    """Run an agent_tooling module's main() function.

    agent_tooling is normally installed alongside palingenesis (it ships in
    the wheel). For older installs or source checkouts, fall back to the
    source-tree root and the current directory.
    """
    import importlib
    from pathlib import Path

    # Fallback locations: source-tree root (editable/src layout) and cwd
    # (running `pgs` from a repo checkout with a site-packages install).
    for candidate in (Path(__file__).parent.parent.parent, Path.cwd()):
        if (candidate / "agent_tooling" / "__init__.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break

    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name and e.name.startswith("agent_tooling"):
            print(
                "Error: the 'agent_tooling' package is not available.\n"
                "It ships with palingenesis >= 0.3.0 wheels — reinstall with:\n"
                "  uv pip install -e '.[train]'   (from the repo root)\n"
                "or run the tool from a source checkout of the repository.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    mod.main()


def _run_prepare_multi():
    """CLI for multi-source data preparation with MSFT allocation."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Prepare multiple data sources with per-source scoring")
    parser.add_argument("--model", required=True, help="Model name/path for scoring")
    parser.add_argument("--sources", required=True, help="YAML file with source configurations")
    parser.add_argument("--output", default="./prepared", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--budget", type=int, help="Budget per source")
    parser.add_argument("--strategy", default="optimal")
    parser.add_argument("--hes", action="store_true", help="Also compute HES reasoning quality scores")
    parser.add_argument("--hes_top_k", type=float, default=0.5, help="Top-k%% for HES")
    args = parser.parse_args()

    from pathlib import Path

    with Path(args.sources).open() as f:
        sources_config = yaml.safe_load(f)

    sources = sources_config if isinstance(sources_config, list) else sources_config.get("sources", [])

    from palingenesis.prepare import prepare_multi_source

    prepare_multi_source(
        model_name=args.model,
        sources=sources,
        output_path=args.output,
        max_seq_length=args.max_seq_length,
        budget_per_source=args.budget,
        strategy=args.strategy,
        compute_hes=args.hes,
        hes_top_k_pct=args.hes_top_k,
    )


def _run_s0_tune():
    """CLI for S0 Tuning of hybrid recurrent-attention models."""
    import argparse

    parser = argparse.ArgumentParser(description="S0 Tuning: zero-overhead PEFT for hybrid models")
    parser.add_argument("--model", required=True, help="Hybrid model name/path (e.g., Qwen/Qwen3.5-4B)")
    parser.add_argument("--data", required=True, help="Training data (JSONL with messages)")
    parser.add_argument("--output", default="./s0_states.pt", help="Output path for learned states")
    parser.add_argument("--alpha", type=float, default=0.07, help="State scaling (0.07 for Qwen3.5, 0.65 for FalconH1)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 penalty on states")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--messages_field", default="messages", help="Messages field in data")
    args = parser.parse_args()

    import json
    import logging
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    # Load data
    data_path = Path(args.data)
    if data_path.suffix == ".jsonl":
        with data_path.open() as f:
            samples = [json.loads(line) for line in f if line.strip()]
    else:
        with data_path.open() as f:
            samples = json.load(f)
    logger.info(f"Loaded {len(samples)} training samples")

    # Load model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Create S0 trainer
    from palingenesis.s0_tuning import S0Trainer

    device = next(model.parameters()).device
    trainer = S0Trainer(model, alpha=args.alpha, weight_decay=args.weight_decay, lr=args.lr, device=device)

    # Prepare batches
    from palingenesis.data import IGNORE_INDEX

    batches = []
    for sample in samples:
        messages = sample.get(args.messages_field, [])
        if not messages:
            continue
        try:
            templated = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                return_dict=True,
                truncation=True,
                max_length=args.max_seq_length,
            )
            input_ids = torch.tensor(templated["input_ids"], dtype=torch.long).unsqueeze(0)
            mask_key = "assistant_masks" if "assistant_masks" in templated else "assistant_tokens_mask"
            assistant_mask = torch.tensor(templated[mask_key], dtype=torch.bool)
            labels = input_ids.clone().squeeze(0)
            labels[~assistant_mask] = IGNORE_INDEX
            attn = torch.ones_like(input_ids)
            batches.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attn,
                    "labels": labels.unsqueeze(0),
                }
            )
        except Exception:
            continue

    logger.info(f"Prepared {len(batches)} training batches")

    # Train
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in batches:
            loss = trainer.step(batch)
            epoch_loss += loss
        avg_loss = epoch_loss / max(len(batches), 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}: avg_loss={avg_loss:.4f}")

    # Save
    trainer.save(args.output)
    logger.info(f"S0 states saved to {args.output}")


def _torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except ImportError:
        return "not installed"


def _transformers_version() -> str:
    try:
        import transformers

        return transformers.__version__
    except ImportError:
        return "not installed"


def _cuda_info() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"{name} ({mem:.0f} GB)"
        return "not available"
    except Exception:
        return "unknown"


def _liger_version() -> str:
    try:
        import liger_kernel

        return getattr(liger_kernel, "__version__", "installed")
    except ImportError:
        return "not installed"


if __name__ == "__main__":
    main()
