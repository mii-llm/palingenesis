"""CLI — the command interface for palingenesis.

Usage:
    palingenesis train --config configs/llama3_8b.yaml
    palingenesis diagnose --config configs/llama3_8b.yaml --mode pre
    palingenesis inspect --config configs/llama3_8b.yaml
    palingenesis profile --config configs/llama3_8b.yaml --gpu 80
    palingenesis version
"""

import sys

from palingenesis import __version__

BANNER = rf"""
             _ _                           _
  _ __  __ _| (_)_ _  __ _ ___ _ _  ___ __(_)___
 | '_ \/ _` | | | ' \/ _` / -_) ' \/ -_|_-< (_-<
 | .__/\__,_|_|_|_||_\__, \___|_||_\___/__/_/__/
 |_|                 |___/
                                        v{__version__}
"""

HELP = f"""{BANNER}
  Fire-and-forget LLM fine-tuning with state-of-the-art defaults.

  Alias: pgs (short form)

Commands:
  init        Write an example config to start from (pgs init --list)
  train       Start training (use with torchrun for multi-GPU); off-policy distillation with distill.enabled
  distill     On-policy distillation: the student samples, its teachers score, KL update
  distill-cache  Precompute a teacher's top-k targets for off-policy distillation (HF or vLLM)
  distill-score  Annotate an mcqa distillation pool with the teacher's answers (filter before training)
  rl          Reinforcement learning: rollouts scored by rewards (math, code, tools, your own)
  harbor      Harbor task directories (Docker envs with tests): check oracle/nop, export RL rows
  env         Environment catalogue: list, info, test (oracle / degenerate audit), run (pass@k), rows, new
  scenario    Multi-role RL scenarios: generate rows, audit verifiers, export episodes, plan a cluster layout
  harness     Harness profiles for Harbor tasks: list, record a real harness's requests (capture)
  quantize    Post-training FP4 / FP8 quantisation into a vLLM checkpoint (NVFP4, MXFP4, FP8; compressed-tensors)
  merge       Weight-space merging: interpolate, linear, soup, task-arithmetic, model-stock, ram, materialize
  surgery     Margin surgery: a minimal weight edit restoring a decision's margins (run, margins, span)
  internals   Read the model's internals: capture, probes, lenses, compare checkpoints, steering, patching
  autopilot   Autonomous training: profile, sweep LR, train, monitor, stop
  prepare     Clean data (broken rows, duplicates), hold out an eval set, select a budget;
              raw text (--mode pretrain): duplicates, near-duplicates, n-gram decontamination
  prepare-multi  Prepare several sources, each on its own, with a row/token share manifest
  s0-tune     S0 Tuning: zero-overhead adaptation for hybrid models (Qwen3.5, FalconH1)
  diagnose    Run health diagnostics (pre/post/full)
  inspect     Visualize tokenization and masking
  validate    Validate masking across many samples
  profile     Estimate GPU memory usage
  fit         Measure the fastest memory settings (checkpointing, SeCO) that fit a GPU memory budget
  monitor     Check status of a running training
  loss        Analyze loss curve from log file
  rollouts    Read RL / distillation rollout dumps: finish reasons, rewards, tool errors, loops, slowest, worst
  eval        Evaluate a checkpoint on tasks (gsm8k, math500, mmlu_pro, ifeval, bfcl_simple, your YAML) through the
              training stack: k samples, thinking modes, results in the pgs compare format
  compare     Evaluation results with intervals; two checkpoints on the same items: paired tests, pass@k
  screen      RL pool difficulty: k rollouts per row with the RL config's rewards, keep rows with 0 < pass rate < 1
  pairs       Preference pairs: build contrastive twin pairs (minimal prompt edits), check a pair file
  version     Show version info

Usage:
  pgs init sft && pgs train --config sft.yaml
  pgs train --config configs/quickstart.yaml
  pgs distill --config configs/distill_math.yaml
  pgs rl --config configs/rl_math.yaml
  pgs autopilot --model Qwen/Qwen3.5-4B --dataset my_data.jsonl
  pgs prepare --model Qwen/Qwen3.5-4B --data traces.jsonl --budget 10000

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
        case "init":
            from palingenesis.examples import main as init_main

            sys.exit(init_main())

        case "train":
            from palingenesis.train import main as train_main

            train_main()

        case "distill":
            from palingenesis.opd.trainer import main as distill_main

            distill_main()

        case "rl":
            from palingenesis.rl.trainer import main as rl_main

            rl_main()

        case "harbor":
            from palingenesis.rl.envs.harbor import main as harbor_main

            sys.exit(harbor_main())

        case "env":
            from palingenesis.rl.catalog.cli import main as env_main

            sys.exit(env_main())

        case "scenario":
            from palingenesis.rl.scenarios.cli import main as scenario_main

            sys.exit(scenario_main())

        case "harness":
            from palingenesis.rl.envs.harness.capture import main as harness_main

            sys.exit(harness_main())

        case "quantize":
            from palingenesis.qat.compressed import main as quantize_main

            sys.exit(quantize_main())

        case "merge":
            from palingenesis.merge import main as merge_main

            sys.exit(merge_main())

        case "surgery":
            from palingenesis.surgery import main as surgery_main

            sys.exit(surgery_main())

        case "internals":
            from palingenesis.internals.cli import main as internals_main

            sys.exit(internals_main())

        case "distill-cache":
            from palingenesis.distill.build import main as distill_cache_main

            sys.exit(distill_cache_main())

        case "distill-score":
            from palingenesis.opd.score_pool import main as distill_score_main

            distill_score_main()

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
            _run_agent_tool("palingenesis.agent_tooling.diagnose")

        case "inspect":
            _run_agent_tool("palingenesis.agent_tooling.inspect_batch")

        case "validate":
            _run_agent_tool("palingenesis.agent_tooling.validate_masking")

        case "profile":
            # Translate --gpu to --gpu_memory_gb for the tool
            _run_agent_tool("palingenesis.agent_tooling.profile_memory")

        case "fit":
            from palingenesis.agent_tooling.fit_memory import main as fit_main

            sys.exit(fit_main())

        case "monitor":
            _run_agent_tool("palingenesis.agent_tooling.monitor_run")

        case "loss":
            _run_agent_tool("palingenesis.agent_tooling.check_loss")

        case "rollouts":
            _run_agent_tool("palingenesis.agent_tooling.rollouts")

        case "eval":
            from palingenesis.eval.cli import main as eval_main

            sys.exit(eval_main())

        case "compare":
            from palingenesis.agent_tooling.compare import main as compare_main

            sys.exit(compare_main())

        case "screen":
            from palingenesis.rl.screen import main as screen_main

            sys.exit(screen_main())

        case "pairs":
            from palingenesis.pairs import main as pairs_main

            sys.exit(pairs_main())

        case "gradients":
            _run_agent_tool("palingenesis.agent_tooling.check_gradients")

        case "version":
            print(f"palingenesis {__version__}")
            print(f"  PyTorch: {_torch_version()}")
            print(f"  Transformers: {_transformers_version()}")
            print(f"  GPU: {_gpu_info()}")
            print(f"  Liger Kernel: {_liger_version()}")

        case _:
            print(f"Unknown command: {cmd}")
            print("Run 'palingenesis --help' for usage.")
            sys.exit(1)


def _run_agent_tool(module_name: str):
    """Run a palingenesis.agent_tooling module's main() function."""
    import importlib

    importlib.import_module(module_name).main()


def _run_prepare_multi():
    """CLI for multi-source data preparation."""
    import argparse

    import yaml

    from palingenesis.prepare import STRATEGIES, prepare_multi_source

    parser = argparse.ArgumentParser(description="Prepare multiple data sources, each on its own")
    parser.add_argument("--model", required=True, help="Model name/path (its tokenizer; its weights only for scoring)")
    parser.add_argument("--sources", required=True, help="YAML file with source configurations")
    parser.add_argument("--output", default="./prepared", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--budget", type=int, help="Budget per source")
    parser.add_argument(
        "--strategy",
        default="random",
        choices=list(STRATEGIES),
        help="Every strategy but random is experimental and turns scoring on",
    )
    parser.add_argument("--score", action="store_true", help="Score every row with the model (response perplexity)")
    parser.add_argument("--min_ppl", type=float, default=0.0, help="Drop rows below this response PPL (0 = off)")
    parser.add_argument("--max_ppl", type=float, default=0.0, help="Drop rows above this response PPL (0 = off)")
    parser.add_argument("--batch_size", type=int, default=4, help="Max samples per scoring forward")
    parser.add_argument("--max_batch_tokens", type=int, default=16384, help="Padded-token cap per scoring forward")
    parser.add_argument("--hes", action="store_true", help="Also compute HES scores (experimental; turns scoring on)")
    parser.add_argument("--hes_top_k", type=float, default=0.5, help="Top-k%% for HES")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Processes that tokenize the rows (0 = the CPUs available, at most 8; 1 = in this process)",
    )
    parser.add_argument(
        "--decontaminate",
        nargs="*",
        default=[],
        help="Raw-text sources (mode: pretrain): evaluation files (JSONL, JSON, parquet, text); documents "
        "sharing a word n-gram with any string in them are dropped",
    )
    parser.add_argument("--decontaminate_ngram", type=int, default=13, help="Words per decontamination n-gram")
    parser.add_argument(
        "--decontaminate_fields",
        nargs="*",
        default=[],
        help="Columns of the evaluation files to read (default: every string column)",
    )
    parser.add_argument(
        "--near_dedupe",
        type=float,
        default=0.0,
        help="Raw-text sources: drop near-duplicate documents at or above this Jaccard similarity (0 = off)",
    )
    parser.add_argument(
        "--decontaminate_shared_share",
        type=float,
        default=-1.0,
        help="N-grams in more than this share of an evaluation file's rows are not matched (0 = match all; "
        "default: 0.2, or off with --decontaminate_fields)",
    )
    parser.add_argument("--min_chars", type=int, default=0, help="Raw-text sources: drop documents shorter than this")
    args = parser.parse_args()

    from pathlib import Path

    with Path(args.sources).open() as f:
        sources_config = yaml.safe_load(f)

    sources = sources_config if isinstance(sources_config, list) else sources_config.get("sources", [])

    prepare_multi_source(
        model_name=args.model,
        sources=sources,
        output_path=args.output,
        max_seq_length=args.max_seq_length,
        budget_per_source=args.budget,
        strategy=args.strategy,
        compute_hes=args.hes,
        hes_top_k_pct=args.hes_top_k,
        score=args.score,
        min_ppl=args.min_ppl,
        max_ppl=args.max_ppl,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        num_workers=args.num_workers,
        decontaminate=args.decontaminate,
        decontaminate_ngram=args.decontaminate_ngram,
        decontaminate_fields=args.decontaminate_fields,
        decontaminate_shared_share=args.decontaminate_shared_share,
        near_dedupe=args.near_dedupe,
        min_chars=args.min_chars,
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
            logger.info(f"Epoch {epoch + 1}/{args.epochs}: avg_loss={avg_loss:.4f}")

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


def _gpu_info() -> str:
    try:
        from palingenesis.accelerator import describe

        return describe()
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
