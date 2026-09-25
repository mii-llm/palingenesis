"""Training configuration — flat, typed, YAML + CLI.

STATUS convention for features:
  # STATUS: proven     — Independently reproduced, ablated at scale, safe default
  # STATUS: validated  — Paper-backed + tested in this codebase, not externally reproduced
  # STATUS: experimental — Single-paper, limited testing, use with monitoring
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass(slots=True)
class ModelConfig:
    name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    trust_remote_code: bool = True
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = "sdpa"
    use_liger_kernel: bool = True  # used when compile is false (compile fuses the same ops)
    compile: bool = True
    compile_backend: str = "inductor"
    compile_mode: str = "default"  # "default", "reduce-overhead", or "max-autotune"


@dataclass(slots=True)
class DataConfig:
    # Single dataset mode (backward compatible)
    dataset: str = "HuggingFaceH4/ultrachat_200k"
    dataset_split: str = "train_sft"
    streaming: bool = True
    max_seq_length: int = 8192
    messages_field: str = "messages"
    # Row field with the tool definitions (a list, or a JSON string), passed to the
    # chat template as `tools=`: the template renders them into the system prompt the
    # way the model sees them at inference. Rows without it render without tools.
    tools_field: str = "tools"
    num_workers: int = 4
    packing: bool = False
    # Without packing, batches are padded to their longest sample. Length-grouped
    # batching buffers N samples, sorts by length, and emits batch-aligned groups
    # so padding (= wasted FLOPs) collapses to the within-group spread.
    # 0 disables. Auto-disabled for curriculum-ordered prepared data.
    length_group_buffer: int = 512
    seed: int = 42
    # Multi-dataset mode: list of source dicts
    # Each: {dataset, split, weight, mode("sft"|"pretrain"), messages_field|text_field}
    sources: list = field(default_factory=list)
    # ECHO-style observation loss: include tool/environment outputs in training
    # Paper: "Terminal Agents Learn World Models for Free" (ICML 2026, arxiv:2605.24517)
    # When true, tool/observation role tokens get loss (not just assistant)
    # Combined with DEFT: model naturally learns more from surprising observations
    # Effect: model becomes a world model, predicting tool behavior internally
    include_observations: bool = False
    # Train on reasoning traces (<think>...</think> blocks / reasoning_content).
    # true (default): reasoning tokens get loss — required for distilling
    #   reasoning behavior from traces (the whole point of reasoning datasets).
    # false: only the post-reasoning response gets loss (use when traces are
    #   low quality and you only want the final-answer style).
    train_on_reasoning: bool = True
    # Delimiters of reasoning baked into the assistant content, e.g. ["[THINK]", "[/THINK]"].
    # A LEADING block is the turn's reasoning (rendered once, by the template); tags later in
    # the text are text. null: the chat template's own delimiters (detected), else
    # <think></think>. Masking always uses the template's, so data written with another
    # model's tags is converted to this model's format. Overridable per source.
    think_tags: list | None = None
    # Chat-template kwargs for every row, e.g. {enable_thinking: true}. A row's own
    # `chat_template_kwargs` column overrides them key by key: thinking and non-thinking
    # rows mix in one dataset. Overridable (merged) per source.
    chat_template_kwargs: dict = field(default_factory=dict)
    # Per-turn loss scaling for multi-turn conversations
    # "uniform": all turns get equal weight (default, standard SFT)
    # "progressive": later turns get more weight (w = (turn_idx/total)^0.5)
    #   Rationale: later turns contain error recovery, iteration, harder reasoning
    # "last_heavy": final turn gets 2x weight, others 1x
    #   Rationale: final answer quality matters most
    turn_scaling: str = "uniform"
    # Train ONLY on the final assistant turn; mask all earlier assistant turns.
    # Use when earlier assistant turns are a FIXED context you must not fit — e.g.
    # n-shot MCQA prompts where the exemplar answers are constant and only the final
    # answer is the target. For single-turn data this is a no-op. Overridable per
    # source via the source dict.
    # Phase-neutral: in training it selects which turns get loss; in eval sources it
    # selects which turns are scored. (Named without a train_/eval_ prefix on purpose.)
    last_turn_only: bool = False
    # Validation / evaluation
    eval_dataset: str = ""  # HF dataset or path for validation (empty = no eval)
    eval_split: str = "test"  # Split to use for evaluation
    eval_samples: int = 200  # Number of eval samples (fixed subset for speed)
    eval_every: int = 100  # Evaluate every N optimizer steps
    # Multi-eval: separate eval sets per capability dimension (arxiv:2603.21606 improved)
    # When defined, replaces single eval_dataset for best-model tracking AND MSFT signals.
    # Each: {name, dataset, split, weight, samples, regression_floor, messages_field}
    eval_sources: list = field(default_factory=list)
    # Pretraining replay: mix generic pretraining data during SFT (arxiv:2603.04964)
    # Surprising finding: replaying pretraining data IMPROVES target task, not just prevents forgetting
    # Recommended: 5-15% of training tokens from generic data
    # Set to empty string to disable, or path/HF dataset for generic corpus
    pretrain_replay_dataset: str = ""
    pretrain_replay_weight: float = 0.1  # 10% of training tokens from replay data
    # MSFT per-source adaptive weight scheduling (arxiv:2603.21606, improved)
    # When sources are defined, track per-source validation loss and dynamically
    # DECAY (never exclude) weights of overfitting sources. Weight decays toward
    # a floor of 10% original (ensures continued exposure for anti-forgetting).
    msft_tracking: bool = False  # Enable adaptive per-source weight scheduling
    msft_eval_every: int = 50  # Check per-source val loss every N steps
    msft_decay_factor: float = 0.7  # Weight decay multiplier when overfitting
    msft_recovery_factor: float = 1.15  # Weight recovery multiplier when improving
    msft_floor_ratio: float = 0.1  # Minimum weight as fraction of original (never zero)
    # Pre-tokenized cache: materialize the fully-assembled (tokenized → masked → mixed
    # → packed) training stream to disk once, then load tensors directly on later runs
    # (skips per-step tokenization AND makes the exact step count a cheap read). A
    # fingerprint over tokenizer/template/seqlen/sources/masking invalidates a stale
    # cache and triggers an automatic rebuild. Incompatible with msft_tracking (it
    # changes the stream during training).
    pretokenize: bool = False
    pretokenize_path: str = "./pretokenized"


@dataclass(slots=True)
class TrainConfig:
    output_dir: str = "./checkpoints"
    resume_from: str | None = None  # path to checkpoint dir, or "auto" to find latest
    epochs: int = 1
    max_steps: int = -1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    # Batch ramp: start with smaller effective batch, increase late in training
    # From arxiv:2602.14208: for hard tasks, small batch early → large batch late
    # STATUS: experimental — single paper, not ablated with DEFT/gradient_release
    ga_ramp_start: int = 0
    learning_rate: float = 2e-5
    min_learning_rate: float = 2e-6
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    lr_scheduler: Literal["cosine", "linear", "constant", "power_decay", "wsd"] = "cosine"
    optimizer: str = "adamw"  # "adamw", "muon", "adamw8bit", "lion8bit", "paged_adamw8bit"
    seed: int = 42
    save_steps: int = 500
    save_final: bool = True  # write the final model at the end (off for dry runs such as `pgs profile --measure`)
    logging_steps: int = 1
    bf16: bool = True
    gradient_checkpointing: Literal["full", "selective", "none"] = "selective"  # STATUS: proven
    spike_detection: bool = True  # STATUS: validated — ZClip-inspired adaptive spike skipping
    spike_z_threshold: float = 5.0  # z-score threshold (higher = fewer skips)
    adagc: bool = False  # STATUS: experimental — AdaGC per-tensor adaptive gradient clipping (ICML 2026)
    adagc_lambda: float = 1.5  # Relative clipping threshold (1.5 = paper default)
    adagc_beta: float = 0.95  # EMA decay for per-tensor norm tracking
    ema: bool = False  # STATUS: proven — EMA of weights (TMLR 2024, widely reproduced)
    ema_decay: float = 0.999  # EMA decay factor (0.999 = ~1000 step window, 0.9999 = ~10000)
    ema_every: int = 10  # Update EMA every N steps (reduces CPU↔GPU overhead)
    base_merge: bool = False  # STATUS: experimental — Periodic merge-back with base model (SFA, arxiv:2501.05559)
    base_merge_ratio: float = 0.1  # How much base to mix in: θ = (1-r)*θ_current + r*θ_base
    base_merge_every: int = 500  # Merge every N steps (set to save_steps for natural alignment)
    base_merge_method: str = "lerp"  # "lerp" (linear) or "slerp" (spherical, preserves weight norms)
    adamc: bool = False  # STATUS: validated — AdamC: corrected WD for normalized layers (arxiv:2506.02285)
    llrd_decay: float = 1.0  # STATUS: proven — Layer-wise LR decay (1.0=off, 0.9=standard, 0.85=aggressive)
    freeze_non_attention: bool = False  # STATUS: validated — Hybrid models: freeze DeltaNet/SSM (arxiv:2604.22127)
    # Advanced optimizer wrappers
    hyperball: bool = False  # STATUS: experimental — Norm-constrained optimization (arxiv:2606.16899)
    # Hyperball's angular step: each update moves an attention/MLP matrix by this
    # fraction of its (fixed) Frobenius norm, scaled by the LR schedule. 0 = calibrate
    # each matrix to the relative step its base optimizer's first update made (for
    # Adam/Lion exactly learning_rate / rms(W)). learning_rate itself stays the base
    # optimizer's rate for embeddings, norms, biases and the head.
    hyperball_lr: float = 0.0
    mona: bool = False  # STATUS: experimental — MONA curvature-aware acceleration (arxiv:2605.26842)
    mona_beta_a: float = 0.975  # MONA acceleration EMA decay
    mona_lite: bool = True  # MONA-Lite: bf16 buffers + streaming (75% overhead reduction)


@dataclass(slots=True)
class ParallelConfig:
    fsdp: bool = True
    context_parallel: bool = False  # enable for multi-GPU long sequences
    cp_rotate_method: Literal["allgather", "alltoall"] = "allgather"
    cpu_offload: bool = False
    reshard_after_forward: bool = True


@dataclass(slots=True)
class MemoryConfig:
    """Memory optimizations for ultra-long sequences."""

    chunked_loss: bool = True  # STATUS: proven — never OOMs on large vocab
    loss_num_chunks: int = 8  # split CE into N chunks along seq dim
    float32_matmul_precision: Literal["highest", "high", "medium"] = "high"
    float8_training: bool = False  # STATUS: validated — enable float8 (H100+ only, 1.2-1.5x speed)
    # STATUS: experimental — Gradient Release (FORGE, arxiv:2606.22932)
    # Eliminates the gradient buffer entirely — only 1 param's gradient lives at a time.
    # Saves ~16 GB for 8B model. ONLY works when gradient_accumulation_steps = 1.
    # Incompatible with: Muon, gradient accumulation, GA ramp, global grad clipping.
    # Compatible with: AdamW, Lion, per-tensor AdaGC, selective_diff.
    gradient_release: bool = False
    # STATUS: validated — Selective Differentiation (arxiv:2404.12406)
    # Skip activation saving for frozen layers. Zero accuracy impact.
    selective_diff: bool = True
    # STATUS: validated — SeCO chunk-wise optimisation (arxiv:2505.16710)
    # Long sequences: forward in chunks of `seco_chunk_size` tokens with a cache,
    # then backprop chunk by chunk in reverse, relaying gradients through the
    # cache. Only one chunk's activations are alive, so activation memory is set
    # by the chunk size, not the sequence length. Exact gradients (tested equal to
    # full backprop, incl. Qwen3.5 hybrids) for one extra no-grad forward (~+33%).
    seco: bool = False
    seco_chunk_size: int = 4096
    # Keep the full-attention K/V (and recurrent start states) in pinned CPU memory,
    # streamed to the GPU block by block: GPU memory then grows only with the K/V
    # gradient. Exact; costs PCIe transfers. Needs attn_implementation: sdpa.
    seco_kv_offload: bool = False
    # SpaCO: backprop only this many random chunks per sequence (0 = SeCO, exact).
    # A stochastic gradient estimate; cuts backward compute on very long inputs.
    spaco_budget: int = 0


@dataclass(slots=True)
class PluginsConfig:
    """Research-backed training plugins (opt-in, torch.compile compatible)."""

    sym_noise: bool = False  # STATUS: proven — Symmetric noisy embeddings (ICLR 2024 + NeurIPS 2025)
    sym_noise_alpha: float = 5.0  # Noise magnitude (default matches NEFTune paper)
    info_sft: bool = False  # STATUS: validated — Information-aware token weighting (arxiv:2605.14967)
    info_sft_pbar: float = 0.93  # Calibration constant (stable at 0.93 across models)
    dft: bool = False  # STATUS: validated — Dynamic Fine-Tuning (Wu et al., 2025)
    cadft: bool = False  # STATUS: experimental — Compatibility-Aware DFT (arxiv:2606.11206)
    cadft_beta: float = 1.0  # Compatibility sensitivity (1.0 = paper default)
    deft: bool = False  # STATUS: validated — DEFT: Dynamic Entropy Fine-Tuning (arxiv:2602.11424)
    schedule_free: bool = False  # STATUS: proven — Schedule-Free AdamW (NeurIPS 2025)
    pre_rl: bool = False  # STATUS: validated — Pre-RL mode: preserve diversity for GRPO/DPO (arxiv:2605.29303)
    pre_rl_entropy_coeff: float = 0.1  # Entropy bonus weight (higher = more diverse)
    pre_rl_kl_coeff: float = 0.5  # KL penalty weight (higher = less drift from base)


@dataclass(slots=True)
class PreprocessConfig:
    """Offline data preparation (scoring, filtering, selection) — `pgs prepare --config`.

    Reuses model.name_or_path, data.dataset, data.dataset_split, data.messages_field
    and data.max_seq_length from the same config used for training, so preprocess
    and training can never drift apart.

    When `enabled: true`, training automatically loads the prepared dataset from
    `output_dir` (instead of data.dataset) and preserves curriculum ordering if
    strategy == "curriculum".
    """

    enabled: bool = False  # train-time: auto-use prepared output from output_dir
    output_dir: str = "./prepared"
    format: Literal["parquet", "jsonl"] = "parquet"
    max_samples: int = 0  # cap samples read from the raw dataset (0 = all)
    budget: int = 0  # samples to keep after scoring/filtering (0 = all)
    # Reserve N samples as a held-out eval set (written to eval_data.parquet,
    # EXCLUDED from the training selection). Training auto-uses it when
    # data.eval_dataset is empty — a true same-distribution holdout.
    eval_holdout: int = 0
    min_ppl: float = 1.5  # outlier filter lower bound
    max_ppl: float = 500.0  # outlier filter upper bound (<=0 disables; useful for OOD/multilingual data)
    filter_score: Literal["response", "full"] = "response"  # filter assistant tokens by default
    strategy: str = "optimal"  # optimal | balanced | medium_focus | curriculum | hard_focus | flow | random
    batch_size: int = 4  # max samples per scoring forward (length-sorted padded batches)
    # Padded-token cap per scoring forward (logits are B×S×V, so THIS bounds
    # memory, not batch_size). 16384 ≈ 5GB of bf16 logits for a 150K vocab;
    # on an 80GB GPU with a 4B model, 32768–49152 is safe and faster.
    max_batch_tokens: int = 16384
    hes: bool = False  # also compute HES reasoning-quality scores (slower)
    hes_top_k_pct: float = 0.5


@dataclass(slots=True)
class DPOConfig:
    """Preference optimisation (DPO and variants) — see docs/dpo.md.

    When enabled, `data.dataset` (and `data.eval_dataset`) hold preference pairs
    instead of SFT conversations; everything else (optimizer, schedule, FSDP,
    checkpointing, chat-template masking) is shared with SFT.
    """

    enabled: bool = False
    # sigmoid (DPO) | hinge (SLiC) | ipo | robust | sigmoid_norm (length-normalised)
    loss_type: str = "sigmoid"
    beta: float = 0.1
    # Assumed preference-label noise; only used by loss_type=robust (must be < 0.5).
    label_smoothing: float = 0.0
    # LD-DPO (arxiv:2409.06411): weight of the longer answer's tail beyond the
    # length both answers share. 1.0 = off (plain DPO); the paper uses ~0.5.
    ld_alpha: float = 1.0
    # Adds sft_weight × mean NLL over chosen tokens (RPO, arxiv:2404.19733). Anchors the
    # chosen answer so the margin cannot grow by only pushing rejected down.
    sft_weight: float = 0.0
    # Frozen reference policy. Empty = the model being trained, as loaded
    # (model.name_or_path) — the standard choice.
    reference_model: str = ""
    # Row fields (conversational preference format). An empty/missing prompt means chosen and
    # rejected are full conversations (implicit prompt).
    prompt_field: str = "prompt"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    # Chosen answers are never truncated (the pair is dropped). Rejected answers
    # longer than data.max_seq_length are truncated when true (suits degenerate,
    # looping rejections), dropped when false.
    truncate_rejected: bool = True
    # Zero every dropout in the policy. Otherwise the policy's
    # log-probs are noisy against a deterministic reference.
    disable_dropout: bool = True


@dataclass(slots=True)
class LoggingConfig:
    project: str = "palingenesis"
    run_name: str | None = None
    use_wandb: bool = True
    use_trackio: bool = True
    log_grad_norm: bool = True
    # Health monitor cadence. Tier 2 (~10ms): grad cosine sim, GNS, CUDA memory.
    # Tier 3 (~200ms): weight norms, stable rank, weight drift.
    # Must be multiples of train.logging_steps or they'll fire less often than set.
    health_tier2_every: int = 10
    health_tier3_every: int = 100
    # RL-readiness monitoring (arxiv:2606.18487, 2606.09932)
    # Enable if you plan to run GRPO/RL after this SFT stage.
    # Monitors output entropy and warns if it collapses (predicts RL failure).
    rl_readiness: bool = False
    rl_entropy_floor: float = 1.0  # Warn if mean output entropy drops below this


@dataclass(slots=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with Path(path).open() as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: expected a mapping of sections (model:, data:, train:, ...).")
        config = cls()
        for section_name, section_data in raw.items():
            section = _section(config, section_name, where=str(path))
            if section_data is None:
                continue
            if not isinstance(section_data, dict):
                raise ConfigError(f"{path}: `{section_name}:` must be a mapping of options.")
            for key, value in section_data.items():
                _set_option(section, section_name, key, value, where=str(path))
        return config

    @classmethod
    def from_cli(cls, args: list[str] | None = None) -> "Config":
        """Parse the config file and --section.field value overrides.

        The config file may be passed either as ``--config file.yaml`` or as a
        bare positional ``file.yaml`` / ``file.yml`` path (so both
        ``pgs train --config cpt.yaml`` and ``pgs train cpt.yaml`` work). If no
        config file is found, the built-in demo defaults are used and a loud
        warning is emitted — silently training the default model is never intended.
        """
        import logging
        import sys

        args = args or sys.argv[1:]
        config = cls()
        config_source: str | None = None

        # First pass: load YAML from --config <path> OR a positional *.yaml/*.yml
        i = 0
        while i < len(args):
            if args[i] == "--config" and i + 1 < len(args):
                config = cls.from_yaml(args[i + 1])
                config_source = args[i + 1]
                i += 2
            elif not args[i].startswith("--") and args[i].lower().endswith((".yaml", ".yml")):
                config = cls.from_yaml(args[i])
                config_source = args[i]
                i += 1
            else:
                i += 1

        if config_source is None:
            logging.getLogger(__name__).warning(
                "No config file passed — falling back to built-in demo defaults "
                "(model=%s, dataset=%s). Pass one with `--config <file>.yaml` or a "
                "positional `<file>.yaml`; this is almost never what you want for a real run.",
                config.model.name_or_path,
                config.data.dataset,
            )

        # Second pass: apply overrides
        i = 0
        while i < len(args):
            if args[i] == "--config":
                i += 2
                continue
            # Skip the positional config path (already consumed above).
            if not args[i].startswith("--") and args[i].lower().endswith((".yaml", ".yml")):
                i += 1
                continue
            if args[i].startswith("--") and i + 1 < len(args):
                key, value = args[i][2:], args[i + 1]
                parts = key.split(".")
                if len(parts) != 2:
                    raise ConfigError(
                        f"--{key}: overrides are --section.option value (e.g. --train.learning_rate 1e-5)."
                    )
                section_name, field_name = parts
                section = _section(config, section_name, where="command line")
                _set_option(section, section_name, field_name, value, where="command line")
                i += 2
            else:
                i += 1
        return config

    def validate(self) -> list[str]:
        """Validate config compatibility. Returns list of warnings.

        Raises ConfigError on hard incompatibilities that would produce silent
        corruption or nonsensical training. Returns warnings for untested-but-not-
        broken combinations.

        Call this after loading config, before training starts:
            config = Config.from_yaml("config.yaml")
            warnings = config.validate()  # raises on errors, returns warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        # ── Hard incompatibilities (raise) ────────────────────────────────
        from palingenesis.validate_data import valid_think_tags

        for where, tags in [("data.think_tags", self.data.think_tags)] + [
            (f"data.sources[{i}].think_tags", s.get("think_tags"))
            for i, s in enumerate(self.data.sources)
            if isinstance(s, dict)
        ]:
            if tags and not valid_think_tags(tags):
                errors.append(f"{where} must be two different non-empty strings [open, close], got {tags!r}.")
        if self.memory.gradient_release:
            if self.train.gradient_accumulation_steps > 1:
                errors.append(
                    "gradient_release=true requires gradient_accumulation_steps=1. "
                    "Gradient release fuses optimizer into backward — no accumulation possible. "
                    "Increase per_device_batch_size instead (freed memory allows it)."
                )
            if self.train.ga_ramp_start > 0:
                errors.append(
                    "gradient_release=true is incompatible with ga_ramp (dynamic accumulation). "
                    "Choose one: gradient_release OR batch ramp, not both."
                )
            if self.train.optimizer == "muon":
                errors.append(
                    "gradient_release=true is incompatible with Muon optimizer. "
                    "Muon needs the full gradient for its orthogonalization step. "
                    "Use adamw, lion8bit, or adamw8bit with gradient_release."
                )
            if self.train.max_grad_norm > 0 and not self.train.adagc:
                warnings.append(
                    "gradient_release=true disables global grad clipping (max_grad_norm). "
                    "Consider enabling adagc=true for per-tensor clipping instead."
                )

        from palingenesis.optim import OPTIMIZERS

        if self.train.optimizer not in OPTIMIZERS:
            errors.append(f"train.optimizer={self.train.optimizer!r} is not one of {', '.join(OPTIMIZERS)}.")
        if self.train.lr_scheduler not in ("cosine", "linear", "constant", "power_decay", "wsd"):
            errors.append(
                f"train.lr_scheduler={self.train.lr_scheduler!r} is not one of cosine, linear, "
                "constant, power_decay, wsd."
            )

        if self.data.packing and self.parallel.context_parallel:
            errors.append(
                "packing=true is incompatible with context_parallel=true. "
                "Context Parallel needs full-length sequences for correct Ring Attention. "
                "Packed sequences break the attention boundary assumptions. "
                "Disable one: use packing for short conversations, CP for long single sequences."
            )

        if self.train.mona and self.plugins.schedule_free:
            errors.append(
                "mona=true is incompatible with schedule_free=true. "
                "MONA wraps the optimizer step; Schedule-Free replaces the optimizer entirely. "
                "Choose one acceleration strategy."
            )

        if self.train.hyperball:
            if self.train.hyperball_lr < 0:
                errors.append(f"train.hyperball_lr must be >= 0 (got {self.train.hyperball_lr}; 0 = calibrated).")
            if self.model.torch_dtype != "float32":
                warnings.append(
                    f"train.hyperball with model.torch_dtype={self.model.torch_dtype}: the weights are "
                    "updated in that dtype, so small angular steps partly round away in bf16. "
                    "float32 weights apply them exactly."
                )
        if self.train.hyperball and self.plugins.schedule_free:
            errors.append(
                "hyperball=true is incompatible with schedule_free=true. "
                "Hyperball projects after optimizer.step(); Schedule-Free has no standard step(). "
                "Choose one."
            )

        # Multiple exclusive loss functions
        active_losses = sum(
            [
                self.plugins.dft,
                self.plugins.cadft,
                self.plugins.deft,
                self.plugins.info_sft,
                self.plugins.pre_rl,
            ]
        )
        if active_losses > 1:
            errors.append(
                f"Only one training objective plugin can be active at a time ({active_losses} enabled). "
                "Enable at most one of: dft, cadft, deft, info_sft, pre_rl."
            )

        if self.preprocess.enabled and self.data.sources:
            errors.append(
                "preprocess.enabled=true is incompatible with data.sources (multi-dataset mode). "
                "The prepared output replaces the single data.dataset. "
                "For multi-source preparation use 'pgs prepare-multi' and point data.sources "
                "at the per-source scored files."
            )

        if self.data.pretokenize:
            if self.data.msft_tracking:
                errors.append(
                    "data.pretokenize=true is incompatible with data.msft_tracking=true. "
                    "MSFT adjusts per-source sampling weights DURING training, so the token "
                    "stream is not static and cannot be baked into a pre-tokenized cache. "
                    "Disable one of them (drop pretokenize to keep adaptive weighting, or "
                    "drop msft_tracking to cache a fixed stream)."
                )

        if self.data.turn_scaling not in ("uniform", "progressive", "last_heavy"):
            errors.append(
                f"data.turn_scaling={self.data.turn_scaling!r} is not one of uniform, progressive, last_heavy."
            )
        elif self.data.turn_scaling != "uniform":
            # Per-token weights reach CE, chunked CE, CCE and the chunked gated objectives.
            gated = self.plugins.dft or self.plugins.cadft or self.plugins.info_sft or self.plugins.deft
            unweighted = [
                name
                for name, on in (
                    ("plugins.pre_rl", self.plugins.pre_rl),
                    (
                        "plugins.deft/dft/cadft/info_sft without memory.chunked_loss",
                        gated and not self.memory.chunked_loss,
                    ),
                    ("memory.seco", self.memory.seco),
                    ("dpo.enabled", self.dpo.enabled),
                )
                if on
            ]
            if unweighted:
                errors.append(
                    f"data.turn_scaling={self.data.turn_scaling!r} is not applied by {', '.join(unweighted)}; "
                    "use turn_scaling: uniform with it."
                )

        if self.dpo.enabled:
            errors.extend(self._dpo_errors())
        if self.memory.seco:
            errors.extend(self._seco_errors())

        # ── Soft warnings (untested combinations) ─────────────────────────
        if self.dpo.enabled and self.model.torch_dtype != "float32":
            warnings.append(
                f"dpo.enabled with model.torch_dtype={self.model.torch_dtype}: the optimizer "
                "updates the weights in that dtype, with no fp32 master copy. A bf16 weight "
                "resolves ~0.4% of its magnitude, so at DPO learning rates (~1e-6) nearly every "
                "update rounds to zero and the policy never leaves the reference. Use "
                "model.torch_dtype: float32 (compute still runs in bf16 under train.bf16)."
            )

        if self.memory.seco and self.model.compile:
            warnings.append(
                "memory.seco with model.compile: SeCO runs every layer against a cache whose length "
                "changes each chunk; this combination is not validated (expect recompilations). "
                "SeCO was validated with model.compile=false."
            )
        if self.memory.seco and self.model.torch_dtype != "float32":
            warnings.append(
                f"memory.seco with model.torch_dtype={self.model.torch_dtype}: SeCO's chunked forward is "
                "verified at startup with a tolerance of 2e-2 in low precision (1e-4 in float32)."
            )

        if self.memory.gradient_release and self.train.hyperball:
            errors.append(
                "gradient_release=true steps the optimizer inside backward, so Hyperball's update "
                "(which wraps the optimizer step) would never run. Disable one of them."
            )

        if self.memory.gradient_release and self.train.mona:
            warnings.append(
                "gradient_release + mona: MONA augments gradients before step, but "
                "gradient_release fuses step into backward. This combination is UNTESTED. "
                "The MONA acceleration may not see the correct gradient state."
            )

        if self.train.ema and self.train.base_merge:
            warnings.append(
                "ema + base_merge: both modify weights outside the optimizer. "
                "EMA averages weights; base_merge pulls toward init. "
                "The interaction is mathematically sound but UNVERIFIED at scale."
            )

        if self.train.adagc and self.train.spike_detection:
            warnings.append(
                "adagc + spike_detection: AdaGC clips per-tensor; spike_detection skips steps globally. "
                "Redundant — AdaGC subsumes spike detection. Consider disabling spike_detection."
            )

        if self.train.ga_ramp_start > 0 and self.train.mona:
            warnings.append(
                "ga_ramp + mona: dynamic batch size changes the gradient noise scale, "
                "which may confuse MONA's curvature estimates. UNTESTED combination."
            )

        if self.data.packing and self.train.max_steps <= 0:
            warnings.append(
                "packing=true with an epochs-based LR horizon: total_steps is derived from ROW "
                "count, but packing merges several rows per sequence, so the epoch ends well "
                "before the schedule completes and training finishes at a barely-decayed LR "
                "(no anneal). Set train.max_steps explicitly (≈ total_dataset_tokens / "
                "(per_device_batch_size × grad_accum × max_seq_length × world_size)), or use "
                "lr_scheduler: wsd, which tolerates an overestimated horizon."
            )

        # ── Raise on errors ───────────────────────────────────────────────
        if errors:
            msg = "Configuration has incompatible settings:\n" + "\n".join(f"  ✗ {e}" for e in errors)
            raise ConfigError(msg)

        return warnings

    def _dpo_errors(self) -> list[str]:
        from palingenesis.dpo import LOSS_TYPES

        d = self.dpo
        errors: list[str] = []
        if d.loss_type not in LOSS_TYPES:
            errors.append(f"dpo.loss_type={d.loss_type!r} is not one of {', '.join(LOSS_TYPES)}.")
        if d.beta <= 0:
            errors.append(f"dpo.beta must be > 0 (got {d.beta}).")
        if not 0.0 <= d.label_smoothing < 0.5:
            errors.append(f"dpo.label_smoothing must be in [0, 0.5) (got {d.label_smoothing}).")
        if d.label_smoothing and d.loss_type != "robust":
            errors.append(
                f"dpo.label_smoothing is only used by loss_type=robust; with {d.loss_type!r} it "
                "would be silently ignored. Set it to 0 or use loss_type: robust."
            )
        if not 0.0 <= d.ld_alpha <= 1.0:
            errors.append(f"dpo.ld_alpha must be in [0, 1] (got {d.ld_alpha}; 1.0 = off).")
        if d.sft_weight < 0:
            errors.append(f"dpo.sft_weight must be >= 0 (got {d.sft_weight}).")
        # Features that assume one SFT sequence per row, or change the token stream.
        unsupported = {
            "data.packing": self.data.packing,
            "data.sources (use one preference dataset)": bool(self.data.sources),
            "data.eval_sources (use data.eval_dataset)": bool(self.data.eval_sources),
            "data.pretokenize": self.data.pretokenize,
            "data.msft_tracking": self.data.msft_tracking,
            "data.pretrain_replay_dataset": bool(self.data.pretrain_replay_dataset),
            "preprocess.enabled": self.preprocess.enabled,
            "parallel.context_parallel": self.parallel.context_parallel,
            "memory.gradient_release": self.memory.gradient_release,
            "plugins.dft": self.plugins.dft,
            "plugins.cadft": self.plugins.cadft,
            "plugins.deft": self.plugins.deft,
            "plugins.info_sft": self.plugins.info_sft,
            "plugins.pre_rl": self.plugins.pre_rl,
        }
        for name, on in unsupported.items():
            if on:
                errors.append(f"dpo.enabled=true does not support {name}; disable it for preference training.")
        return errors

    def _seco_errors(self) -> list[str]:
        m = self.memory
        errors: list[str] = []
        if m.seco_chunk_size < 1:
            errors.append(f"memory.seco_chunk_size must be >= 1 (got {m.seco_chunk_size}).")
        if m.spaco_budget < 0:
            errors.append(f"memory.spaco_budget must be >= 0 (got {m.spaco_budget}; 0 = exact SeCO).")
        unsupported = {
            "data.packing (packed documents need per-document attention)": self.data.packing,
            "parallel.context_parallel (both split the sequence)": self.parallel.context_parallel,
            "dpo.enabled": self.dpo.enabled,
            "memory.gradient_release (it steps inside every backward; SeCO runs one per chunk)": self.memory.gradient_release,
            "plugins.sym_noise (noise would differ between the two passes)": self.plugins.sym_noise,
            "plugins.dft": self.plugins.dft,
            "plugins.cadft": self.plugins.cadft,
            "plugins.deft": self.plugins.deft,
            "plugins.info_sft": self.plugins.info_sft,
            "plugins.pre_rl": self.plugins.pre_rl,
            "logging.rl_readiness (needs full-sequence hidden states)": self.logging.rl_readiness,
        }
        for name, on in unsupported.items():
            if on:
                errors.append(f"memory.seco=true does not support {name}.")
        return errors


class ConfigError(Exception):
    """Raised when config has hard incompatibilities that prevent safe training."""

    pass


# Options that existed once: a config that still sets them gets told why they are gone.
_REMOVED_OPTIONS = {
    ("data", "seq_len_curriculum"): "it never took effect (the curriculum was not wired into the data "
    "pipeline). Remove it; set data.max_seq_length directly.",
    ("data", "seq_len_curriculum_min"): "see data.seq_len_curriculum.",
    ("data", "seq_len_curriculum_ramp_steps"): "see data.seq_len_curriculum.",
}


def _section(config: Config, name: str, where: str):
    import dataclasses
    import difflib

    names = [f.name for f in dataclasses.fields(config)]
    if name not in names:
        hint = difflib.get_close_matches(name, names, n=1)
        raise ConfigError(
            f"{where}: unknown config section `{name}`"
            + (f" (did you mean `{hint[0]}`?)" if hint else f"; sections: {', '.join(names)}")
        )
    return getattr(config, name)


def _set_option(section, section_name: str, key: str, value, where: str) -> None:
    """Set section.key = value, coerced to the option's type. Unknown or removed options
    are errors: a misspelt option silently keeping its default is a wrong training run."""
    import dataclasses
    import difflib

    names = [f.name for f in dataclasses.fields(section)]
    if key not in names:
        if (section_name, key) in _REMOVED_OPTIONS:
            raise ConfigError(f"{where}: {section_name}.{key} was removed: {_REMOVED_OPTIONS[(section_name, key)]}")
        hint = difflib.get_close_matches(key, names, n=1)
        raise ConfigError(
            f"{where}: unknown option {section_name}.{key}"
            + (f" (did you mean {section_name}.{hint[0]}?)" if hint else "")
        )
    current = getattr(section, key)
    if isinstance(value, str):
        # YAML reads `2e-5` (no dot) as a string, and command-line values are strings.
        text = value.strip()
        try:
            if isinstance(current, bool):
                if text.lower() not in ("true", "false", "1", "0", "yes", "no"):
                    raise ValueError
                value = text.lower() in ("true", "1", "yes")
            elif isinstance(current, int):
                value = int(text)
            elif isinstance(current, float):
                value = float(text)
            elif current is None and text.lower() in ("none", "null", ""):
                value = None
        except ValueError:
            raise ConfigError(
                f"{where}: {section_name}.{key}={value!r} is not a valid {type(current).__name__}."
            ) from None
    elif isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    setattr(section, key, value)
