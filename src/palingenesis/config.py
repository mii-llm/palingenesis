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
    use_liger_kernel: bool = True
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
    # Per-turn loss scaling for multi-turn conversations
    # "uniform": all turns get equal weight (default, standard SFT)
    # "progressive": later turns get more weight (w = (turn_idx/total)^0.5)
    #   Rationale: later turns contain error recovery, iteration, harder reasoning
    # "last_heavy": final turn gets 2x weight, others 1x
    #   Rationale: final answer quality matters most
    turn_scaling: str = "uniform"
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
    # Sequence length curriculum (arxiv:2405.13226, Dataset Decomposition)
    # Ramps max_seq_length from short to full over training. Short seqs early = faster
    # attention (quadratic), model learns basics fast. Long seqs later = full context.
    seq_len_curriculum: bool = False
    seq_len_curriculum_min: int = 1024  # Starting max sequence length
    seq_len_curriculum_ramp_steps: int = 1000  # Steps to ramp from min to max_seq_length


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
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with Path(path).open() as f:
            raw = yaml.safe_load(f) or {}
        config = cls()
        for section_name, section_data in raw.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section = getattr(config, section_name)
                for k, v in section_data.items():
                    if hasattr(section, k):
                        # Type coercion: ensure YAML values match dataclass field types
                        current = getattr(section, k)
                        if isinstance(current, float) and isinstance(v, str):
                            v = float(v)
                        elif isinstance(current, int) and isinstance(v, str):
                            v = int(v)
                        elif isinstance(current, bool) and isinstance(v, str):
                            v = v.lower() in ("true", "1", "yes")
                        setattr(section, k, v)
        return config

    @classmethod
    def from_cli(cls, args: list[str] | None = None) -> "Config":
        """Parse --config file.yaml and --section.field value overrides."""
        import sys

        args = args or sys.argv[1:]
        config = cls()

        # First pass: load YAML if specified
        i = 0
        while i < len(args):
            if args[i] == "--config" and i + 1 < len(args):
                config = cls.from_yaml(args[i + 1])
                i += 2
            else:
                i += 1

        # Second pass: apply overrides
        i = 0
        while i < len(args):
            if args[i] == "--config":
                i += 2
                continue
            if args[i].startswith("--") and i + 1 < len(args):
                key, value = args[i][2:], args[i + 1]
                parts = key.split(".")
                if len(parts) == 2:
                    section_name, field_name = parts
                    if hasattr(config, section_name):
                        section = getattr(config, section_name)
                        if hasattr(section, field_name):
                            current = getattr(section, field_name)
                            if isinstance(current, bool):
                                setattr(section, field_name, value.lower() in ("true", "1", "yes"))
                            elif isinstance(current, int):
                                setattr(section, field_name, int(value))
                            elif isinstance(current, float):
                                setattr(section, field_name, float(value))
                            else:
                                setattr(section, field_name, value)
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

        if self.train.hyperball and self.plugins.schedule_free:
            errors.append(
                "hyperball=true is incompatible with schedule_free=true. "
                "Hyperball projects after optimizer.step(); Schedule-Free has no standard step(). "
                "Choose one."
            )

        # Multiple exclusive loss functions
        active_losses = sum([
            self.plugins.dft,
            self.plugins.cadft,
            self.plugins.deft,
            self.plugins.info_sft,
        ])
        if active_losses > 1:
            errors.append(
                f"Only one token-weighting loss can be active at a time ({active_losses} enabled). "
                "Enable exactly one of: dft, cadft, deft, info_sft. "
                "Hierarchy: DEFT > CADFT > DFT > InfoSFT > standard CE."
            )

        if self.preprocess.enabled and self.data.sources:
            errors.append(
                "preprocess.enabled=true is incompatible with data.sources (multi-dataset mode). "
                "The prepared output replaces the single data.dataset. "
                "For multi-source preparation use 'pgs prepare-multi' and point data.sources "
                "at the per-source scored files."
            )

        # ── Soft warnings (untested combinations) ─────────────────────────
        if self.memory.gradient_release and self.train.hyperball:
            warnings.append(
                "gradient_release + hyperball: both modify the parameter update path. "
                "Individually tested, but their interaction is UNVERIFIED. "
                "Monitor weight norms via health metrics to confirm Hyperball is projecting correctly."
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

        if self.data.seq_len_curriculum and self.data.packing:
            warnings.append(
                "seq_len_curriculum + packing: curriculum ramps max_seq_length, but packing "
                "concatenates sequences into fixed-length blocks. The curriculum may be ineffective "
                "because packing already handles variable lengths efficiently."
            )

        # The trainer picks ONE loss objective per run (priority: chunked DEFT >
        # chunked CE > CADFT > DEFT > DFT > InfoSFT > pre_rl > CE), so pre_rl is
        # silently shadowed by earlier branches — warn instead of ignoring.
        if self.plugins.pre_rl:
            shadowed_by = []
            if self.plugins.deft:
                shadowed_by.append("plugins.deft")
            if self.plugins.cadft:
                shadowed_by.append("plugins.cadft")
            if self.plugins.dft:
                shadowed_by.append("plugins.dft")
            if self.plugins.info_sft:
                shadowed_by.append("plugins.info_sft")
            if self.memory.chunked_loss:
                shadowed_by.append("memory.chunked_loss")
            if shadowed_by:
                warnings.append(
                    f"pre_rl is enabled but will be SILENTLY IGNORED: {', '.join(shadowed_by)} "
                    "take(s) precedence in the loss selection. To actually use pre_rl "
                    "(entropy preservation for RL), disable those options — note that "
                    "chunked_loss off means full logits are materialized (more memory)."
                )

        # ── Raise on errors ───────────────────────────────────────────────
        if errors:
            msg = "Configuration has incompatible settings:\n" + "\n".join(f"  ✗ {e}" for e in errors)
            raise ConfigError(msg)

        return warnings


class ConfigError(Exception):
    """Raised when config has hard incompatibilities that prevent safe training."""
    pass
