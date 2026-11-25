"""Training configuration — flat, typed, YAML + CLI.

STATUS convention for features:
  # STATUS: proven     — Independently reproduced, ablated at scale, safe default
  # STATUS: validated  — Paper-backed + tested in this codebase, not externally reproduced
  # STATUS: experimental — Single-paper, limited testing, use with monitoring
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from palingenesis.distill.config import DistillConfig


@dataclass(slots=True)
class ModelConfig:
    name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    trust_remote_code: bool = True
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    # sdpa | flash_attention_2 | flash_attention_3 | flash_attention_4 | flex_attention | eager, or a Hub kernel
    # "org/repo[@rev]" loaded by transformers through the `kernels` package (AMD: "kernels-community/aiter-flash-attn@v2")
    attn_implementation: str = "sdpa"
    # The flash kernel behind sdpa (dense and variable-length; packed rows and SeCO included): "builtin" (torch's
    # FlashAttention-2), "fa3" (FlashAttention-3, Hopper), "fa4" (FlashAttention-4, Hopper / Blackwell) or "auto"
    # (FA3 on Hopper, FA4 on Blackwell, when installed; else builtin). Probed at startup against the built-in kernel
    # (palingenesis.accelerator.activate_flash_kernel). Only with attn_implementation: sdpa.
    flash_kernel: str = "builtin"
    use_liger_kernel: bool = True  # used when compile is false (compile fuses the same ops)
    compile: bool = True
    compile_backend: str = "inductor"
    compile_mode: str = "default"  # "default", "reduce-overhead", or "max-autotune"
    # Where the weights come from. "pretrained": the checkpoint at name_or_path. "random": the architecture of
    # name_or_path's config.json with freshly initialised weights (training from scratch); name_or_path then needs
    # only config.json and the tokenizer files, and any weights there are ignored. `pgs train` only.
    init: str = "pretrained"
    # Seed of the random initialisation (the same on every rank). -1: train.seed.
    init_seed: int = -1


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
    # Each: {dataset, split, weight, mode("sft"|"pretrain"), messages_field|text_field}. Raw text (mode: pretrain)
    # also takes append_eos: true (default: the tokenizer's EOS token ends every document), a token string
    # (that token instead, e.g. "<|endoftext|>"), or false.
    sources: list = field(default_factory=list)
    # What an epoch of data.sources is. "total": as many rows as the sources hold together,
    # drawn by weight (a source weighted above its share of the rows is repeated, one below
    # it contributes a fresh random subset each epoch); a source without a weight gets its
    # share of the rows, so unweighted sources behave like one shuffled concatenation. A row
    # counts when its last training sequence is drawn: a raw-text document split into several
    # chunks counts once, with all its chunks (one raw-text source: one pass over every chunk).
    # "first_exhausted": the epoch ends when a source runs out (sizes unknown fall back to it).
    mix_epoch: str = "total"
    # What the `weight` of a data source is a share of. "sequences": of the training sequences drawn (a source of
    # long sequences then takes a larger share of the tokens than its weight). "tokens": of the tokens trained on:
    # each source is drawn in proportion to its weight divided by the mean length of its sequences (measured as the
    # run reads them), so `weight: 0.8` is 80% of the tokens whatever the documents' lengths. Needs a weight on
    # every source. Sources without any weight are mixed by their rows (one shuffled concatenation) either way.
    mix_weights: str = "sequences"
    # Observation loss: the text of tool / environment messages gets loss too (never the template's wrapper: role
    # headers, <tool_response> tags), so the model is also trained to predict what the environment returns. After
    # ECHO (arXiv:2605.24517), which adds this loss with weight 0.05 to GRPO on the policy's own rollouts (pgs rl:
    # loss.observation_weight); on SFT traces its effect is not measured (see the agentic guide).
    include_observations: bool = False
    # Weight λ of the observation loss. With observation_normalization "conversation" each conversation's loss is
    # its assistant loss + λ x the mean NLL of its observation tokens (ECHO's per-sequence normalisation by the
    # observation length), and the loss's token count (its denominator) counts assistant tokens only, so tool
    # outputs no longer dilute the assistant's loss. "token": observation tokens are trained like assistant tokens
    # with weight λ per token and counted in the denominator; observation_weight 1.0 with "token" is the behaviour
    # of include_observations before observation_weight existed.
    observation_weight: float = 0.05
    observation_normalization: str = "conversation"
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
    # Loop-recovery augmentation: the share of multi-turn rows in which assistant tool calls (one per 10, at least one)
    # and their observations are pasted again 1-3 times right after themselves, untrained ("loss": false), so the
    # following gold turn is
    # learned in a context where the agent has just repeated itself, i.e. the way out of a loop. Distilled students
    # loop at repeated decision points their teachers handle (arxiv:2512.12895); masking the bad action and
    # supervising the correction follows arxiv:2609.40111. Deterministic per row content; a row whose copies would
    # not fit max_seq_length stays unaugmented (so the share achieved leans to shorter rows); 0 = off.
    loop_recovery: float = 0.0
    # The same inside one reply: the share of rows in which a sentence of trained turns' reasoning (one turn in
    # three, at least one) is written 1-4 more times right after itself; the copies get no loss, the rest of the gold reasoning does (continue after a
    # repeated sentence, the within-reply loop of distilled thinking students). Needs a fast tokenizer; 0 = off.
    reasoning_loop_recovery: float = 0.0
    # Repetition unlikelihood at decision points: weight of -log(1 - p(repeat token)) at the trained positions where
    # the gold stops repeating a previous action or reasoning sentence (ChatDataset._repeat_targets: after an exact
    # repeat, or late in a near-repeat), added to the CE sum with the same denominator (one target counts like this many
    # tokens). Distilled students fall back on copying their context at hard decision points (arxiv:2512.12895); plain
    # CE never pushes that option down. Only these few positions: legitimate copying (paths, code) is untouched,
    # unlike token-level unlikelihood. Needs memory.seco, no pretokenize. 0 = off.
    repeat_unlikelihood: float = 0.0
    # The same for stopping: at the final reply of an agent trace, the ways a tool call could start instead (the call
    # opener at the reply's first token and after paragraph breaks, the paragraph break at its end); the summed term is
    # -log P(no call where the gold stopped). Weight per target. Same requirements. 0 = off.
    stop_unlikelihood: float = 0.0
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
    # Each: {name, dataset, split, weight, samples, regression_floor, mode, messages_field|text_field}; dataset is
    # a local .jsonl / .json / .parquet file or a Hub dataset. A source that cannot be read is an error.
    eval_sources: list = field(default_factory=list)
    # Pretraining replay: mix raw text (split "train", field "text", all-token loss, no end-of-document token)
    # into SFT. An epoch is one pass over the target data's rows, with the replay sequences drawn alongside.
    # Kotha & Liang (arXiv:2603.04964) find generic replay can also help the target task,
    # mostly in 150M-param pretraining simulations (best at 33-67% of fine-tuning steps).
    # Empty string disables.
    pretrain_replay_dataset: str = ""
    # Per-sequence sampling probability of replay text (not a token share: replay
    # sequences are up to max_seq_length tokens, so their token share is usually larger).
    pretrain_replay_weight: float = 0.1
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
    # Epoch-based runs size the LR schedule from an estimate (row counts + a sample of rows
    # through the pipeline: no pass over the data). True: scan the whole pipeline once first.
    exact_steps: bool = False
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
    # "adamw", "adamw_bf16", "muon", "adamw8bit", "lion8bit", "paged_adamw8bit". adamw_bf16: bf16 weights and moments
    # with stochastic rounding (palingenesis.bf16_adamw): updates below a bf16 ulp still accumulate, at 8 B/param with
    # the gradient against 16 for fp32 master weights. Needs model.torch_dtype: bfloat16.
    optimizer: str = "adamw"
    # adamw_bf16: how the weight keeps updates below its ulp. "stochastic" (unbiased rounding; its error is a random
    # walk) or "kahan" (a bf16 compensation buffer, +2 B/param: no rounding noise; for very small learning rates).
    bf16_rounding: str = "stochastic"
    # adamw / adamw_bf16: step every parameter with one multi-tensor Triton launch per dtype bucket
    # (palingenesis.fused_optim) instead of one kernel per parameter, with gradient clipping folded into the step on
    # one process. adamw_bf16 keeps its exact rounding (same bits as its eager update); adamw is torch AdamW's update.
    # Without CUDA or Triton the usual optimizer runs (logged). Not with memory.gradient_release (per-parameter steps).
    fused_optimizer: bool = False
    # Muon (optimizer: muon; palingenesis.muon). muon_ns: "gram" (Gram Newton-Schulz with a restart, ~half the FLOPs
    # on rectangular matrices) or "standard"; muon_scale: "moonlight" (update RMS matched to AdamW: the same learning
    # rate and weight decay apply) or "spectral" (needs a larger LR); muon_min_dim: thinner matrices stay on AdamW;
    # muon_backend: "auto" (quack symmetric-GEMM kernels on Hopper/Blackwell when installed, torch otherwise),
    # "torch" or "quack"; muon_compile: torch.compile the Newton-Schulz step (off: the step is GEMM-bound, compiling
    # gave 0-5% on an A100 for seconds of compile time per matrix shape).
    muon_ns: str = "gram"
    muon_scale: str = "moonlight"
    muon_momentum: float = 0.95
    muon_min_dim: int = 32
    muon_backend: str = "auto"
    muon_compile: bool = False

    def muon_options(self) -> dict:
        return {
            "ns_method": self.muon_ns,
            "scale": self.muon_scale,
            "momentum": self.muon_momentum,
            "min_dim": self.muon_min_dim,
            "backend": self.muon_backend,
            "compile": self.muon_compile,
        }

    seed: int = 42
    save_steps: int = 500
    # Step checkpoints kept on disk (the oldest are deleted after each save; best/ and final/ are never deleted).
    # 0 keeps all. A full checkpoint holds the weights and the optimizer state (about 3x the fp32 weights for AdamW).
    keep_checkpoints: int = 5
    save_final: bool = True  # write the final model at the end (off for dry runs such as `pgs profile --measure`)
    logging_steps: int = 1
    bf16: bool = True
    # Cross-entropy, and the dense forward KL of off-policy distillation, computed so the target's gradient
    # never rounds to zero and equal student and teacher logits give a zero gradient (arXiv:2610.31642;
    # palingenesis.stable_softmax). A few extra element-wise passes over the logits; not measured on quality.
    stable_softmax: bool = False
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
    # Split every sequence over a group of ranks (palingenesis.context_parallel): exact attention
    # across them (K/V all-gather), head-tail load balancing. Needs fsdp: true.
    context_parallel: bool = False
    # Ranks per context-parallel group; 0 = min(world_size, 8) when it divides world_size, else all.
    cp_degree: int = 0
    # How K/V reach the other ranks: allgather (the only method implemented).
    cp_rotate_method: Literal["allgather"] = "allgather"
    cpu_offload: bool = False
    reshard_after_forward: bool = True
    # MoE models: every rank keeps experts / ranks experts of each layer and the tokens are sent to
    # the rank that owns their experts (all-to-all), instead of FSDP gathering every layer's expert
    # weights on every rank (palingenesis.moe_experts). Needs fsdp: true and a number of ranks that
    # divides the number of experts. STATUS: experimental (gradients tested equal to one process on
    # CPU ranks; not measured on several GPUs).
    expert_parallel: bool = False
    # Tensor parallelism (palingenesis.tensor_parallel): every layer's attention heads, linear-attention
    # heads and MLP columns split over tp_degree ranks (Megatron-style column / row parallel linears with
    # sequence parallelism between them). The ranks of a group read the same rows; FSDP2 shards over the
    # other dimensions. 1 = off. Must divide the number of processes and the (key / value) head counts.
    tp_degree: int = 1
    # Overlap the tensor-parallel all-gathers / reduce-scatters with the matmuls next to them (async TP:
    # torch's micro-pipelined collectives over symmetric memory). Needs model.compile and NVLink GPUs.
    async_tp: bool = False
    # Pipeline parallelism (palingenesis.pipeline_parallel): the layers split into pp_degree stages, one
    # per rank group, with micro-batches flowing through them. 1 = off.
    pp_degree: int = 1
    # gpipe, 1f1b, interleaved_1f1b, looped_bfs, interleaved_zero_bubble, zbv (zero bubble, V-shaped), dualpipev.
    pp_schedule: str = "1f1b"
    # Pipeline micro-batches per train.per_device_batch_size rows (must divide it); 0 = one row each.
    pp_microbatches: int = 0
    # Stages per pipeline rank: 1 for gpipe / 1f1b, 2 or more for the interleaved / looped / zero-bubble
    # schedules (zbv, dualpipev: exactly 2). 0 = the schedule's default.
    pp_stages_per_rank: int = 0
    # Layers the first / last stage gives up for the embeddings / output head (balance the stages).
    pp_first_stage_less_layers: int = 1
    pp_last_stage_less_layers: int = 1


@dataclass(slots=True)
class MoEConfig:
    """Mixture-of-Experts models (palingenesis.moe); ignored for models without MoE layers."""

    # Router balancing and routing statistics. false: the routers train on the task loss alone.
    balance: bool = True
    # Coefficient of the load-balancing loss, measured over the global batch (every rank and
    # micro-batch of the optimizer step). -1: the model config's router_aux_loss_coef, at most
    # 0.01; 0: off.
    balance_loss_coef: float = -1.0
    # Coefficient of the router z-loss (ST-MoE used 0.001); 0: off.
    z_loss_coef: float = 0.0
    # Auxiliary-loss-free balancing: step size of the selection bias, for routers that have one in
    # their checkpoint (DeepSeek-V3 family). -1: 0.001 for those models, off for the others; 0: off.
    bias_update_rate: float = -1.0
    # Also log the per-layer loads and router entropies (moe/layer_<i>/...).
    log_layers: bool = False
    # Expert compute: "auto" (transformers' default: grouped GEMM where available), "grouped" (this
    # library's sort-free dispatch and grouped GEMM), or a transformers name (eager, grouped_mm, batched_mm).
    experts_implementation: str = "auto"

    def validate(self) -> list[str]:
        """Errors in the section (shared by the pgs train, pgs rl and pgs distill configs)."""
        errors = []
        if self.balance_loss_coef < 0 and self.balance_loss_coef != -1:
            errors.append(
                f"moe.balance_loss_coef must be >= 0, or -1 for the model config's value (got {self.balance_loss_coef})."
            )
        if self.bias_update_rate < 0 and self.bias_update_rate != -1:
            errors.append(
                f"moe.bias_update_rate must be >= 0, or -1 for the model's default (got {self.bias_update_rate})."
            )
        if self.z_loss_coef < 0:
            errors.append(f"moe.z_loss_coef must be >= 0 (got {self.z_loss_coef}).")
        return errors


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
    # full backprop, incl. Qwen3.5 hybrids) for an extra no-grad forward of every
    # chunk but the last. With data.packing, every packed document is trained on
    # its own: long ones chunk by chunk, short ones batched in one forward.
    # Cost: +0-25% against full activation checkpointing, which is cheaper until it stops fitting.
    seco: bool = False
    seco_chunk_size: int = 4096
    # Keep the full-attention K/V (and recurrent start states) in pinned CPU memory,
    # streamed to the GPU block by block: GPU memory then grows only with the K/V
    # gradient. Exact; costs PCIe transfers. Needs attn_implementation: sdpa.
    seco_kv_offload: bool = False
    # SpaCO: backprop only this many random chunks per sequence (0 = SeCO, exact).
    # A stochastic gradient estimate; cuts backward compute on very long inputs.
    spaco_budget: int = 0
    # Activation offload (palingenesis.activation_offload): the activations the transformer layers save for the
    # backward wait in pinned host memory, copied out in the forward and back in the backward on a side CUDA stream,
    # overlapped with other layers' compute. Exact. What moves depends on train.gradient_checkpointing: every saved
    # activation (none), each layer's input (full), or its input plus what selective checkpointing keeps (selective).
    # The last layer stays on the GPU. Host memory: about the activations' size, at most 40% of RAM.
    activation_offload: bool = False
    activation_offload_min_bytes: int = 1 << 20  # smaller tensors stay on the GPU
    # layers whose device-to-host copies may still run while later layers compute (each keeps its activations on the
    # GPU until done); layers fetched ahead of the backward (0: fetched when their backward starts)
    activation_offload_in_flight: int = 1
    activation_offload_prefetch: int = 1
    # only each checkpointed layer's inputs; what selective checkpointing keeps stays on the GPU
    activation_offload_boundaries_only: bool = False


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
    # Your own trainer plugins (palingenesis.hooks.TrainerPlugin): "module:Class", "path/file.py:Class",
    # an entry point of group "palingenesis.plugins", or {class: ..., args: {...}}. Run in order.
    custom: list = field(default_factory=list)


@dataclass(slots=True)
class PreprocessConfig:
    """Offline data preparation (cleaning, held-out split, selection) — `pgs prepare --config`.

    Reuses model.name_or_path, data.dataset, data.dataset_split, data.messages_field,
    data.max_seq_length and the masking options from the same config used for training,
    so preprocess and training can never drift apart.

    By default nothing here runs the model: rows training would drop and exact duplicates
    are removed, and what was removed is reported. Scoring with the model (one forward
    pass over the data) runs when `score` is set or an option that needs it is (min_ppl,
    max_ppl, hes, any strategy but random); those options are experimental.

    When `enabled: true`, training automatically loads the prepared dataset from
    `output_dir` (instead of data.dataset) and preserves curriculum ordering if
    strategy == "curriculum".
    """

    enabled: bool = False  # train-time: auto-use prepared output from output_dir
    output_dir: str = "./prepared"
    format: Literal["parquet", "jsonl"] = "parquet"
    max_samples: int = 0  # cap samples read from the raw dataset (0 = all)
    budget: int = 0  # samples to keep after cleaning/filtering (0 = all)
    # Tokens to keep (0 = all): a random subset whose rows add up to at most this many
    # tokens, the unit training cost is paid in. Applied after `budget`; strategy random only.
    token_budget: int = 0
    dedupe: bool = True  # drop exact duplicate conversations (same messages and tools)
    # Raw-text sources (data.sources entries with mode: pretrain; `pgs prepare --config` cleans each into
    # output_dir/<name>/). dedupe drops documents whose normalised text (case, whitespace) is the same.
    # near_dedupe > 0 also drops near-duplicates: documents whose word 5-gram sets have an estimated Jaccard
    # similarity of at least this value (MinHash; 0.8 is a common choice), keeping the first of each group.
    near_dedupe: float = 0.0
    # Decontamination: files (JSONL, JSON, parquet, .txt) of the evaluation sets you will report. A document that
    # shares a word n-gram of decontaminate_ngram words with any string in them is dropped.
    decontaminate: list = field(default_factory=list)
    decontaminate_ngram: int = 13
    # Columns of the evaluation files to read (e.g. [question, answer]); empty = every string column.
    decontaminate_fields: list = field(default_factory=list)
    # The share of an evaluation file's rows above which an n-gram counts as shared scaffolding and is not
    # matched. -1 (default): 0.2 when every column is read, off when decontaminate_fields names the columns.
    # 0 (or 1): off, every n-gram is matched.
    decontaminate_shared_share: float = -1.0
    # Documents with fewer characters are dropped (0 = keep all).
    min_chars: int = 0
    # Keep at most N rows that share a prompt (system prompt + first user turn), e.g.
    # several trajectories of one task. 0 = no cap.
    max_per_prompt: int = 0
    # Caps per value of `group_field` (e.g. per source): rows and tokens. 0 = no cap.
    max_per_group: int = 0
    max_tokens_per_group: int = 0
    # Reserve N samples as a held-out eval set (written to eval_data.parquet,
    # EXCLUDED from the training selection, and sharing no prompt with it). Training
    # auto-uses it when data.eval_dataset is empty — a true same-distribution holdout.
    eval_holdout: int = 0
    # Score every row with the model (response perplexity). Off by default: it costs a
    # forward pass over the data, and nothing selected by the score has beaten random.
    score: bool = False
    # Perplexity outlier filter, off by default (0). A bound > 0 turns scoring on. Absolute
    # perplexity depends on model, domain and language: read the distribution first.
    min_ppl: float = 0.0
    max_ppl: float = 0.0
    filter_score: Literal["response", "full"] = "response"  # filter assistant tokens by default
    # Selection when budget > 0 (prepare.select_by_budget). Every strategy but `random`
    # is experimental and turns scoring on: on Qwen3.5-0.8B math SFT no familiarity-based
    # strategy beat a random subset, and `optimal` (the J-shaped mix) was worse (see
    # guides/data.md); `curriculum` also makes training keep the stored order.
    strategy: Literal["optimal", "balanced", "medium_focus", "curriculum", "hard_focus", "flow", "random"] = "random"
    batch_size: int = 4  # max samples per scoring forward (length-sorted padded batches)
    # Padded-token cap per scoring forward (logits are B×S×V, so THIS bounds
    # memory, not batch_size). 16384 ≈ 5GB of bf16 logits for a 150K vocab;
    # on an 80GB GPU with a 4B model, 32768–49152 is safe and faster.
    max_batch_tokens: int = 16384
    # Also write HES (arXiv:2605.22389) from the scoring forward. Experimental: validated on
    # long math/STEM reasoning only, and no selection strategy reads it.
    hes: bool = False
    hes_top_k_pct: float = 0.5
    # Row column that names a row's group (e.g. "source"): the caps above apply per group,
    # and prepared_meta.json profiles each group (count, tokens, mean response tokens and,
    # when scored, median response NLL and familiarity buckets) before and after selection.
    group_field: str = ""
    # Processes that tokenize the rows (0 = the CPUs available, at most 8; 1 = in this process).
    # The output is the same whatever the number.
    num_workers: int = 0


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
    # Which chosen tokens the SFT term covers: "all" (RPO as published) or "edits" (only
    # where chosen and rejected differ, ± sft_edit_window tokens; for minimal-edit pairs
    # whose chosen side is mostly the policy's own uncorrected text).
    sft_scope: str = "all"
    sft_edit_window: int = 8
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
class MultimodalConfig:
    """Images in the training data (palingenesis.multimodal; the multimodal guide)."""

    # Load the checkpoint in its multimodal architecture and feed the images in the data's content
    # parts ({"type": "image", "image": ...}) to its vision tower. false: the language model alone is
    # loaded and trained, image parts are ignored, and exports copy the vision tower unchanged.
    enabled: bool = False
    # Train the vision encoder / the projector (Qwen: the patch merger) or keep them frozen.
    # Frozen modules still run forward on every image; they take no optimizer state.
    train_vision_tower: bool = False
    train_projector: bool = False
    # Bounds of the image processor's resize, in pixels (0: the checkpoint's own). They set each
    # image's token count: (pixels / patch^2) / merge^2. Serve the model with the same bounds.
    min_pixels: int = 0
    max_pixels: int = 0


@dataclass(slots=True)
class LoggingConfig:
    project: str = "palingenesis"
    run_name: str | None = None
    use_wandb: bool = True
    use_trackio: bool = True
    # Further metric backends: `module:Class` or `file.py:Class` (palingenesis.logging.Tracker documents the interface)
    trackers: list = field(default_factory=list)
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
    moe: MoEConfig = field(default_factory=MoEConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
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

        args = sys.argv[1:] if args is None else args
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
                if len(parts) < 2:
                    raise ConfigError(
                        f"--{key}: overrides are --section.option value (e.g. --train.learning_rate 1e-5)."
                    )
                section_name, field_name = parts[0], parts[1]
                section = _section(config, section_name, where="command line")
                if len(parts) > 2:  # into a mapping option: --distill.teachers.big.placement shard
                    _set_nested(section, section_name, field_name, parts[2:], value)
                else:
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
        from palingenesis.accelerator import FLASH_ATTENTION_NAMES, FLASH_KERNELS, is_hub_kernel
        from palingenesis.validate_data import valid_think_tags

        attn = self.model.attn_implementation
        if attn not in ("sdpa", *FLASH_ATTENTION_NAMES, "flex_attention", "eager") and not is_hub_kernel(attn):
            errors.append(
                f"model.attn_implementation={attn!r}: expected sdpa, {', '.join(FLASH_ATTENTION_NAMES)}, "
                'flex_attention, eager or a Hub kernel "org/repo[@rev]" (e.g. kernels-community/aiter-flash-attn@v2)'
            )
        if self.model.flash_kernel not in FLASH_KERNELS:
            errors.append(
                f"model.flash_kernel must be one of {', '.join(FLASH_KERNELS)}, got {self.model.flash_kernel!r}."
            )
        elif self.model.flash_kernel in ("fa3", "fa4") and attn != "sdpa":
            errors.append(
                f"model.flash_kernel={self.model.flash_kernel} replaces the kernel behind sdpa; attn_implementation is "
                f"{attn}. Use attn_implementation: sdpa, or flash_attention_3 / flash_attention_4 with flash_kernel: "
                "builtin."
            )

        for where, tags in [("data.think_tags", self.data.think_tags)] + [
            (f"data.sources[{i}].think_tags", s.get("think_tags"))
            for i, s in enumerate(self.data.sources)
            if isinstance(s, dict)
        ]:
            if tags and not valid_think_tags(tags):
                errors.append(f"{where} must be two different non-empty strings [open, close], got {tags!r}.")
        if self.multimodal.enabled:
            mm = self.multimodal
            for name, why in (
                (
                    "parallel.context_parallel" if self.parallel.context_parallel else "",
                    "an image's tokens and its M-RoPE positions cannot be split across the ranks of a sequence",
                ),
                (
                    "memory.seco" if self.memory.seco else "",
                    "the chunked forward runs the language model on token ids, without the vision tower",
                ),
                (
                    "data.pretokenize" if self.data.pretokenize else "",
                    "the cache stores token tensors, not pixels",
                ),
            ):
                if name:
                    errors.append(f"multimodal.enabled with {name} is not supported: {why}.")
            for name in ("min_pixels", "max_pixels"):
                if getattr(mm, name) < 0:
                    errors.append(f"multimodal.{name} must be a number of pixels (0: the checkpoint's own).")
            if mm.min_pixels and mm.max_pixels and mm.min_pixels > mm.max_pixels:
                errors.append(f"multimodal.min_pixels ({mm.min_pixels}) is above max_pixels ({mm.max_pixels}).")
            if self.distill.enabled:
                for name, teacher in (self.distill.teachers or {}).items():
                    if isinstance(teacher, dict) and teacher.get("cache"):
                        errors.append(
                            f"multimodal.enabled with a cached teacher (distill.teachers.{name}.cache) is not "
                            "supported: cached targets are keyed by the tokens, and the same tokens with other "
                            "pixels have other targets. Use an online teacher (model: ...)."
                        )
        elif self.multimodal.train_vision_tower or self.multimodal.train_projector:
            errors.append("multimodal.train_vision_tower / train_projector need multimodal.enabled: true.")
        if self.train.keep_checkpoints < 0:
            errors.append(f"train.keep_checkpoints must be >= 0 (0 keeps all), got {self.train.keep_checkpoints}.")
        if self.train.stable_softmax:
            uncovered = [
                name
                for name, on in (
                    (
                        "plugins.deft/dft/cadft/info_sft",
                        self.plugins.deft or self.plugins.dft or self.plugins.cadft or self.plugins.info_sft,
                    ),
                    ("plugins.pre_rl", self.plugins.pre_rl),
                    ("dpo.enabled", self.dpo.enabled),
                )
                if on
            ]
            if uncovered:
                errors.append(
                    "train.stable_softmax covers cross-entropy and distillation's forward KL, not "
                    f"{', '.join(uncovered)}: turn one of them off."
                )
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
        if self.train.optimizer == "muon":
            if self.train.muon_ns not in ("gram", "standard"):
                errors.append(f"train.muon_ns must be gram or standard, got {self.train.muon_ns!r}.")
            if self.train.muon_scale not in ("moonlight", "spectral"):
                errors.append(f"train.muon_scale must be moonlight or spectral, got {self.train.muon_scale!r}.")
            if not 0 <= self.train.muon_momentum < 1:
                errors.append(f"train.muon_momentum must be in [0, 1), got {self.train.muon_momentum}.")
            if self.train.llrd_decay < 1.0:
                warnings.append("train.llrd_decay is ignored with optimizer=muon (one learning rate per group kind).")
            if self.train.muon_scale == "spectral":
                warnings.append(
                    "train.muon_scale=spectral: Muon updates are not RMS-matched to AdamW, so the AdamW learning rate "
                    "is far too small for the Muon matrices (moonlight scaling avoids this)."
                )
        if self.train.bf16_rounding not in ("stochastic", "kahan"):
            errors.append(f"train.bf16_rounding must be stochastic or kahan, got {self.train.bf16_rounding!r}.")
        errors += self._fused_optimizer_errors()
        errors += self._activation_offload_errors(warnings)
        if self.train.optimizer == "adamw_bf16":
            if self.model.torch_dtype != "bfloat16":
                errors.append(
                    f"train.optimizer=adamw_bf16 steps bfloat16 weights with stochastic rounding; model.torch_dtype is "
                    f"{self.model.torch_dtype}. Set model.torch_dtype: bfloat16, or use optimizer: adamw (float32 "
                    "weights are fp32 masters already)."
                )
            if self.plugins.schedule_free:
                errors.append(
                    "train.optimizer=adamw_bf16 with plugins.schedule_free: Schedule-Free replaces the optimizer."
                )
        if self.train.lr_scheduler not in ("cosine", "linear", "constant", "power_decay", "wsd"):
            errors.append(
                f"train.lr_scheduler={self.train.lr_scheduler!r} is not one of cosine, linear, "
                "constant, power_decay, wsd."
            )
        if self.data.mix_epoch not in ("total", "first_exhausted"):
            errors.append(f"data.mix_epoch must be total or first_exhausted, got {self.data.mix_epoch!r}.")
        errors += self._source_errors()
        errors += self._init_errors()
        if not 0.0 <= self.preprocess.near_dedupe <= 1.0:
            errors.append(
                f"preprocess.near_dedupe must be in [0, 1] (a Jaccard similarity; 0 = off), got "
                f"{self.preprocess.near_dedupe}."
            )
        if self.preprocess.decontaminate_ngram < 1:
            errors.append(
                f"preprocess.decontaminate_ngram must be >= 1 (words), got {self.preprocess.decontaminate_ngram}."
            )
        if self.preprocess.min_chars < 0:
            errors.append(f"preprocess.min_chars must be >= 0, got {self.preprocess.min_chars}.")
        if not isinstance(self.preprocess.decontaminate, list) or not all(
            isinstance(f, str) and f for f in self.preprocess.decontaminate
        ):
            errors.append("preprocess.decontaminate must be a list of file paths.")
        if not isinstance(self.preprocess.decontaminate_fields, list) or not all(
            isinstance(f, str) and f for f in self.preprocess.decontaminate_fields
        ):
            errors.append("preprocess.decontaminate_fields must be a list of column names.")
        share = self.preprocess.decontaminate_shared_share
        if share != -1 and not 0.0 <= share <= 1.0:
            errors.append(
                f"preprocess.decontaminate_shared_share must be in [0, 1] (0 = off), or -1 for the default, got {share}."
            )
        if self.model.init == "random" and self.train.learning_rate <= 1e-4:
            warnings.append(
                f"model.init: random with train.learning_rate={self.train.learning_rate:g}: the defaults are tuned "
                "for fine-tuning a trained model. Training from random weights usually needs a much higher peak "
                "learning rate (1e-4 to 1e-3 with AdamW), a longer warmup and far more tokens."
            )
        strategies = ("optimal", "balanced", "medium_focus", "curriculum", "hard_focus", "flow", "random")
        if self.preprocess.strategy not in strategies:
            errors.append(f"preprocess.strategy={self.preprocess.strategy!r} is not one of {', '.join(strategies)}.")
        if self.preprocess.num_workers < 0:
            errors.append(f"preprocess.num_workers must be >= 0 (got {self.preprocess.num_workers}; 0 = automatic).")
        if self.preprocess.token_budget and self.preprocess.strategy != "random":
            errors.append(
                "preprocess.token_budget selects a random subset: it cannot be combined with "
                f"preprocess.strategy={self.preprocess.strategy!r}."
            )
        if (self.preprocess.max_per_group or self.preprocess.max_tokens_per_group) and not self.preprocess.group_field:
            errors.append(
                "preprocess.max_per_group / max_tokens_per_group need preprocess.group_field "
                "(the row column that names the group, e.g. source)."
            )

        errors += self._context_parallel_errors()
        errors += self._model_parallel_errors()
        errors += self.moe.validate()

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

        if self.data.msft_tracking and not self.data.pretokenize:
            warnings.append(
                "data.msft_tracking has no effect yet: msft.AdaptiveSourceTracker is not wired into the "
                "training loop, so per-source weights stay fixed at data.sources[*].weight."
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

        for knob in ("repeat_unlikelihood", "stop_unlikelihood"):
            if getattr(self.data, knob) < 0:
                errors.append(f"data.{knob} must be >= 0, got {getattr(self.data, knob)}.")
            elif getattr(self.data, knob) > 0 and (not self.memory.seco or self.data.pretokenize):
                errors.append(f"data.{knob} is implemented on the memory.seco path without pretokenize.")
        for knob in ("loop_recovery", "reasoning_loop_recovery"):
            if not 0.0 <= getattr(self.data, knob) <= 1.0:
                errors.append(f"data.{knob} must be in [0, 1], got {getattr(self.data, knob)}.")
        errors.extend(self._observation_errors())
        if self.data.turn_scaling not in ("uniform", "progressive", "last_heavy"):
            errors.append(
                f"data.turn_scaling={self.data.turn_scaling!r} is not one of uniform, progressive, last_heavy."
            )
        elif self.data.turn_scaling != "uniform":
            # Per-token weights reach CE, chunked CE, CCE, SeCO and the chunked gated objectives.
            gated = self.plugins.dft or self.plugins.cadft or self.plugins.info_sft or self.plugins.deft
            unweighted = [
                name
                for name, on in (
                    ("plugins.pre_rl", self.plugins.pre_rl),
                    (
                        "plugins.deft/dft/cadft/info_sft without memory.chunked_loss",
                        gated and not self.memory.chunked_loss,
                    ),
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
        if self.distill.enabled:
            from palingenesis.distill.config import distill_errors

            errors.extend(distill_errors(self))
        elif self.distill.teachers or self.distill.features:
            warnings.append("distill.teachers / distill.features are set but distill.enabled is false: ignored.")
        if not isinstance(self.plugins.custom, list):
            errors.append("plugins.custom must be a list of plugin entries (module:Class or {class, args}).")

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
        from palingenesis.dpo import SFT_SCOPES

        if d.sft_scope not in SFT_SCOPES:
            errors.append(f"dpo.sft_scope={d.sft_scope!r} is not one of {', '.join(SFT_SCOPES)}.")
        if d.sft_edit_window < 0:
            errors.append(f"dpo.sft_edit_window must be >= 0 (got {d.sft_edit_window}).")
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

    def _source_errors(self) -> list[str]:
        """data.sources / data.eval_sources entries, and data.mix_weights."""
        errors: list[str] = []
        for section in ("sources", "eval_sources"):
            entries = getattr(self.data, section)
            if not isinstance(entries, list):
                errors.append(f"data.{section} must be a list of sources.")
                continue
            for i, src in enumerate(entries):
                where = f"data.{section}[{i}]"
                if not isinstance(src, dict):
                    errors.append(f"{where} must be a mapping (dataset, mode, weight, ...), got {src!r}.")
                    continue
                mode = src.get("mode", "sft")
                if mode not in ("sft", "pretrain"):
                    errors.append(f"{where}.mode must be sft or pretrain, got {mode!r}.")
                if "append_eos" in src:
                    eos = src["append_eos"]
                    if mode != "pretrain":
                        errors.append(
                            f"{where}.append_eos applies to raw text (mode: pretrain): a conversation ends with "
                            "the chat template's own end-of-turn token."
                        )
                    elif not isinstance(eos, bool) and not (isinstance(eos, str) and eos):
                        errors.append(
                            f"{where}.append_eos must be true (the tokenizer's EOS token), false, or a token "
                            f"string such as '<|endoftext|>', got {eos!r}."
                        )
                weight = src.get("weight")
                if weight is not None and (
                    isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not math.isfinite(weight)
                    or weight < 0
                ):
                    errors.append(f"{where}.weight must be a finite number >= 0, got {weight!r}.")
        if self.data.mix_weights not in ("sequences", "tokens"):
            errors.append(f"data.mix_weights must be sequences or tokens, got {self.data.mix_weights!r}.")
        elif self.data.mix_weights == "tokens":
            sources = [s for s in self.data.sources if isinstance(s, dict)]
            if not sources:
                errors.append("data.mix_weights: tokens applies to data.sources, and there are none.")
            missing = [source_label(s, i) for i, s in enumerate(sources) if s.get("weight") is None]
            if missing:
                errors.append(
                    "data.mix_weights: tokens needs a weight (its share of the tokens) on every source; none on "
                    f"{', '.join(missing)}."
                )
            if self.data.msft_tracking:
                errors.append("data.mix_weights: tokens is not supported with data.msft_tracking.")
        return errors

    def _init_errors(self) -> list[str]:
        """model.init / model.init_seed, and what cannot start from random weights."""
        m = self.model
        errors: list[str] = []
        if m.init not in ("pretrained", "random"):
            errors.append(f"model.init must be pretrained or random, got {m.init!r}.")
        if isinstance(m.init_seed, bool) or not isinstance(m.init_seed, int) or m.init_seed < -1:
            errors.append(f"model.init_seed must be an integer >= 0, or -1 for train.seed, got {m.init_seed!r}.")
        if m.init != "random":
            return errors
        unsupported = {
            "dpo.enabled (the reference policy would be the untrained model)": self.dpo.enabled,
            "multimodal.enabled (the vision tower and projector would be untrained too)": self.multimodal.enabled,
            "train.base_merge (it pulls the weights back to the random initialisation)": self.train.base_merge,
            "plugins.pre_rl (its KL term keeps the model near the random initialisation)": self.plugins.pre_rl,
            "train.freeze_non_attention (the frozen layers would stay random)": self.train.freeze_non_attention,
            "preprocess.score / a perplexity bound or strategy (scores from an untrained model)": (
                self.preprocess.enabled
                and (
                    self.preprocess.score
                    or self.preprocess.hes
                    or self.preprocess.min_ppl > 0
                    or self.preprocess.max_ppl > 0
                    or self.preprocess.strategy != "random"
                )
            ),
        }
        errors += [f"model.init: random does not support {name}." for name, on in unsupported.items() if on]
        return errors

    def _context_parallel_errors(self) -> list[str]:
        p = self.parallel
        errors: list[str] = []
        if p.cp_degree < 0:
            errors.append(f"parallel.cp_degree must be >= 0 (got {p.cp_degree}; 0 = automatic).")
        if p.cp_rotate_method != "allgather":
            errors.append(
                f"parallel.cp_rotate_method={p.cp_rotate_method!r}: only allgather is implemented (each attention "
                "layer gathers the sequence's K/V from the context-parallel group)."
            )
        if p.cpu_offload:
            unsupported = {
                "memory.seco (its chunked backward runs on a plain module)": self.memory.seco,
                "memory.gradient_release (it steps inside the backward, before FSDP has offloaded the gradient)": (
                    self.memory.gradient_release
                ),
            }
            errors += [f"parallel.cpu_offload does not support {name}." for name, on in unsupported.items() if on]
        if p.context_parallel and not p.fsdp:
            errors.append(
                "parallel.context_parallel needs parallel.fsdp: true (FSDP sums the gradients of the ranks that "
                "split a sequence)."
            )
        if p.expert_parallel:
            unsupported = {
                "parallel.fsdp: false (the other parameters stay under FSDP)": not p.fsdp,
                "parallel.context_parallel": p.context_parallel,
                "memory.seco (ranks run different numbers of chunks; the token exchange is a collective)": (
                    self.memory.seco
                ),
                "parallel.cpu_offload": p.cpu_offload,
            }
            errors += [f"parallel.expert_parallel does not support {name}." for name, on in unsupported.items() if on]
        return errors

    def _model_parallel_errors(self) -> list[str]:
        """parallel.tp_degree / pp_degree and what they compose with (the world size is checked at startup:
        palingenesis.distributed.build_mesh)."""
        p = self.parallel
        errors: list[str] = []
        for name in ("tp_degree", "pp_degree"):
            value = getattr(p, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                errors.append(f"parallel.{name} must be an integer >= 1 (1 = off), got {value!r}.")
        for name in (
            "pp_microbatches",
            "pp_stages_per_rank",
            "pp_first_stage_less_layers",
            "pp_last_stage_less_layers",
        ):
            value = getattr(p, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"parallel.{name} must be an integer >= 0, got {value!r}.")
        if errors:
            return errors
        tp, pp = p.tp_degree > 1, p.pp_degree > 1
        if p.async_tp and not tp:
            errors.append("parallel.async_tp overlaps tensor-parallel collectives: it needs parallel.tp_degree > 1.")
        if p.async_tp and not self.model.compile:
            errors.append(
                "parallel.async_tp needs model.compile: true (inductor rewrites each all-gather / reduce-scatter and "
                "its matmul into a micro-pipelined pair)."
            )
        if not (tp or pp):
            return errors
        if not p.fsdp:
            errors.append(
                "parallel.tp_degree / pp_degree need parallel.fsdp: true (FSDP2 is the data-parallel dimension and "
                "sums the gradients of the ranks that hold different tokens; with one data-parallel rank it shards "
                "nothing and costs nothing)."
            )
        # what the sharded layers or the pipeline cannot run (both: one row of the composition table)
        shared = {
            "parallel.cpu_offload (offload is a single-GPU memory path; use more ranks instead)": p.cpu_offload,
            "parallel.expert_parallel (experts would need a tensor-parallel and an expert-parallel layout at once)": (
                p.expert_parallel
            ),
            "memory.seco (its chunked forward runs a plain module on one process)": self.memory.seco,
            "dpo.enabled (sequence log-probabilities and the reference model are not split yet)": self.dpo.enabled,
            "distill.enabled (teacher targets are not split yet)": self.distill.enabled,
            "multimodal.enabled (the vision tower is not split)": self.multimodal.enabled,
            "memory.float8_training (float8 layouts are not split)": self.memory.float8_training,
            "memory.gradient_release (it steps inside the backward, per parameter)": self.memory.gradient_release,
            "plugins.pre_rl (its per-rank normalisation and reference logits assume whole sequences)": (
                self.plugins.pre_rl
            ),
            "train.optimizer=muon (it orthogonalises whole matrices)": self.train.optimizer == "muon",
            "train.adagc (per-tensor clipping state assumes unsplit tensors)": self.train.adagc,
            "train.hyperball (norm constraints on unsplit matrices)": self.train.hyperball,
            "train.mona": self.train.mona,
            "plugins.schedule_free": self.plugins.schedule_free,
            "train.ema / train.base_merge (their weight copies are plain tensors)": self.train.ema
            or self.train.base_merge,
            "data.eval_sources (the multi-source evaluator scores whole sequences; use data.eval_dataset)": bool(
                self.data.eval_sources
            ),
        }
        which = " / ".join(name for name, on in (("parallel.tp_degree", tp), ("parallel.pp_degree", pp)) if on)
        errors += [f"{which} > 1 does not support {name}." for name, on in shared.items() if on]
        if tp and p.context_parallel:
            errors.append(
                "parallel.tp_degree > 1 with parallel.context_parallel is not validated yet (the sequence would be split "
                "twice); use one of them: context parallelism for very long sequences, tensor parallelism for large layers."
            )
        if pp:
            from palingenesis.parallel_dims import PP_SCHEDULES, schedule_errors

            if p.pp_schedule not in PP_SCHEDULES:
                errors.append(f"parallel.pp_schedule must be one of {', '.join(PP_SCHEDULES)}, got {p.pp_schedule!r}.")
            else:
                errors += schedule_errors(p.pp_schedule, p.pp_stages_per_rank)
            bs = self.train.per_device_batch_size
            if p.pp_microbatches and bs % p.pp_microbatches:
                errors.append(
                    f"parallel.pp_microbatches={p.pp_microbatches} must divide train.per_device_batch_size={bs} "
                    "(every pipeline micro-batch has the same shape)."
                )
            pp_unsupported = {
                "parallel.context_parallel (not validated with pipeline stages)": p.context_parallel,
                "plugins.dft / cadft / deft / info_sft (their losses are not computed on the last stage)": (
                    self.plugins.dft or self.plugins.cadft or self.plugins.deft or self.plugins.info_sft
                ),
                "logging.rl_readiness (it needs the final hidden states on every rank)": self.logging.rl_readiness,
            }
            errors += [f"parallel.pp_degree > 1 does not support {name}." for name, on in pp_unsupported.items() if on]
        return errors

    def _observation_errors(self) -> list[str]:
        """data.observation_weight / observation_normalization, and the objectives that cannot apply them."""
        d = self.data
        errors: list[str] = []
        if not math.isfinite(d.observation_weight) or d.observation_weight < 0:
            errors.append(f"data.observation_weight must be a finite number >= 0, got {d.observation_weight}.")
        if d.observation_normalization not in ("conversation", "token"):
            errors.append(
                f"data.observation_normalization={d.observation_normalization!r} is not one of conversation, token."
            )
        weighted = d.observation_normalization == "conversation" or d.observation_weight != 1.0
        if d.include_observations and weighted:
            # the observation term is carried by per-token weights and, per conversation, an assistant-only
            # denominator: objectives that ignore the weights would train observations at full weight
            gated = self.plugins.dft or self.plugins.cadft or self.plugins.info_sft or self.plugins.deft
            unweighted = [
                name
                for name, on in (
                    ("plugins.pre_rl", self.plugins.pre_rl),
                    (
                        "plugins.deft/dft/cadft/info_sft without memory.chunked_loss",
                        gated and not self.memory.chunked_loss,
                    ),
                )
                if on
            ]
            if unweighted:
                errors.append(
                    f"data.include_observations with observation_weight={d.observation_weight} / "
                    f"observation_normalization={d.observation_normalization!r} is not applied by "
                    f"{', '.join(unweighted)}; use memory.chunked_loss, or observation_normalization: token with "
                    "observation_weight: 1.0 (observation tokens trained like the assistant's)."
                )
        return errors

    def _fused_optimizer_errors(self) -> list[str]:
        t = self.train
        if not t.fused_optimizer:
            return []
        unsupported = {
            f"train.optimizer={t.optimizer} (only adamw and adamw_bf16 have multi-tensor kernels)": t.optimizer
            not in ("adamw", "adamw_bf16"),
            "plugins.schedule_free (it replaces the optimizer)": self.plugins.schedule_free,
            "memory.gradient_release (it steps each parameter in its backward hook: nothing to batch)": (
                self.memory.gradient_release
            ),
        }
        return [f"train.fused_optimizer does not support {name}." for name, on in unsupported.items() if on]

    def _activation_offload_errors(self, warnings: list[str]) -> list[str]:
        m = self.memory
        errors: list[str] = []
        if m.activation_offload_min_bytes < 0:
            errors.append(f"memory.activation_offload_min_bytes must be >= 0, got {m.activation_offload_min_bytes}.")
        if m.activation_offload_in_flight < 1:
            errors.append(
                "memory.activation_offload_in_flight must be >= 1 (layers whose copies may overlap later layers), "
                f"got {m.activation_offload_in_flight}."
            )
        if m.activation_offload_prefetch < 0:
            errors.append(f"memory.activation_offload_prefetch must be >= 0, got {m.activation_offload_prefetch}.")
        if not m.activation_offload:
            tuned = (
                m.activation_offload_min_bytes != 1 << 20
                or m.activation_offload_in_flight != 1
                or m.activation_offload_prefetch != 1
                or m.activation_offload_boundaries_only
            )
            if tuned:
                warnings.append("memory.activation_offload_* options are set but memory.activation_offload is false.")
            return errors
        if m.activation_offload_boundaries_only and self.train.gradient_checkpointing == "none":
            errors.append(
                "memory.activation_offload_boundaries_only offloads the inputs of checkpointed layers; "
                "train.gradient_checkpointing is none. Set it to selective or full, or offload every activation "
                "(activation_offload_boundaries_only: false)."
            )
        if m.seco and m.seco_kv_offload:
            warnings.append(
                "memory.activation_offload with memory.seco_kv_offload: both stream over PCIe in the same backward; "
                "measure the step time against either alone (pgs fit)."
            )
        return errors

    def _seco_errors(self) -> list[str]:
        m = self.memory
        errors: list[str] = []
        if m.seco_chunk_size < 1:
            errors.append(f"memory.seco_chunk_size must be >= 1 (got {m.seco_chunk_size}).")
        if m.spaco_budget < 0:
            errors.append(f"memory.spaco_budget must be >= 0 (got {m.spaco_budget}; 0 = exact SeCO).")
        unsupported = {
            "data.packing with memory.spaco_budget (SpaCO samples the chunks of one sequence)": (
                self.data.packing and m.spaco_budget > 0
            ),
            "data.packing with distill.enabled (not validated)": self.data.packing and self.distill.enabled,
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


def source_label(src: dict, index: int) -> str:
    """How messages name a data.sources entry: its name, else its dataset, else its position."""
    return str(src.get("name") or src.get("dataset") or f"source[{index}]")


# Version of the raw-text training stream, recorded in checkpoints (the position in the epoch counts in it) and
# in the pre-tokenized cache's fingerprint. 2: documents of mode: pretrain sources end with an end-of-document
# token (append_eos), a short tail reaches back into the previous chunk instead of being dropped, a mix epoch
# counts rows (every chunk of a document), and replay (no end-of-document token) is drawn alongside the target.
DATA_STREAM_VERSION = 2


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


def _set_nested(section, section_name: str, key: str, path: list[str], value: str) -> None:
    """A command-line value inside a mapping option (`--distill.teachers.big.model x`): the value is
    read as YAML (numbers, booleans, lists), intermediate mappings are created."""
    import dataclasses

    names = [f.name for f in dataclasses.fields(section)]
    if key not in names or not isinstance(getattr(section, key), dict):
        raise ConfigError(
            f"command line: --{section_name}.{key}.{'.'.join(path)}: only mapping options take nested keys "
            f"({section_name}.{key} is not one)."
        )
    node = getattr(section, key)
    for part in path[:-1]:
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        elif isinstance(child, str):  # a shorthand (teacher: model id) becomes a mapping
            child = node[part] = {"model": child}
        elif not isinstance(child, dict):
            raise ConfigError(f"command line: {section_name}.{key}.{part} is not a mapping.")
        node = child
    node[path[-1]] = yaml.safe_load(value) if isinstance(value, str) else value


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
