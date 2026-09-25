"""RL configuration — typed sections, YAML + CLI overrides, strict validation.

    config = RLConfig.from_yaml("configs/rl_math.yaml")
    pgs rl --config configs/rl_math.yaml --train.learning_rate 2e-6 --rollout.group_size 16

Rewards are a named mapping (``rewards: {name: {fn, weight, args}}``); each name is
logged separately. Everything else is a fixed section. Unknown options are errors,
and options that would do nothing in the selected mode are rejected up front.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

LOSSES = ("masked_is", "cispo", "icepop", "gspo")
ADVANTAGE_STD = ("batch", "group", "none")
AGGREGATIONS = ("prompt", "token", "constant")
TRUNCATION = ("score", "mask")
ROLLOUT_BACKENDS = ("hf", "vllm", "vllm_server")
ENV_TYPES = ("single_turn", "tools")  # or "module:Class" / "path/to/file.py:Class"
TOOL_PARSERS = ("auto", "hermes", "xml")
SANDBOX_BACKENDS = ("docker", "agent_sandbox", "subprocess")


@dataclass(slots=True)
class RLModelConfig:
    # The policy to train (a Hugging Face id or local path). No default on purpose.
    policy: str = ""
    # Extra apply_chat_template arguments, e.g. {enable_thinking: false}.
    chat_template_kwargs: dict = field(default_factory=dict)
    # Tokens that end a turn besides the template's end-of-turn token and eos.
    stop_tokens: list = field(default_factory=list)
    # Activation checkpointing: "none", "selective" (keep matmul outputs, recompute the cheap
    # ops: most of the memory for a few % compute), "full" (recompute every layer) or "auto"
    # ("full" with train.cpu_offload, which asks for the smallest footprint; otherwise "none").
    gradient_checkpointing: str = "auto"
    # Liger's fused kernels (RMSNorm, SwiGLU, RoPE, ...) in the policy, on CUDA.
    use_liger_kernel: bool = True


@dataclass(slots=True)
class RLDataConfig:
    # JSONL / JSON / parquet path, or a Hugging Face dataset id. Every column reaches the
    # reward functions and the environment's reset().
    dataset: str = ""
    split: str = "train"
    # The rows' format: "chat" (palingenesis rows), "nemo_gym" (NeMo Gym / Nemotron-RL rows with
    # responses_create_params), or "auto" (detected). See palingenesis.rl.formats.
    format: str = "auto"
    # The column holding the prompt: a chat (list of messages) or a string (one user
    # turn). Empty = the first of messages, prompt, question, problem.
    prompt_field: str = ""
    # Prepended as a system message when the row has none.
    system_prompt: str = ""
    # A column with the row's own tool schemas (OpenAI or Responses-API form), used when the
    # environment defines none: e.g. function-calling data scored on the calls themselves.
    tools_field: str = "tools"
    # Held-out rows: `eval_dataset` (with `eval_split`), else `eval_size` rows split off
    # `dataset` (deterministic by content).
    eval_dataset: str = ""
    eval_split: str = "test"
    eval_size: int = 0
    # Rows whose rendered prompt is longer are dropped (0 = keep all that fit the context).
    max_prompt_tokens: int = 0
    # Retire a prompt once its running mean reward reaches this (0 = never). With a 0/1
    # reward, 0.9 retires prompts the policy has learned (ScaleRL's no-positive-resampling).
    retire_above: float = 0.0
    seed: int = 0


@dataclass(slots=True)
class RLRolloutConfig:
    # "hf": the trainer's model.generate (no dependency, slow; tests and debugging).
    # "vllm": an in-process vLLM engine on the policy's GPU (sleeps while training).
    # "vllm_server": a separate vLLM server fed weights over CUDA IPC.
    backend: str = "vllm"
    batch_prompts: int = 32  # groups per optimizer step
    group_size: int = 8  # rollouts per prompt
    temperature: float = 1.0  # sampling is plain temperature: top-p/top-k would bias the ratios
    max_new_tokens: int = 2048  # per assistant turn
    # Tokens the policy may generate over a whole trajectory, all turns (0 = max_new_tokens).
    max_completion_tokens: int = 0
    # Policy versions a batch may lag behind the weights it trains (0 = on-policy).
    # > 0 overlaps rollouts (and environment/reward latency) with training.
    max_staleness: int = 0
    # Dynamic sampling: groups whose rewards are all equal carry no gradient; draw more
    # prompts until batch_prompts informative groups are collected, at most this many
    # extra prompts per batch, as a multiple of batch_prompts (0 = never refill).
    max_refill: float = 2.0
    # vllm
    gpu_memory_utilization: float = 0.3
    max_model_len: int = 8192  # prompt + completion tokens of a trajectory
    enforce_eager: bool = False
    prefix_caching: bool = True  # a group's rollouts share their prompt's prefill
    sleep: bool = True  # release the engine's memory while training (max_staleness 0)
    max_num_seqs: int = 0  # concurrent sequences (0 = batch_prompts x group_size, >= 256)
    url: str = ""  # vllm_server: a running server (empty = launch one)
    # vllm: FP8 rollouts ("fp8") trade sampler precision for speed; the importance weights and
    # the sequence mask absorb the mismatch (watch policy/abs_ratio_dev). Empty = bf16.
    kv_cache_dtype: str = ""
    quantization: str = ""
    micro_seqs: int = 64  # hf: sequences per generate() call


@dataclass(slots=True)
class RLEnvConfig:
    # "single_turn": one completion per prompt.
    # "tools": multi-turn tool calling with the functions in `tools`.
    # "module:Class" or "path/to/file.py:Class": your environment (see palingenesis.rl.env).
    type: str = "single_turn"
    # For type "tools": "module:function" or "file.py:function" names.
    tools: list = field(default_factory=list)
    max_turns: int = 8  # assistant turns per trajectory (tools/custom envs)
    tool_parser: str = "auto"  # "hermes" (JSON), "xml" (Qwen3-Coder/Qwen3.5), or auto
    max_tool_output_tokens: int = 512  # longer tool results keep their head and tail
    tool_timeout: float = 30.0  # seconds per tool call
    max_concurrent: int = 0  # live environment instances at once (0 = no limit; remote envs have a capacity)
    # Keyword arguments for the environment's constructor.
    args: dict = field(default_factory=dict)


@dataclass(slots=True)
class RewardConfig:
    # A built-in (math, exact, choice, regex, code, judge), or "module:function".
    fn: str = ""
    weight: float = 1.0
    # Keyword arguments bound to the function (e.g. {field: answer} or {pattern: ...}).
    args: dict = field(default_factory=dict)


@dataclass(slots=True)
class SandboxConfig:
    # "docker": a warm pool of locked-down local containers (no network, read-only, capped).
    # "agent_sandbox": a pool of Kubernetes Agent Sandbox pods (agent-sandbox.sigs.k8s.io),
    #                  claimed once from a SandboxWarmPool and reused; isolation comes from
    #                  the pool's RuntimeClass (gVisor, Kata). pip install 'k8s-agent-sandbox[async]'.
    # "subprocess": the same limits without isolation. Untrusted code can read your files:
    #               development only, and only with allow_unsafe.
    backend: str = "docker"
    image: str = "python:3.12-slim"  # docker
    workers: int = 4  # containers / pods
    slots_per_worker: int = 4  # concurrent programs per container / pod
    memory_mb: int = 1024  # per program
    timeout: float = 6.0  # seconds per program (tests may set their own)
    allow_unsafe: bool = False
    # agent_sandbox: the SandboxWarmPool to claim from, its namespace, and how to reach the
    # pods: "in_cluster", "gateway:<name>[/<namespace>]" or "url:<router url>".
    warmpool: str = ""
    namespace: str = "default"
    connection: str = "in_cluster"


@dataclass(slots=True)
class RLLossConfig:
    # masked_is: policy gradient with the sampler's importance weight capped at is_cap and
    #            PPO's directional clip applied as a mask (ε_low/ε_high), anchored to the
    #            rollout policy (no recompute). The default.
    # cispo:     sg(min(ratio, is_cap)) · A · log π, no mask (MiniMax-M1, ScaleRL).
    # icepop:    ratio · A inside [icepop_low, icepop_high], zero outside (INTELLECT-3).
    # gspo:      sequence-level ratio (geometric mean) with a tight clip (Qwen GSPO).
    type: str = "masked_is"
    eps_low: float = 0.2
    eps_high: float = 0.28
    is_cap: float = 5.0
    icepop_low: float = 0.5
    icepop_high: float = 5.0
    gspo_eps: float = 4e-3
    # Sequence mask: drop a trajectory whose mean |ratio - 1| over its tokens exceeds this
    # (Trust Region Masking, "SER"; 0 = off). Guards against sampler/trainer mismatch.
    seq_mask: float = 0.1
    # Advantage = reward - group mean, divided by the std of the batch / of the group / not.
    advantage_std: str = "batch"
    # prompt: mean over tokens within a prompt's group, then over prompts (ScaleRL);
    # token: mean over all tokens (DAPO); constant: sum / (sequences x max tokens) (Dr. GRPO).
    aggregation: str = "prompt"
    # A trajectory cut by the token budget: scored like any other, or kept out of the loss.
    truncation: str = "score"
    # Soft overlong penalty (DAPO): the last `overlong_buffer` tokens of the budget cost up
    # to `overlong_penalty` reward, linearly (0 = off).
    overlong_buffer: int = 0
    overlong_penalty: float = 1.0
    log_entropy: bool = True


@dataclass(slots=True)
class RLTrainConfig:
    output_dir: str = "runs/rl"
    steps: int = 200
    learning_rate: float = 1e-6
    lr_scheduler: str = "constant"  # "constant" or "cosine"
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-15  # small: RL gradients are tiny (MiniMax-M1, ScaleRL)
    weight_decay: float = 0.0
    micro_tokens: int = 16384  # padded tokens per forward/backward micro-batch
    # FSDP2: shard the policy over data-parallel ranks (always under torchrun with > 1 process;
    # set it for one GPU only with cpu_offload). cpu_offload keeps parameters, gradients and
    # optimizer state in CPU memory: the largest model a GPU can train, at PCIe speed.
    fsdp: bool = False
    cpu_offload: bool = False
    reshard_after_forward: bool = True
    # A group's rollouts share their prompt: "on" forwards it once per group (an exact tree
    # forward/backward, palingenesis.seco_tree) instead of once per rollout; "auto" does so
    # when prompts average >= 1024 tokens (short prompts run faster batched). Not with FSDP.
    prompt_sharing: str = "auto"
    seed: int = 0
    eval_every: int = 0  # steps (0 = only at the end, when eval rows exist)
    eval_samples: int = 256  # held-out prompts per evaluation
    eval_temperature: float = 0.0
    save_steps: int = 0
    keep_checkpoints: int = 2  # resumable step_* checkpoints kept (0 = all)
    resume_from: str = ""  # a step_* dir, or "auto" (newest complete one in output_dir)
    save_final: bool = True  # export the trained policy at the end (off: benchmarks, dry runs)


@dataclass(slots=True)
class RLLoggingConfig:
    log_every: int = 1
    use_wandb: bool = False
    project: str = "palingenesis-rl"
    run_name: str = ""


@dataclass(slots=True)
class RLConfig:
    model: RLModelConfig = field(default_factory=RLModelConfig)
    data: RLDataConfig = field(default_factory=RLDataConfig)
    rollout: RLRolloutConfig = field(default_factory=RLRolloutConfig)
    env: RLEnvConfig = field(default_factory=RLEnvConfig)
    rewards: dict = field(default_factory=dict)  # name -> RewardConfig
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    loss: RLLossConfig = field(default_factory=RLLossConfig)
    train: RLTrainConfig = field(default_factory=RLTrainConfig)
    logging: RLLoggingConfig = field(default_factory=RLLoggingConfig)

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RLConfig":
        """Load a YAML config. Unknown sections and options are errors (with a hint)."""
        from palingenesis.config import ConfigError

        with Path(path).open() as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: expected a mapping of sections (model:, data:, rewards:, ...).")
        config = cls()
        for section, options in raw.items():
            if options is None:
                continue
            if not isinstance(options, dict):
                raise ConfigError(f"{path}: `{section}:` must be a mapping.")
            if section == "rewards":
                for name, entry in options.items():
                    if isinstance(entry, str):  # rewards: {correct: math}
                        entry = {"fn": entry}
                    if not isinstance(entry, dict):
                        raise ConfigError(f"{path}: `rewards.{name}:` must be a function name or a mapping.")
                    for key, value in entry.items():
                        config.set(f"rewards.{name}.{key}", value, where=str(path))
            else:
                for key, value in options.items():
                    config.set(f"{section}.{key}", value, where=str(path))
        return config

    @classmethod
    def from_cli(cls, args: list[str] | None = None) -> "RLConfig":
        """--config file.yaml plus --section.option value overrides (--rewards.<name>.option too)."""
        import sys

        from palingenesis.config import ConfigError

        args = sys.argv[1:] if args is None else args
        config = cls()
        if "--config" in args:
            i = args.index("--config")
            if i + 1 >= len(args):
                raise ConfigError("--config needs a file.")
            config = cls.from_yaml(args[i + 1])
        i = 0
        while i < len(args):
            if args[i] == "--config":
                i += 2
            elif args[i].startswith("--") and i + 1 < len(args):
                config.set(args[i][2:], args[i + 1], where="command line")
                i += 2
            else:
                raise ConfigError(
                    f"command line: unexpected argument {args[i]!r} (overrides are --section.option value)."
                )
        return config

    def set(self, key: str, value, where: str = "code") -> None:
        """Set one option by dotted key: section.option, or rewards.<name>.option."""
        from palingenesis.config import ConfigError, _section, _set_option

        parts = key.split(".")
        if parts[0] == "rewards":
            if len(parts) != 3:
                raise ConfigError(f"{where}: {key}: use rewards.<name>.<option> (names cannot contain dots).")
            section, section_name, option = (
                self.rewards.setdefault(parts[1], RewardConfig()),
                key.rsplit(".", 1)[0],
                parts[2],
            )
        else:
            if len(parts) != 2:
                raise ConfigError(f"{where}: {key}: use <section>.<option>.")
            section, section_name, option = (
                _section(self, parts[0], where=where),
                parts[0],
                parts[1],
            )
        names = {f.name for f in dataclasses.fields(section)}
        if option in names and isinstance(getattr(section, option), (dict, list)) and isinstance(value, str):
            value = yaml.safe_load(value)  # mappings/lists given on the command line
        _set_option(section, section_name, option, value, where=where)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    # --------------------------------------------------------------- validation

    @property
    def completion_budget(self) -> int:
        return self.rollout.max_completion_tokens or self.rollout.max_new_tokens

    def validate(self, python_rewards: bool = False, python_env: bool = False) -> list[str]:
        """Raise RLConfigError on impossible or no-op settings; return warnings. The flags say
        rewards / an environment are passed in Python (RLTrainer(rewards=..., env=...)).
        """
        errors: list[str] = []
        warnings: list[str] = []
        m, d, r, e, s, loss, t = (
            self.model,
            self.data,
            self.rollout,
            self.env,
            self.sandbox,
            self.loss,
            self.train,
        )

        if not m.policy:
            errors.append("model.policy is required (see configs/rl_*.yaml).")
        if r.backend not in ROLLOUT_BACKENDS:
            errors.append(f"rollout.backend must be one of {ROLLOUT_BACKENDS}, got {r.backend!r}.")
        if r.batch_prompts < 1 or r.group_size < 1:
            errors.append("rollout.batch_prompts and rollout.group_size must be >= 1.")
        if r.group_size == 1:
            warnings.append(
                "rollout.group_size 1: every advantage is 0 (the baseline is the sample itself); "
                "use >= 4 (8 is a good default)."
            )
        if not r.temperature > 0:
            errors.append(f"rollout.temperature must be > 0 for training rollouts, got {r.temperature}.")
        if r.max_new_tokens < 1:
            errors.append("rollout.max_new_tokens must be >= 1.")
        if r.max_completion_tokens and r.max_completion_tokens < r.max_new_tokens:
            errors.append(
                "rollout.max_completion_tokens (whole trajectory) must be >= rollout.max_new_tokens (one turn)."
            )
        if r.backend != "hf" and r.max_new_tokens >= r.max_model_len:
            errors.append(
                f"rollout.max_new_tokens ({r.max_new_tokens}) must be < rollout.max_model_len "
                f"({r.max_model_len}), which also holds the prompt."
            )
        if r.max_staleness < 0:
            errors.append("rollout.max_staleness must be >= 0.")
        if r.max_staleness > 0 and r.backend == "hf":
            errors.append(
                "rollout.max_staleness > 0 needs a vLLM backend (hf generates with the trained model itself)."
            )
        if r.max_staleness > 8:
            warnings.append(
                f"rollout.max_staleness {r.max_staleness}: published recipes stay within 1-8 policy versions."
            )
        if r.max_refill < 0:
            errors.append("rollout.max_refill must be >= 0.")
        if r.backend == "vllm" and r.max_staleness == 0 and not r.sleep:
            warnings.append(
                "rollout.sleep false with max_staleness 0: the engine keeps its memory while training; "
                "fine only if gpu_memory_utilization leaves room for the trainer."
            )

        if not d.dataset:
            errors.append(
                "data.dataset is required (JSONL/parquet path or a Hugging Face dataset id), "
                "or pass dataset= to RLTrainer."
            )
        from palingenesis.rl.formats import FORMATS

        if d.format not in FORMATS:
            errors.append(f"data.format must be one of {FORMATS}, got {d.format!r}.")
        if d.eval_size < 0 or t.eval_samples < 0:
            errors.append("data.eval_size and train.eval_samples must be >= 0.")
        if d.retire_above < 0:
            errors.append("data.retire_above must be >= 0 (0 = never retire prompts).")

        custom_env = e.type not in ENV_TYPES or python_env
        if e.type not in ENV_TYPES and ":" not in e.type:
            errors.append(f"env.type must be one of {ENV_TYPES} or 'module:Class' / 'file.py:Class', got {e.type!r}.")
        if e.type == "tools" and not e.tools:
            errors.append("env.type tools needs env.tools: ['module:function', ...].")
        if e.tools and e.type != "tools":
            errors.append(
                "env.tools is only used with env.type tools (a custom environment's public methods are its tools)."
            )
        if e.type == "single_turn" and e.args:
            errors.append("env.args are constructor arguments for a custom environment; single_turn has none.")
        if e.max_turns < 1:
            errors.append("env.max_turns must be >= 1.")
        if e.tool_parser not in TOOL_PARSERS:
            errors.append(f"env.tool_parser must be one of {TOOL_PARSERS}, got {e.tool_parser!r}.")

        if not self.rewards and not python_rewards and not custom_env:
            errors.append(
                "at least one reward is required: rewards: {<name>: {fn: math}} (or an environment with get_reward)."
            )
        for name, spec in self.rewards.items():
            if not spec.fn:
                errors.append(f"rewards.{name}.fn is required (a built-in or 'module:function').")
            if spec.weight == 0:
                warnings.append(f"rewards.{name}.weight is 0: it is logged but does not train.")
        uses_code = any(spec.fn == "code" for spec in self.rewards.values())
        if s.backend not in SANDBOX_BACKENDS:
            errors.append(f"sandbox.backend must be one of {SANDBOX_BACKENDS}, got {s.backend!r}.")
        if s.backend == "agent_sandbox" and uses_code and not s.warmpool:
            errors.append("sandbox.backend agent_sandbox needs sandbox.warmpool (the SandboxWarmPool to claim from).")
        if not (s.connection == "in_cluster" or s.connection.startswith(("gateway:", "url:"))):
            errors.append(
                f"sandbox.connection must be in_cluster, gateway:<name>[/<namespace>] or url:<router url>, "
                f"got {s.connection!r}."
            )
        if uses_code and s.backend == "subprocess" and not s.allow_unsafe:
            errors.append(
                "sandbox.backend subprocess runs untrained model code with your permissions "
                "(it can read and delete your files). Use docker, or set sandbox.allow_unsafe true "
                "for development."
            )

        if loss.type not in LOSSES:
            errors.append(f"loss.type must be one of {LOSSES}, got {loss.type!r}.")
        if loss.advantage_std not in ADVANTAGE_STD:
            errors.append(f"loss.advantage_std must be one of {ADVANTAGE_STD}, got {loss.advantage_std!r}.")
        if loss.aggregation not in AGGREGATIONS:
            errors.append(f"loss.aggregation must be one of {AGGREGATIONS}, got {loss.aggregation!r}.")
        if loss.type == "gspo" and loss.aggregation != "prompt":
            errors.append(
                "loss.type gspo is a sequence-level objective: its aggregation is fixed "
                "(mean over a group's sequences); leave loss.aggregation at prompt."
            )
        if loss.truncation not in TRUNCATION:
            errors.append(f"loss.truncation must be one of {TRUNCATION}, got {loss.truncation!r}.")
        if not 0 < loss.eps_low < 1 or loss.eps_high <= 0:
            errors.append("loss.eps_low must be in (0, 1) and loss.eps_high > 0.")
        if loss.is_cap < 1:
            errors.append(f"loss.is_cap must be >= 1, got {loss.is_cap}.")
        if not 0 < loss.icepop_low < 1 < loss.icepop_high:
            errors.append("loss.icepop_low must be in (0, 1) and loss.icepop_high > 1.")
        if loss.seq_mask < 0:
            errors.append("loss.seq_mask must be >= 0 (0 = off).")
        if loss.overlong_buffer < 0 or loss.overlong_buffer >= self.completion_budget:
            errors.append(f"loss.overlong_buffer must be in [0, the completion budget {self.completion_budget}).")

        if t.lr_scheduler not in ("constant", "cosine"):
            errors.append(f"train.lr_scheduler must be 'constant' or 'cosine', got {t.lr_scheduler!r}.")
        if t.prompt_sharing not in ("auto", "on", "off"):
            errors.append(f"train.prompt_sharing must be auto, on or off, got {t.prompt_sharing!r}.")
        if t.prompt_sharing == "on" and (t.fsdp or t.cpu_offload or os.environ.get("WORLD_SIZE", "1") != "1"):
            errors.append("train.prompt_sharing on runs the tree forward on an unsharded model: not with FSDP.")
        if m.gradient_checkpointing not in ("auto", "none", "selective", "full"):
            errors.append(
                f"model.gradient_checkpointing must be auto, none, selective or full, got {m.gradient_checkpointing!r}."
            )
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world > 1 and r.batch_prompts % world:
            errors.append(f"rollout.batch_prompts ({r.batch_prompts}) must be divisible by the {world} ranks.")
        if world > 1 and r.backend == "hf":
            errors.append(
                "rollout.backend hf generates with the sharded model: under torchrun every rank would "
                "run FSDP collectives inside generate() out of step. Use vllm (one engine per rank)."
            )
        if world > 1 and r.backend == "vllm_server":
            errors.append(
                "rollout.backend vllm_server is single-process; under torchrun use vllm (one engine per rank)."
            )
        if (t.fsdp or t.cpu_offload) and r.backend == "hf" and r.max_staleness:
            errors.append("rollout.backend hf generates with the sharded trainer model: max_staleness must be 0.")
        if t.steps < 1 or t.micro_tokens < 1:
            errors.append("train.steps and train.micro_tokens must be >= 1.")
        if t.learning_rate > 1e-5:
            warnings.append(f"train.learning_rate {t.learning_rate}: RL recipes use 5e-7 to 2e-6.")

        if errors:
            raise RLConfigError("RL configuration has incompatible settings:\n" + "\n".join(f"  ✗ {x}" for x in errors))
        return warnings


class RLConfigError(Exception):
    """Raised when the RL config has settings that would train wrongly or not at all."""


def max_num_seqs(config: RLConfig) -> int:
    """vLLM's max_num_seqs for the policy engine: rollout.max_num_seqs, or every sequence one
    step generates at once (x the steps in flight), at least vLLM's own 256, at most 2048.
    """
    r = config.rollout
    if r.max_num_seqs > 0:
        return r.max_num_seqs
    # speculative refill launches up to ~2x the batch at once (see pipeline._collect)
    return min(2048, max(256, 2 * r.batch_prompts * r.group_size * (1 + r.max_staleness)))
