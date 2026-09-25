"""Reinforcement learning for causal LMs: async, single- and multi-turn, verifiable rewards.

    pgs rl --config configs/rl_math.yaml

    from palingenesis.rl import RLConfig, RLTrainer

    def correct(completion, answer, **row):          # any dataset column is an argument
        return float(answer in completion)

    config = RLConfig.from_yaml("configs/rl_math.yaml")
    RLTrainer(config, rewards=[correct]).train()

Modules: config (options), trainer (the loop), pipeline (rollouts, rewards, dynamic
sampling), losses (objectives), rewards (built-ins and the reward signature), env
(multi-turn environments and tools), chat (token-in/token-out templates, tool calls),
sandbox + grading (code execution and verification), data (prompts).
"""

from palingenesis.rl.config import RLConfig, RLConfigError
from palingenesis.rl.env import ToolEnv
from palingenesis.rl.rewards import Reward, SkipSample
from palingenesis.rl.sandbox import ExecJob, ExecResult, make_sandbox


def __getattr__(name):  # the trainer pulls in torch and transformers: load on use
    if name == "RLTrainer":
        from palingenesis.rl.trainer import RLTrainer

        return RLTrainer
    raise AttributeError(name)


__all__ = [
    "RLConfig",
    "RLConfigError",
    "RLTrainer",
    "Reward",
    "SkipSample",
    "ToolEnv",
    "ExecJob",
    "ExecResult",
    "make_sandbox",
]
