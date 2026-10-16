"""Harness profiles for HarborEnv: train and evaluate the agent under the dialects of real agent harnesses.

A HarnessProfile (profile.py) is one harness at a pinned version: system / user prompt, exact tool contracts bound
to a closed op set on the episode's container (ops.py), an action protocol (protocols.py: native function calling or
a text protocol), observation and error wording, truncation and context policies (context.py), stop and answer rules
(rules.py). Built-in profiles (profiles.py): palingenesis-default (HarborEnv's own, the default), mini-swe-agent and
its text variant, opencode, pi, openhands-sdk.

    env:
      type: palingenesis.rl.envs.harbor:HarborEnv
      args: {profiles: {palingenesis-default: 1, opencode: 1, pi: 1, mini-swe-agent: 1, openhands-sdk: 1}}

One profile per GRPO group (HarborEnv.episode_group, set by the trainer), varied across groups; logged as
metric/profile/<name> (and metric/profile/<name>/reward) and in the rollout's info. See docs/harness_profiles.md.
"""

from palingenesis.rl.envs.harness.profile import (
    FORMAT_ERROR,
    HarnessMix,
    HarnessProfile,
    Session,
    ToolSpec,
    available,
    get_profile,
    register,
)

__all__ = ["FORMAT_ERROR", "HarnessMix", "HarnessProfile", "Session", "ToolSpec", "available", "get_profile",
           "register"]
