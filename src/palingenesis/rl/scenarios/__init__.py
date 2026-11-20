"""Multi-role RL: episodes in which several roles (models, adapters, frozen partners) interact.

    scenario: {type: take_away}                 # in the RL config: `pgs rl` then runs episodes of the scenario
    roles:
      first:  {}                                # the trained policy
      second: {partner: "scripted:oracle"}      # a frozen partner

runtime    episodes, sessions, channels, branches and credit; `play` runs a scenario without a model
base       the Scenario class (rows, roles, verifier, scripted partners, audit) and the registry
library    starter scenarios: take_away (a two-player game), ledger_sum (a lead and its workers), checked_answer
           (a solver and a verifier)
pipeline   the RL pipeline's episodes: trained sessions through the rollout loop, comparison groups
credit     per-role advantages, their normalisation and each role's share of the loss
export     episodes from the rollout dumps as SFT, DPO and distillation rows
placement  a cluster description turned into a layout of trainers and rollout engines
"""

from palingenesis.rl.scenarios.base import BUILTIN, Scenario, build_scenario, scenario_class
from palingenesis.rl.scenarios.runtime import Channel, Episode, LiveSession, Session, play

__all__ = [
    "BUILTIN",
    "Channel",
    "Episode",
    "LiveSession",
    "Scenario",
    "Session",
    "build_scenario",
    "play",
    "scenario_class",
]
