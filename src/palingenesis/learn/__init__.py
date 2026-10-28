"""Scoped continual learning (experimental): consolidate notes and documents into one LoRA per scope.

A small model deployed for a company, team or person needs to know things no base model knows. `pgs learn`
keeps that knowledge per scope (base -> org -> team -> user): notes stay in the prompt until a session
distills them into the scope's adapter, with a teacher that sees the scope's current notes in context.
Adapters of a chain stack exactly into one LoRA, so one base model serves every scope.

Modules:
  config    LearnConfig (strict YAML + overrides)
  store     the store layout, scopes, records, locking, atomic writes
  ingest    documents into chunks, source diffs, atomic facts, the contradiction check
  signals   deliberate signals: explicit commands, corrections; quoted / pasted / tool text excluded
  pii       the PII filter run before anything is stored
  generate  facts, questions, probes and near-miss prompts (model-backed, or the deterministic stub)
  render    the notes block shared by the teacher and serving; BM25
  recipe    the rows of a session and their teachers
  adapter   MLP-only LoRA via PEFT, the frozen ancestor stack, exact stacking and export
  trainer   one training session of a scope
  evaluate  metrics of a version paired against the live one, the exclusive matcher, the detectors
  gates     publishing gates over the metrics (uncalibrated defaults until null-session calibration)
  report    report.json, report.md and diagnostics.jsonl per session
  forget    purge a note from every file, prune the versions that held it, verify
  serving   publish a scope's stacked adapter as <scope>@v<n> (vLLM, SGLang), parity canary, rollback, unload
  vllm_plugin  vLLM LoRA resolver plugin: loads published scope adapters from the store on demand
  proxy     OpenAI-compatible proxy: API key -> scope, notes in the prompt, routing, deliberate signals
  scheduler pgs learn sleep: run, evaluate, set live and publish the scopes with pending notes, under a budget
  learner   Learner, the Python API
  cli       pgs learn

Experimental: the recipe defaults and the gate thresholds are hypotheses until benchmarked and calibrated,
and the serving path is tested on CPU against fake servers until the GPU integration tests run.
"""

import warnings


class ExperimentalWarning(UserWarning):
    """palingenesis.learn is experimental: its interfaces and defaults may change."""


warnings.warn(
    "palingenesis.learn is experimental: interfaces and defaults may change, and the evaluation gates use "
    "uncalibrated thresholds.",
    ExperimentalWarning,
    stacklevel=2,
)

from palingenesis.learn.config import LearnConfig, LearnConfigError  # noqa: E402
from palingenesis.learn.learner import Learner, Scope  # noqa: E402
from palingenesis.learn.store import ScopeId, Store, StoreError  # noqa: E402

__all__ = [
    "ExperimentalWarning",
    "LearnConfig",
    "LearnConfigError",
    "Learner",
    "Scope",
    "ScopeId",
    "Store",
    "StoreError",
]
