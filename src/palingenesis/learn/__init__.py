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
  forget    purge a note from every file, prune the versions that held it, verify
  learner   Learner, the Python API
  cli       pgs learn

Experimental: the recipe defaults are hypotheses until benchmarked, and evaluation gates, publishing and
serving are not implemented yet.
"""

import warnings


class ExperimentalWarning(UserWarning):
    """palingenesis.learn is experimental: its interfaces and defaults may change."""


warnings.warn(
    "palingenesis.learn is experimental: interfaces and defaults may change, and no evaluation gate checks a "
    "trained adapter yet.",
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
