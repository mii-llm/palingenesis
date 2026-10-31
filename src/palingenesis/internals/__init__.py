"""Reading a model's internals: residual-stream capture, linear probes, lenses, checkpoint comparison, causal checks.

    sites       where the residual stream is (decoder layers, final norm, head), residual index r = 0..n
    hooks       Recorder (read at positions, bounded memory, early stop), edit_residual, steer, ablate, patch
    positions   positions from the chat template: last_prompt, assistant / answer / reasoning spans
    capture     capture_dataset -> ActivationStore (sharded safetensors on disk)
    probes      grouped splits, logistic / diff-of-means / LDA probes, control task, calibration, sweep, Probe artifact
    stats       AUROC with a cluster bootstrap, Platt scaling, Brier, ECE, TPR at a fixed FPR
    similarity  linear CKA (debiased), compare_stores: what a training stage changed per layer
    lens        logit lens, tuned lens, lens_report
    causal      direction_effect (steering dose-response + ablation vs random directions), patching_sweep, generate
    score       ProbeScorer (apply / gate), score_rows (dataset labels), probe_reward (RL reward function)

The design, its research grounding and the worked example are in the guide (mkdocs_docs/guides/internals.md) and the
research notes (mkdocs_docs/architecture/interpretability.md). CLI: `pgs internals --help`.
"""

from palingenesis.internals.capture import ActivationStore, capture_dataset, load_rows
from palingenesis.internals.causal import decision_margins, direction_effect, generate, patching_sweep
from palingenesis.internals.hooks import Recorder, Selection, ablate, active_hooks, edit_residual, patch, steer
from palingenesis.internals.lens import TunedLens, check_readout, fit_tuned_lens, lens_report
from palingenesis.internals.positions import POSITIONS, Encoder
from palingenesis.internals.probes import Probe, evaluate, grouped_split, sweep, train_probe
from palingenesis.internals.score import ProbeScorer, probe_reward, score_rows
from palingenesis.internals.similarity import compare_stores, linear_cka
from palingenesis.internals.sites import Sites, find_sites, parse_layers

__all__ = [
    "POSITIONS",
    "ActivationStore",
    "Encoder",
    "Probe",
    "ProbeScorer",
    "Recorder",
    "Selection",
    "Sites",
    "TunedLens",
    "ablate",
    "active_hooks",
    "capture_dataset",
    "check_readout",
    "compare_stores",
    "decision_margins",
    "direction_effect",
    "edit_residual",
    "evaluate",
    "find_sites",
    "fit_tuned_lens",
    "generate",
    "grouped_split",
    "lens_report",
    "linear_cka",
    "load_rows",
    "parse_layers",
    "patch",
    "patching_sweep",
    "probe_reward",
    "score_rows",
    "steer",
    "sweep",
    "train_probe",
]
