#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# palingenesis — the asynchronous evaluator of a training run, in an allocation of its own
# ═══════════════════════════════════════════════════════════════════════════════
#
# The training config says eval_tasks.placement: async and eval_tasks.service: auto. Submit this before or after
# the training job (the evaluator waits for the trainer's <output_dir>/eval_tasks/spec.json); both jobs need the
# same shared file system for train.output_dir:
#
#   sbatch scripts/eval_slurm.sh runs/rl            # transport push:       pgs eval serve --dir runs/rl
#   sbatch scripts/eval_slurm.sh runs/sft watch     # transport checkpoint: pgs eval watch runs/sft
#   sbatch scripts/eval_slurm.sh runs/sft watch --once    # after training: every checkpoint there, then exit
#
# The evaluator uses every GPU of the allocation (one engine each; eval_tasks.engine.replicas for more per GPU),
# refuses to start on a GPU the trainer uses, and ends after the run's last evaluation. Its results go to
# <output_dir>/eval_tasks/step_<n>/ and the trainer logs them at step n.
#
#SBATCH --job-name=pgs-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.eval.out
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

OUTPUT_DIR="${1:?usage: sbatch scripts/eval_slurm.sh <train.output_dir> [serve | watch] [options]}"
MODE="${2:-serve}"
shift $(( $# >= 2 ? 2 : 1 ))

case "$MODE" in
    serve) exec python -m palingenesis.eval.service serve --dir "$OUTPUT_DIR" "$@" ;;
    watch) exec python -m palingenesis.eval.service watch "$OUTPUT_DIR" "$@" ;;
    *) echo "mode must be serve (transport push) or watch (transport checkpoint), got $MODE" >&2; exit 1 ;;
esac
