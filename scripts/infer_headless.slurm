#!/usr/bin/env bash
###############################################################################
# SLURM script — Headless batch inference on HPC
#
# Usage:
#   sbatch scripts/infer_headless.slurm \
#       -- --source data/test_images/ --output-dir results/batch
#
# Everything after '--' is passed through to infer_headless.py
###############################################################################

#SBATCH --job-name=anyface-infer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/logs/infer_%j.out
#SBATCH --error=hpc/logs/infer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@domain.edu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── load modules ────────────────────────────────────────────────────────────
# module purge
# module load cuda/12.4
# module load python/3.11

# ── activate ────────────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate anyface
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "ERROR: No conda env 'anyface' and no .venv found."
    exit 1
fi

mkdir -p hpc/logs

# ── pass through all user arguments ────────────────────────────────────────
python scripts/infer_headless.py "$@"

echo "=== Inference finished: $SLURM_JOB_ID ==="
