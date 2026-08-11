#!/usr/bin/env bash
###############################################################################
# SLURM script — Train the mood/emotion classifier on HPC
#
# Usage:
#   sbatch scripts/train_mood.slurm
###############################################################################

#SBATCH --job-name=anyface-mood
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/mood_%j.out
#SBATCH --error=hpc/logs/mood_%j.err
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

# ── activate environment ───────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate anyface
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "ERROR: No conda env 'anyface' and no .venv found."
    exit 1
fi

mkdir -p hpc/logs checkpoints output

python train_mood.py \
    --data   "$PROJECT_ROOT/data/emotions.csv" \
    --img-dir "$PROJECT_ROOT/data/fer2013" \
    --output  checkpoints/mood.pth \
    --epochs  30 \
    --batch-size 64 \
    --lr      1e-3 \
    --device  cuda

echo "=== Mood training finished: $SLURM_JOB_ID ==="
