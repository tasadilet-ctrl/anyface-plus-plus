#!/usr/bin/env bash
###############################################################################
# SLURM script — Train the age-estimation module on HPC
#
# Usage:
#   sbatch scripts/train_age.slurm
#
# Customise the #SBATCH flags below to match your cluster's GPUs & quotas.
###############################################################################

#SBATCH --job-name=anyface-age
#SBATCH --partition=gpu              # adjust to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1                     # A100 / V100 / RTX …
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/age_%j.out
#SBATCH --error=hpc/logs/age_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@domain.edu

set -euo pipefail

# ── resolve project root (two levels up from this script) ──────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── load modules (uncomment & adjust for your cluster) ──────────────────────
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

# ── create log directory ───────────────────────────────────────────────────
mkdir -p hpc/logs checkpoints output

# ── run training ───────────────────────────────────────────────────────────
python train_age.py \
    --data   "$PROJECT_ROOT/data/ages.csv" \
    --img-dir "$PROJECT_ROOT/data/utkface" \
    --output  checkpoints/age.pth \
    --epochs  50 \
    --batch-size 64 \
    --lr      1e-3 \
    --device  cuda

echo "=== Age training finished: $SLURM_JOB_ID ==="
