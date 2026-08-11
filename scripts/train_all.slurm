#!/usr/bin/env bash
###############################################################################
# SLURM script — Train both age + mood modules sequentially on one GPU
#
# Usage:
#   sbatch scripts/train_all.slurm
###############################################################################

#SBATCH --job-name=anyface-all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=hpc/logs/train_all_%j.out
#SBATCH --error=hpc/logs/train_all_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@domain.edu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# module purge
# module load cuda/12.4
# module load python/3.11

if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate anyface
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

mkdir -p hpc/logs checkpoints output

echo "========== Age Training =========="
python train_age.py \
    --data   data/ages.csv \
    --img-dir data/utkface \
    --output checkpoints/age.pth \
    --epochs 50 --batch-size 64 --lr 1e-3 --device cuda

echo ""
echo "========== Mood Training =========="
python train_mood.py \
    --data   data/emotions.csv \
    --img-dir data/fer2013 \
    --output checkpoints/mood.pth \
    --epochs 30 --batch-size 64 --lr 1e-3 --device cuda

echo ""
echo "=== All training complete: $SLURM_JOB_ID ==="
