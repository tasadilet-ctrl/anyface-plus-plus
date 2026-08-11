#!/usr/bin/env bash
###############################################################################
# SLURM — Train YOLO26-AnyFace++ MTL model
###############################################################################

#SBATCH --job-name=anyface-mtl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=hpc/logs/mtl_%j.out
#SBATCH --error=hpc/logs/mtl_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@domain.edu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate
mkdir -p hpc/logs

python yolo26_mtl/scripts/train_mtl.py \
    --data datasets/anyface.yaml \
    --model yolo26n-mtl.yaml \
    --epochs 300 \
    --batch 16 \
    --imgsz 640 \
    --device cuda \
    --workers 8 \
    --cache

echo "=== MTL training done: $SLURM_JOB_ID ==="
