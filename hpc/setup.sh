#!/usr/bin/env bash
###############################################################################
# HPC setup helper — run once after SSHing into your cluster
#
# Usage:
#   bash hpc/setup.sh
#
# This script:
#   1. Creates the conda environment from environment.yml
#   2. Verifies CUDA is available in PyTorch
#   3. Pre-downloads YOLO26n weights (so the first SLURM job doesn't stall)
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           Anyface++ HPC Setup                           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Create conda env ────────────────────────────────────────────────────
echo "▶ Creating conda environment 'anyface' ..."
conda env create -f "$SCRIPT_DIR/environment.yml" --quiet
echo "  ✓ Environment created."

# ── 2. Activate and verify ─────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate anyface

echo ""
echo "▶ Verifying CUDA ..."
python -c "
import torch
print(f'  PyTorch  {torch.__version__}')
print(f'  CUDA avail {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU count  {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'    GPU {i}:  {torch.cuda.get_device_name(i)}  (mem={torch.cuda.get_device_properties(i).total_mem / 1e9:.0f} GB)')
else:
    print('  ⚠  CUDA NOT detected — check module load cuda/xx or GPU partition.')
"

# ── 3. Pre-download YOLO26n weights ────────────────────────────────────────
echo ""
echo "▶ Pre-downloading YOLO26n weights (this takes ~30s) ..."
python -c "
from ultralytics import YOLO
print('  Downloading yolo26n.pt ...')
model = YOLO('yolo26n.pt')
print('  ✓ Weights cached.')
" 2>&1 | grep -v '^$'

echo ""
echo "═══ Setup complete! ═══"
echo ""
echo "Next steps:"
echo "  1. Edit #SBATCH flags in scripts/*.slurm for your cluster"
echo "  2. Transfer your datasets to the HPC"
echo "  3. Submit training:  sbatch scripts/train_all.slurm"
echo "  4. Submit inference: sbatch scripts/infer_headless.slurm -- --source data/ --output-dir results/"
echo ""
