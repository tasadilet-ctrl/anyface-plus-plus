#!/usr/bin/env bash
###############################################################################
# YOLO26 MTL Setup — patches ultralytics with AnyFace++ MTL head
# Run ON TRX50 (or wherever ultralytics is installed)
###############################################################################
set -euo pipefail

cd ~/srp/anyface-plus-plus
source .venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ULTRA=$(python -c "import ultralytics; print(ultralytics.__file__)" | head -1)
echo "Patching ultralytics at: $ULTRA"

# ── 1. Copy MTL head module ──
cp "$PROJECT_ROOT/yolo26_mtl/head_module/head_mtl.py" "$ULTRA/nn/modules/head_mtl.py"
echo "  ✓ Copied head_mtl.py"

# ── 2. Register in __all__ ──
if ! grep -q "MTLPose" "$ULTRA/nn/modules/head.py"; then
    sed -i 's/"v10Detect",/"v10Detect",\n    "MTLPose",/' "$ULTRA/nn/modules/head.py"
    echo "" >> "$ULTRA/nn/modules/head.py"
    echo "from .head_mtl import MTLPose  # noqa: F401" >> "$ULTRA/nn/modules/head.py"
    echo "  ✓ Registered MTLPose in head.py __all__"
else
    echo "  ✓ MTLPose already registered"
fi

# ── 3. Copy model YAML ──
cp "$PROJECT_ROOT/yolo26_mtl/configs/yolo26n-mtl.yaml" "$ULTRA/cfg/models/26/yolo26n-mtl.yaml"
echo "  ✓ Copied yolo26n-mtl.yaml"

# ── 4. Verify ──
python -c "
from ultralytics.nn.modules.head_mtl import MTLPose
import torch

ch = (256, 512, 1024)
h = MTLPose(nc=1, kpt_shape=(5, 3), reg_max=1, end2end=False, ch=ch)
h.train()
x = [torch.randn(2, c, 80, 80) for c in ch]
o = h(x)

print('MTLPose forward test:')
print(f'  boxes:   {o[\"boxes\"].shape}   # (B, 4*reg_max, anchors)')
print(f'  scores:  {o[\"scores\"].shape}   # (B, nc, anchors)')
print(f'  kpts:    {o[\"kpts\"].shape}     # (B, nk, anchors)')
print(f'  gender:  {o[\"gender\"].shape}   # (B, 3)')
print(f'  age:     {o[\"age\"].shape}      # (B,)')
print(f'  emotion: {o[\"emotion\"].shape}  # (B, 8)')
print('All shapes OK ✓')
"

echo ""
echo "═══════════════════════════════════════"
echo "  YOLO26 MTL Setup Complete! ✓"
echo "═══════════════════════════════════════"
echo ""
echo "Next: train_mtl.py or sbatch scripts/train_mtl.slurm"
