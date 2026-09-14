#!/usr/bin/env python3
"""Trains the YOLO26-AnyFace++ MTL model's detection and landmark branches.

READ THIS FIRST. The attribute branches (gender, age, emotion) have NO LOSS.
Ultralytics' task="pose" optimises boxes and keypoints; it has no targets for
face attributes and never touches those weights, which stay at their random
initialisation for the whole run. Training here is therefore a face detector
with landmarks. Making the attributes learn needs three things this repo does
not have: per-face attribute labels in the dataset, a loss that masks them to
positive anchors, and a trainer that passes them through. See the README
section "The MTL subtree".

    python3 yolo26_mtl/scripts/train_mtl.py --data datasets/anyface.yaml \
        --epochs 300 --batch 8 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo26_mtl import DEFAULT_CONFIG, registered_mtl_head  # noqa: E402

BANNER = """
------------------------------------------------------------------
 The gender / age / emotion branches have no loss and will NOT be
 trained by this run. Boxes and landmarks will be. See the README
 section "The MTL subtree" before reading anything into the
 attribute outputs afterwards.
------------------------------------------------------------------
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset YAML")
    ap.add_argument("--model", default=str(DEFAULT_CONFIG), help="model config")
    ap.add_argument("--weights", default="", help="pretrained .pt to resume from")
    ap.add_argument("--epochs", "-e", type=int, default=300)
    ap.add_argument("--batch", "-b", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", "-w", type=int, default=8)
    ap.add_argument("--project", default="runs/mtl")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--save-period", type=int, default=10)
    ap.add_argument("--cache", action="store_true")
    args = ap.parse_args()

    print(BANNER, file=sys.stderr)

    from ultralytics import YOLO

    src = args.weights if args.weights and os.path.exists(args.weights) else args.model
    print(f"Building model from: {src}")

    # The head must be registered for the whole run: ultralytics re-parses the
    # config when it builds the trainer's model, not only here.
    with registered_mtl_head():
        model = YOLO(src)
        print(model.info())
        print(f"\nTraining for {args.epochs} epochs (task=pose)...")
        model.train(
            data=args.data,
            task="pose",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            patience=args.patience,
            save_period=args.save_period,
            cache=args.cache,
        )

    print("Training complete.")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")
    print(BANNER, file=sys.stderr)


if __name__ == "__main__":
    main()
