#!/usr/bin/env python3
"""Train YOLO26-AnyFace++ MTL model."""

import argparse
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to dataset YAML")
    ap.add_argument("--model", default="yolo26n-mtl.yaml", help="Model config")
    ap.add_argument("--weights", default="", help="Pretrained weights .pt")
    ap.add_argument("--epochs", "-e", type=int, default=300)
    ap.add_argument("--batch", "-b", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", "-w", type=int, default=8)
    ap.add_argument("--project", default="runs/mtl")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--save-period", type=int, default=10)
    ap.add_argument("--cache", action="store_true", help="Cache images in RAM")
    args = ap.parse_args()

    # Build model — load from weights if provided, else from config YAML
    print(f"Building model from: {args.model}")
    if args.weights and os.path.exists(args.weights):
        model = YOLO(args.weights)
    else:
        model = YOLO(args.model)

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

    print("Training complete!")
    print(f"Best model: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
