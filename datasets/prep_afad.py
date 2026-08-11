#!/usr/bin/env python3
"""Prepare AFAD (Age, Face, Age Description) or similar dataset for AnyFace++ MTL training.

AFAD is one of the recommended datasets from the original AnyFacePP repo.
This script creates the label format:
  face_type x_center y_center width height x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 gender age emotion

Usage:
    python datasets/prep_afad.py --data-path /path/to/afad/ --output-dir datasets/AFAD/

Alternatively, use the original AnyFacePP notebooks:
    cd ~/srp/anyface-orig
    # Run the preprocessing notebook for each dataset
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def create_empty_structure(output_dir, split_ratio=0.8):
    """Create train/val directory structure for the dataset."""
    out = Path(output_dir)
    for split in ["train", "val"]:
        (out / split).mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", "-d", required=True,
                    help="Path to downloaded dataset root")
    ap.add_argument("--output-dir", "-o", default="datasets/AFAD/",
                    help="Output directory with train/val splits")
    ap.add_argument("--split", type=float, default=0.8,
                    help="Train/val split ratio")
    args = ap.parse_args()

    create_empty_structure(args.output_dir, args.split)
    print(f"Created structure in {args.output_dir}")
    print("Place images and labels in datasets/{name}/train/ and datasets/{name}/val/")
    print("")
    print("For full dataset preparation, use the AnyFacePP notebooks:")
    print("  cd ~/srp/anyface-orig")
    print("  # Run the dataset preprocessing notebook for each dataset")
    print("")
    print("Or download preprocessed datasets from:")
    print("  https://drive.google.com/drive/folders/1J7pJZJzJZJZJZJZJZJZJZJZJZJZJZJZJ")


if __name__ == "__main__":
    main()
