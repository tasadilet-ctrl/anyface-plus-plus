#!/usr/bin/env python3
"""
Dataset Preprocessor for AnyFace++ MTL
Converts various face datasets to the MTL label format:
  face_type x_center y_center width height x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 gender age emotion

Supported datasets:
- RAF-DB: Real-world Affective Faces Dataset (emotion + landmarks)
- UTKFace: Age and gender dataset
- FER2013: Facial Expression Recognition
- Custom: Any dataset with face detection annotations
"""

import os
import sys
import json
import csv
import argparse
import random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


def detect_faces_yolo26(image_path, model=None):
    """Detect faces using YOLO26 and return bounding boxes."""
    if model is None:
        from ultralytics import YOLO
        model = YOLO("yolo26n.pt")  # Will download if not present

    image = cv2.imread(str(image_path))
    if image is None:
        return []

    results = model(image, conf=0.5, verbose=False)
    faces = []

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            faces.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": conf
            })

    return faces


def preprocess_rafdb(rafdb_dir, output_dir, split="train"):
    """
    Preprocess RAF-DB dataset.

    RAF-DB structure:
    - Image/Predict/ or Image/Train/
    - Emotion/ - contains emotion annotations
    - Identity/ - contains face landmarks
    """
    rafdb_path = Path(rafdb_dir)
    output_path = Path(output_dir)

    # Create output directories
    for split_dir in ["train", "val"]:
        (output_path / split_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Load emotion annotations
    emotion_file = rafdb_path / "Emotion" / "emotion_list.txt"
    emotions = {}
    with open(emotion_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                emotions[parts[0]] = parts[1]

    # Load face landmarks
    identity_dir = rafdb_path / "Identity"
    landmarks = {}
    for identity_file in identity_dir.glob("*.txt"):
        with open(identity_file) as f:
            content = f.read().strip()
            if content:
                landmarks[identity_file.stem] = content

    # Process images
    image_dir = rafdb_path / "Image" / "Align_croppped"
    image_files = list(image_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} RAF-DB images")

    # Shuffle and split
    random.seed(42)
    random.shuffle(image_files)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    def process_image(img_path, output_split):
        """Process a single image and create label file."""
        img_name = img_path.name
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Get emotion (last 2 chars of filename)
        emotion_idx = emotions.get(img_name, "neutral")

        # Map emotion to index (RAF-DB uses: angry, disgust, fear, happy, sad, surprise, neutral)
        emotion_map = {
            "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
            "sad": 4, "surprise": 5, "neutral": 6
        }
        emotion_val = emotion_map.get(emotion_idx, 6)

        # Copy image
        cv2.imwrite(str(output_path / output_split / "images" / img_name), img)

        # Create label - face is centered in cropped image
        # RAF-DB images are face-cropped, so face is the whole image
        # Format: face_type x_center y_center width height x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 gender age emotion
        label = f"0 0.5 0.5 1.0 1.0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 {emotion_val}\n"

        with open(output_path / output_split / "labels" / img_name.replace(".jpg", ".txt"), "w") as f:
            f.write(label)

    # Process train/val splits
    for img_path in train_files:
        process_image(img_path, "train")

    for img_path in val_files:
        process_image(img_path, "val")

    print(f"RAF-DB preprocessing complete:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")


def preprocess_utkface(utkface_dir, output_dir):
    """
    Preprocess UTKFace dataset.

    UTKFace filenames: age_gender_blur_date.jpg
    """
    utkface_path = Path(utkface_dir)
    output_path = Path(output_dir)

    # Create output directories
    for split_dir in ["train", "val"]:
        (output_path / split_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = list(utkface_path.rglob("*.jpg"))
    print(f"Found {len(image_files)} UTKFace images")

    # Filter valid files
    valid_files = []
    for img_path in image_files:
        parts = img_path.stem.split("_")
        if len(parts) >= 3:
            try:
                age = int(parts[0])
                gender = int(parts[1])
                if 0 <= age <= 116 and 0 <= gender <= 2:
                    valid_files.append((img_path, age, gender))
            except ValueError:
                continue

    # Shuffle and split
    random.seed(42)
    random.shuffle(valid_files)
    split_idx = int(len(valid_files) * 0.8)
    train_files = valid_files[:split_idx]
    val_files = valid_files[split_idx:]

    def process_image(img_path, age, gender, output_split):
        """Process a single image and create label file."""
        img_name = img_path.name
        img = cv2.imread(str(img_path))
        if img is None:
            return

        # Copy image
        cv2.imwrite(str(output_path / output_split / "images" / img_name), img)

        # Create label - face is centered in UTKFace images
        label = f"0 0.5 0.5 1.0 1.0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 {gender} {age} -1\n"

        with open(output_path / output_split / "labels" / img_name.replace(".jpg", ".txt"), "w") as f:
            f.write(label)

    # Process splits
    for img_path, age, gender in train_files:
        process_image(img_path, age, gender, "train")

    for img_path, age, gender in val_files:
        process_image(img_path, age, gender, "val")

    print(f"UTKFace preprocessing complete:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")


def preprocess_fer2013(fer2013_csv, output_dir):
    """
    Preprocess FER2013 dataset from CSV.

    FER2013 CSV format: emotion,pixels
    Emotion mapping: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
    """
    output_path = Path(output_dir)

    # Create output directories
    for split_dir in ["train", "val"]:
        (output_path / split_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Load and process CSV
    images = []
    with open(fer2013_csv) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for i, row in enumerate(reader):
            if len(row) >= 2:
                emotion = int(row[0])
                pixels = np.array([int(p) for p in row[1].split()], dtype=np.uint8)
                image = pixels.reshape(48, 48)
                images.append((i, emotion, image))

    print(f"Found {len(images)} FER2013 images")

    # Split (FER2013 has predefined splits, but we'll use 80/20 for simplicity)
    random.seed(42)
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    def process_image(idx, emotion, image, output_split):
        """Process a single image and create label file."""
        img_name = f"fer2013_{idx:06d}.jpg"

        # Save image (convert to 3-channel for consistency)
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(output_path / output_split / "images" / img_name), image_color)

        # Create label
        label = f"0 0.5 0.5 1.0 1.0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 {emotion}\n"

        with open(output_path / output_split / "labels" / img_name.replace(".jpg", ".txt"), "w") as f:
            f.write(label)

    # Process splits
    for idx, emotion, image in train_images:
        process_image(idx, emotion, image, "train")

    for idx, emotion, image in val_images:
        process_image(idx, emotion, image, "val")

    print(f"FER2013 preprocessing complete:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")


def create_dataset_yaml(output_dir, dataset_names):
    """Create dataset.yaml for training."""
    yaml_content = """# AnyFace++ MTL Dataset Config
# Combined from multiple datasets

train: [
"""
    for name in dataset_names:
        yaml_content += f'    "{output_dir}/{name}/train/",\n'

    yaml_content += """]
val: [
"""
    for name in dataset_names:
        yaml_content += f'    "{output_dir}/{name}/val/",\n'

    yaml_content += """]

nc: 1
kpt_shape: [5, 3]

names: ["face"]
"""

    yaml_path = Path(output_dir) / "combined.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Created dataset.yaml at {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="AnyFace++ Dataset Preprocessor")
    parser.add_argument("--dataset", required=True,
                        choices=["rafdb", "utkface", "fer2013", "combine"],
                        help="Dataset to preprocess")
    parser.add_argument("--input", "-i", required=True,
                        help="Input path (dataset dir or CSV file)")
    parser.add_argument("--output", "-o", default="datasets/",
                        help="Output directory")
    parser.add_argument("--datasets", nargs="*",
                        help="Dataset names to combine (for --dataset combine)")

    args = parser.parse_args()

    if args.dataset == "rafdb":
        preprocess_rafdb(args.input, args.output)
    elif args.dataset == "utkface":
        preprocess_utkface(args.input, args.output)
    elif args.dataset == "fer2013":
        preprocess_fer2013(args.input, args.output)
    elif args.dataset == "combine":
        if args.datasets:
            create_dataset_yaml(args.output, args.datasets)
        else:
            print("For combine mode, use: --datasets dataset1 dataset2 ...")

if __name__ == "__main__":
    main()
