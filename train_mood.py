#!/usr/bin/env python3
"""Training script for the mood/emotion classification module.

Trains the MobileNetV2-based mood classifier on a dataset of face images
with emotion labels.  Expects a CSV with columns ``image_path`` and
``emotion`` (one of: angry, disgusted, fearful, happy, sad, surprised, neutral).

Example
-------
python train_mood.py \
    --data fer2013_labeled.csv \
    --output checkpoints/mood_fer2013.pth \
    --epochs 30 --batch-size 64 --device mps
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms

from anyface_pp.models.mood_classifier import _MoodNet, EMOTION_LABELS, INPUT_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}


# ------------------------------------------------------------------ dataset
# ------------------------------------------------------------------
class MoodDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str = "."):
        self.samples = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["emotion"].strip().lower()
                if label not in EMOTION_TO_IDX:
                    continue
                self.samples.append((str(Path(img_dir) / row["image_path"]), EMOTION_TO_IDX[label]))

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = np.asarray(__import__("PIL").Image.open(img_path).convert("RGB"))
        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.long)


# ------------------------------------------------------------------ train loop
# ------------------------------------------------------------------
def train(args):
    device = torch.device(args.device)

    # split: 80% train, 20% val
    from torch.utils.data import random_split

    dataset = MoodDataset(args.data, args.img_dir)
    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    net = _MoodNet().to(device)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        net.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = net(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        scheduler.step()

        # validation
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = net(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total if total else 0

        if acc > best_acc:
            best_acc = acc
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": net.state_dict(), "labels": EMOTION_LABELS}, args.output)
            logger.info("Epoch %3d  loss=%.4f  val_acc=%.2f  ★ saved", epoch + 1, total_loss / n_train, acc)
        elif (epoch + 1) % 5 == 0:
            logger.info("Epoch %3d  loss=%.4f  val_acc=%.2f", epoch + 1, total_loss / n_train, acc)

    logger.info("Best val accuracy = %.2f%% → %s", best_acc * 100, args.output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with columns: image_path, emotion")
    ap.add_argument("--img-dir", default=".", help="Root directory for images")
    ap.add_argument("--output", "-o", default="checkpoints/mood.pth")
    ap.add_argument("--epochs", "-e", type=int, default=30)
    ap.add_argument("--batch-size", "-b", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="mps")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
