#!/usr/bin/env python3
"""Training script for the age-estimation module.

Trains the binary-code age network on a dataset of face images with age
annotations.  Expects a CSV file with columns ``image_path`` and ``age``.

Example
-------
python train_age.py \
    --data utkface.csv \
    --output checkpoints/age_utkface.pth \
    --epochs 50 --batch-size 32 --device mps
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

from anyface_pp.models.age_estimator import _AgeNet, INPUT_SIZE, NUM_BITS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ dataset
# ------------------------------------------------------------------
class AgeDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str = "."):
        self.samples = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((str(Path(img_dir) / row["image_path"]), int(row["age"])))

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]
        img = np.asarray(__import__("PIL").Image.open(img_path).convert("RGB"))
        tensor = self.transform(img)
        # age → binary code
        code = np.array([(age >> b) & 1 for b in range(NUM_BITS - 1, -1, -1)], dtype=np.float32)
        return tensor, torch.tensor(code, dtype=torch.float32)


# ------------------------------------------------------------------ train loop
# ------------------------------------------------------------------
def train(args):
    device = torch.device(args.device)
    dataset = AgeDataset(args.data, args.img_dir)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    net = _AgeNet().to(device)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.BCELoss()

    for epoch in range(args.epochs):
        net.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = net(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        scheduler.step()
        avg = total_loss / len(dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("Epoch %3d  loss=%.4f  lr=%.6f", epoch + 1, avg, scheduler.get_last_lr()[0])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.output)
    logger.info("Saved → %s", args.output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with columns: image_path, age")
    ap.add_argument("--img-dir", default=".", help="Root directory for images")
    ap.add_argument("--output", "-o", default="checkpoints/age.pth")
    ap.add_argument("--epochs", "-e", type=int, default=50)
    ap.add_argument("--batch-size", "-b", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="mps")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
