#!/usr/bin/env python3
"""Trains the age-estimation network.

Expects a CSV with columns ``image_path`` and ``age``.

    python train_age.py --data data/ages.csv --img-dir data/utkface \
        --output checkpoints/age.pth --epochs 50 --batch-size 64 --device cuda

The age code is chosen with --encoding and is written into the checkpoint, so
AgeEstimator always decodes a model the way it was trained. See
benchmarks/age_encoding.py for why the default is ordinal.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from anyface_pp.models.age_estimator import (ENCODINGS, INPUT_SIZE, _AgeNet,
                                             decode_age, encode_age)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

TRAIN_TF = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    _NORM,
])
EVAL_TF = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    _NORM,
])


class AgeDataset(Dataset):
    """Face crops with integer age labels."""

    def __init__(self, csv_path: str, img_dir: str = ".",
                 encoding: str = "ordinal", train: bool = True):
        self.encoding = encoding
        self.transform = TRAIN_TF if train else EVAL_TF
        with open(csv_path) as f:
            self.samples = [(str(Path(img_dir) / r["image_path"]), int(r["age"]))
                            for r in csv.DictReader(f)]
        if not self.samples:
            raise ValueError(f"No rows in {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, age = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        target = encode_age(np.array([age]), self.encoding)[0]
        return self.transform(img), torch.from_numpy(target), age


@torch.no_grad()
def evaluate(net, loader, device, encoding) -> float:
    """Mean absolute error in years on *loader*."""
    net.eval()
    errs = []
    for xb, _, ages in loader:
        probs = torch.sigmoid(net(xb.to(device))).cpu().numpy()
        errs.append(np.abs(decode_age(probs, encoding) - ages.numpy()))
    return float(np.concatenate(errs).mean())


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    full = AgeDataset(args.data, args.img_dir, args.encoding, train=True)
    n_val = max(1, int(len(full) * args.val_frac))
    train_set, val_set = random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    # random_split shares one underlying dataset, so the validation half would
    # otherwise be read through the training augmentations.
    val_set.dataset = AgeDataset(args.data, args.img_dir, args.encoding, train=False)

    pin = device.type == "cuda"        # pinned memory is a no-op off CUDA
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=pin)
    logger.info("%d train / %d val images, encoding=%s",
                len(train_set), len(val_set), args.encoding)

    net = _AgeNet(encoding=args.encoding).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # The network emits logits; BCEWithLogitsLoss folds in the sigmoid and is
    # numerically stable where a separate sigmoid + BCELoss is not.
    criterion = torch.nn.BCEWithLogitsLoss()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(args.epochs):
        net.train()
        total = 0.0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(net(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        scheduler.step()

        val_mae = evaluate(net, val_loader, device, args.encoding)
        logger.info("Epoch %3d  loss=%.4f  val_MAE=%.2f y  lr=%.6f",
                    epoch + 1, total / len(train_set), val_mae,
                    scheduler.get_last_lr()[0])

        if val_mae < best:
            best = val_mae
            torch.save({"model": net.state_dict(), "encoding": args.encoding,
                        "val_mae": val_mae, "epoch": epoch + 1}, out)

    logger.info("Best val MAE %.2f y -> %s", best, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with columns: image_path, age")
    ap.add_argument("--img-dir", default=".", help="root directory for images")
    ap.add_argument("--output", "-o", default="checkpoints/age.pth")
    ap.add_argument("--encoding", choices=ENCODINGS, default="ordinal")
    ap.add_argument("--epochs", "-e", type=int, default=50)
    ap.add_argument("--batch-size", "-b", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
