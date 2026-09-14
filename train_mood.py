#!/usr/bin/env python3
"""Trains the mood/emotion classifier.

Expects a CSV with columns ``image_path`` and ``emotion``, the latter one of:
angry, disgusted, fearful, happy, sad, surprised, neutral.

    python train_mood.py --data data/emotions.csv --img-dir data/fer2013 \
        --output checkpoints/mood.pth --epochs 30 --batch-size 64 --device cuda

Emotion datasets are badly imbalanced -- FER2013 is 1.5% "disgusted" -- so
this selects checkpoints on macro-F1 rather than accuracy, splits the
validation set stratified by class, and reports per-class recall every epoch.
See benchmarks/mood_metrics.py for why.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from anyface_pp.models.mood_classifier import (EMOTION_LABELS, INPUT_SIZE,
                                               _MoodNet)
from anyface_pp.utils.metrics import format_report, summary
from anyface_pp.utils.splits import stratified_split

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}

_TAIL = [
    transforms.Grayscale(num_output_channels=3),   # FER2013 is grayscale
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
]
TRAIN_TF = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    *_TAIL,
])
# No flip, no rotation: validation must measure the model, not the dice.
EVAL_TF = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    *_TAIL,
])


class MoodDataset(Dataset):
    """Face crops with emotion labels."""

    def __init__(self, csv_path: str, img_dir: str = ".", train: bool = True):
        self.transform = TRAIN_TF if train else EVAL_TF
        self.samples = []
        skipped = 0
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                label = row["emotion"].strip().lower()
                if label not in EMOTION_TO_IDX:
                    skipped += 1
                    continue
                self.samples.append((str(Path(img_dir) / row["image_path"]),
                                     EMOTION_TO_IDX[label]))
        if not self.samples:
            raise ValueError(f"No usable rows in {csv_path}")
        if skipped:
            # Silently dropping rows used to hide label typos entirely.
            logger.warning("Skipped %d row(s) with an unrecognised emotion", skipped)

    @property
    def labels(self):
        return [lab for _, lab in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


@torch.no_grad()
def evaluate(net, loader, device):
    """Return (y_true, y_pred) over *loader*."""
    net.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        y_pred.extend(net(xb.to(device)).argmax(1).cpu().tolist())
        y_true.extend(yb.tolist())
    return y_true, y_pred


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    train_ds = MoodDataset(args.data, args.img_dir, train=True)
    eval_ds = MoodDataset(args.data, args.img_dir, train=False)
    tr_idx, va_idx = stratified_split(train_ds.labels, args.val_frac, args.seed)
    # Two dataset objects, so the validation half is never read through the
    # training augmentations.
    train_set, val_set = Subset(train_ds, tr_idx), Subset(eval_ds, va_idx)

    counts = Counter(train_ds.labels)
    logger.info("%d train / %d val images", len(train_set), len(val_set))
    logger.info("class balance: %s",
                ", ".join(f"{EMOTION_LABELS[c]} {counts[c]}"
                          for c in sorted(counts)))

    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=pin)

    net = _MoodNet(pretrained=not args.from_scratch).to(device)
    logger.info("backbone init: %s",
                "random" if args.from_scratch else "ImageNet (MobileNetV2)")

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.class_weights:
        # Inverse-frequency weights, normalised to mean 1 so the loss scale
        # stays comparable to the unweighted run.
        w = np.array([len(train_ds) / (len(counts) * max(counts[c], 1))
                      for c in range(len(EMOTION_LABELS))], dtype=np.float32)
        w /= w.mean()
        weight = torch.tensor(w, device=device)
        logger.info("class weights: %s",
                    ", ".join(f"{l} {v:.2f}" for l, v in zip(EMOTION_LABELS, w)))
    else:
        weight = None
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, history = -1.0, 0, []

    for epoch in range(args.epochs):
        net.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(net(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        scheduler.step()

        y_true, y_pred = evaluate(net, val_loader, device)
        m = summary(y_true, y_pred, EMOTION_LABELS)
        history.append({"epoch": epoch + 1, "loss": total / len(train_set),
                        "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                        "never_predicted": m["never_predicted"]})

        # Selected on macro-F1: accuracy would rank a model that ignores the
        # rare classes above one that does not.
        improved = m["macro_f1"] > best
        if improved:
            best, best_epoch = m["macro_f1"], epoch + 1
            torch.save({"model": net.state_dict(), "labels": EMOTION_LABELS,
                        "macro_f1": best, "accuracy": m["accuracy"],
                        "epoch": best_epoch}, out)
        # Every epoch is logged -- the old script printed only on improvement
        # or every fifth epoch, so most runs looked stalled.
        logger.info("Epoch %3d  loss=%.4f  acc=%.4f  macro-F1=%.4f%s%s",
                    epoch + 1, total / len(train_set), m["accuracy"],
                    m["macro_f1"], "  * saved" if improved else "",
                    f"  [never predicted: {', '.join(m['never_predicted'])}]"
                    if m["never_predicted"] else "")

    logger.info("Best macro-F1 %.4f at epoch %d -> %s", best, best_epoch, out)
    y_true, y_pred = evaluate(net, val_loader, device)
    logger.info("Final-epoch validation report:\n%s",
                format_report(y_true, y_pred, EMOTION_LABELS))

    if args.history_json:
        Path(args.history_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_json).write_text(json.dumps(history, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with columns: image_path, emotion")
    ap.add_argument("--img-dir", default=".", help="root directory for images")
    ap.add_argument("--output", "-o", default="checkpoints/mood.pth")
    ap.add_argument("--epochs", "-e", type=int, default=30)
    ap.add_argument("--batch-size", "-b", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--class-weights", action="store_true",
                    help="inverse-frequency weighting in the loss")
    ap.add_argument("--from-scratch", action="store_true",
                    help="random backbone init instead of ImageNet")
    ap.add_argument("--history-json", default=None,
                    help="write per-epoch metrics to this file")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
