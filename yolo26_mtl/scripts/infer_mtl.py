#!/usr/bin/env python3
"""Inference for the YOLO26-AnyFace++ MTL model.

Face boxes + 5 landmarks + gender/age/emotion per face. Headless.

    python yolo26_mtl/scripts/infer_mtl.py \
        --weights runs/mtl/exp/weights/best.pt \
        --source data/test_images/ --output-dir results/mtl --device cpu

This drives the underlying torch model rather than ``YOLO.predict``. The head
appends attribute columns after the keypoints, and the stock pose predictor
knows nothing about them -- the previous version of this file read
``result.mtl``, an attribute that nothing in ultralytics or in this repo ever
sets, so gender/age/emotion were silently absent from every result it wrote.

IMPORTANT: the attribute branches have no loss (see the README section "The
MTL subtree"), so unless you have trained them yourself the values below are
whatever the initialisation produced. The detection and landmark outputs are
the parts a pose-trained checkpoint actually optimises.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo26_mtl import build_mtl_model, registered_mtl_head  # noqa: E402
from yolo26_mtl.head_module.head_mtl import ATTR_DIM, decode_attrs  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
LANDMARK_COLORS = [(0, 255, 0), (0, 0, 255), (0, 255, 255),
                   (255, 0, 255), (255, 0, 0)]


def load_model(weights, device):
    """Load a trained checkpoint, or build an untrained model from the config."""
    if weights:
        with registered_mtl_head():
            ckpt = torch.load(weights, map_location=device, weights_only=False)
        model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model = model.float()
    else:
        model = build_mtl_model()
        print("WARNING: no --weights given; running an untrained model. "
              "Output is structurally valid and numerically meaningless.",
              file=sys.stderr)
    return model.to(device).eval()


def letterbox(image, imgsz):
    """Resize preserving aspect ratio, pad to a square. Returns (tensor, ratio, pad)."""
    h, w = image.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    x = torch.from_numpy(canvas[..., ::-1].transpose(2, 0, 1).copy()).float() / 255.0
    return x.unsqueeze(0), r, (left, top)


@torch.no_grad()
def predict(image, model, head, conf=0.3, iou=0.45, imgsz=640, device="cpu"):
    """Detect faces in one BGR image and decode each one's attributes."""
    from ultralytics.utils.nms import non_max_suppression

    x, ratio, (padx, pady) = letterbox(image, imgsz)
    y = model(x.to(device))
    y = y[0] if isinstance(y, (tuple, list)) else y

    if head.end2end_flag:
        # The head already ran top-k postprocess: rows are
        # [x1,y1,x2,y2, conf, cls, kpts..., attrs...]
        rows = y[0]
        rows = rows[rows[:, 4] >= conf]
    else:
        dets = non_max_suppression(y, conf_thres=conf, iou_thres=iou, nc=head.nc)
        rows = dets[0]

    nk = head.nk
    img_h, img_w = image.shape[:2]
    faces = []
    for row in rows:
        row = row.cpu()
        x1, y1, x2, y2 = ((row[:4].numpy() - [padx, pady, padx, pady]) / ratio)
        # Clip to the frame: undoing the letterbox can place a predicted box
        # in the padding, which is not part of the image.
        x1, x2 = float(np.clip(x1, 0, img_w)), float(np.clip(x2, 0, img_w))
        y1, y2 = float(np.clip(y1, 0, img_h)), float(np.clip(y2, 0, img_h))
        kpts = row[6:6 + nk].reshape(-1, 3).numpy()
        kpts[:, 0] = np.clip((kpts[:, 0] - padx) / ratio, 0, img_w)
        kpts[:, 1] = np.clip((kpts[:, 1] - pady) / ratio, 0, img_h)
        face = {
            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "confidence": round(float(row[4]), 4),
            "landmarks": [[round(float(k[0]), 1), round(float(k[1]), 1)]
                          for k in kpts],
        }
        face.update(decode_attrs(row[6 + nk:6 + nk + ATTR_DIM]))
        faces.append(face)
    return faces


def annotate(image, faces):
    """Draw boxes, landmarks and attribute labels. Returns a new image."""
    out = image.copy()
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    for f in faces:
        x, y, w, h = f["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        lines = [f"face {f['confidence']:.2f}",
                 f"{f['gender']} {f['gender_conf']:.2f}",
                 f"age {f['age']}",
                 f"{f['emotion']} {f['emotion_conf']:.2f}"]
        for i, text in enumerate(lines):
            cv2.putText(out, text, (x + 2, max(y - 6 - i * 14, 12 + i * 14)),
                        font, fs, (0, 255, 255), 1, cv2.LINE_AA)
        for k, (lx, ly) in enumerate(f["landmarks"]):
            cv2.circle(out, (int(lx), int(ly)), 2,
                       LANDMARK_COLORS[k % len(LANDMARK_COLORS)], -1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", "-w", default=None,
                    help="trained .pt; omitted builds an untrained model")
    ap.add_argument("--source", "-s", required=True,
                    help="image file, video file, or directory")
    ap.add_argument("--output-dir", "-o", default="results/mtl")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.weights, args.device)
    head = model.model[-1]

    source = Path(args.source)
    if source.is_dir():
        images = sorted(f for f in source.iterdir() if f.suffix.lower() in IMG_EXTS)
    elif source.suffix.lower() in IMG_EXTS:
        images = [source]
    elif source.suffix.lower() in VIDEO_EXTS:
        return run_video(source, model, head, args, out_dir)
    else:
        raise SystemExit(f"Unsupported source: {source}")

    results = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  skipped (unreadable): {path.name}", file=sys.stderr)
            continue
        faces = predict(image, model, head, args.conf, args.iou,
                        args.imgsz, args.device)
        cv2.imwrite(str(out_dir / f"{path.stem}_annotated.jpg"),
                    annotate(image, faces))
        results.append({"file": path.name, "faces": faces})
        print(f"  {path.name}: {len(faces)} face(s)")

    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"  {len(results)} image(s) -> {json_path}")


def run_video(source, model, head, args, out_dir):
    """Annotate a video.

    The previous version called the model with stream=True, never consumed the
    returned generator -- so inference never actually ran -- and wrote the
    untouched input frame to the output file.
    """
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path = out_dir / f"{source.stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            faces = predict(frame, model, head, args.conf, args.iou,
                            args.imgsz, args.device)
            writer.write(annotate(frame, faces))
            n += 1
    finally:
        cap.release()
        writer.release()
    print(f"  {source.name}: {n} frames -> {out_path}")


if __name__ == "__main__":
    main()
