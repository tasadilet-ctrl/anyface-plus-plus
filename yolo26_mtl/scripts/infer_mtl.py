#!/usr/bin/env python3
"""Inference for YOLO26-AnyFace++ MTL model.

Runs face detection + landmarks + gender + age + emotion on images/videos.
Headless (no GUI) — works on HPC via SSH.

Example:
    python yolo26_mtl/scripts/infer_mtl.py \
        --weights runs/mtl/exp/weights/best.pt \
        --source data/test_images/ \
        --output-dir results/mtl/ \
        --device cuda
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

GENDER_LABELS = ["female", "male", "unsure"]
EMOTION_LABELS = ["angry", "happy", "fear", "sad", "surprise", "disgust", "neutral", "unsure"]
FACE_LABELS = {0: "human", 1: "animal", 2: "cartoon"}


def predict(image_path, model, conf=0.3, iou=0.1, device="cuda"):
    """Run inference on a single image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    results = model.predict(
        source=image, imgsz=640, conf=conf, iou=iou, device=device, verbose=False
    )

    h, w, _ = image.shape
    faces = []

    for result in results:
        boxes = result.boxes.cpu().numpy()
        if len(boxes) == 0:
            continue

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf_val = float(box.conf[0])
            label = int(box.cls[0]) if hasattr(box, 'cls') and box.cls is not None else 0

            face = {
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "confidence": round(conf_val, 4),
                "face_type": FACE_LABELS.get(label, "unknown"),
            }

            # MTL predictions
            if hasattr(result, 'mtl') and result.mtl is not None:
                mtl = result.mtl[i]

                # Gender
                gender_logits = mtl[0:3].cpu().numpy()
                gender_idx = np.argmax(gender_logits)
                face["gender"] = GENDER_LABELS[gender_idx]
                face["gender_conf"] = round(float(gender_logits[gender_idx]), 4)

                # Age
                age = int(mtl[3:4][0].cpu().numpy())
                face["age"] = max(0, min(116, age))

                # Emotion
                emo_logits = mtl[4:].cpu().numpy()
                emo_idx = np.argmax(emo_logits)
                face["emotion"] = EMOTION_LABELS[emo_idx]
                face["emotion_conf"] = round(float(emo_logits[emo_idx]), 4)

            # Landmarks
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                kps = result.keypoints.data[i].cpu().numpy()
                face["landmarks"] = [
                    [round(float(kp[0]), 1), round(float(kp[1]), 1)]
                    for kp in kps
                ]

            # Annotate
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = max(0.4, min(0.7, (w + h) / 1280))
            color = (0, 255, 0) if face["face_type"] == "human" else (255, 0, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            y_off = max(y1 - 5, 15)

            cv2.putText(image, f"Face: {face['face_type']}",
                        (x1 + 2, y_off), font, fs, (0, 255, 0), 1)
            y_off += int(h / 30)
            if "gender" in face:
                cv2.putText(image, f"Gender: {face['gender']}",
                            (x1 + 2, y_off), font, fs, (255, 255, 0), 1)
                y_off += int(h / 30)
            if "age" in face:
                cv2.putText(image, f"Age: {face['age']}",
                            (x1 + 2, y_off), font, fs, (255, 0, 255), 1)
                y_off += int(h / 30)
            if "emotion" in face:
                cv2.putText(image, f"Emotion: {face['emotion']}",
                            (x1 + 2, y_off), font, fs, (0, 255, 255), 1)

            # Draw landmarks
            if "landmarks" in face:
                colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255),
                          (255, 0, 255), (255, 0, 0)]
                for k, (lx, ly) in enumerate(face["landmarks"]):
                    cv2.circle(image, (int(lx), int(ly)), 2,
                               colors[k % len(colors)], -1)

            faces.append(face)

    return faces, image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", "-w", required=True,
                    help="Path to trained .pt weights")
    ap.add_argument("--source", "-s", required=True,
                    help="Image file, video file, or directory")
    ap.add_argument("--output-dir", "-o", default="results/mtl",
                    help="Output directory")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)
    iou = args.iou

    source = Path(args.source)
    results_all = {}

    # Single image
    if source.is_file() and source.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
        faces, annotated = predict(source, model, args.conf, args.iou, args.device)
        save = out_dir / f"{source.stem}_annotated.jpg"
        cv2.imwrite(str(save), annotated)
        entry = {"file": source.name, "faces": faces}
        json_path = out_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(entry, f, indent=2)
        print(f"  {source.name}: {len(faces)} face(s) → {save}")
        print(f"  Results → {json_path}")

    # Directory
    elif source.is_dir():
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = sorted([f for f in source.iterdir() if f.suffix.lower() in exts])
        all_results = []

        for img in images:
            try:
                faces, annotated = predict(img, model, args.conf, args.iou, args.device)
                save = out_dir / f"{img.stem}_annotated.jpg"
                cv2.imwrite(str(save), annotated)
                all_results.append({"file": img.name, "faces": faces})
                print(f"  {img.name}: {len(faces)} face(s)")
            except Exception as e:
                print(f"  ✗ {img.name}: {e}")

        json_path = out_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Total: {len(images)} images → {json_path}")

    # Video
    elif source.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
        cap = cv2.VideoCapture(str(source))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = out_dir / f"{source.stem}_annotated.mp4"
        writer = cv2.VideoWriter(str(out_vid), fourcc, fps, (w, h))
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                ret = model([frame], imgsz=640, conf=args.conf, iou=iou,
                            device=args.device, verbose=False, stream=True)
                writer.write(frame)
                frame_idx += 1
        finally:
            cap.release()
            writer.release()

        print(f"  {source.name}: {frame_idx} frames → {out_vid}")


if __name__ == "__main__":
    main()
