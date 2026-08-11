#!/usr/bin/env python3
"""Headless Anyface++ inference for HPC batch processing.

No GUI dependencies (no cv2.imshow).  Processes images or videos from a
directory, writes annotated outputs and a JSON results file.

Designed to run inside a SLURM job over SSH with no display server.

Examples
--------
# Single image
python scripts/infer_headless.py \
    --source photo.jpg \
    --output-dir results/img \
    --device cuda

# Batch: every image in a directory
python scripts/infer_headless.py \
    --source data/test_images/ \
    --output-dir results/batch \
    --device cuda \
    --extensions jpg png jpeg bmp

# Video (headless — no window, just writes output file + JSON)
python scripts/infer_headless.py \
    --source data/test_clip.mp4 \
    --output-dir results/video \
    --device cuda \
    --video-fps 24 \
    --video-every 5            # process every Nth frame

# Resume / append results
python scripts/infer_headless.py \
    --source data/test_images/ \
    --output-dir results/batch \
    --results-file results/batch/all.jsonl \
    --device cuda

# With custom weights
python scripts/infer_headless.py \
    --source photo.jpg \
    --output-dir results \
    --age-weights checkpoints/age.pth \
    --mood-weights checkpoints/mood.pth \
    --face-model checkpoints/yolo26m_face.pt \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from anyface_pp.pipeline import AnyfacePP, FaceResult

# Image extensions we process
DEFAULT_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ helpers
# ------------------------------------------------------------------
def _collect_files(
    source: Path,
    extensions: set,
    recursive: bool = False,
) -> List[Path]:
    """Collect sorted file list from *source* (file or directory)."""
    if source.is_file():
        return [source]
    if recursive:
        return sorted(source.rglob("*"))
    return sorted(source.glob("*"))


def _is_image(p: Path, exts: set) -> bool:
    return p.is_file() and p.suffix.lower() in exts


def _is_video(p: Path, exts: set) -> bool:
    return p.is_file() and p.suffix.lower() in exts


def _result_to_dict(r: FaceResult) -> Dict:
    return {
        "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
        "confidence": round(r.confidence, 4),
        "age": round(r.age, 1),
        "emotion": r.emotion,
        "emotion_conf": round(r.emotion_conf, 4),
        "emotions": {k: round(v, 4) for k, v in r.emotions.items()},
    }


# ------------------------------------------------------------------ modes
# ------------------------------------------------------------------
def process_image(
    analyzer: AnyfacePP,
    img_path: Path,
    out_dir: Path,
    results_fh,
) -> int:
    """Process one image. Returns face count."""
    try:
        results = analyzer.run(str(img_path), annotate_image=True)
    except Exception as e:
        logger.error("Failed on %s: %s", img_path, e)
        return 0

    annotated = analyzer._annotated
    if annotated is not None:
        save = out_dir / f"{img_path.stem}_annotated.jpg"
        cv2.imwrite(str(save), annotated)

    # write JSONL result
    entry = {
        "file": str(img_path.name),
        "faces": [_result_to_dict(r) for r in results],
    }
    json.dump(entry, results_fh)
    results_fh.write("\n")
    results_fh.flush()

    logger.info("%-40s → %d face(s) → %s", img_path.name, len(results), save.name)
    return len(results)


def process_video(
    analyzer: AnyfacePP,
    vid_path: Path,
    out_dir: Path,
    results_fh,
    fps: float = 30.0,
    every: int = 1,
    max_frames: int = 0,
) -> int:
    """Process one video (headless). Returns total frames processed."""
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", vid_path)
        return 0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = out_dir / f"{vid_path.stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_vid), fourcc, fps, (w, h))

    total_faces = 0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every == 0:
                results = analyzer._analyze(frame)
                total_faces += len(results)

                # draw annotations
                annotated = analyzer._draw(frame.copy(), results)
                writer.write(annotated)

                # JSONL
                entry = {
                    "file": vid_path.name,
                    "frame": frame_idx,
                    "faces": [_result_to_dict(r) for r in results],
                }
                json.dump(entry, results_fh)
                results_fh.write("\n")
                results_fh.flush()

            frame_idx += 1
            if 0 < max_frames <= frame_idx:
                break

    finally:
        cap.release()
        writer.release()

    logger.info(
        "%-40s → %d frames processed, %d total faces → %s",
        vid_path.name, frame_idx, total_faces, out_vid.name,
    )
    return frame_idx


# ------------------------------------------------------------------ main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Headless Anyface++ inference for HPC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--source", "-s", required=True,
                    help="Image file, video file, or directory")
    ap.add_argument("--output-dir", "-o", default="output/inference",
                    help="Directory for annotated outputs & JSONL")
    ap.add_argument("--results-file", default=None,
                    help="JSONL results file (default: <output-dir>/results.jsonl)")

    # model weights
    ap.add_argument("--age-weights", default=None, help="Age-estimator .pth")
    ap.add_argument("--mood-weights", default=None, help="Mood-classifier .pth")
    ap.add_argument("--face-model", default=None, help="YOLO26 face-detector .pt")

    # device & perf
    ap.add_argument("--device", default="cuda",
                    choices=["cuda", "cpu", "mps"], help="Torch device")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="Detection confidence threshold")

    # batch / directory options
    ap.add_argument("--extensions", default=None,
                    help="Comma-separated image extensions (default: jpg jpeg png bmp tiff webp)")
    ap.add_argument("--recursive", "-r", action="store_true",
                    help="Recursively search subdirectories")

    # video options
    ap.add_argument("--video-fps", type=float, default=30.0,
                    help="Output video FPS")
    ap.add_argument("--video-every", type=int, default=1,
                    help="Process every Nth frame (1 = all frames)")
    ap.add_argument("--video-max-frames", type=int, default=0,
                    help="Max frames to process per video (0 = unlimited)")

    args = ap.parse_args()

    # ── parse extensions ────────────────────────────────────────────────────
    if args.extensions:
        exts = {("." + e.strip()) for e in args.extensions.split(",")}
    else:
        exts = DEFAULT_IMAGE_EXTS

    source = Path(args.source)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = Path(args.results_file) if args.results_file else out_dir / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # ── detect device ───────────────────────────────────────────────────────
    if args.device in ("cuda", "mps"):
        import torch
        available = torch.cuda.is_available() if args.device == "cuda" else (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if not available:
            logger.warning("%s unavailable — falling back to CPU", args.device.upper())
            args.device = "cpu"

    # ── initialise pipeline ─────────────────────────────────────────────────
    analyzer = AnyfacePP(
        device=args.device,
        face_model_path=args.face_model,
        age_weights_path=args.age_weights,
        mood_weights_path=args.mood_weights,
        conf_thres=args.conf,
    )
    logger.info("Pipeline ready: device=%s  conf=%.2f", analyzer._device, args.conf)

    # ── open results file ───────────────────────────────────────────────────
    results_fh = open(results_path, "a")
    t0 = time.time()
    total_faces = 0
    total_files = 0

    try:
        # ── single file ────────────────────────────────────────────────────
        if source.is_file():
            total_files = 1
            if _is_image(source, exts):
                total_faces = process_image(analyzer, source, out_dir, results_fh)
            elif _is_video(source, VIDEO_EXTS):
                total_faces = process_video(
                    analyzer, source, out_dir, results_fh,
                    fps=args.video_fps, every=args.video_every,
                    max_frames=args.video_max_frames,
                )
            else:
                logger.warning("Skipping unsupported file: %s", source)

        # ── directory ──────────────────────────────────────────────────────
        elif source.is_dir():
            files = _collect_files(source, exts, recursive=args.recursive)
            images = [f for f in files if _is_image(f, exts)]
            videos = [f for f in files if _is_video(f, VIDEO_EXTS)]
            total_files = len(images) + len(videos)

            logger.info("Found %d image(s), %d video(s) in %s",
                        len(images), len(videos), source)

            for i, img in enumerate(images):
                logger.info("[%d/%d] %s", i + 1, len(images), img.name)
                total_faces += process_image(analyzer, img, out_dir, results_fh)

            for i, vid in enumerate(videos):
                logger.info("[video %d/%d] %s", i + 1, len(videos), vid.name)
                total_faces += process_video(
                    analyzer, vid, out_dir, results_fh,
                    fps=args.video_fps, every=args.video_every,
                    max_frames=args.video_max_frames,
                )

        else:
            logger.error("Source not found: %s", source)
            sys.exit(1)

    finally:
        results_fh.close()

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Done. %d file(s), %d face(s) detected in %.1fs",
                total_files, total_faces, elapsed)
    logger.info("Results → %s", results_path)
    logger.info("Annotated   → %s/", out_dir)


if __name__ == "__main__":
    main()
