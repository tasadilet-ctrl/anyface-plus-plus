#!/usr/bin/env python3
"""Anyface++ demo — analyse an image, video, or live webcam.

Requires a display server (X11 / macOS Quartz) for video/webcam modes.
For headless HPC use, run ``scripts/infer_headless.py`` instead.

Examples
--------
# Single image (works headless — just saves file)
python demo.py image photo.jpg

# Video file (opens a preview window)
python demo.py video clip.mp4 --output output.mp4

# Live webcam (press 'q' to quit, 's' to save a frame)
python demo.py webcam

# Custom model weights
python demo.py image photo.jpg --age-weights checkpoints/age.pth --mood-weights checkpoints/mood.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

from anyface_pp.pipeline import AnyfacePP, FaceResult

SAVE_DIR = Path("output")


def _has_display() -> bool:
    """Check whether a display server is available."""
    if sys.platform == "darwin":
        return True  # macOS always has a display
    return "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ


def _print_results(results: list[FaceResult], frame_num: int | None = None):
    if not results:
        return
    tag = f"[frame {frame_num}] " if frame_num is not None else ""
    print(f"{tag}Detected {len(results)} face(s):")
    for i, r in enumerate(results, 1):
        print(
            f"  Face #{i}:  box=({r.x1},{r.y1})-{(r.x2),(r.y2)}  "
            f"age~{r.age:.0f}  mood={r.emotion}({r.emotion_conf:.1%})  "
            f"det-conf={r.confidence:.2f}"
        )


# ------------------------------------------------------------------ modes
# ------------------------------------------------------------------
def mode_image(analyzer: AnyfacePP, path: str):
    results, annotated = analyzer.run_with_image(path)
    _print_results(results)

    SAVE_DIR.mkdir(exist_ok=True)
    out = SAVE_DIR / "demo_image.jpg"
    cv2.imwrite(str(out), annotated)
    print(f"Annotated image saved → {out}")


def mode_video(analyzer: AnyfacePP, path: str, output: str, fps: float = 30.0):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    headless = not _has_display()
    if headless:
        print("⚠ No display detected — running headless (writing output only, no preview).")
        print("  For full headless batch processing use scripts/infer_headless.py")

    frame_num = 0
    try:
        for results, annotated in analyzer.run_video(path):
            _print_results(results, frame_num)
            writer.write(annotated)
            if not headless:
                cv2.imshow("Anyface++", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_num += 1
    finally:
        cap.release()
        writer.release()
        if not headless:
            cv2.destroyAllWindows()

    print(f"Output video → {output}")


def mode_webcam(analyzer: AnyfacePP, cam_id: int = 0):
    if not _has_display():
        print("ERROR: Webcam mode requires a display server.")
        print("On HPC / SSH, use scripts/infer_headless.py for batch inference instead.")
        sys.exit(1)

    SAVE_DIR.mkdir(exist_ok=True)
    snap = 0

    for results, annotated in analyzer.run_video(cam_id):
        _print_results(results)
        cv2.imshow("Anyface++ Webcam", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            out = SAVE_DIR / f"webcam_snap_{snap:03d}.jpg"
            cv2.imwrite(str(out), annotated)
            print(f"Snapshot saved → {out}")
            snap += 1

    cv2.destroyAllWindows()


# ------------------------------------------------------------------ main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Anyface++ demo (requires display for video/webcam)")
    sub = ap.add_subparsers(dest="mode", required=True)

    # image
    p_img = sub.add_parser("image", help="Analyse a single image (works headless)")
    p_img.add_argument("path", type=str, help="Image file path")

    # video
    p_vid = sub.add_parser("video", help="Analyse a video file")
    p_vid.add_argument("path", type=str, help="Video file path")
    p_vid.add_argument("--output", "-o", default="output/demo_video.mp4", help="Output path")
    p_vid.add_argument("--fps", type=float, default=30.0)

    # webcam
    p_cam = sub.add_parser("webcam", help="Live webcam analysis (requires display)")
    p_cam.add_argument("--cam-id", type=int, default=0)

    # global opts
    ap.add_argument("--device", default="mps",
                    help='Torch device (default: "mps" on Mac, "cuda" on GPU)')
    ap.add_argument("--face-model", default=None,
                    help="Custom YOLO26 face-detector .pt (default: yolo26n.pt)")
    ap.add_argument("--age-weights", default=None, help="Age-estimator .pth")
    ap.add_argument("--mood-weights", default=None, help="Mood-classifier .pth")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="Detection confidence threshold")

    args = ap.parse_args()

    # Auto-select device
    if args.device == "mps":
        import torch
        if not torch.backends.mps.is_available():
            args.device = "cpu"
            print("MPS not available, falling back to CPU")

    analyzer = AnyfacePP(
        device=args.device,
        face_model_path=args.face_model,
        age_weights_path=args.age_weights,
        mood_weights_path=args.mood_weights,
        conf_thres=args.conf,
    )
    print(f"Anyface++ initialised (device={analyzer._device})")

    if args.mode == "image":
        mode_image(analyzer, args.path)
    elif args.mode == "video":
        mode_video(analyzer, args.path, args.output, args.fps)
    elif args.mode == "webcam":
        mode_webcam(analyzer, args.cam_id)


if __name__ == "__main__":
    main()
