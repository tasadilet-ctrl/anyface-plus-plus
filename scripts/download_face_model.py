#!/usr/bin/env python3
"""Downloads YOLO face-detection weights.

Kept as an explicit step rather than an implicit download inside
FaceDetector: the weights come from a third-party HuggingFace repository, so
fetching them should be a decision the user makes, not a side effect of
constructing an object.

    python3 scripts/download_face_model.py
    python3 scripts/download_face_model.py --out /path/to/weights.pt
"""
import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyface_pp.models.face_detector import (  # noqa: E402
    DEFAULT_FACE_WEIGHTS, FACE_WEIGHTS_URL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_FACE_WEIGHTS)
    ap.add_argument("--url", default=FACE_WEIGHTS_URL)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file already exists")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} already exists ({out.stat().st_size / 1e6:.1f} MB). "
              f"Use --force to re-download.")
        return 0

    print(f"Downloading {args.url}\n         -> {out}")
    urllib.request.urlretrieve(args.url, out)
    size_mb = out.stat().st_size / 1e6
    print(f"Saved {out} ({size_mb:.1f} MB)")

    # Confirm the weights are what they claim to be, rather than trusting the
    # filename -- a failed download that lands as HTML would otherwise only
    # surface much later, inside YOLO().
    try:
        from ultralytics import YOLO
        names = YOLO(str(out)).names
        print(f"Classes: {names}")
        if not any("face" in str(n).lower() for n in names.values()):
            print("WARNING: these weights have no 'face' class; FaceDetector "
                  "will reject them.", file=sys.stderr)
            return 1
    except Exception as e:  # pragma: no cover - depends on the downloaded file
        print(f"ERROR: downloaded file did not load as a YOLO model: {e}",
              file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
