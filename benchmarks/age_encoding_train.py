#!/usr/bin/env python3
"""Trains the age head with each code on one identical synthetic task.

benchmarks/age_encoding.py argues from the structure of the codes alone. This
script checks that the argument survives contact with gradient descent: same
backbone, same schedule, same images, same seed -- only the code differs.

The task is deliberately trivial. Age is readable straight off image
brightness, so a model that cannot do well here is not being limited by
vision. That isolates the decoding step, which is the thing under test. None
of this says anything about accuracy on real faces; it says that a code whose
units carry place value turns small unit errors into large age errors, and
that this happens in training and not just in theory.

    python3 benchmarks/age_encoding_train.py --seeds 0 1 2
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyface_pp.models.age_estimator import ENCODINGS  # noqa: E402


def make_dataset(root: Path, n: int, seed: int) -> Path:
    """Faces stand in as flat colour patches whose brightness encodes age."""
    from PIL import Image
    import csv

    rng = np.random.default_rng(seed)
    (root / "imgs").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        age = int(rng.integers(5, 80))
        value = int(255 * age / 100)
        px = np.clip(value + rng.normal(0, 6, (64, 64, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(px).save(root / "imgs" / f"{i:03d}.png")
        rows.append({"image_path": f"{i:03d}.png", "age": age})

    csv_path = root / "ages.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, ["image_path", "age"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def run_one(csv_path: Path, img_dir: Path, out: Path, encoding: str,
            epochs: int, seed: int) -> float:
    """Train once and return the best validation MAE in years."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "train_age.py"),
         "--data", str(csv_path), "--img-dir", str(img_dir),
         "--output", str(out), "--encoding", encoding,
         "--epochs", str(epochs), "--batch-size", "16",
         "--workers", "0", "--device", "cpu", "--seed", str(seed)],
        capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"training failed for encoding={encoding}")
    for line in reversed(proc.stdout.splitlines() + proc.stderr.splitlines()):
        if "Best val MAE" in line:
            return float(line.split("Best val MAE")[1].split("y")[0])
    raise SystemExit(f"no MAE reported for encoding={encoding}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=160, help="synthetic images")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--out-json",
                    default=str(ROOT / "benchmarks" / "age_encoding_train.json"))
    args = ap.parse_args()

    results = {enc: [] for enc in ENCODINGS}
    for seed in args.seeds:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = make_dataset(tmp, args.n, seed)
            for enc in ENCODINGS:
                mae = run_one(csv_path, tmp / "imgs", tmp / f"{enc}.pth",
                              enc, args.epochs, seed)
                results[enc].append(mae)
                print(f"seed {seed}  {enc:<8} best val MAE {mae:6.2f} y", flush=True)

    print(f"\n{args.n} synthetic images, {args.epochs} epochs, "
          f"{len(args.seeds)} seeds -- best validation MAE, years:")
    print(f"{'encoding':<10}" + "".join(f"{f'seed {s}':>10}" for s in args.seeds)
          + f"{'mean':>10}")
    summary = {"n": args.n, "epochs": args.epochs, "seeds": args.seeds,
               "encodings": {}}
    for enc in ENCODINGS:
        vals = results[enc]
        print(f"{enc:<10}" + "".join(f"{v:>10.2f}" for v in vals)
              + f"{np.mean(vals):>10.2f}")
        summary["encodings"][enc] = {"per_seed": vals, "mean": float(np.mean(vals))}

    ratio = summary["encodings"]["binary"]["mean"] / summary["encodings"]["ordinal"]["mean"]
    summary["binary_over_ordinal"] = float(ratio)
    print(f"\nbinary is {ratio:.1f}x worse on a task where brightness gives "
          f"the answer away.")

    out = Path(args.out_json)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
