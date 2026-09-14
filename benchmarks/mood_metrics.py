#!/usr/bin/env python3
"""Why the mood trainer selects on macro-F1 and splits stratified.

Emotion datasets are badly imbalanced. Everything here is computed from
FER2013's published class counts, so it is exact, needs no images, no weights
and no GPU, and is rerun in CI.

Three things are shown:

1. What accuracy costs a model for dropping a rare class entirely -- the
   failure mode that plain accuracy is least able to see.
2. How that compares to macro-F1, which averages over classes rather than
   over samples.
3. How often an unstratified validation split loses a rare class altogether,
   which makes the class invisible to model selection at any metric.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyface_pp.models.mood_classifier import EMOTION_LABELS  # noqa: E402
from anyface_pp.utils.metrics import summary  # noqa: E402

# FER2013 training-set class counts (Goodfellow et al., 2013; 35,887 images).
FER2013_COUNTS = {"angry": 4953, "disgusted": 547, "fearful": 5121,
                  "happy": 8989, "sad": 6077, "surprised": 4002,
                  "neutral": 6198}


def _labels_from_counts(counts):
    return np.concatenate([[EMOTION_LABELS.index(k)] * v for k, v in counts.items()])


def drop_class_cost(y_true, dropped_idx, fallback_idx):
    """Metrics for an otherwise-perfect model that never predicts one class."""
    y_pred = y_true.copy()
    y_pred[y_true == dropped_idx] = fallback_idx
    return summary(y_true, y_pred, EMOTION_LABELS)


def unstratified_miss_rate(counts, n, val_frac, trials, rng):
    """P(the rarest class has no sample in an unstratified validation split)."""
    total = sum(counts.values())
    rare = min(counts.values())
    k = max(1, round(n * rare / total))
    n_val = int(n * val_frac)
    misses = sum((rng.permutation(n)[:n_val] < k).sum() == 0 for _ in range(trials))
    return k, misses / trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=str(ROOT / "benchmarks" / "mood_metrics.json"))
    args = ap.parse_args()

    total = sum(FER2013_COUNTS.values())
    out = {"dataset": "FER2013", "total": total,
           "counts": FER2013_COUNTS, "drop_class": {}, "split": {}}

    print(f"FER2013 class balance ({total:,} images)")
    for k, v in sorted(FER2013_COUNTS.items(), key=lambda kv: kv[1]):
        print(f"  {k:<11}{v:>6}{v / total:>8.2%}")

    y_true = _labels_from_counts(FER2013_COUNTS)
    print("\nAn otherwise-perfect model that never predicts one class:")
    print(f"{'dropped class':<15}{'share':>8}{'accuracy':>11}{'macro-F1':>11}")
    for name in sorted(FER2013_COUNTS, key=FER2013_COUNTS.get):
        idx = EMOTION_LABELS.index(name)
        fallback = EMOTION_LABELS.index("sad" if name != "sad" else "neutral")
        m = drop_class_cost(y_true, idx, fallback)
        share = FER2013_COUNTS[name] / total
        print(f"{name:<15}{share:>8.2%}{m['accuracy']:>11.4f}{m['macro_f1']:>11.4f}")
        out["drop_class"][name] = {"share": share, "accuracy": m["accuracy"],
                                   "macro_f1": m["macro_f1"]}

    worst = min(FER2013_COUNTS, key=FER2013_COUNTS.get)
    d = out["drop_class"][worst]
    print(f"\nDropping '{worst}' entirely costs {1 - d['accuracy']:.1%} of accuracy "
          f"but {1 - d['macro_f1']:.1%} of macro-F1 -- a "
          f"{(1 - d['macro_f1']) / (1 - d['accuracy']):.0f}x stronger signal.")
    out["accuracy_cost"] = 1 - d["accuracy"]
    out["macro_f1_cost"] = 1 - d["macro_f1"]

    rng = np.random.default_rng(args.seed)
    print(f"\nP(rarest class absent from an unstratified 20% val split), "
          f"{args.trials:,} trials:")
    print(f"{'subset size':>12}{'rare samples':>14}{'miss rate':>11}")
    for n in (200, 500, 1000, 5000, 35887):
        k, rate = unstratified_miss_rate(FER2013_COUNTS, n, 0.2, args.trials, rng)
        print(f"{n:>12,}{k:>14}{rate:>11.1%}")
        out["split"][str(n)] = {"rare_samples": k, "miss_rate": rate}
    print("\nThe full dataset is safe; quick runs on a few hundred images are not,"
          "\nwhich is exactly when the split is most often taken at random.")

    p = Path(args.out_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
