"""Classification metrics for imbalanced label sets.

Plain accuracy is close to useless on an emotion dataset. FER2013 is 1.5%
"disgusted", so a model that never predicts that class at all still scores
98.5% of whatever the accuracy ceiling is -- the number barely moves while a
whole class is being dropped. Selecting checkpoints on accuracy therefore
selects, in part, for ignoring rare classes.

Macro-F1 averages over classes instead of over samples, so an unpredicted
class costs a full 1/num_classes. These are implemented here in numpy rather
than pulled from scikit-learn, which is not otherwise a dependency.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int],
                     num_classes: int) -> np.ndarray:
    """Counts matrix ``C`` where ``C[t, p]`` is true class *t* called *p*."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.trace(cm) / total) if total else 0.0


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    """Recall per class; NaN for classes with no samples present."""
    support = cm.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.diag(cm) / support
    return np.where(support > 0, rec, np.nan)


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    """F1 per class; NaN for classes with no samples present.

    A class that is never predicted and has support scores 0, not NaN -- that
    is the case the metric exists to expose.
    """
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1).astype(np.float64)
    predicted = cm.sum(0).astype(np.float64)
    denom = support + predicted
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * tp / denom
    f1 = np.where(denom > 0, f1, 0.0)
    return np.where(support > 0, f1, np.nan)


def macro_f1(cm: np.ndarray) -> float:
    """Mean F1 over the classes actually present in the labels."""
    f1 = per_class_f1(cm)
    return float(np.nanmean(f1)) if not np.all(np.isnan(f1)) else 0.0


def classes_never_predicted(cm: np.ndarray) -> List[int]:
    """Classes that have support but were never predicted for any sample."""
    return [int(i) for i in np.where((cm.sum(1) > 0) & (cm.sum(0) == 0))[0]]


def summary(y_true: Sequence[int], y_pred: Sequence[int],
            labels: Sequence[str]) -> Dict[str, object]:
    """All of the above in one dict, keyed by label name where per-class."""
    cm = confusion_matrix(y_true, y_pred, len(labels))
    rec, f1 = per_class_recall(cm), per_class_f1(cm)
    return {
        "accuracy": accuracy(cm),
        "macro_f1": macro_f1(cm),
        "per_class_recall": {l: (None if np.isnan(r) else float(r))
                             for l, r in zip(labels, rec)},
        "per_class_f1": {l: (None if np.isnan(v) else float(v))
                         for l, v in zip(labels, f1)},
        "support": {l: int(s) for l, s in zip(labels, cm.sum(1))},
        "never_predicted": [labels[i] for i in classes_never_predicted(cm)],
        "confusion_matrix": cm.tolist(),
    }


def format_report(y_true: Sequence[int], y_pred: Sequence[int],
                  labels: Sequence[str]) -> str:
    """Human-readable per-class table."""
    s = summary(y_true, y_pred, labels)
    lines = [f"{'class':<12}{'support':>8}{'recall':>9}{'f1':>8}"]
    for l in labels:
        rec, f1 = s["per_class_recall"][l], s["per_class_f1"][l]
        lines.append(f"{l:<12}{s['support'][l]:>8}"
                     f"{'   --  ' if rec is None else f'{rec:>9.3f}'}"
                     f"{'   -- ' if f1 is None else f'{f1:>8.3f}'}")
    lines.append(f"{'':12}{'':8}{'':9}{'':8}")
    lines.append(f"accuracy {s['accuracy']:.4f}   macro-F1 {s['macro_f1']:.4f}")
    if s["never_predicted"]:
        lines.append(f"never predicted: {', '.join(s['never_predicted'])}")
    return "\n".join(lines)
