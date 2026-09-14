"""Dataset splitting helpers.

Lives in the package rather than in train_mood.py so that both the trainer and
the tests import it the same way. A root-level script is not part of the
installed package, so a test that imported it would pass locally and fail
under a clean install -- which has bitten this repo before.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def stratified_split(labels: Sequence[int], val_frac: float,
                     seed: int) -> Tuple[List[int], List[int]]:
    """Split indices into (train, val), preserving each class's proportion.

    A uniformly random split can drop a rare class from validation entirely.
    With FER2013's balance and a few hundred images that happens about half
    the time (benchmarks/mood_metrics.py), and the class is then invisible to
    model selection no matter which metric is used.

    Every class with at least two samples is guaranteed to appear on both
    sides. A class with exactly one sample goes to train, since a validation
    set is useless for a class the model has never seen.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("no labels to split")

    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        if len(idx) == 1:
            train_idx.extend(idx.tolist())
            continue
        n_val = int(round(len(idx) * val_frac))
        n_val = min(max(n_val, 1), len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx
