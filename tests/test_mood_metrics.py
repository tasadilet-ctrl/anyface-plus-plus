"""Tests for the mood branch: imbalance-aware metrics and splitting.

The defects being guarded:

  * checkpoints were selected on plain accuracy, which barely moves when a
    rare class is dropped entirely;
  * the validation half was read through the training augmentations, so the
    number model selection used was partly random;
  * the split was uniform, so a rare class could be missing from validation.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from anyface_pp.models.mood_classifier import EMOTION_LABELS
from anyface_pp.utils.metrics import (accuracy, classes_never_predicted,
                                      confusion_matrix, macro_f1,
                                      per_class_f1, per_class_recall, summary)
from anyface_pp.utils.splits import stratified_split

ROOT = Path(__file__).resolve().parent.parent

# FER2013's real balance: 'disgusted' is 1.5% of the data.
FER = {"angry": 4953, "disgusted": 547, "fearful": 5121, "happy": 8989,
       "sad": 6077, "surprised": 4002, "neutral": 6198}


def _fer_labels():
    return np.concatenate([[EMOTION_LABELS.index(k)] * v for k, v in FER.items()])


# ---------------------------------------------------------------- metrics
def test_confusion_matrix_counts():
    cm = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 0], 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]
    assert cm.sum() == 4
    assert accuracy(cm) == pytest.approx(2 / 4)


def test_confusion_matrix_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="shape mismatch"):
        confusion_matrix([0, 1], [0], 2)


def test_dropping_the_rarest_class_barely_dents_accuracy():
    """The reason macro-F1 is the selection metric."""
    y = _fer_labels()
    pred = y.copy()
    pred[y == EMOTION_LABELS.index("disgusted")] = EMOTION_LABELS.index("sad")
    m = summary(y, pred, EMOTION_LABELS)

    assert m["never_predicted"] == ["disgusted"]
    # A whole class gone, and accuracy notices almost nothing...
    assert m["accuracy"] > 0.98
    # ...while macro-F1 takes an order-of-magnitude larger hit.
    assert (1 - m["macro_f1"]) > 9 * (1 - m["accuracy"])


def test_macro_f1_is_insensitive_to_which_class_is_dropped():
    """Accuracy's penalty tracks class frequency; macro-F1's does not."""
    y = _fer_labels()
    accs, f1s = [], []
    for name in FER:
        pred = y.copy()
        other = "sad" if name != "sad" else "neutral"
        pred[y == EMOTION_LABELS.index(name)] = EMOTION_LABELS.index(other)
        m = summary(y, pred, EMOTION_LABELS)
        accs.append(m["accuracy"])
        f1s.append(m["macro_f1"])
    assert max(accs) - min(accs) > 0.20      # 0.75 .. 0.98
    assert max(f1s) - min(f1s) < 0.07        # 0.80 .. 0.85


def test_never_predicted_class_scores_zero_f1_not_nan():
    cm = confusion_matrix([0, 0, 1], [0, 0, 0], 2)
    f1 = per_class_f1(cm)
    assert f1[1] == 0.0 and not np.isnan(f1[1])
    assert classes_never_predicted(cm) == [1]


def test_absent_class_is_nan_not_zero():
    """A class with no samples must not drag macro-F1 down."""
    cm = confusion_matrix([0, 0], [0, 0], 3)      # classes 1 and 2 absent
    assert np.isnan(per_class_recall(cm)[1])
    assert np.isnan(per_class_f1(cm)[2])
    assert macro_f1(cm) == pytest.approx(1.0)


def test_perfect_prediction_scores_one():
    y = _fer_labels()
    m = summary(y, y.copy(), EMOTION_LABELS)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["never_predicted"] == []


# ---------------------------------------------------------------- splitting
def test_stratified_split_keeps_every_class_on_both_sides():
    labels = [0] * 100 + [1] * 4 + [2] * 50        # class 1 is the rare one
    tr, va = stratified_split(labels, 0.2, seed=0)
    assert set(labels[i] for i in va) == {0, 1, 2}
    assert set(labels[i] for i in tr) == {0, 1, 2}


def test_stratified_split_is_a_partition():
    labels = [i % 7 for i in range(304)]
    tr, va = stratified_split(labels, 0.2, seed=3)
    assert sorted(tr + va) == list(range(len(labels)))
    assert not (set(tr) & set(va))


def test_stratified_split_is_seed_deterministic():
    labels = [i % 7 for i in range(304)]
    assert stratified_split(labels, 0.2, 1) == stratified_split(labels, 0.2, 1)
    assert stratified_split(labels, 0.2, 1) != stratified_split(labels, 0.2, 2)


def test_stratified_split_beats_uniform_on_rare_class_retention():
    """The uniform split loses the rare class often; stratified never does."""
    labels = np.array([0] * 160 + [1] * 3 + [2] * 37)
    rng = np.random.default_rng(0)
    uniform_misses = 0
    for _ in range(400):
        perm = rng.permutation(len(labels))
        if 1 not in labels[perm[: int(0.2 * len(labels))]]:
            uniform_misses += 1
    assert uniform_misses / 400 > 0.3            # measured ~0.5
    for seed in range(50):
        _, va = stratified_split(labels, 0.2, seed)
        assert 1 in labels[va]


def test_two_sample_class_still_reaches_validation():
    """round(2 * 0.2) is 0, so without the clamp this class vanishes from val."""
    labels = [0] * 80 + [1, 1]
    tr, va = stratified_split(labels, 0.2, seed=0)
    assert 1 in [labels[i] for i in va]
    assert 1 in [labels[i] for i in tr]


def test_class_is_never_entirely_consumed_by_validation():
    """A large val_frac must still leave every class something to train on."""
    labels = [0] * 80 + [1, 1] + [2] * 3
    tr, va = stratified_split(labels, 0.9, seed=0)
    for cls in (0, 1, 2):
        assert cls in [labels[i] for i in tr], f"class {cls} has no training data"
        assert cls in [labels[i] for i in va], f"class {cls} has no validation data"


def test_single_sample_class_goes_to_train():
    labels = [0] * 20 + [1]
    tr, va = stratified_split(labels, 0.2, seed=0)
    assert labels.index(1) in tr and labels.index(1) not in va


def test_split_rejects_impossible_fraction():
    with pytest.raises(ValueError, match="val_frac"):
        stratified_split([0, 1], 0.0, 0)


# ------------------------------------------------- validation preprocessing
def _load_train_mood():
    """Import the root-level script by path.

    It is not part of the installed package, so a plain `import train_mood`
    would depend on the working directory.
    """
    spec = importlib.util.spec_from_file_location("train_mood", ROOT / "train_mood.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_transform_is_deterministic_and_train_is_not():
    """Validation must measure the model, not the augmentation dice.

    The old trainer used torch random_split, which shares one dataset object,
    so the validation half went through RandomHorizontalFlip and
    RandomRotation. The same model then scored differently every epoch.
    """
    from PIL import Image
    tm = _load_train_mood()
    img = Image.fromarray((np.random.default_rng(0)
                           .integers(0, 255, (64, 64, 3))).astype(np.uint8))

    a, b = tm.EVAL_TF(img), tm.EVAL_TF(img)
    assert np.array_equal(a.numpy(), b.numpy()), "eval transform must be deterministic"

    # The training transform is expected to vary -- that is its job.
    draws = [tm.TRAIN_TF(img).numpy() for _ in range(12)]
    assert any(not np.array_equal(draws[0], d) for d in draws[1:])
