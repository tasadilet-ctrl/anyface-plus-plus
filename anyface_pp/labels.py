"""Single source of truth for the label sets used across the repo.

These were previously declared in three places that disagreed:
``mood_classifier.py`` had seven emotions, the MTL head declared eight, and
``infer_mtl.py`` listed eight in a different order. Nothing checked them
against each other, so a prediction from one path could not be compared with,
or fed into, another.
"""

from __future__ import annotations

# Emotion classes. Order is significant: it fixes the meaning of every index
# produced by MoodClassifier and by the MTL head's emotion branch.
EMOTION_LABELS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "sad",
    "surprised",
    "neutral",
]

# Gender classes for the MTL head. "unsure" is a real class, not a fallback:
# a model that must answer female/male on an ambiguous crop is being forced
# into a confident wrong answer.
GENDER_LABELS = ["female", "male", "unsure"]

NUM_EMOTIONS = len(EMOTION_LABELS)
NUM_GENDERS = len(GENDER_LABELS)

# Age is a single regressed scalar, clamped to this range on decode.
AGE_MIN, AGE_MAX = 0, 116
