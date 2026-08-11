"""Annotation helpers — draw bounding boxes, age labels and mood badges on images."""

from __future__ import annotations

import cv2
import numpy as np

EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    "happy":      (78, 205, 196),    # teal
    "neutral":    (158, 158, 158),   # grey
    "sad":        (100, 149, 237),   # cornflower blue
    "angry":      (220, 50, 50),     # red
    "surprised":  (255, 193, 7),     # yellow
    "fearful":    (147, 112, 219),   # medium purple
    "disgusted":  (205, 133, 63),    # peru / brown
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    age: float,
    emotion: str,
    emotion_conf: float,
    conf: float = 0.0,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a bounding box with age + mood annotation on *image* (BGR, in-place).

    Returns the (same) image for chaining.
    """
    color = EMOTION_COLORS.get(emotion, (200, 200, 200))

    # bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # label bar at top
    label_text = f"{emotion} ({emotion_conf:.0f}%)  age~{age:.0f}"
    (tw, th), _ = cv2.getTextSize(label_text, FONT, 0.6, 1)
    bar_y = max(y1 - 22, 0)
    cv2.rectangle(image, (x1, bar_y), (x1 + tw + 10, bar_y + th + 10), color, -1)
    cv2.putText(image, label_text, (x1 + 5, bar_y + th), FONT, 0.6, (255, 255, 255), 1)

    return image


def annotate_batch(
    image: np.ndarray,
    boxes: list,
) -> np.ndarray:
    """Annotate multiple face results on *image*.

    Each element in *boxes* must have attributes:
    ``x1, y1, x2, y2, age, emotion, emotion_conf, confidence``.
    """
    for b in boxes:
        annotate(
            image,
            x1=b.x1,
            y1=b.y1,
            x2=b.x2,
            y2=b.y2,
            age=b.age,
            emotion=b.emotion,
            emotion_conf=b.emotion_conf * 100,
            conf=b.confidence,
        )
    return image
