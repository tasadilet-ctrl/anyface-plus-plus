"""Anyface++ pipeline — orchestrates face detection, age estimation, and mood classification.

Usage
-----
.. code-block:: python

    from anyface_pp.pipeline import AnyfacePP

    analyzer = AnyfacePP(device="mps")  # or "cuda", "cpu"
    results = analyzer.run("photo.jpg")

    for r in results:
        print(f"Face at {r.x1},{r.y1}: age~{r.age:.0f}, mood={r.emotion}")

    # annotated image ready for display / saving
    cv2.imwrite("output.jpg", results.image)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from .models.face_detector import FaceDetector, FaceBox
from .models.age_estimator import AgeEstimator
from .models.mood_classifier import MoodClassifier
from .utils.visualizer import annotate

logger = logging.getLogger(__name__)


@dataclass
class FaceResult:
    """Structured result for one detected face."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float        # detection confidence
    age: float               # estimated age
    emotion: str             # top mood label
    emotion_conf: float      # top mood probability [0-1]
    emotions: dict           # full probability distribution


class AnyfacePP:
    """Unified face-analysis pipeline.

    Parameters
    ----------
    device : str
        Torch device (``"cpu"``, ``"cuda"``, ``"mps"``).
    face_model_path : str or Path, optional
        Custom YOLO26 face-detector weights.
    age_weights_path : str or Path, optional
        Pretrained age-estimator ``.pth``.
    mood_weights_path : str or Path, optional
        Pretrained mood-classifier ``.pth``.
    conf_thres : float
        Detection confidence threshold.
    """

    def __init__(
        self,
        device: str = "cpu",
        face_model_path: Optional[Union[str, Path]] = None,
        age_weights_path: Optional[Union[str, Path]] = None,
        mood_weights_path: Optional[Union[str, Path]] = None,
        conf_thres: float = 0.5,
    ):
        self.detector = FaceDetector(model_path=face_model_path, conf_thres=conf_thres, device=device)
        self.age_estimator = AgeEstimator(weights_path=age_weights_path, device=device)
        self.mood_classifier = MoodClassifier(weights_path=mood_weights_path, device=device)
        self._device = device

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(
        self,
        source: Union[str, Path, np.ndarray],
        annotate_image: bool = True,
    ) -> List[FaceResult]:
        """Analyse *source* (image path or BGR ``np.ndarray``).

        Returns a list of :class:`FaceResult` and, when ``annotate_image`` is
        true, attaches the annotated frame as ``._image`` on the pipeline so
        you can access it via ``results[0]._image`` — or more conveniently,
        use :meth:`run_with_image`.
        """
        image = self._load_image(source)
        results = self._analyze(image)

        if annotate_image:
            self._annotated = self._draw(image.copy(), results)
        else:
            self._annotated = None

        return results

    def run_with_image(
        self, source: Union[str, Path, np.ndarray]
    ) -> tuple[List[FaceResult], np.ndarray]:
        """Return ``(results, annotated_image)``."""
        results = self.run(source, annotate_image=True)
        return results, self._annotated

    def run_video(self, source: Union[int, str, Path], fps: float = 30.0):
        """Generator yielding ``(frame_results, annotated_frame)`` for each frame.

        *source* can be a camera index (``0``) or a video-file path.
        """
        cap = cv2.VideoCapture(int(source) if isinstance(source, int) else str(source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {source}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = self._analyze(frame)
                annotated = self._draw(frame.copy(), results)
                yield results, annotated
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _load_image(source: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {source}")
        return img

    def _analyze(self, image: np.ndarray) -> List[FaceResult]:
        boxes = self.detector.detect(image)
        if not boxes:
            return []

        face_crops = [b.crop(image) for b in boxes]
        ages = self.age_estimator.estimate_batch(face_crops)
        emotions = self.mood_classifier.classify_batch(face_crops)

        results: List[FaceResult] = []
        for box, age, emo_dict in zip(boxes, ages, emotions):
            top_emotion = next(iter(emo_dict))
            results.append(
                FaceResult(
                    x1=box.x1,
                    y1=box.y1,
                    x2=box.x2,
                    y2=box.y2,
                    confidence=box.confidence,
                    age=age,
                    emotion=top_emotion,
                    emotion_conf=emo_dict[top_emotion],
                    emotions=emo_dict,
                )
            )

        logger.debug("Detected %d face(s) on device %s", len(results), self._device)
        return results

    @staticmethod
    def _draw(image: np.ndarray, results: List[FaceResult]) -> np.ndarray:
        for r in results:
            annotate(
                image,
                x1=r.x1,
                y1=r.y1,
                x2=r.x2,
                y2=r.y2,
                age=r.age,
                emotion=r.emotion,
                emotion_conf=r.emotion_conf,
            )
        return image

    def __repr__(self) -> str:
        return f"AnyfacePP(device={self._device!r})"
