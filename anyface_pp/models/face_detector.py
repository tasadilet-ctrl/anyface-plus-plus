"""Face detection built on an Ultralytics YOLO model.

Important: a general object detector such as the stock ``yolo26n.pt`` is
trained on COCO and has no "face" class at all. Pointing this module at one
does not give a weak face detector -- it gives buses, cars and full-body
person boxes labelled as faces, which then get cropped and fed to the age and
mood classifiers. Detections are therefore filtered by class, and a model with
no face class is rejected outright rather than silently misused.

Getting weights:

    python3 scripts/download_face_model.py          # fetches a YOLO face model

or pass ``model_path`` pointing at any face-trained YOLO checkpoint. If you
must use a non-face model, pass ``face_class_ids`` explicitly so the choice is
recorded in the code rather than assumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# A YOLO detector fine-tuned for faces. scripts/download_face_model.py fetches
# this; it is never downloaded implicitly.
DEFAULT_FACE_WEIGHTS = "yolov8n-face.pt"
FACE_WEIGHTS_URL = (
    "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt"
)


class NotAFaceModelError(ValueError):
    """Raised when the loaded weights have no face class.

    Deliberately fatal: the alternative -- returning whatever the model
    happened to detect -- produces confident, entirely wrong output instead of
    an obvious failure.
    """


@dataclass
class FaceBox:
    """A detected face bounding box."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop the face region from *image* (BGR or RGB, shape preserved)."""
        return image[self.y1 : self.y2, self.x1 : self.x2].copy()


def _resolve_face_class_ids(names: dict, explicit: Optional[Sequence[int]]) -> set:
    """Decide which class ids count as faces.

    An explicit list always wins. Otherwise any class whose name mentions
    "face" is used. A model with neither is rejected.
    """
    if explicit is not None:
        unknown = [c for c in explicit if c not in names]
        if unknown:
            raise ValueError(
                f"face_class_ids {unknown} are not classes of this model "
                f"(valid ids: {sorted(names)})"
            )
        return set(explicit)

    found = {i for i, n in names.items() if "face" in str(n).lower()}
    if found:
        return found

    sample = ", ".join(str(names[i]) for i in sorted(names)[:5])
    raise NotAFaceModelError(
        f"These weights have no 'face' class -- they look like a general "
        f"object detector (classes include: {sample}...). Such a model cannot "
        f"detect faces; using it would return objects and full-body person "
        f"boxes as though they were faces.\n"
        f"Fix: run 'python3 scripts/download_face_model.py' and use the "
        f"resulting {DEFAULT_FACE_WEIGHTS}, or pass face_class_ids=[...] to "
        f"state deliberately which classes to treat as faces."
    )


class FaceDetector:
    """YOLO-based face detector.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to face-detection weights. Defaults to ``yolov8n-face.pt`` in the
        working directory (see scripts/download_face_model.py).
    conf_thres, iou_thres : float
        Confidence and NMS IoU thresholds.
    device : str
        Torch device string, e.g. ``"cpu"``, ``"cuda"``, ``"mps"``.
    face_class_ids : sequence of int, optional
        Class ids to treat as faces. Only needed for models whose face class
        is not named "face".
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_thres: float = 0.5,
        iou_thres: float = 0.5,
        device: str = "cpu",
        face_class_ids: Optional[Sequence[int]] = None,
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        weights = Path(model_path) if model_path else Path(DEFAULT_FACE_WEIGHTS)
        if model_path is None and not weights.exists():
            raise FileNotFoundError(
                f"No face weights at '{weights}'. Run "
                f"'python3 scripts/download_face_model.py' to fetch them, or "
                f"pass model_path=... explicitly."
            )

        logger.info("Loading face detector from %s", weights)
        self.model = YOLO(str(weights))
        self.model.to(device)
        self._device = device
        self._face_class_ids = _resolve_face_class_ids(self.model.names, face_class_ids)
        logger.info(
            "Treating class ids %s (%s) as faces",
            sorted(self._face_class_ids),
            ", ".join(str(self.model.names[i]) for i in sorted(self._face_class_ids)),
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _boxes_from_result(self, result) -> List[FaceBox]:
        """Convert one Ultralytics result to FaceBoxes, keeping only faces."""
        boxes: List[FaceBox] = []
        for box in result.boxes:
            if int(box.cls[0]) not in self._face_class_ids:
                continue
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            boxes.append(
                FaceBox(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    confidence=float(box.conf[0]),
                )
            )
        return boxes

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[FaceBox]:
        """Run face detection on a single *image* (HWC, BGR or RGB)."""
        results = self.model(
            image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
            device=self._device,
        )
        return self._boxes_from_result(results[0])

    def detect_batch(
        self, images: List[np.ndarray], batch_size: int = 8
    ) -> List[List[FaceBox]]:
        """Run detection on a batch of images."""
        all_boxes: List[List[FaceBox]] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            results = self.model(
                batch,
                conf=self.conf_thres,
                iou=self.iou_thres,
                verbose=False,
                device=self._device,
            )
            all_boxes.extend(self._boxes_from_result(r) for r in results)
        return all_boxes

    def __repr__(self) -> str:
        return (
            f"FaceDetector(device={self._device!r}, conf={self.conf_thres}, "
            f"face_classes={sorted(self._face_class_ids)})"
        )
