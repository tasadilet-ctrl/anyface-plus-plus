"""Face detection module using Ultralytics YOLO26.

Uses a pretrained YOLO26n model from the latest Ultralytics release.
For best face-detection results, provide a WIDER-FACE / COCO-Face tuned
weights file via ``model_path``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Ultralytics YOLO26n — downloads automatically from the latest release
DEFAULT_FACE_MODEL = "yolo26n.pt"


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


class FaceDetector:
    """YOLO26-based face detector.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to a YOLO26 face-detection weights file (.pt). If ``None`` the
        stock ``yolo26n.pt`` is downloaded automatically from Ultralytics.
    conf_thres : float
        Minimum confidence to keep a detection (default ``0.5``).
    iou_thres : float
        NMS IoU threshold (default ``0.5``).
    device : str
        Torch device string, e.g. ``"cpu"``, ``"cuda"``, ``"mps"``.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_thres: float = 0.5,
        iou_thres: float = 0.5,
        device: str = "cpu",
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Try to load a face-specific model; fall back to generic yolo26n
        if model_path and Path(model_path).exists():
            logger.info("Loading face detector from %s", model_path)
            self.model = YOLO(str(model_path))
        else:
            logger.info(
                "Using default YOLO26n model (%s). "
                "For better face detection provide a WIDER-FACE-tuned weights file.",
                DEFAULT_FACE_MODEL,
            )
            self.model = YOLO(DEFAULT_FACE_MODEL)

        self.model.to(device)
        self._device = device

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

        boxes: List[FaceBox] = []
        result = results[0]

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            boxes.append(FaceBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3], confidence=conf))

        return boxes

    def detect_batch(self, images: List[np.ndarray], batch_size: int = 8) -> List[List[FaceBox]]:
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
            for result in results:
                boxes: List[FaceBox] = []
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    boxes.append(
                        FaceBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3], confidence=conf)
                    )
                all_boxes.append(boxes)
        return all_boxes

    def __repr__(self) -> str:
        return f"FaceDetector(device={self._device!r}, conf={self.conf_thres})"
