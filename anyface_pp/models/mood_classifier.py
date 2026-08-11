"""Mood / emotion classification module.

A lightweight MobilenetV2-based classifier trained on the FER2013 / AffectNet
emotion space.  Seven classes:

    0. angry       1. disgusted  2. fearful
    3. happy       4. sad        5. surprised
    6. neutral
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

logger = logging.getLogger(__name__)

EMOTION_LABELS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "sad",
    "surprised",
    "neutral",
]

INPUT_SIZE = 224


class _MoodNet(nn.Module):
    """MobileNetV2 backbone → single FC head for 7-class emotion."""

    def __init__(self, num_classes: int = len(EMOTION_LABELS)):
        super().__init__()
        backbone = mobilenet_v2(weights=None)
        self.features = backbone.features
        self._classifier_in = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential()  # drop original head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self._classifier_in, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).mean([-2, -1]))


class MoodClassifier:
    """Classify mood/emotion from a face-crop image.

    Parameters
    ----------
    weights_path : str or Path, optional
        Pretrained ``.pth`` with full ``MoodClassifier`` state (model + metadata).
        If ``None`` the network is initialised with random weights.
    device : str
        Torch device string.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ):
        self._device = device
        self._net = _MoodNet().to(device)
        self._net.eval()

        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=device, weights_only=False)
            self._net.load_state_dict(state["model"])
            logger.info("Loaded mood-classifier weights from %s", weights_path)
        else:
            logger.warning(
                "No mood-classifier weights provided — model has random weights. "
                "Train or download weights for meaningful predictions."
            )

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.Grayscale(num_output_channels=3),  # FER2013 is grayscale
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def classify(self, face_crop: np.ndarray) -> Dict[str, float]:
        """Classify *face_crop* and return ``{label: probability}`` dict.

        The dictionary is sorted by probability descending.
        """
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB) if face_crop.shape[-1] == 3 else face_crop
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        logits = self._net(tensor)[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()

        return dict(
            sorted(zip(EMOTION_LABELS, probs.astype(float)), key=lambda kv: -kv[1])
        )

    @torch.no_grad()
    def classify_batch(
        self, face_crops: List[np.ndarray], batch_size: int = 16
    ) -> List[Dict[str, float]]:
        """Classify multiple face crops in a single forward pass."""
        results: List[Dict[str, float]] = []
        for i in range(0, len(face_crops), batch_size):
            batch = face_crops[i : i + batch_size]
            tensors = []
            for fc in batch:
                rgb = cv2.cvtColor(fc, cv2.COLOR_BGR2RGB) if fc.shape[-1] == 3 else fc
                tensors.append(self._transform(rgb))
            tensor_batch = torch.stack(tensors).to(self._device)
            logits = self._net(tensor_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for p in probs:
                results.append(
                    dict(
                        sorted(zip(EMOTION_LABELS, p.astype(float)), key=lambda kv: -kv[1])
                    )
                )
        return results

    def predict(self, face_crop: np.ndarray) -> Tuple[str, float]:
        """Return ``(top_emotion_label, confidence)`` for quick access."""
        probs = self.classify(face_crop)
        top_label = next(iter(probs))
        return top_label, probs[top_label]

    def save_weights(self, path: Union[str, Path]) -> None:
        torch.save({"model": self._net.state_dict(), "labels": EMOTION_LABELS}, str(path))

    def get_backbone(self) -> nn.Module:
        """Return the raw network for training / transfer-learning."""
        return self._net

    def __repr__(self) -> str:
        return f"MoodClassifier(device={self._device!r})"

# ---------- standalone import guard for cv2 (used above) ----------
import cv2  # noqa: E402  (placed here so the module-level cv2 calls in _transform work)
