"""Age estimation module.

A lightweight deep-residual CNN (based on the well-known *Deep Residual
Learning for Human Age Approximation* — Zhang & Sun, ACCV 2017) that
predicts a numeric age from a face crop.

The network learns an 8-bit binary age code (8 output bits → 256 bins)
and decodes it to a continuous age estimate, which generalises better
than direct regression.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------- constants ----------
AGE_WEIGHTS_URL = (
    "https://github.com/Nolzzan/Human-Age-estimation-with-CNN-in-TensorFlow/"
    "releases/download/v1.0/age_model_8bit.pth"
)

INPUT_SIZE = 224
NUM_BITS = 8  # 2^8 = 256 bins → age 0-127

# ---------- PyTorch model ----------


class _ResBlock(nn.Module):
    """Pre-activation residual block (GroupNorm variant for low batch-size)."""

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride, stride),
                nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        return out + self.shortcut(x)


class _AgeNet(nn.Module):
    """Binary-code age network: 8 independent bit-branches on a shared backbone."""

    def __init__(self, num_bits: int = NUM_BITS):
        super().__init__()
        # stem
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)

        # residual stages
        self.block1 = self._make_layer(64, 3, 1)
        self.block2 = self._make_layer(128, 3, 2)
        self.block3 = self._make_layer(256, 3, 2)
        self.block4 = self._make_layer(512, 3, 2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        # 8 bit-classifiers
        self.bit_layers = nn.ModuleList(
            [nn.Linear(512, 1) for _ in range(num_bits)]
        )

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(_ResBlock(channels) if stride == 1 else self._first_block(channels, stride))
        for _ in range(1, blocks):
            layers.append(_ResBlock(channels))
        return nn.Sequential(*layers)

    @staticmethod
    def _first_block(channels: int, stride: int) -> nn.Module:
        class FirstBlock(nn.Module):
            def __init__(self, ch: int, st: int):
                super().__init__()
                self.bn1 = nn.GroupNorm(8, ch)
                self.conv1 = nn.Conv2d(ch, ch, 3, st, 1, bias=False)
                self.bn2 = nn.GroupNorm(8, ch)
                self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(st, st),
                    nn.Conv2d(ch, ch, 1, 1, 0, bias=False),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = F.relu(self.bn1(x))
                out = self.conv1(out)
                out = F.relu(self.bn2(out))
                out = self.conv2(out)
                return out + self.shortcut(x)

        return FirstBlock(channels, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).flatten(1)
        return torch.stack([bn(x).sigmoid() for bn in self.bit_layers], dim=1)


# ---------- decode helper ----------


def _binary_code_to_age(codes: np.ndarray) -> np.ndarray:
    """Convert binary codes [N, 8] to age values."""
    powers = 2 ** np.arange(NUM_BITS - 1, -1, -1)  # [128, 64, …, 1]
    # clamp to [0, 127] (bit 0 should ideally be 0)
    ages = codes.dot(powers)
    ages = np.clip(ages, 0, 127).astype(np.float32)
    return ages


# ---------- public estimator ----------


class AgeEstimator:
    """Estimate age from a face-crop image.

    Parameters
    ----------
    weights_path : str or Path, optional
        Pretrained ``.pth`` file. If ``None`` the model runs untrained
        (random weights) — useful as a starting point for fine-tuning.
    device : str
        Torch device.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ):
        self._device = device
        self._net = _AgeNet().to(device)
        self._net.eval()

        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=device, weights_only=True)
            self._net.load_state_dict(state)
            logger.info("Loaded age-estimator weights from %s", weights_path)
        else:
            logger.warning(
                "No age-estimator weights provided — model has random weights. "
                "Train or download weights for meaningful predictions."
            )

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def estimate(self, face_crop: np.ndarray) -> float:
        """Return estimated *age* (float) for a single BGR/RGB face crop."""
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB) if face_crop.shape[-1] == 3 else face_crop
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        codes = self._net(tensor)[0].cpu().numpy()  # [8]
        code_bin = (codes > 0.5).astype(np.float32)
        return float(_binary_code_to_age(code_bin[None]))

    @torch.no_grad()
    def estimate_batch(self, face_crops: List[np.ndarray], batch_size: int = 16) -> List[float]:
        """Estimate ages for multiple face crops efficiently."""
        ages: List[float] = []
        for i in range(0, len(face_crops), batch_size):
            batch = face_crops[i : i + batch_size]
            tensors = []
            for fc in batch:
                rgb = cv2.cvtColor(fc, cv2.COLOR_BGR2RGB) if fc.shape[-1] == 3 else fc
                tensors.append(self._transform(rgb))
            tensor_batch = torch.stack(tensors).to(self._device)
            codes = self._net(tensor_batch).cpu().numpy()  # [B, 8]
            code_bin = (codes > 0.5).astype(np.float32)
            ages.extend(_binary_code_to_age(code_bin).tolist())
        return ages

    def save_weights(self, path: Union[str, Path]) -> None:
        torch.save(self._net.state_dict(), str(path))

    def get_backbone(self) -> nn.Module:
        """Return the raw network for training / transfer-learning."""
        return self._net

    def __repr__(self) -> str:
        return f"AgeEstimator(device={self._device!r})"
