"""Age estimation from a face crop.

A pre-activation residual CNN predicts a set of binary "is the age above X"
decisions, which are then decoded to a single number.

How the age is coded matters more than the backbone. Two codes are supported:

``ordinal`` (default)
    Thermometer code: unit *k* answers "is this person older than *k*?", and
    the age is the number of units that say yes. Every unit's target is a
    single monotone step in age, and one wrong unit costs one year.

``binary``
    The eight bits of the integer age, weighted 128...1. Compact, but each
    unit is worth its place value, so one wrong unit can cost 64 years, and
    the low bits ask the network to predict things like whether the age is
    odd. Kept because it is what this module used to do, and because
    benchmarks/age_encoding.py quantifies the difference.

See the "Age coding" section of the README for the measurements.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)

INPUT_SIZE = 224
NUM_BITS = 8        # binary code: 8 bits, place values 128...1
MAX_AGE = 100       # ordinal code: one unit per year, ages 0..MAX_AGE

ENCODINGS = ("ordinal", "binary")


# ---------------------------------------------------------------- age codes
def num_outputs(encoding: str) -> int:
    """Number of network outputs required by *encoding*."""
    _check_encoding(encoding)
    return MAX_AGE if encoding == "ordinal" else NUM_BITS


def _check_encoding(encoding: str) -> None:
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")


def encode_age(ages: np.ndarray, encoding: str = "ordinal") -> np.ndarray:
    """Encode integer ages ``[N]`` as training targets ``[N, num_outputs]``."""
    _check_encoding(encoding)
    ages = np.asarray(ages).reshape(-1)
    if encoding == "ordinal":
        # unit k is 1 iff age > k, so the number of ones *is* the age.
        return (ages[:, None] > np.arange(MAX_AGE)[None, :]).astype(np.float32)
    bits = np.clip(ages, 0, 2 ** NUM_BITS - 1).astype(np.int64)
    shifts = np.arange(NUM_BITS - 1, -1, -1)[None, :]        # MSB first
    return ((bits[:, None] >> shifts) & 1).astype(np.float32)


def decode_age(probs: np.ndarray, encoding: str = "ordinal") -> np.ndarray:
    """Decode network probabilities ``[N, num_outputs]`` to ages ``[N]``."""
    _check_encoding(encoding)
    probs = np.atleast_2d(np.asarray(probs, dtype=np.float32))
    on = probs > 0.5
    if encoding == "ordinal":
        # Counting rather than reading off the first zero: the units are
        # predicted independently, so the sequence need not be monotone, and
        # a single early false negative must not truncate the whole estimate.
        return on.sum(1).astype(np.float32)
    place = (2 ** np.arange(NUM_BITS - 1, -1, -1)).astype(np.float32)
    return np.clip(on.astype(np.float32) @ place, 0, 2 ** NUM_BITS - 1)


# ------------------------------------------------------------------ network
class _ResBlock(nn.Module):
    """Pre-activation residual block.

    Handles a change of width and/or stride in the shortcut. The previous
    version took a single ``channels`` argument used as both input and output
    width, so every stage transition was a shape mismatch.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)

        if stride == 1 and in_ch == out_ch:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride, stride) if stride != 1 else nn.Identity(),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)


def _stage(in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
    """One resolution stage: a projecting block, then identity blocks."""
    return nn.Sequential(
        _ResBlock(in_ch, out_ch, stride),
        *[_ResBlock(out_ch, out_ch) for _ in range(blocks - 1)],
    )


class _AgeNet(nn.Module):
    """Residual backbone with one output unit per code position.

    Returns raw logits. Training therefore uses BCEWithLogitsLoss, which is
    numerically stable; callers that want probabilities apply sigmoid.

    ``downsample_stem`` selects the standard ResNet stem (stride-2 conv plus
    max-pool, so the stages run at 56x56 for a 224 input). Disabling it keeps
    the earlier full-resolution stem, which is 11x slower for the same
    parameter count.
    """

    def __init__(self, encoding: str = "ordinal", downsample_stem: bool = True):
        super().__init__()
        _check_encoding(encoding)
        self.encoding = encoding

        if downsample_stem:
            self.stem: nn.Module = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                nn.MaxPool2d(3, 2, 1),
            )
        else:
            self.stem = nn.Conv2d(3, 64, 3, 1, 1, bias=False)

        self.block1 = _stage(64, 64, 3, 1)
        self.block2 = _stage(64, 128, 3, 2)
        self.block3 = _stage(128, 256, 3, 2)
        self.block4 = _stage(256, 512, 3, 2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, num_outputs(encoding))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(self.gap(x).flatten(1))


# ---------------------------------------------------------------- estimator
class AgeEstimator:
    """Estimate age from a face crop.

    Parameters
    ----------
    weights_path : str or Path, optional
        Checkpoint written by :meth:`save_weights` or ``train_age.py``. With
        no weights the network is random and its output is meaningless.
    device : str
        Torch device.
    encoding : str
        ``"ordinal"`` or ``"binary"``. Ignored when *weights_path* records its
        own encoding, so a checkpoint is always decoded the way it was trained.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        encoding: str = "ordinal",
    ):
        _check_encoding(encoding)
        self._device = device
        self.encoding = encoding

        state = None
        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=device, weights_only=True)
            if isinstance(state, dict) and "encoding" in state:
                # The checkpoint knows how it was trained; trusting the
                # constructor argument instead would decode a binary model as
                # ordinal and silently return nonsense.
                if state["encoding"] != encoding:
                    logger.info(
                        "Checkpoint was trained with encoding=%r; using that "
                        "instead of the requested %r.",
                        state["encoding"], encoding,
                    )
                self.encoding = state["encoding"]

        self._net = _AgeNet(encoding=self.encoding).to(device)
        self._net.eval()

        if state is not None:
            self._net.load_state_dict(state["model"] if "model" in state else state)
            logger.info("Loaded age-estimator weights from %s (encoding=%s)",
                        weights_path, self.encoding)
        else:
            if weights_path:
                raise FileNotFoundError(f"No age weights at '{weights_path}'")
            logger.warning(
                "No age-estimator weights provided -- the model is randomly "
                "initialised and its age output is meaningless. Train with "
                "train_age.py or pass weights_path."
            )

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        )

    # ------------------------------------------------------------------
    def _prepare(self, crop: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.ndim == 3 else crop
        return self._transform(rgb)

    @torch.no_grad()
    def estimate(self, face_crop: np.ndarray) -> float:
        """Return the estimated age for a single BGR face crop."""
        return self.estimate_batch([face_crop])[0]

    @torch.no_grad()
    def estimate_batch(self, face_crops: List[np.ndarray],
                       batch_size: int = 16) -> List[float]:
        """Estimate ages for several face crops."""
        ages: List[float] = []
        for i in range(0, len(face_crops), batch_size):
            batch = [self._prepare(fc) for fc in face_crops[i: i + batch_size]]
            logits = self._net(torch.stack(batch).to(self._device))
            probs = torch.sigmoid(logits).cpu().numpy()
            ages.extend(decode_age(probs, self.encoding).tolist())
        return ages

    def save_weights(self, path: Union[str, Path]) -> None:
        """Save the network together with the encoding it was trained for."""
        torch.save({"model": self._net.state_dict(), "encoding": self.encoding},
                   str(path))

    def get_backbone(self) -> nn.Module:
        """Return the raw network for training / transfer learning."""
        return self._net

    def __repr__(self) -> str:
        return f"AgeEstimator(device={self._device!r}, encoding={self.encoding!r})"
