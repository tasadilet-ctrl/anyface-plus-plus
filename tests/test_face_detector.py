"""Tests for the face-detection class filtering.

These run offline and need no weights: the class-resolution logic is the part
that had the bug, and it depends only on a model's ``names`` mapping. An
integration test that needs real weights is skipped unless they are present.

The bug being guarded: FaceDetector used to return every detection from
whatever model was loaded. With the stock COCO ``yolo26n.pt`` that meant a
bus, and four full-body person boxes, were returned as faces and cropped for
the age and mood classifiers.
"""
import numpy as np
import pytest

from anyface_pp.models.face_detector import (DEFAULT_FACE_WEIGHTS, FaceBox,
                                             NotAFaceModelError,
                                             _resolve_face_class_ids)

# The first few COCO classes, i.e. what the stock yolo26n.pt reports.
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus"}
FACE_NAMES = {0: "FACE"}


def test_coco_model_is_rejected():
    """A general detector has no face class and must not be used as one."""
    with pytest.raises(NotAFaceModelError) as exc:
        _resolve_face_class_ids(COCO_NAMES, None)
    msg = str(exc.value)
    assert "no 'face' class" in msg
    assert "download_face_model" in msg          # error is actionable


def test_face_model_is_detected_case_insensitively():
    assert _resolve_face_class_ids(FACE_NAMES, None) == {0}
    assert _resolve_face_class_ids({0: "face"}, None) == {0}
    assert _resolve_face_class_ids({3: "human face", 4: "car"}, None) == {3}


def test_explicit_ids_override_and_are_validated():
    # Deliberate opt-in to a non-face model is allowed...
    assert _resolve_face_class_ids(COCO_NAMES, [0]) == {0}
    # ...but a nonexistent class id is a mistake, not a silent no-op.
    with pytest.raises(ValueError, match="not classes of this model"):
        _resolve_face_class_ids(COCO_NAMES, [99])


def test_facebox_geometry_and_crop():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[10:40, 50:90] = 255
    b = FaceBox(x1=50, y1=10, x2=90, y2=40, confidence=0.9)
    assert (b.width, b.height) == (40, 30)
    crop = b.crop(img)
    assert crop.shape == (30, 40, 3)
    assert (crop == 255).all()
    crop[:] = 0                                   # crop must be a copy
    assert (img[10:40, 50:90] == 255).all()


@pytest.mark.skipif(not __import__("pathlib").Path(DEFAULT_FACE_WEIGHTS).exists(),
                    reason=f"{DEFAULT_FACE_WEIGHTS} not present "
                           "(run scripts/download_face_model.py)")
def test_integration_detects_only_face_sized_boxes():
    """With real weights on a real photo, detections should be face-sized --
    a fraction of the frame, not whole-person or whole-vehicle boxes."""
    import cv2
    from anyface_pp.models.face_detector import FaceDetector

    path = __import__("pathlib").Path(__file__).parent / "assets" / "bus.jpg"
    if not path.exists():
        pytest.skip("test image not present")

    img = cv2.imread(str(path))
    boxes = FaceDetector(model_path=DEFAULT_FACE_WEIGHTS, device="cpu",
                         conf_thres=0.4).detect(img)
    assert boxes, "expected at least one face"
    h, w = img.shape[:2]
    for b in boxes:
        assert b.width < 0.25 * w and b.height < 0.25 * h
