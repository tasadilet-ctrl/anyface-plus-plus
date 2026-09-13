"""End-to-end pipeline wiring, without weights or images.

The age branch used to raise NameError on the first face, so AnyfacePP.run()
could never return. Stubbing out the detector lets the rest of the chain --
crop, age, mood, annotate, FaceResult -- run offline in CI, which is where the
break actually was.
"""
import numpy as np
import pytest

from anyface_pp.models.face_detector import FaceBox


class _StubDetector:
    """Stands in for FaceDetector so no YOLO weights are needed."""

    def __init__(self, boxes):
        self._boxes = boxes

    def detect(self, image):
        return self._boxes


@pytest.fixture
def pipeline(monkeypatch):
    from anyface_pp import pipeline as pl

    boxes = [FaceBox(10, 10, 60, 70, 0.91), FaceBox(80, 20, 120, 75, 0.77)]
    monkeypatch.setattr(pl, "FaceDetector", lambda **kw: _StubDetector(boxes))
    return pl.AnyfacePP(device="cpu"), boxes


def test_run_returns_one_result_per_detection(pipeline):
    analyzer, boxes = pipeline
    image = (np.random.default_rng(0).random((150, 200, 3)) * 255).astype(np.uint8)

    results = analyzer.run(image)

    assert len(results) == len(boxes)
    for r, b in zip(results, boxes):
        assert (r.x1, r.y1, r.x2, r.y2) == (b.x1, b.y1, b.x2, b.y2)
        assert isinstance(r.age, float)
        assert r.emotion in r.emotions
        # emotions is sorted by probability, so the reported one is the top one
        assert r.emotion == next(iter(r.emotions))
        assert r.emotion_conf == pytest.approx(max(r.emotions.values()))
        assert sum(r.emotions.values()) == pytest.approx(1.0, abs=1e-5)


def test_annotation_draws_on_a_copy_not_the_input(pipeline):
    analyzer, _ = pipeline
    image = np.zeros((150, 200, 3), dtype=np.uint8)

    results, annotated = analyzer.run_with_image(image)

    assert annotated.shape == image.shape
    assert annotated.any(), "nothing was drawn"
    assert not image.any(), "the caller's image was modified in place"


def test_no_detections_gives_no_results(monkeypatch):
    from anyface_pp import pipeline as pl

    monkeypatch.setattr(pl, "FaceDetector", lambda **kw: _StubDetector([]))
    analyzer = pl.AnyfacePP(device="cpu")
    assert analyzer.run(np.zeros((80, 80, 3), dtype=np.uint8)) == []


def test_each_face_is_cropped_from_its_own_box(pipeline):
    """Every model must see the crop belonging to the box it is reported under.

    Asserting on the returned ages instead would not catch this: with random
    weights two different crops can still decode to the same number.
    """
    analyzer, boxes = pipeline
    image = (np.random.default_rng(1).random((150, 200, 3)) * 255).astype(np.uint8)

    seen = []
    real_estimate = analyzer.age_estimator.estimate_batch
    analyzer.age_estimator.estimate_batch = lambda crops, **kw: (
        seen.extend(crops), real_estimate(crops, **kw))[1]

    analyzer.run(image, annotate_image=False)

    assert len(seen) == len(boxes)
    for crop, b in zip(seen, boxes):
        assert np.array_equal(crop, image[b.y1:b.y2, b.x1:b.x2])
