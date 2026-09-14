"""Tests for the YOLO26 multi-task head.

None of this subtree had ever been executed. The faults being guarded:

  * the attribute heads were Linear layers sized for ch[1] but fed the cv4
    output, so the first forward pass raised a shape error -- including in
    the verification block inside setup_mtl.sh itself;
  * they pooled the whole feature map, producing one gender/age/emotion per
    *image* rather than per detected face;
  * the sigma branch reshaped a 10-channel conv to 5 channels, doubling its
    anchor axis and misaligning it with every other branch;
  * fuse() set the detection and keypoint branches to None;
  * setup_mtl.sh never registered the head with parse_model, so the model
    could not be built even after running it.

Everything here runs on CPU with random weights; no dataset is involved.
"""
import pytest
import torch

from anyface_pp.labels import (AGE_MAX, AGE_MIN, EMOTION_LABELS,
                               GENDER_LABELS, NUM_EMOTIONS, NUM_GENDERS)
from yolo26_mtl import build_mtl_model, registered_mtl_head
from yolo26_mtl.head_module.head_mtl import (ATTR_AGE, ATTR_DIM, ATTR_EMOTION,
                                             ATTR_GENDER, decode_attrs)

KPT_SHAPE = (5, 3)


def _head(ch, end2end=False):
    with registered_mtl_head() as MTLPose:
        h = MTLPose(nc=1, kpt_shape=KPT_SHAPE, reg_max=1, end2end=end2end, ch=ch)
    h.stride = torch.tensor([8.0, 16.0, 32.0])
    return h


def _feats(ch, base=80, bs=2):
    return [torch.randn(bs, c, base // (2 ** i), base // (2 ** i))
            for i, c in enumerate(ch)]


# ------------------------------------------------------------------ layout
def test_attr_layout_is_contiguous_and_complete():
    """The slices must tile [0, ATTR_DIM) exactly -- consumers index by them."""
    covered = (list(range(*ATTR_GENDER.indices(ATTR_DIM)))
               + list(range(*ATTR_AGE.indices(ATTR_DIM)))
               + list(range(*ATTR_EMOTION.indices(ATTR_DIM))))
    assert sorted(covered) == list(range(ATTR_DIM))
    assert ATTR_DIM == NUM_GENDERS + 1 + NUM_EMOTIONS


def test_label_sets_are_shared_not_duplicated():
    """Three copies of the emotion list used to disagree on length and order."""
    from anyface_pp.models.mood_classifier import EMOTION_LABELS as from_mood
    from anyface_pp import labels
    assert from_mood is labels.EMOTION_LABELS
    assert len(set(EMOTION_LABELS)) == len(EMOTION_LABELS)
    assert len(set(GENDER_LABELS)) == len(GENDER_LABELS)


# ------------------------------------------------------------------ forward
@pytest.mark.parametrize("ch", [(256, 512, 1024), (64, 128, 256)],
                         ids=["setup_mtl.sh-channels", "yolo26n-real-channels"])
def test_forward_runs_at_both_channel_configurations(ch):
    """The old head raised 'mat1 and mat2 shapes cannot be multiplied' here."""
    h = _head(ch)
    h.train()
    out = h(_feats(ch))
    assert set(out) >= {"boxes", "scores", "kpts", "kpts_sigma", "attrs"}


@pytest.mark.parametrize("ch", [(64, 128, 256)])
def test_every_branch_shares_one_anchor_axis(ch):
    """kpts_sigma used to report twice as many anchors as the other branches."""
    h = _head(ch)
    h.train()
    out = h(_feats(ch))
    anchors = {k: v.shape[-1] for k, v in out.items() if k != "feats"}
    assert len(set(anchors.values())) == 1, anchors


def test_attributes_are_per_anchor_not_per_image():
    """A pooled head gives one gender for a whole group photo."""
    ch = (64, 128, 256)
    h = _head(ch)
    h.train()
    out = h(_feats(ch))
    attrs = out["attrs"]
    assert attrs.ndim == 3 and attrs.shape[1] == ATTR_DIM
    assert attrs.shape[-1] == out["kpts"].shape[-1]
    # Different anchors must be able to disagree; a pooled head cannot.
    assert attrs[0].std(dim=-1).max() > 0


def test_inference_tensor_carries_attributes_after_keypoints():
    ch = (64, 128, 256)
    h = _head(ch)
    h.eval()
    with torch.no_grad():
        y, _ = h(_feats(ch))
    assert y.shape[1] == 4 + h.nc + h.nk + ATTR_DIM


# ---------------------------------------------------------------------- NMS
def test_attributes_stay_matched_to_their_detection_through_nms():
    """The property that makes per-anchor attributes usable at all."""
    from ultralytics.utils.nms import non_max_suppression

    ch = (64, 128, 256)
    h = _head(ch)
    h.eval()
    with torch.no_grad():
        _, raw = h(_feats(ch, bs=1))

    n_anchors = raw["attrs"].shape[-1]
    marker = torch.arange(n_anchors, dtype=torch.float32)
    raw["attrs"][0, 0, :] = marker              # stamp each anchor
    y = h._inference(raw)

    dets, keep = non_max_suppression(y, conf_thres=0.001, iou_thres=0.7,
                                     nc=h.nc, max_det=10, return_idxs=True)
    attrs = dets[0][:, 6 + h.nk:]
    assert attrs.shape[1] == ATTR_DIM
    assert torch.allclose(attrs[:, 0], marker[keep[0]])


# ------------------------------------------------------------------- decode
def test_decode_attrs_returns_valid_labels_and_probabilities():
    out = decode_attrs(torch.randn(ATTR_DIM))
    assert out["gender"] in GENDER_LABELS
    assert out["emotion"] in EMOTION_LABELS
    assert 0.0 <= out["gender_conf"] <= 1.0
    assert 0.0 <= out["emotion_conf"] <= 1.0
    assert AGE_MIN <= out["age"] <= AGE_MAX


def test_decode_attrs_confidence_is_a_probability_not_a_logit():
    """The old consumer reported the raw logit, so 'confidence' could be < 0."""
    v = torch.zeros(ATTR_DIM)
    v[ATTR_GENDER] = torch.tensor([-8.0, -9.0, -10.0])   # all logits negative
    out = decode_attrs(v)
    assert out["gender"] == "female"
    assert 0.0 < out["gender_conf"] <= 1.0


def test_decode_attrs_clamps_age_to_the_declared_range():
    v = torch.zeros(ATTR_DIM)
    v[ATTR_AGE] = 1e4
    assert decode_attrs(v)["age"] == AGE_MAX
    v[ATTR_AGE] = -1e4
    assert decode_attrs(v)["age"] == AGE_MIN


def test_decode_attrs_reads_the_argmax_of_each_group():
    v = torch.full((ATTR_DIM,), -10.0)
    v[ATTR_GENDER.start + 1] = 10.0                       # "male"
    v[ATTR_EMOTION.start + EMOTION_LABELS.index("sad")] = 10.0
    out = decode_attrs(v)
    assert out["gender"] == "male" and out["emotion"] == "sad"


# --------------------------------------------------------------------- misc
def test_fuse_keeps_the_model_runnable():
    """fuse() used to None out cv2/cv3/cv4, breaking every later forward."""
    ch = (64, 128, 256)
    h = _head(ch)
    h.eval()
    h.fuse()
    with torch.no_grad():
        y, _ = h(_feats(ch))
    assert y.shape[1] == 4 + h.nc + h.nk + ATTR_DIM


def test_registration_restores_ultralytics_afterwards():
    import ultralytics.nn.tasks as tasks
    before = tasks.Pose26
    with registered_mtl_head():
        assert tasks.Pose26 is not before
    assert tasks.Pose26 is before


def test_model_builds_from_config():
    """setup_mtl.sh's cp/sed left parse_model unable to build this at all."""
    model = build_mtl_model()
    head = model.model[-1]
    assert head.attr_dim == ATTR_DIM
    assert len(head.cv5) == len(head.cv5_attr) == head.nl


# ------------------------------------------------------- inference plumbing
def _load_infer():
    """Import the script by path; yolo26_mtl/scripts is not a package."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "infer_mtl", root / "yolo26_mtl" / "scripts" / "infer_mtl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("shape", [(480, 640), (720, 405), (300, 300)])
def test_letterbox_maps_back_to_original_coordinates(shape):
    """Undoing the letterbox must recover the original pixel geometry."""
    import numpy as np
    infer = _load_infer()
    h, w = shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    x, ratio, (padx, pady) = infer.letterbox(img, 640)

    assert x.shape == (1, 3, 640, 640)
    # The image corners, mapped forward then back, land where they started.
    for px, py in [(0, 0), (w, h), (w / 2, h / 2)]:
        fx, fy = px * ratio + padx, py * ratio + pady
        assert (fx - padx) / ratio == pytest.approx(px)
        assert (fy - pady) / ratio == pytest.approx(py)
        assert -1e-6 <= fx <= 640 + 1e-6 and -1e-6 <= fy <= 640 + 1e-6


def test_predict_returns_in_frame_boxes_and_full_attributes():
    """End to end on an untrained model: structure must be right even though
    the numbers are not, and nothing may fall outside the frame."""
    import numpy as np
    infer = _load_infer()
    model = infer.load_model(None, "cpu")
    head = model.model[-1]
    img = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)

    faces = infer.predict(img, model, head, conf=0.05, device="cpu")
    assert faces, "expected the untrained model to emit rows"
    for f in faces[:20]:
        x, y, bw, bh = f["bbox"]
        assert 0 <= x <= 640 and 0 <= y <= 480
        assert bw >= 0 and bh >= 0 and x + bw <= 640 and y + bh <= 480
        assert set(f) >= {"gender", "gender_conf", "age", "emotion",
                          "emotion_conf", "landmarks", "confidence"}
        assert len(f["landmarks"]) == 5
        for lx, ly in f["landmarks"]:
            assert 0 <= lx <= 640 and 0 <= ly <= 480


def test_end2end_refuses_rather_than_allocating_dead_branches():
    """end2end=True used to deep-copy branches forward never called.

    They took 240,708 parameters (+36% on this head) and received no gradient,
    because forward returned a flat dict instead of Pose26's
    {"one2many", "one2one"} pair.
    """
    with registered_mtl_head() as MTLPose:
        with pytest.raises(NotImplementedError, match="one2many"):
            MTLPose(nc=1, kpt_shape=KPT_SHAPE, reg_max=1, end2end=True,
                    ch=(64, 128, 256))


def test_shipped_config_does_not_request_end2end():
    from yolo26_mtl import DEFAULT_CONFIG
    text = DEFAULT_CONFIG.read_text()
    assert "end2end: False" in text
