"""Multi-task head for YOLO26 -- AnyFace++.

Face detection + 5 landmarks + gender + age + emotion from one backbone.

The attribute branches predict *per anchor*, exactly like the keypoint branch,
so every detected face carries its own gender/age/emotion. An earlier version
pooled the whole feature map to a single vector per image, which gave one
gender for a group photo and could not be matched to a detection at all.

Attributes are concatenated onto the inference tensor after the keypoints, in
the same way Pose26 appends keypoints after the class scores, so Ultralytics'
NMS carries them through to the surviving detections untouched. Use
``decode_attrs`` on an NMS row to read them back.

What is NOT implemented: a loss for the attribute branches. Training through
``task="pose"`` optimises boxes and keypoints only, so the attribute weights
stay at their initialisation. See the README section "The MTL subtree".
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import RealNVP
from ultralytics.nn.modules.conv import Conv, DWConv

from anyface_pp.labels import (AGE_MAX, AGE_MIN, EMOTION_LABELS,
                               GENDER_LABELS, NUM_EMOTIONS, NUM_GENDERS)

# Layout of the per-anchor attribute vector. One place, so the head and every
# consumer agree; infer_mtl.py used to hard-code a different layout entirely.
ATTR_GENDER = slice(0, NUM_GENDERS)
ATTR_AGE = slice(NUM_GENDERS, NUM_GENDERS + 1)
ATTR_EMOTION = slice(NUM_GENDERS + 1, NUM_GENDERS + 1 + NUM_EMOTIONS)
ATTR_DIM = NUM_GENDERS + 1 + NUM_EMOTIONS


def decode_attrs(attrs):
    """Decode a per-anchor attribute vector into labelled predictions.

    *attrs* is ``(..., ATTR_DIM)`` raw head output. Probabilities come from a
    softmax: the previous consumer reported the raw logit as a confidence, so
    a "confidence" could be negative or above 1.
    """
    gender_p = attrs[..., ATTR_GENDER].softmax(-1)
    emotion_p = attrs[..., ATTR_EMOTION].softmax(-1)
    gi = int(gender_p.argmax(-1))
    ei = int(emotion_p.argmax(-1))
    age = float(attrs[..., ATTR_AGE].reshape(-1)[0])
    return {
        "gender": GENDER_LABELS[gi],
        "gender_conf": float(gender_p.reshape(-1)[gi]),
        "age": int(min(max(round(age), AGE_MIN), AGE_MAX)),
        "emotion": EMOTION_LABELS[ei],
        "emotion_conf": float(emotion_p.reshape(-1)[ei]),
    }


class MTLPose(nn.Module):
    """YOLO26 MTL head for AnyFace++.

    Per-anchor outputs, all sharing one anchor axis:
        - face box + confidence
        - 5 landmarks (left eye, right eye, nose, mouth L, mouth R)
        - gender, NUM_GENDERS classes from anyface_pp.labels
        - age, one regressed scalar clamped to [AGE_MIN, AGE_MAX] on decode
        - emotion, NUM_EMOTIONS classes from anyface_pp.labels

    The class counts are taken from the shared label module rather than
    written here; the previous docstring said eight emotions while the rest
    of the repo used seven.
    """

    def __init__(self, nc=1, kpt_shape=(5, 3), reg_max=1, end2end=False, ch=()):
        super().__init__()
        self.nc = nc
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.end2end_flag = end2end
        self.stride = torch.zeros(self.nl)
        self.dynamic = False
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)
        self.inplace = True
        self.export = False

        # ── Detection heads ──
        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            ) for x in ch
        )

        # ── Keypoint heads (like Pose26) ──
        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch
        )
        self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
        # One (x, y) sigma pair per keypoint.
        self.nk_sigma = kpt_shape[0] * 2
        self.cv4_sigma = nn.ModuleList(
            nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)
        self.flow_model = RealNVP()

        # ── Attribute heads: one prediction per anchor, per level ──
        # Built like cv4/cv4_kpts rather than as pooled Linear layers. The
        # pooled version had two faults: its Linear was sized for ch[1] but
        # fed the cv4 output (c4 channels), so the first forward pass raised a
        # shape error; and pooling collapses the map to one vector per image,
        # which cannot be attached to an individual detection.
        self.attr_dim = ATTR_DIM
        c5 = max(ch[0] // 2, 64)
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3)) for x in ch
        )
        self.cv5_attr = nn.ModuleList(nn.Conv2d(c5, ATTR_DIM, 1) for _ in ch)

        # ── One-to-one branches for end2end ──
        if end2end:
            # Pose26 runs both branches in forward and returns
            # {"one2many": ..., "one2one": ...}; the v26 loss and the NMS-free
            # inference path both depend on that shape. This head never did
            # either: forward returned a single flat dict and the deep copies
            # below were never called, so they took 240,708 parameters (+36%
            # on this head) that received no gradient and did nothing. Rather
            # than keep silently allocating them, refuse.
            raise NotImplementedError(
                "MTLPose does not implement the end2end one2many/one2one "
                "contract. Set end2end: False in the model YAML. The dual "
                "branches would otherwise be allocated but never run."
            )

    # ── Properties ──
    @property
    def one2many(self):
        return dict(
            box_head=self.cv2, cls_head=self.cv3, pose_head=self.cv4,
            kpts_head=self.cv4_kpts, kpts_sigma_head=self.cv4_sigma,
        )

    @property
    def one2one(self):
        raise NotImplementedError(
            "MTLPose has no one2one branch; see __init__ for why."
        )

    # ── Forward ──
    def forward(self, x):
        bs = x[0].shape[0]

        # Box regression
        boxes = torch.cat(
            [self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], -1
        )
        # Class scores
        scores = torch.cat(
            [self.cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], -1
        )
        # Keypoints
        features = [self.cv4[i](x[i]) for i in range(self.nl)]
        kpts = torch.cat(
            [self.cv4_kpts[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1
        )

        preds = dict(boxes=boxes, scores=scores, kpts=kpts, feats=x)

        # Sigma for NME loss (training only)
        if self.training:
            # Reshaped to the channel count the conv actually emits. Using
            # nk // kpt_shape[1] (= 5) against a 10-channel conv folded half
            # of each level into extra anchor columns, so the sigma tensor had
            # twice as many anchors as every other branch and none of them
            # lined up. Nothing consumed it, so nothing complained.
            preds["kpts_sigma"] = torch.cat(
                [self.cv4_sigma[i](features[i]).view(bs, self.nk_sigma, -1)
                 for i in range(self.nl)], -1
            )

        # Attributes, per anchor, laid out exactly like kpts: (B, ATTR_DIM, A)
        preds["attrs"] = torch.cat(
            [self.cv5_attr[i](self.cv5[i](x[i])).view(bs, self.attr_dim, -1)
             for i in range(self.nl)], -1
        )

        # ── Inference ──
        if not self.training:
            y = self._inference(preds)
            if self.end2end_flag:
                y = self.postprocess(y.permute(0, 2, 1))
            return y if self.export else (y, preds)

        return preds

    # ── Inference helpers ──
    def _inference(self, x):
        from ultralytics.utils.tal import dist2bbox, make_anchors

        shape = x["feats"][0].shape
        if self.dynamic or not hasattr(self, "_sh") or self._sh != shape:
            self.anchors, self.strides = (
                a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5)
            )
            self._sh = shape

        dbox = dist2bbox(
            x["boxes"], self.anchors.unsqueeze(0), dim=1
        ) * self.strides
        # Attributes ride along after the keypoints. NMS treats everything
        # past 4 + nc as opaque extra columns and carries it to the surviving
        # rows, so each detection keeps its own attribute vector.
        return torch.cat(
            (dbox, x["scores"].sigmoid(), self.kpts_decode(x["kpts"]),
             x["attrs"]), 1
        )

    def kpts_decode(self, kpts):
        ndim = self.kpt_shape[1]
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
        y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
        return y

    def postprocess(self, preds):
        boxes, scores, kpts, attrs = preds.split(
            [4, self.nc, self.nk, self.attr_dim], -1)
        b, a, nc = scores.shape
        k = min(300, a)
        ori = scores.max(-1)[0].topk(k)[1].unsqueeze(-1)
        sc = scores.gather(1, ori.repeat(1, 1, nc))
        sc, idx = sc.flatten(1).topk(k)
        idx2 = ori[torch.arange(b)[:, None], idx // nc]
        boxes = boxes.gather(1, idx2.repeat(1, 1, 4))
        kpts = kpts.gather(1, idx2.repeat(1, 1, self.nk))
        attrs = attrs.gather(1, idx2.repeat(1, 1, self.attr_dim))
        return torch.cat(
            [boxes, sc[..., None], (idx % nc)[..., None].float(), kpts, attrs], -1
        )

    # ── Init ──
    def bias_init(self):
        for i in range(self.nl):
            self.cv2[i][-1].bias.data[:] = 2.0
            self.cv3[i][-1].bias.data[: self.nc] = -5.0


    def fuse(self):
        """Drop training-only branches.

        This used to set cv2/cv3/cv4/cv4_kpts to None as well, which deletes
        the detection and keypoint branches and makes any later forward pass
        raise. Only the pieces that exist purely for the loss are removed.
        """
        self.cv4_sigma = None
        self.flow_model = None
