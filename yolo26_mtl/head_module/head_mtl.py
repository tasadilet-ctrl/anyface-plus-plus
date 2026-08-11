"""Multi-Task Learning head for YOLO26 — AnyFace++

Detects faces + 5 landmarks + gender + age + emotion in one forward pass.
Based on the Pose26 head with additional MTL branches for gender/age/emotion.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import RealNVP
from ultralytics.nn.modules.conv import Conv, DWConv


class MTLPose(nn.Module):
    """YOLO26 MTL head for AnyFace++.

    Tasks:
        - Face detection (bounding box + confidence)
        - 5 facial landmarks (left eye, right eye, nose, mouth L, mouth R)
        - Gender classification (3 classes: female, male, unsure)
        - Age regression (0-116)
        - Emotion classification (8 classes)
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
        nk_sig = kpt_shape[0] * 2
        self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, nk_sig, 1) for _ in ch)
        self.flow_model = RealNVP()

        # ── MTL heads on P4 feature (ch[1], intermediate scale) ──
        mtl_ch = ch[1]  # P4: 512 channels at 1/16 resolution

        self.gender_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mtl_ch, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

        self.age_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mtl_ch, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        self.emotion_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mtl_ch, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 8),
        )

        # ── One-to-one branches for end2end ──
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    # ── Properties ──
    @property
    def one2many(self):
        return dict(
            box_head=self.cv2, cls_head=self.cv3, pose_head=self.cv4,
            kpts_head=self.cv4_kpts, kpts_sigma_head=self.cv4_sigma,
        )

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2, cls_head=self.one2one_cv3,
            pose_head=self.one2one_cv4, kpts_head=self.one2one_cv4_kpts,
            kpts_sigma_head=self.one2one_cv4_sigma,
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
            preds["kpts_sigma"] = torch.cat(
                [self.cv4_sigma[i](features[i]).view(
                    bs, self.nk // self.kpt_shape[1], -1
                ) for i in range(self.nl)], -1
            )

        # MTL heads from P4 feature (features[1])
        mtl_feat = features[1]
        preds["gender"] = self.gender_head(mtl_feat)          # (B, 3)
        preds["age"] = self.age_head(mtl_feat).squeeze(-1)    # (B,)
        preds["emotion"] = self.emotion_head(mtl_feat)        # (B, 8)

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
        return torch.cat(
            (dbox, x["scores"].sigmoid(), self.kpts_decode(x["kpts"])), 1
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
        boxes, scores, kpts = preds.split([4, self.nc, self.nk], -1)
        b, a, nc = scores.shape
        k = min(300, a)
        ori = scores.max(-1)[0].topk(k)[1].unsqueeze(-1)
        sc = scores.gather(1, ori.repeat(1, 1, nc))
        sc, idx = sc.flatten(1).topk(k)
        idx2 = ori[torch.arange(b)[:, None], idx // nc]
        boxes = boxes.gather(1, idx2.repeat(1, 1, 4))
        kpts = kpts.gather(1, idx2.repeat(1, 1, self.nk))
        return torch.cat(
            [boxes, sc[..., None], (idx % nc)[..., None].float(), kpts], -1
        )

    # ── Init ──
    def bias_init(self):
        for i in range(self.nl):
            self.cv2[i][-1].bias.data[:] = 2.0
            self.cv3[i][-1].bias.data[: self.nc] = -5.0
        if self.end2end_flag:
            for i in range(self.nl):
                self.one2one_cv2[i][-1].bias.data[:] = 2.0
                self.one2one_cv3[i][-1].bias.data[: self.nc] = -5.0

    def fuse(self):
        self.cv2 = self.cv3 = self.cv4 = None
        self.cv4_kpts = self.cv4_sigma = None
        self.flow_model = None
