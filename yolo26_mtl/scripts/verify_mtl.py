#!/usr/bin/env python3
"""Checks that the MTL head builds and runs. CPU-only, no weights, no data.

Replaces setup_mtl.sh, which copied head_mtl.py into the installed ultralytics
package and ran sed on its head.py. That was undone by any reinstall of
ultralytics, and it did not work: parse_model dispatches on a hard-coded set
of head classes that the sed never touched, so building the model failed with
"TypeError: list indices must be integers or slices, not list". The
verification block at the end of that script had never been run either -- it
used ch=(256, 512, 1024), where the old attribute heads raised a shape error.

    python3 yolo26_mtl/scripts/verify_mtl.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo26_mtl import build_mtl_model, registered_mtl_head  # noqa: E402
from yolo26_mtl.head_module.head_mtl import ATTR_DIM, decode_attrs  # noqa: E402


def check(label, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    return ok


def main():
    ok = True
    print("Head, at both the documented and the real channel widths:")
    for ch in [(256, 512, 1024), (64, 128, 256)]:
        with registered_mtl_head() as MTLPose:
            h = MTLPose(nc=1, kpt_shape=(5, 3), reg_max=1, end2end=False, ch=ch)
        h.stride = torch.tensor([8.0, 16.0, 32.0])
        h.train()
        feats = [torch.randn(2, c, 80 // (2 ** i), 80 // (2 ** i))
                 for i, c in enumerate(ch)]
        out = h(feats)
        anchors = {k: v.shape[-1] for k, v in out.items() if k != "feats"}
        ok &= check(f"forward at ch={ch}", True,
                    f"attrs {tuple(out['attrs'].shape)}")
        ok &= check("  every branch on one anchor axis",
                    len(set(anchors.values())) == 1, str(anchors))

    print("\nModel from config:")
    model = build_mtl_model()
    head = model.model[-1]
    ok &= check("built from yolo26n-mtl.yaml", True, type(head).__name__)

    import ultralytics.nn.tasks as tasks
    ok &= check("ultralytics left unpatched afterwards",
                tasks.Pose26.__name__ == "Pose26")

    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, 640, 640))
    y = y[0] if isinstance(y, (tuple, list)) else y
    ok &= check("inference forward", True, f"{tuple(y.shape)}")

    print("\nAttribute decode:")
    d = decode_attrs(torch.randn(ATTR_DIM))
    ok &= check("decodes to labels", isinstance(d["gender"], str), str(d))
    ok &= check("confidences are probabilities",
                0 <= d["gender_conf"] <= 1 and 0 <= d["emotion_conf"] <= 1)

    print("\nNOTE: the attribute branches have no loss. Training with"
          "\ntask='pose' optimises boxes and keypoints only; gender, age and"
          "\nemotion stay at their initialisation. See the README.")
    print("\n" + ("All checks passed." if ok else "FAILURES above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
