"""YOLO26 multi-task head for AnyFace++ (faces + landmarks + attributes).

``setup_mtl.sh`` used to install this by copying head_mtl.py into the
installed ultralytics package and running sed on its head.py. That approach
had three problems: it edited a third-party package in place, it was undone by
any reinstall or upgrade of ultralytics, and it did not actually work --
``parse_model`` dispatches on a hard-coded set of head classes that the sed
never touched, so building the model still failed with

    TypeError: list indices must be integers or slices, not list

``build_mtl_model`` registers the head at runtime instead, for the duration of
the call, and leaves ultralytics untouched.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional, Union

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "yolo26n-mtl.yaml"


@contextlib.contextmanager
def registered_mtl_head():
    """Make MTLPose visible to ultralytics' model parser, then undo it.

    ``parse_model`` decides how to build a head by testing membership in a
    frozenset that it rebuilds from its own module globals on every call. The
    set is a literal, so it cannot be extended -- but it is built from names,
    so binding one of those names to MTLPose for the duration of the parse
    puts our head on the head path and gets it the (reg_max, end2end, ch)
    arguments it expects. Pose26 is the name borrowed, and it is restored on
    exit, so parsing an ordinary pose model afterwards is unaffected.
    """
    import ultralytics.nn.tasks as tasks

    from .head_module.head_mtl import MTLPose

    saved_pose26 = tasks.Pose26
    saved_mtl = getattr(tasks, "MTLPose", None)
    tasks.Pose26 = MTLPose
    tasks.MTLPose = MTLPose
    try:
        yield MTLPose
    finally:
        tasks.Pose26 = saved_pose26
        if saved_mtl is None:
            delattr(tasks, "MTLPose")
        else:
            tasks.MTLPose = saved_mtl


def build_mtl_model(config: Optional[Union[str, Path]] = None, ch: int = 3,
                    nc: Optional[int] = None, verbose: bool = False):
    """Build the MTL model from a YAML config. Returns an ultralytics PoseModel."""
    from ultralytics.nn.tasks import PoseModel

    cfg = str(config or DEFAULT_CONFIG)
    with registered_mtl_head():
        return PoseModel(cfg, ch=ch, nc=nc, verbose=verbose)


__all__ = ["build_mtl_model", "registered_mtl_head", "DEFAULT_CONFIG"]
