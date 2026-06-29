"""Pre-flight validation checks before training."""

import os
import streamlit as st

from flashstudio.utils.device import has_cuda
from flashstudio.pages.training._common import _get_save_dir


def _run_preflight_checks() -> list:
    """Run pre-flight validation checks before training. Returns list of (label, ok, msg)."""
    checks = []

    train_path = st.session_state.get("train_img_path", "")
    if train_path and os.path.isdir(train_path):
        checks.append(("Dataset", True, ""))
    else:
        checks.append(("Dataset", False, "No train data path"))

    if train_path and os.path.isdir(train_path):
        ann_file = os.path.join(train_path, "_annotations.coco.json")
        json_files = [f for f in os.listdir(train_path) if f.endswith(".json")] if os.path.isdir(train_path) else []
        if os.path.isfile(ann_file) or json_files:
            checks.append(("Annotations", True, ""))
        else:
            checks.append(("Annotations", False, "No COCO JSON found"))
    else:
        checks.append(("Annotations", False, "No data path"))

    nc = st.session_state.get("num_classes", 0)
    cls_names = st.session_state.get("class_names", "")
    if isinstance(cls_names, list):
        cls_names = "\n".join(cls_names)
    if nc and nc > 0:
        checks.append(("Classes", True, f"{nc} classes"))
    elif isinstance(cls_names, str) and cls_names.strip():
        n = len([c for c in cls_names.strip().split("\n") if c.strip()])
        checks.append(("Classes", True, f"{n} classes"))
    else:
        checks.append(("Classes", False, "No classes — go to Data → Upload"))

    from flashstudio.utils import get_state
    arch = get_state("model_arch") or ""
    if arch:
        checks.append(("Model", True, ""))
    else:
        checks.append(("Model", False, "No model selected"))

    if has_cuda():
        checks.append(("GPU", True, ""))
    else:
        checks.append(("GPU", False, "CPU only (slow)"))
        checks[-1] = ("GPU", True, "CPU mode")

    save_dir = _get_save_dir()
    try:
        stat = os.statvfs(os.path.dirname(save_dir) if not os.path.isdir(save_dir) else save_dir)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        if free_gb > 1:
            checks.append(("Disk", True, ""))
        else:
            checks.append(("Disk", False, f"Only {free_gb:.1f} GB free"))
    except (OSError, AttributeError):
        checks.append(("Disk", True, ""))

    return checks
