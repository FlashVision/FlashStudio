"""Pre-flight validation checks before training."""

import os
import streamlit as st

from flashstudio.utils.device import has_cuda
from flashstudio.pages.training._common import _get_save_dir


def _other_active_runs():
    """Return live training runs in the workspace other than the active one."""
    from flashstudio.pages.training.processes import running_trainer_manifests

    save_root = os.path.abspath(_get_save_dir())
    current_run = os.path.abspath(
        str(st.session_state.get("active_run_path") or "")
    )
    found = []
    try:
        manifests = running_trainer_manifests()
    except Exception:
        return found
    for manifest, pids in manifests.items():
        run_dir = os.path.dirname(manifest)
        try:
            inside = os.path.commonpath([manifest, save_root]) == save_root
        except ValueError:
            inside = False
        if inside and run_dir != current_run:
            found.append({
                "name": os.path.basename(run_dir),
                "path": run_dir,
                "pids": pids,
            })
    found.sort(key=lambda run: run["name"])
    return found


def _run_preflight_checks() -> list:
    """Run pre-flight validation checks before training. Returns list of (label, ok, msg)."""
    checks = []

    from flashstudio.utils.dataset_paths import find_validation_dir

    train_path = st.session_state.get("train_img_path", "")
    if train_path and os.path.isdir(train_path):
        checks.append(("Training data", True, ""))
    else:
        checks.append(("Training data", False, "No training images directory"))

    val_path = st.session_state.get("val_img_path", "")
    if not val_path or not os.path.isdir(val_path):
        val_path = find_validation_dir(train_path) if train_path else ""
    if val_path and os.path.isdir(val_path):
        checks.append(("Validation split", True, ""))
    else:
        checks.append(("Validation split", False, "No validation images directory"))

    from flashstudio.utils.dataset_paths import resolve_annotation_file

    if train_path and os.path.isdir(train_path):
        ann_file = resolve_annotation_file(
            train_path,
            split="train",
            hint=st.session_state.get("train_ann_path", ""),
        )
        ann_found = bool(ann_file)
        checks.append(("Train annotations", ann_found, "" if ann_found else "No COCO annotations found"))
    else:
        checks.append(("Train annotations", False, "No training data directory"))

    if val_path and os.path.isdir(val_path):
        val_ann = resolve_annotation_file(
            val_path,
            split="val",
            hint=st.session_state.get("val_ann_path", ""),
        )
        val_ok = bool(val_ann)
        checks.append(("Validation annotations", val_ok, "" if val_ok else "No COCO annotations found"))
    else:
        checks.append(("Validation annotations", False, "No validation data directory"))

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
        checks.append(("Classes", False, "Load a dataset or enter class names"))

    from flashstudio.utils import get_state
    arch = get_state("model_arch") or ""
    if arch:
        checks.append(("Model", True, ""))
    else:
        checks.append(("Model", False, "No architecture selected"))

    backend = st.session_state.get("train_backend", "FlashDet")
    if backend == "Ultralytics":
        family = str(get_state("arch_family") or "")
        if not family.startswith("YOLOv"):
            checks.append((
                "Backend model", False,
                "Ultralytics needs YOLOv8/9/10/11 in Model → Architecture",
            ))
        try:
            import ultralytics

            checks.append(("Backend", True, f"Ultralytics {ultralytics.__version__}"))
        except Exception as exc:
            checks.append(("Backend", False, f"Ultralytics unavailable: {exc}"))
    else:
        try:
            import re
            import flashdet
            from flashdet.engine import Trainer  # noqa: F401

            version = str(getattr(flashdet, "__version__", "0"))
            parts = tuple(int(x) for x in re.findall(r"\d+", version)[:2])
            if parts and parts < (1, 3):
                raise RuntimeError(
                    f"flashdet {version} is too old; install flashdet>=1.3.0 "
                    "(a stale copy may be shadowing an editable install)"
                )
            checks.append(("Backend", True, f"flashdet {version}"))
        except Exception as exc:
            checks.append(("Backend", False, f"FlashDet Trainer unavailable: {exc}"))

    if has_cuda():
        checks.append(("GPU", True, "CUDA"))
    else:
        checks.append(("GPU", True, "CPU only"))

    ignored = []
    if st.session_state.get("multiscale"):
        ignored.append("Multi-Scale")
    if st.session_state.get("chunked_loss"):
        ignored.append("Chunked Loss")
    if ignored:
        checks.append(("Ignored options", True, ", ".join(ignored)))

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

    # Never launch a second run while another run is still training. Without
    # this, clicking Start on a new run leaves two trainers competing for the
    # same CPU/GPU.
    active_others = _other_active_runs()
    if active_others:
        other = active_others[0]
        checks.append((
            "Other run active", False,
            f"{other['name']} (PID {other['pids'][0]}) — stop it before starting another run",
        ))

    return checks
