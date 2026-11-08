"""Repair inconsistent dataset selections stored in session state.

Earlier FlashStudio versions could persist a ``train_img_path`` that pointed at
a temporary conversion folder (``_yolo_*``) or at a validation split.  Because
the selection is saved with the project, upgrading the package alone does not
repair it: the stale path keeps being reused on every launch.

:func:`repair_dataset_paths` detects this situation and rebuilds a consistent
train/val pair from the project's own datasets.  It is safe to call on every
startup and before launching training; a healthy selection is left untouched.
"""

from __future__ import annotations

import os

from flashstudio.utils.dataset_paths import (
    has_images,
    load_coco_classes,
    resolve_annotation_file,
    split_dir,
)

# Folders FlashStudio creates while converting/uploading datasets.  They are
# never a user-selected dataset root.
_INTERNAL_PREFIXES = ("_yolo_", "_subset_", "_upload_", "_tmp_", ".downloads")

_SPLIT_ALIASES = {
    "train": ("train", "training", "train2017"),
    "val": ("valid", "val", "validation", "val2017", "test"),
}


def _is_internal(path: str) -> bool:
    parts = str(path or "").replace("\\", "/").split("/")
    return any(
        part.lower().startswith(_INTERNAL_PREFIXES) for part in parts if part
    )


def _split_name(path: str) -> str:
    return os.path.basename(split_dir(path)).lower()


def _looks_like_val(path: str) -> bool:
    return _split_name(path).startswith(("val", "test"))


def _looks_like_train(path: str) -> bool:
    return _split_name(path).startswith("train")


def _images_dir(path: str) -> str:
    """Return *path* when it directly contains images, else its ``images/`` child."""
    if not path or not os.path.isdir(path):
        return ""
    if has_images(path):
        return path
    for child in ("images", "imgs", "JPEGImages"):
        candidate = os.path.join(path, child)
        if os.path.isdir(candidate) and has_images(candidate):
            return candidate
    return ""


def _valid_split_images(path: str, split: str) -> str:
    """Return the image folder for *split* when it is usable, else ``""``."""
    if not path or not os.path.isdir(path):
        return ""
    if _is_internal(path):
        return ""
    if split == "train" and _looks_like_val(path):
        return ""
    if split == "val" and _looks_like_train(path):
        return ""

    images = _images_dir(path)
    if not images:
        return ""
    annotation = resolve_annotation_file(images, split=split)
    if not annotation:
        # Layouts where the annotation sits next to the split's images folder.
        annotation = resolve_annotation_file(split_dir(path), split=split)
    return images if annotation else ""


def _root_from_path(path: str) -> str:
    """Best-effort dataset root for an images/split path."""
    if not path:
        return ""
    return os.path.dirname(split_dir(path))


def _find_split(root: str, split: str) -> str:
    """Find a usable *split* image folder inside *root*."""
    if not root or not os.path.isdir(root):
        return ""
    for alias in _SPLIT_ALIASES[split]:
        direct = os.path.join(root, alias)
        images = _valid_split_images(direct, split)
        if images:
            return images
    return ""


def _pair_for_root(root: str) -> tuple[str, str]:
    """Return ``(train_images, val_images)`` when *root* holds a full pair."""
    if not root or not os.path.isdir(root) or _is_internal(root):
        return "", ""
    train = _find_split(root, "train")
    if not train:
        return "", ""
    val = _find_split(root, "val")
    if not val:
        from flashstudio.utils.dataset_paths import find_validation_dir

        val = find_validation_dir(train, root=root)
        val = _valid_split_images(val, "val")
    return (train, val) if val else ("", "")


def _candidate_roots(state) -> list[str]:
    roots: list[str] = []

    def _add(path):
        root = _root_from_path(path) if path else ""
        if root and root not in roots:
            roots.append(root)
        if path and os.path.isdir(path) and path not in roots:
            roots.append(os.path.abspath(path))

    # The validation split and dataset_output_path usually identify the
    # dataset the user actually wants; the stale path is normally the train
    # one, so it comes last.
    _add(str(state.get("val_img_path") or ""))
    _add(str(state.get("dataset_output_path") or ""))

    train_hint = str(state.get("train_ann_path") or "")
    if train_hint:
        _add(os.path.dirname(train_hint))
    _add(str(state.get("train_img_path") or ""))

    download_root = str(state.get("download_output_dir") or "")
    if download_root and os.path.isdir(download_root):
        try:
            for name in sorted(os.listdir(download_root)):
                child = os.path.join(download_root, name)
                if os.path.isdir(child):
                    _add(child)
                    try:
                        for sub in sorted(os.listdir(child)):
                            sub_path = os.path.join(child, sub)
                            if os.path.isdir(sub_path):
                                _add(sub_path)
                    except OSError:
                        pass
        except OSError:
            pass
    return roots


def _friendly_name(root: str) -> str:
    """Human-readable dataset name derived from its folder."""
    name = os.path.basename(os.path.normpath(root)) or "dataset"
    lowered = name.lower()
    for suffix in ("-master_coco", "_dataset-master_coco", "_coco", "-coco"):
        if lowered.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _publish(state, root: str, train: str, val: str) -> None:
    """Write a consistent pair into *state*."""
    train_ann = resolve_annotation_file(train, split="train")
    val_ann = resolve_annotation_file(val, split="val")
    state["train_img_path"] = train
    state["val_img_path"] = val
    state["dataset_output_path"] = root
    state["dataset_name"] = _friendly_name(root)
    if train_ann:
        state["train_ann_path"] = train_ann
    if val_ann:
        state["val_ann_path"] = val_ann

    names = load_coco_classes(train_ann) if train_ann else []
    if names:
        state["class_names"] = "\n".join(names)
        state["num_classes"] = len(names)
        state["dataset_classes"] = len(names)


def _current_pair_is_healthy(state) -> bool:
    train = str(state.get("train_img_path") or "")
    val = str(state.get("val_img_path") or "")
    if not train or not val:
        return False
    if _is_internal(train) or _is_internal(val):
        return False
    if _looks_like_val(train) or _looks_like_train(val):
        return False
    train_images = _valid_split_images(train, "train")
    val_images = _valid_split_images(val, "val")
    if not train_images or not val_images:
        return False
    # Both splits must belong to the same dataset.
    return os.path.abspath(_root_from_path(train_images)) == os.path.abspath(
        _root_from_path(val_images)
    )


def repair_dataset_paths(save: bool = False) -> dict:
    """Repair the active session's dataset paths.

    Returns a report dict with ``repaired`` and ``root`` keys.  When the
    current train/val pair is already consistent nothing is changed.
    """
    import streamlit as st

    state = st.session_state
    if _current_pair_is_healthy(state):
        return {"repaired": False, "root": ""}

    for root in _candidate_roots(state):
        train, val = _pair_for_root(root)
        if train and val:
            _publish(state, root, train, val)
            report = {"repaired": True, "root": root}
            if save:
                try:
                    from flashstudio.components.project_manager import (
                        save_project_state,
                    )

                    save_project_state()
                except Exception:
                    pass
            state["_dataset_repair_message"] = (
                f"Dataset paths were repaired to use `{root}`."
            )
            return report

    return {"repaired": False, "root": ""}
