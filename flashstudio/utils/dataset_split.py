"""Automatic train/validation split when a dataset has no validation set."""

import json
import os
import random
import shutil

import streamlit as st

from PIL import Image

from flashstudio.utils.universal_convert import _write_split


def _images(root):
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                found.append(os.path.join(dirpath, name))
    return found


def _copy_or_link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _collect_coco(img_dir, ann_file):
    with open(ann_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cat_names = {
        cat["id"]: str(cat.get("name") or cat["id"])
        for cat in data.get("categories", [])
    }
    anns = {}
    for ann in data.get("annotations", []):
        anns.setdefault(ann.get("image_id"), []).append(ann)
    index = {os.path.basename(p): p for p in _images(img_dir)}
    entries = []
    classes = set(cat_names.values())
    for image in data.get("images", []):
        name = str(image.get("file_name") or "")
        path = os.path.join(img_dir, name)
        if not os.path.isfile(path):
            path = index.get(os.path.basename(name), "")
        if not path:
            continue
        boxes = []
        for ann in anns.get(image.get("id"), []):
            bbox = ann.get("bbox")
            label = cat_names.get(ann.get("category_id"))
            if label and bbox and len(bbox) == 4:
                boxes.append((label, list(bbox)))
        entries.append({
            "path": path,
            "name": os.path.basename(path),
            "width": image.get("width"),
            "height": image.get("height"),
            "boxes": boxes,
        })
    return entries, classes


def _collect_yolo(img_dir, label_dir):
    pairs = []
    for path in _images(img_dir):
        label = ""
        if label_dir:
            candidate = os.path.join(
                label_dir, os.path.splitext(os.path.basename(path))[0] + ".txt"
            )
            if os.path.isfile(candidate):
                label = candidate
        pairs.append((path, label))
    return pairs


def auto_validation_split(train_img_dir, ann_file="", label_dir="",
                          val_ratio=20, seed=42, out_dir=None):
    """Create an 80/20-style split from a train-only dataset.

    Returns a dict of session-state paths, or ``None`` when there is no data.
    """
    if not train_img_dir or not os.path.isdir(train_img_dir):
        return None
    out = out_dir or (
        train_img_dir.rstrip("/\\") + f"_val{int(val_ratio)}"
    )
    shutil.rmtree(out, ignore_errors=True)
    rng = random.Random(int(seed))

    if ann_file and os.path.isfile(ann_file):
        entries, classes = _collect_coco(train_img_dir, ann_file)
        if not entries:
            return None
        rng.shuffle(entries)
        split_at = max(1, int(len(entries) * (100 - val_ratio) / 100.0))
        train_entries = entries[:split_at]
        val_entries = entries[split_at:] or entries[-1:]
        names = sorted(classes) or ["object"]
        cat_ids = {name: i + 1 for i, name in enumerate(names)}
        _write_split(os.path.join(out, "train"), train_entries, cat_ids)
        _write_split(os.path.join(out, "valid"), val_entries, cat_ids)
        return {
            "train_img_path": os.path.join(out, "train", "images"),
            "val_img_path": os.path.join(out, "valid", "images"),
            "train_ann_path": os.path.join(
                out, "train", "images", "_annotations.coco.json"
            ),
            "val_ann_path": os.path.join(
                out, "valid", "images", "_annotations.coco.json"
            ),
            "train_label_path": "",
            "val_label_path": "",
            "class_names": "\n".join(names),
            "num_classes": len(names),
            "_pending_ann_format": "COCO JSON",
            "detected_format": "coco",
        }

    pairs = _collect_yolo(train_img_dir, label_dir)
    if not pairs:
        return None
    rng.shuffle(pairs)
    split_at = max(1, int(len(pairs) * (100 - val_ratio) / 100.0))
    train_pairs = pairs[:split_at]
    val_pairs = pairs[split_at:] or pairs[-1:]
    for split, split_pairs in (("train", train_pairs), ("valid", val_pairs)):
        img_out = os.path.join(out, split, "images")
        lbl_out = os.path.join(out, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for image, label in split_pairs:
            name = os.path.basename(image)
            _copy_or_link(image, os.path.join(img_out, name))
            target = os.path.join(lbl_out, os.path.splitext(name)[0] + ".txt")
            if label:
                _copy_or_link(label, target)
            else:
                open(target, "w").close()
    return {
        "train_img_path": os.path.join(out, "train", "images"),
        "val_img_path": os.path.join(out, "valid", "images"),
        "train_label_path": os.path.join(out, "train", "labels"),
        "val_label_path": os.path.join(out, "valid", "labels"),
        "train_ann_path": "",
        "val_ann_path": "",
        "_pending_ann_format": "YOLO",
        "detected_format": "txt",
    }


def maybe_auto_split(train_img_dir="", val_img_dir="", ann_file="", label_dir=""):
    """Create a validation split only when none exists.

    The percentage/seed come from the Data page controls and default to 20%.
    """
    if val_img_dir and os.path.isdir(val_img_dir) and _images(val_img_dir):
        return None
    if not train_img_dir or not os.path.isdir(train_img_dir):
        return None
    pct = int(st.session_state.get("auto_val_pct", 20) or 20)
    seed = int(st.session_state.get("auto_val_seed", 42) or 42)
    state = auto_validation_split(
        train_img_dir, ann_file=ann_file, label_dir=label_dir,
        val_ratio=pct, seed=seed,
    )
    if state:
        st.session_state.update(state)
        st.session_state["_auto_split_done_for"] = train_img_dir
    return state
