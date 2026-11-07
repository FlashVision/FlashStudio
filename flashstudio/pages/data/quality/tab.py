"""Data — Dataset quality report."""

import hashlib
import os

import streamlit as st
from PIL import Image

from flashstudio.pages.data.fix.annotations import load_coco
from flashstudio.pages.data.fix.tab import _coco_path, _yolo_label_dir


def _image_files(img_dir: str) -> list:
    if not os.path.isdir(img_dir):
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted(f for f in os.listdir(img_dir) if f.lower().endswith(exts))


def dataset_quality(images_dir: str, ann_path: str = "", labels_dir: str = "") -> dict:
    """Compute a practical quality report for COCO or YOLO labels."""
    images = _image_files(images_dir)
    report = {
        "images": len(images),
        "annotations": 0,
        "classes": {},
        "empty_images": [],
        "corrupt_images": [],
        "duplicates": {},
        "widths": [],
        "heights": [],
        "tiny_boxes": 0,
        "small_boxes": 0,
        "medium_boxes": 0,
        "large_boxes": 0,
        "format": "COCO" if ann_path else ("YOLO" if labels_dir else "unknown"),
    }

    hashes = {}
    for name in images:
        path = os.path.join(images_dir, name)
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                w, h = im.size
            report["widths"].append(w)
            report["heights"].append(h)
            with open(path, "rb") as f:
                digest = hashlib.md5(f.read(65536)).hexdigest() + f":{os.path.getsize(path)}"
            hashes.setdefault(digest, []).append(name)
        except Exception:
            report["corrupt_images"].append(name)

    report["duplicates"] = {k: v for k, v in hashes.items() if len(v) > 1}

    image_ids = {}
    if ann_path and os.path.isfile(ann_path):
        data = load_coco(ann_path)
        cats = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
        for img in data.get("images", []):
            image_ids[img.get("id")] = img.get("file_name")
            report["widths"].append(int(img.get("width", 0) or 0)) if img.get("width") else None
            report["heights"].append(int(img.get("height", 0) or 0)) if img.get("height") else None
        counts_by_image = {}
        for ann in data.get("annotations", []):
            report["annotations"] += 1
            name = cats.get(ann.get("category_id"), str(ann.get("category_id")))
            report["classes"][name] = report["classes"].get(name, 0) + 1
            counts_by_image[ann.get("image_id")] = counts_by_image.get(ann.get("image_id"), 0) + 1
            bbox = ann.get("bbox") or [0, 0, 0, 0]
            img = next((i for i in data.get("images", []) if i.get("id") == ann.get("image_id")), {})
            iw, ih = float(img.get("width", 0) or 0), float(img.get("height", 0) or 0)
            if iw > 0 and ih > 0 and len(bbox) == 4:
                ratio = (float(bbox[2]) * float(bbox[3])) / (iw * ih)
                if ratio < 0.001:
                    report["tiny_boxes"] += 1
                elif ratio < 0.01:
                    report["small_boxes"] += 1
                elif ratio < 0.1:
                    report["medium_boxes"] += 1
                else:
                    report["large_boxes"] += 1
        for iid, name in image_ids.items():
            if counts_by_image.get(iid, 0) == 0 and name:
                report["empty_images"].append(name)
    elif labels_dir and os.path.isdir(labels_dir):
        for name in images:
            stem = os.path.splitext(name)[0]
            path = os.path.join(labels_dir, stem + ".txt")
            if not os.path.isfile(path):
                report["empty_images"].append(name)
                continue
            count = 0
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    try:
                        cid = int(float(parts[0]))
                        w, h = float(parts[3]), float(parts[4])
                    except ValueError:
                        continue
                    count += 1
                    report["annotations"] += 1
                    report["classes"][f"class_{cid}"] = report["classes"].get(f"class_{cid}", 0) + 1
                    ratio = w * h
                    if ratio < 0.001:
                        report["tiny_boxes"] += 1
                    elif ratio < 0.01:
                        report["small_boxes"] += 1
                    elif ratio < 0.1:
                        report["medium_boxes"] += 1
                    else:
                        report["large_boxes"] += 1
            if count == 0:
                report["empty_images"].append(name)

    widths = [w for w in report["widths"] if w]
    heights = [h for h in report["heights"] if h]
    report["avg_width"] = sum(widths) / len(widths) if widths else 0
    report["avg_height"] = sum(heights) / len(heights) if heights else 0
    counts = list(report["classes"].values())
    report["imbalance"] = (max(counts) / max(min(counts), 1)) if counts else 0
    report["duplicate_images"] = sum(len(v) - 1 for v in report["duplicates"].values())
    return report


def _render_quality():
    st.caption("Check class balance, empty images, duplicates, corrupt files, and box sizes before training.")
    split = st.radio("Split", ["Train", "Validation"], horizontal=True, key="quality_split")
    img_key = "train_img_path" if split == "Train" else "val_img_path"
    label_key = "train_label_path" if split == "Train" else "val_label_path"
    img_dir = st.session_state.get(img_key, "")
    if not img_dir or not os.path.isdir(img_dir):
        st.info(f"No {split.lower()} images directory is configured.")
        return

    ann_path = _coco_path(img_dir)
    labels_dir = _yolo_label_dir(img_dir, st.session_state.get(label_key))
    if not ann_path and not (os.path.isdir(labels_dir) and any(
        f.endswith(".txt") for f in os.listdir(labels_dir)
    )):
        st.info("No COCO JSON or YOLO labels found.")
        return

    if st.button("Run quality report", type="primary", key="quality_run"):
        with st.spinner("Checking images and annotations..."):
            st.session_state["_quality_report"] = dataset_quality(img_dir, ann_path, labels_dir)

    report = st.session_state.get("_quality_report")
    if not report:
        return

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Images", report["images"])
    with c2:
        st.metric("Annotations", report["annotations"])
    with c3:
        st.metric("Classes", len(report["classes"]))
    with c4:
        st.metric("Empty", len(report["empty_images"]))
    with c5:
        st.metric("Corrupt", len(report["corrupt_images"]))
    with c6:
        st.metric("Duplicates", report.get("duplicate_images", 0))

    if report["classes"]:
        st.markdown("#### Class distribution")
        st.bar_chart(report["classes"])
        if report.get("imbalance", 0) > 5:
            st.warning(f"Class imbalance is high: max/min ratio ≈ {report['imbalance']:.1f}×.")
        else:
            st.caption(f"Class imbalance ratio: {report.get('imbalance', 0):.1f}×")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Tiny boxes", report["tiny_boxes"])
    with s2:
        st.metric("Small boxes", report["small_boxes"])
    with s3:
        st.metric("Medium boxes", report["medium_boxes"])
    with s4:
        st.metric("Large boxes", report["large_boxes"])

    if report["empty_images"]:
        with st.expander(f"Images without annotations ({len(report['empty_images'])})"):
            st.code("\n".join(report["empty_images"][:100]))
    if report["corrupt_images"]:
        with st.expander(f"Corrupt images ({len(report['corrupt_images'])})"):
            st.code("\n".join(report["corrupt_images"]))
    if report["duplicates"]:
        with st.expander(f"Duplicate image groups ({len(report['duplicates'])})"):
            for names in list(report["duplicates"].values())[:50]:
                st.caption(" · ".join(names))
