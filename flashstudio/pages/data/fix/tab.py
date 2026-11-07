"""Data — Repair annotations tab."""

import json
import os
import tempfile

import streamlit as st

from flashstudio.pages.data.fix.annotations import (
    fix_coco, fix_yolo, scan_coco, scan_yolo, _yolo_label_dir,
)
from flashstudio.pages.data._common import _load_data_prompt


def _coco_path(img_dir: str, split: str = "") -> str:
    """Find the COCO annotation file for a split.

    Searches the split directory, its parent, and common ``annotations/``
    folders.  ``split`` ("train"/"val") is used to avoid mixing the train and
    validation annotation files.
    """
    from flashstudio.utils.dataset_paths import resolve_annotation_file

    hint = st.session_state.get(f"{split}_ann_path", "") if split else ""
    return resolve_annotation_file(img_dir, split=split, hint=hint)


def _issue_rows(report: dict) -> list:
    rows = []
    if report.get("format") == "COCO":
        for name in report.get("missing_images", []):
            rows.append({"Type": "Missing image", "Detail": name})
        for ann in report.get("missing_image_refs", []):
            rows.append({"Type": "Missing image reference", "Detail": f"annotation {ann.get('id')}"})
        for ann in report.get("invalid_boxes", []):
            rows.append({"Type": "Invalid bbox", "Detail": f"annotation {ann.get('id')}"})
        for ann in report.get("zero_area", []):
            rows.append({"Type": "Zero-area bbox", "Detail": f"annotation {ann.get('id')}"})
        for ann in report.get("out_of_bounds", []):
            rows.append({"Type": "Bbox outside image", "Detail": f"annotation {ann.get('id')}"})
        for ann in report.get("unknown_categories", []):
            rows.append({"Type": "Unknown category", "Detail": f"annotation {ann.get('id')} → {ann.get('category_id')}"})
        for ann in report.get("duplicate_boxes", []):
            rows.append({"Type": "Duplicate bbox", "Detail": f"annotation {ann.get('id')}"})
        for item in report.get("bad_categories", []):
            rows.append({"Type": "Bad metadata", "Detail": item})
    else:
        for item in report.get("invalid_lines", []):
            rows.append({"Type": "Invalid line", "Detail": item})
        for item in report.get("out_of_range_classes", []):
            rows.append({"Type": "Class out of range", "Detail": item})
        for item in report.get("out_of_bounds", []):
            rows.append({"Type": "Coords outside 0–1", "Detail": item})
        for item in report.get("missing_labels", [])[:50]:
            rows.append({"Type": "Missing label", "Detail": item})
        for item in report.get("orphan_labels", [])[:50]:
            rows.append({"Type": "Label without image", "Detail": item})
    return rows


def _render_fix():
    st.caption("Scan annotations, repair boxes and class IDs, and review the result before training.")

    split = st.radio("Split", ["Train", "Validation"], horizontal=True, key="fix_split")
    img_key = "train_img_path" if split == "Train" else "val_img_path"
    label_key = "train_label_path" if split == "Train" else "val_label_path"
    img_dir = st.session_state.get(img_key, "")

    if not img_dir or not os.path.isdir(img_dir):
        _load_data_prompt(
            f"No {split.lower()} images directory is configured. Load a dataset first.",
            "fix_load_data",
        )
        return

    ann_path = _coco_path(img_dir, "train" if split == "Train" else "val")
    labels_dir = _yolo_label_dir(img_dir, st.session_state.get(label_key))
    fmt = "COCO" if ann_path else ("YOLO" if os.path.isdir(labels_dir) and any(
        f.endswith(".txt") for f in os.listdir(labels_dir)
    ) else "")

    if not fmt:
        st.info("No COCO JSON or YOLO labels found for this split.")
        st.caption(f"Images: `{img_dir}`")
        return

    st.caption(f"Images: `{img_dir}`")
    if fmt == "COCO":
        st.caption(f"Annotations: `{ann_path}`")
    else:
        st.caption(f"Labels: `{labels_dir}`")

    report_key = f"_fix_report_{split.lower()}"
    scan_col, note_col = st.columns([1, 4])
    with scan_col:
        if st.button("Scan", type="primary", key="fix_scan_btn", use_container_width=True):
            try:
                if fmt == "COCO":
                    report = scan_coco(ann_path, img_dir)
                else:
                    report = scan_yolo(
                        labels_dir, img_dir,
                        int(st.session_state.get("num_classes", 0) or 0) or None,
                    )
                st.session_state[report_key] = report
            except Exception as exc:
                st.error(f"Scan failed: {exc}")
                return
    with note_col:
        st.caption("A `.bak` backup is created before any in-place fix.")

    report = st.session_state.get(report_key)
    if not report:
        return

    if fmt == "COCO":
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("Images", report.get("images", 0))
        with c2:
            st.metric("Annotations", report.get("annotations", 0))
        with c3:
            st.metric("Categories", report.get("categories", 0))
        with c4:
            st.metric("Missing files", len(report.get("missing_images", [])))
        with c5:
            st.metric("Bad boxes", len(report.get("invalid_boxes", [])) + len(report.get("zero_area", [])))
        with c6:
            st.metric("Issues", report.get("issues", 0))
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Label files", report.get("files", 0))
        with c2:
            st.metric("Boxes", report.get("boxes", 0))
        with c3:
            st.metric("Invalid lines", len(report.get("invalid_lines", [])))
        with c4:
            st.metric("Issues", report.get("issues", 0))

    if report.get("issues", 0):
        st.warning(f"Found {report['issues']} annotation issue(s).")
        rows = _issue_rows(report)
        if rows:
            st.dataframe(rows[:100], use_container_width=True, hide_index=True, height=220)
    else:
        st.success("No annotation issues detected.")

    action_cols = st.columns([1, 1, 1, 3])
    with action_cols[0]:
        if st.button("Fix in place", type="primary", key="fix_apply_btn", use_container_width=True):
            try:
                if fmt == "COCO":
                    result = fix_coco(ann_path, img_dir, backup=True)
                    st.session_state["class_names"] = "\n".join(
                        c["name"] for c in json.load(open(ann_path, encoding="utf-8"))["categories"]
                    )
                    st.session_state["num_classes"] = len(
                        json.load(open(ann_path, encoding="utf-8"))["categories"]
                    )
                else:
                    result = fix_yolo(
                        labels_dir,
                        int(st.session_state.get("num_classes", 0) or 0) or None,
                        backup=True,
                    )
                st.success(f"Annotations repaired: {result}")
                st.session_state.pop(report_key, None)
                st.session_state.pop("_last_verify_ok", None)
                st.session_state["_auto_verify"] = True
                st.session_state["_data_tab_next"] = "Verify"
                st.rerun()
            except Exception as exc:
                st.error(f"Fix failed: {exc}")

    with action_cols[1]:
        if fmt == "COCO":
            if st.button("Prepare download", key="fix_download_btn", use_container_width=True):
                try:
                    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
                    tmp.close()
                    fix_coco(ann_path, img_dir, output_path=tmp.name, backup=False)
                    with open(tmp.name, "r", encoding="utf-8") as f:
                        st.session_state["_fixed_coco_download"] = f.read()
                    st.session_state["_fixed_coco_name"] = f"{split.lower()}_annotations_fixed.json"
                except Exception as exc:
                    st.error(f"Could not prepare download: {exc}")
        else:
            st.caption("YOLO fixes are applied in place.")

    with action_cols[2]:
        if st.button("Reload classes", key="fix_reload_classes", use_container_width=True):
            if fmt == "COCO":
                cats = json.load(open(ann_path, encoding="utf-8")).get("categories", [])
                names = [c.get("name", f"class_{c.get('id')}") for c in sorted(cats, key=lambda c: c.get("id", 0))]
            else:
                names = st.session_state.get("class_names", "").splitlines()
            st.session_state["class_names"] = "\n".join(names)
            st.session_state["num_classes"] = len(names)
            st.success(f"Loaded {len(names)} classes")
            st.rerun()

    data = st.session_state.get("_fixed_coco_download")
    if fmt == "COCO" and data:
        st.download_button(
            "Download corrected COCO",
            data,
            file_name=st.session_state.get("_fixed_coco_name", "annotations_fixed.json"),
            mime="application/json",
            use_container_width=False,
        )
