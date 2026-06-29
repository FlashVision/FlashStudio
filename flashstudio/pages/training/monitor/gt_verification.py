"""Ground truth verification display."""

import os
import re
import json
import streamlit as st

from flashstudio.pages.training.monitor.parsers import _find_log_file
from flashstudio.pages.training.monitor.visualizations import _render_image_grid


def _render_gt_verification(run_dir):
    """GT verification — from dedicated directory, or extracted from log file."""
    from flashstudio.constants import GT_VERIFICATION_DIR, GT_REPORT_FILE, GT_SUMMARY_FILE
    gt_dir = os.path.join(run_dir, GT_VERIFICATION_DIR)

    if os.path.isdir(gt_dir):
        report_path = os.path.join(gt_dir, GT_REPORT_FILE)
        summary_path = os.path.join(gt_dir, GT_SUMMARY_FILE)

        if os.path.isfile(report_path):
            with open(report_path) as f:
                report = json.load(f)
            st.success("Verification: PASSED") if report.get("passed") else st.error("Verification: FAILED")

            tc = report.get("splits", {}).get("train", {}).get("coco", {})
            vc = report.get("splits", {}).get("val", {}).get("coco", {})
            td = report.get("splits", {}).get("train", {}).get("dataloader", {})

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                st.metric("Train Imgs", tc.get("num_images", 0))
            with m2:
                st.metric("Train Ann", tc.get("num_annotations", 0))
            with m3:
                st.metric("Val Imgs", vc.get("num_images", 0))
            with m4:
                st.metric("Val Ann", vc.get("num_annotations", 0))
            with m5:
                st.metric("Classes", report.get("num_classes", "?"))
            with m6:
                st.metric("Avg Boxes/Img", f"{td.get('avg_boxes_per_sample', 0):.1f}")

        if os.path.isfile(summary_path):
            with st.expander("Verification Summary", expanded=False):
                with open(summary_path) as f:
                    st.code(f.read(), language="text")

        raw_dir = os.path.join(gt_dir, "images", "raw")
        dl_dir = os.path.join(gt_dir, "images", "dataloader")
        gt_tab_raw, gt_tab_dl = st.tabs(["Raw GT Images", "Dataloader GT Images"])
        with gt_tab_raw:
            _render_image_grid(raw_dir, "Raw ground truth (before augmentations)")
        with gt_tab_dl:
            _render_image_grid(dl_dir, "After dataloader transforms & augmentations")
        return

    # No dedicated directory — extract verification info from the log file
    log_file = _find_log_file(run_dir)
    if not log_file:
        st.info("No GT verification data yet. Start training to generate.")
        return

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        log_content = f.read()

    # Extract dataset verification block from log
    ver_start = log_content.find("Dataset Verification")
    if ver_start == -1:
        st.info("No dataset verification found in log.")
        return

    ver_end = log_content.find("Starting training", ver_start)
    if ver_end == -1:
        ver_end = ver_start + 800
    ver_block = log_content[ver_start:ver_end].strip()

    # Parse verification info
    train_imgs = re.search(r"train.*?Images:\s*(\d+)", ver_block, re.DOTALL)
    train_ann = re.search(r"train.*?Annotations:\s*(\d+)", ver_block, re.DOTALL)
    train_found = re.search(r"train.*?Files found:\s*(\d+)/(\d+)", ver_block, re.DOTALL)
    val_imgs = re.search(r"(?:valid|val).*?Images:\s*(\d+)", ver_block, re.DOTALL)
    val_ann = re.search(r"(?:valid|val).*?Annotations:\s*(\d+)", ver_block, re.DOTALL)

    # Extract classes from header
    cls_m = re.search(r"Classes \((\d+)\): \[(.+?)\]", log_content)
    num_cls = int(cls_m.group(1)) if cls_m else "?"
    class_names = [c.strip().strip("'") for c in cls_m.group(2).split(",")] if cls_m else []

    # Check if all passed
    train_ok = "✓ train" in ver_block
    val_ok = "✓ valid" in ver_block or "✓ val" in ver_block
    if train_ok and val_ok:
        st.success("Dataset Verification: PASSED")
    elif train_ok:
        st.warning("Dataset Verification: Train OK, Val missing")
    else:
        st.error("Dataset Verification: FAILED")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Train Imgs", train_imgs.group(1) if train_imgs else "—")
    with m2:
        st.metric("Train Ann", train_ann.group(1) if train_ann else "—")
    with m3:
        st.metric("Val Imgs", val_imgs.group(1) if val_imgs else "—")
    with m4:
        st.metric("Val Ann", val_ann.group(1) if val_ann else "—")
    with m5:
        st.metric("Classes", num_cls)
    with m6:
        if train_found:
            st.metric("Files OK", f"{train_found.group(1)}/{train_found.group(2)}")
        else:
            st.metric("Files OK", "—")

    if class_names:
        with st.expander(f"Class Names ({len(class_names)})", expanded=False):
            st.code("\n".join(class_names), language="text")

    with st.expander("Verification Log", expanded=False):
        st.code(ver_block, language="text")
