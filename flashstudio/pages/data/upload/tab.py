"""Upload tab — file upload, class configuration, and format conversion hint."""

import os
import streamlit as st

from flashstudio.pages.data._common import (
    _HAS_FLASHDET, _auto_detect_classes,
)
from flashstudio.pages.data.helpers import _extract_upload, _convert_to_coco

try:
    from flashdet.data import detect_dataset_format
except ImportError:
    detect_dataset_format = None


def _render_upload():
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Train")
            train_upload = st.file_uploader("ZIP/TAR", type=["zip", "tar", "gz"], key="train_data_upload")
            manual_train = st.text_input(
                "Images dir", placeholder="/path/to/train/",
                value=st.session_state.get("train_img_path", ""),
                key="_train_img_input",
            )
            if manual_train:
                st.session_state["train_img_path"] = manual_train

    with col2:
        with st.container(border=True):
            st.markdown("#### Val")
            val_upload = st.file_uploader("ZIP/TAR", type=["zip", "tar", "gz"], key="val_data_upload")
            manual_val = st.text_input(
                "Images dir", placeholder="/path/to/valid/",
                value=st.session_state.get("val_img_path", ""),
                key="_val_img_input",
            )
            if manual_val:
                st.session_state["val_img_path"] = manual_val

    fc1, fc2 = st.columns([2, 2])
    with fc1:
        st.selectbox("Format", ["COCO JSON", "Pascal VOC", "YOLO TXT"], key="ann_format")
    with fc2:
        if train_upload:
            st.session_state["dataset_name"] = train_upload.name
            if not st.session_state.get("train_img_path"):
                if st.button("Extract", key="extract_train", use_container_width=True, type="primary"):
                    _extract_upload(train_upload, "train")
        if val_upload and not st.session_state.get("val_img_path"):
            if st.button("Extract Val", key="extract_val", use_container_width=True):
                _extract_upload(val_upload, "val")

    # ── Classes section ──
    _render_class_config()

    _render_conversion_hint()


def _render_class_config():
    """Render class configuration: auto-detect from annotation, load from .txt, or manual edit."""
    from flashstudio.utils import flash

    with st.container(border=True):
        st.markdown("#### Classes")

        # Auto-detect button + .txt upload in one row
        cc1, cc2, cc3 = st.columns([2, 2, 1])
        with cc1:
            if st.button("Auto-detect from annotation", key="btn_auto_classes",
                         use_container_width=True, type="primary"):
                detected = _auto_detect_classes()
                if detected:
                    st.session_state["class_names"] = "\n".join(detected)
                    st.session_state["num_classes"] = len(detected)
                    flash(f"Detected {len(detected)} classes from annotation", "success")
                    st.rerun()
                else:
                    flash("No classes found — load a dataset with COCO annotations first", "warning")
                    st.rerun()

        with cc2:
            cls_file = st.file_uploader("Load classes.txt", type=["txt"],
                                        key="class_txt_upload", label_visibility="collapsed")
            if cls_file:
                content = cls_file.read().decode("utf-8", errors="ignore")
                names = [line.strip() for line in content.strip().split("\n") if line.strip()]
                if names:
                    st.session_state["class_names"] = "\n".join(names)
                    st.session_state["num_classes"] = len(names)
                    flash(f"Loaded {len(names)} classes from file", "success")
                    st.rerun()

        with cc3:
            nc = st.session_state.get("num_classes", 0)
            st.metric("Count", nc if nc else "—")

        # Auto-detect on first load if classes not set
        current_classes = st.session_state.get("class_names", "")
        if isinstance(current_classes, list):
            current_classes = "\n".join(current_classes)
        if not current_classes.strip() and st.session_state.get("train_img_path"):
            detected = _auto_detect_classes()
            if detected:
                st.session_state["class_names"] = "\n".join(detected)
                st.session_state["num_classes"] = len(detected)

        # Editable text area
        _cv = st.session_state.get("class_names", "")
        if isinstance(_cv, list):
            _cv = "\n".join(_cv)
        class_text = st.text_area(
            "Class names (one per line)",
            value=_cv,
            height=100,
            key="_class_names_input",
            placeholder="person\ncar\nbicycle\n...",
        )
        if class_text != _cv:
            names = [n.strip() for n in class_text.strip().split("\n") if n.strip()]
            st.session_state["class_names"] = "\n".join(names)
            st.session_state["num_classes"] = len(names)

        # Download button for current classes
        if current_classes.strip():
            st.download_button(
                "Download classes.txt",
                current_classes.strip() + "\n",
                file_name="classes.txt",
                mime="text/plain",
                use_container_width=True,
            )


def _render_conversion_hint():
    train_path = st.session_state.get("train_img_path", "")
    if not train_path or not os.path.isdir(train_path):
        return
    ann_format = st.session_state.get("ann_format", "COCO JSON")
    detected = st.session_state.get("detected_format", "")
    needs = "YOLO" in ann_format or "VOC" in ann_format or detected in ("txt", "voc")
    if not needs:
        coco_ann = os.path.join(train_path, "_annotations.coco.json")
        if not os.path.isfile(coco_ann) and _HAS_FLASHDET:
            parent = os.path.dirname(train_path)
            det_fmt = detect_dataset_format(parent)
            if det_fmt in ("txt", "voc"):
                needs = True
                st.session_state["detected_format"] = det_fmt
    if needs:
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Convert to COCO", type="primary", key="manual_convert", use_container_width=True):
                source = os.path.dirname(train_path) if os.path.basename(train_path) in ("images", "train") else train_path
                tfmt = "txt" if "YOLO" in ann_format or detected == "txt" else "voc"
                _convert_to_coco(source, "train", tfmt)
        with c2:
            st.caption(f"Format detected: **{detected or ann_format}** → auto-convert to COCO JSON")
