"""Upload tab — file upload, class configuration, and format conversion hint."""

import os
import streamlit as st

from flashstudio.pages.data._common import (
    _HAS_FLASHDET, _auto_detect_classes,
)
from flashstudio.pages.data.helpers import (
    _convert_to_coco, _extract_upload, _find_labels, _image_dir,
)

try:
    from flashdet.data import detect_dataset_format
except ImportError:
    detect_dataset_format = None


def _sync_path_input(widget_key: str, state_key: str) -> None:
    """Keep a text input widget in sync with a programmatically changed path."""
    value = st.session_state.get(state_key, "")
    marker_key = f"{widget_key}_synced"
    if st.session_state.get(marker_key) != value:
        st.session_state[widget_key] = value
        st.session_state[marker_key] = value


def _render_upload(show_uploaders: bool = True):
    pending_format = st.session_state.pop("_pending_ann_format", None)
    if pending_format:
        st.session_state["ann_format"] = pending_format
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Train")
            if show_uploaders:
                train_upload = st.file_uploader(
                    "ZIP/TAR", type=["zip", "tar", "gz", "tgz"],
                    key="train_data_upload",
                )
                if train_upload:
                    if st.button("Extract", key="extract_train", use_container_width=True, type="primary"):
                        _extract_upload(train_upload, "train")
            _sync_path_input("_train_img_input", "train_img_path")
            manual_train = st.text_input(
                "Images folder", placeholder="/path/to/train/",
                key="_train_img_input",
            )
            if manual_train != st.session_state.get("train_img_path", ""):
                normalized = _image_dir(manual_train) or manual_train
                st.session_state["train_img_path"] = normalized
                st.session_state["_train_img_input_synced"] = manual_train
                if normalized and os.path.isdir(normalized):
                    st.session_state["_data_tab_next"] = "Preview"
                for key in ("dataset_output_path", "train_ann_path",
                            "_last_verify_ok", "detected_format",
                            "_quality_report_train", "_fix_report_train",
                            "class_names", "num_classes", "_class_names_synced"):
                    st.session_state.pop(key, None)

    with col2:
        with st.container(border=True):
            st.markdown("#### Validation")
            if show_uploaders:
                val_upload = st.file_uploader(
                    "ZIP/TAR", type=["zip", "tar", "gz", "tgz"],
                    key="val_data_upload",
                )
                if val_upload:
                    if st.button("Extract", key="extract_val", use_container_width=True, type="primary"):
                        _extract_upload(val_upload, "val")
            _sync_path_input("_val_img_input", "val_img_path")
            manual_val = st.text_input(
                "Images folder", placeholder="/path/to/valid/",
                key="_val_img_input",
            )
            if manual_val != st.session_state.get("val_img_path", ""):
                normalized = _image_dir(manual_val) or manual_val
                st.session_state["val_img_path"] = normalized
                st.session_state["_val_img_input_synced"] = manual_val
                if normalized and os.path.isdir(normalized):
                    st.session_state["_data_tab_next"] = "Preview"
                for key in ("dataset_output_path", "val_ann_path",
                            "_last_verify_ok", "detected_format",
                            "_quality_report_val", "_fix_report_val"):
                    st.session_state.pop(key, None)

    with st.container(border=True):
        st.markdown("#### Dataset format")
        st.selectbox("Annotation format", ["COCO JSON", "Pascal VOC", "YOLO TXT"],
                     key="ann_format")

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
            if st.button("Detect from annotations", key="btn_auto_classes",
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
                content = cls_file.read().decode("utf-8-sig", errors="ignore")
                names = [line.strip() for line in content.strip().split("\n") if line.strip()]
                if names:
                    st.session_state["class_names"] = "\n".join(names)
                    st.session_state["num_classes"] = len(names)
                    flash(f"Loaded {len(names)} classes from file", "success")
                    st.rerun()

        with cc3:
            nc = st.session_state.get("num_classes", 0)
            st.metric("Class count", nc if nc else "—")

        # Auto-detect on first load if classes not set
        current_classes = st.session_state.get("class_names", "")
        if isinstance(current_classes, list):
            current_classes = "\n".join(current_classes)
        if not current_classes.strip() and st.session_state.get("train_img_path"):
            detected = _auto_detect_classes()
            if detected:
                st.session_state["class_names"] = "\n".join(detected)
                st.session_state["num_classes"] = len(detected)

        # Editable text area. Keep the widget key in sync when class names are
        # set programmatically (dataset download, annotation repair, etc.).
        _cv = st.session_state.get("class_names", "")
        if isinstance(_cv, list):
            _cv = "\n".join(_cv)
        if st.session_state.get("_class_names_synced") != _cv:
            st.session_state["_class_names_input"] = _cv
            st.session_state["_class_names_synced"] = _cv
        class_text = st.text_area(
            "Class names (one per line)",
            height=100,
            key="_class_names_input",
            placeholder="person\ncar\nbicycle\n...",
        )
        if class_text != _cv:
            names = [n.strip() for n in class_text.strip().split("\n") if n.strip()]
            new_value = "\n".join(names)
            st.session_state["class_names"] = new_value
            st.session_state["num_classes"] = len(names)
            st.session_state["_class_names_synced"] = new_value

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
            from flashstudio.utils.dataset_paths import split_dir

            parent = os.path.dirname(split_dir(train_path))
            det_fmt = detect_dataset_format(parent)
            if det_fmt in ("txt", "voc"):
                needs = True
                st.session_state["detected_format"] = det_fmt
                detected = det_fmt
            elif (
                _find_labels(train_path)
                or _find_labels(parent)
                or _find_labels(os.path.dirname(parent))
            ):
                # Non-standard YOLO layout (images/train2017 + labels/...).
                needs = True
                detected = "txt"
                st.session_state["detected_format"] = "txt"
    if needs:
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Convert to COCO", type="primary", key="manual_convert", use_container_width=True):
                from flashstudio.utils.dataset_paths import split_dir

                source = os.path.dirname(split_dir(train_path))
                if not os.path.isdir(source):
                    source = split_dir(train_path)
                tfmt = "txt" if "YOLO" in ann_format or detected == "txt" else "voc"
                _convert_to_coco(source, "train", tfmt)
        with c2:
            st.caption(f"Format detected: **{detected or ann_format}** → auto-convert to COCO JSON")
