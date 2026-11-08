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
    """Keep a text input widget in sync with a programmatically changed path.

    A widget can hold an empty value left over from before a dataset was
    loaded while the session state already points at a directory.  Treat that
    as stale and push the real path back into the widget instead of letting
    the empty value overwrite the dataset below.
    """
    value = st.session_state.get(state_key, "")
    marker_key = f"{widget_key}_synced"
    widget_value = st.session_state.get(widget_key)
    stale_empty = bool(value) and not str(widget_value or "").strip()
    if st.session_state.get(marker_key) != value or stale_empty:
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
            current_train = st.session_state.get("train_img_path", "")
            if manual_train != current_train and (manual_train.strip() or not current_train):
                normalized = _image_dir(manual_train) or manual_train
                st.session_state["train_img_path"] = normalized
                st.session_state["_train_img_input_synced"] = manual_train
                for key in ("dataset_output_path", "train_ann_path",
                            "_last_verify_ok", "detected_format",
                            "_quality_report_train", "_fix_report_train",
                            "class_names", "num_classes", "_class_names_synced"):
                    st.session_state.pop(key, None)
                if normalized and os.path.isdir(normalized):
                    # Re-render with the dataset loaded, but stay on Load data
                    # so the user can finish setting up train + validation
                    # before opening Preview.
                    st.rerun()

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
            current_val = st.session_state.get("val_img_path", "")
            if manual_val != current_val and (manual_val.strip() or not current_val):
                normalized = _image_dir(manual_val) or manual_val
                st.session_state["val_img_path"] = normalized
                st.session_state["_val_img_input_synced"] = manual_val
                for key in ("dataset_output_path", "val_ann_path",
                            "_last_verify_ok", "detected_format",
                            "_quality_report_val", "_fix_report_val"):
                    st.session_state.pop(key, None)
                if normalized and os.path.isdir(normalized):
                    st.rerun()

    # Format selector + conversion share one row — no separate box.
    _render_conversion_hint()

    # ── Classes section ──
    _render_class_config()


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
            st.caption(f"**{nc}** classes" if nc else "—")

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
                use_container_width=False,
            )


def _render_conversion_hint():
    """One row: annotation-format selector plus a convert action when needed."""
    c1, c2, c3 = st.columns([1.6, 1.3, 3.1], gap="small")
    with c1:
        st.selectbox(
            "Annotation format",
            [
                "COCO JSON", "Pascal VOC", "YOLO TXT", "KITTI TXT",
                "LabelMe JSON", "CSV bboxes", "Supervisely",
            ],
            key="ann_format",
        )

    train_path = st.session_state.get("train_img_path", "")
    if not train_path or not os.path.isdir(train_path):
        return
    ann_format = st.session_state.get("ann_format", "COCO JSON")
    detected = st.session_state.get("detected_format", "")
    extra_formats = {
        "KITTI": "kitti", "LabelMe": "labelme",
        "CSV": "csv", "Supervisely": "supervisely",
    }
    needs = (
        "YOLO" in ann_format
        or "VOC" in ann_format
        or any(key in ann_format for key in extra_formats)
        or detected in ("txt", "voc", "kitti", "labelme", "csv", "supervisely")
    )
    if not needs:
        coco_ann = os.path.join(train_path, "_annotations.coco.json")
        if not os.path.isfile(coco_ann):
            from flashstudio.utils.dataset_paths import split_dir

            parent = os.path.dirname(split_dir(train_path))
            candidates = [
                train_path,
                os.path.dirname(train_path),
                os.path.dirname(os.path.dirname(train_path)),
                parent,
            ]
            from flashstudio.utils.universal_convert import detect_extra_format

            extra = ""
            for candidate in candidates:
                if candidate and os.path.isdir(candidate):
                    extra = detect_extra_format(candidate)
                    if extra:
                        break
            if extra:
                needs = True
                detected = extra
                st.session_state["detected_format"] = extra
            elif _HAS_FLASHDET:
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
    if not needs:
        return
    with c2:
        if st.button("Convert to COCO", type="primary", key="manual_convert",
                     use_container_width=True):
            from flashstudio.utils.dataset_paths import split_dir

            source = os.path.dirname(split_dir(train_path))
            if not os.path.isdir(source):
                source = split_dir(train_path)
            if "KITTI" in ann_format or detected == "kitti":
                tfmt = "kitti"
            elif "LabelMe" in ann_format or detected == "labelme":
                tfmt = "labelme"
            elif "CSV" in ann_format or detected == "csv":
                tfmt = "csv"
            elif "Supervisely" in ann_format or detected == "supervisely":
                tfmt = "supervisely"
            elif "YOLO" in ann_format or detected == "txt":
                tfmt = "txt"
            else:
                tfmt = "voc"
            _convert_to_coco(source, "train", tfmt)
    with c3:
        st.caption(
            f"Detected **{detected or ann_format}** → auto-convert to COCO JSON"
        )
