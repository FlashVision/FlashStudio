"""Convert tab — normalize any supported annotation format to COCO."""

import os

import streamlit as st

from flashstudio.pages.data._common import _get_data_root
from flashstudio.pages.data.helpers import _convert_to_coco


_FORMATS = {
    "COCO JSON": "coco",
    "Pascal VOC": "voc",
    "YOLO TXT": "txt",
    "KITTI TXT": "kitti",
    "LabelMe JSON": "labelme",
    "CSV bboxes": "csv",
    "Supervisely": "supervisely",
}


def _guess_source():
    root = st.session_state.get("dataset_output_path") or ""
    if root and os.path.isdir(root):
        return root
    train = st.session_state.get("train_img_path") or ""
    if train:
        for candidate in (
            os.path.dirname(train),
            os.path.dirname(os.path.dirname(train)),
        ):
            if candidate and os.path.isdir(candidate):
                return candidate
    return ""


def _detect(source, hint):
    from flashstudio.utils.universal_convert import detect_extra_format

    if not source or not os.path.isdir(source):
        return ""
    if hint != "Auto-detect":
        return _FORMATS.get(hint, "")
    extra = detect_extra_format(source)
    if extra:
        return extra
    if os.path.isfile(os.path.join(source, "train", "images",
                                   "_annotations.coco.json")) or any(
        name.endswith(".coco.json") for name in os.listdir(source)
    ):
        return "coco"
    try:
        from flashdet.data import detect_dataset_format

        detected = detect_dataset_format(source)
        if detected:
            return str(detected).lower()
    except Exception:
        pass
    return ""


def _render_convert():
    st.caption(
        "Convert any supported annotation format to **COCO JSON** so it can be "
        "trained, verified, and previewed in FlashStudio."
    )

    source = st.text_input(
        "Dataset folder",
        value=st.session_state.get("_convert_source", _guess_source()),
        key="_convert_source",
        placeholder="/path/to/dataset",
    )
    fmt_label = st.selectbox(
        "Source format",
        ["Auto-detect", *_FORMATS.keys()],
        key="_convert_format",
    )

    detected = _detect(source, fmt_label)
    if detected:
        st.success(f"Detected format: **{detected.upper()}**")
    elif source:
        st.info("Format not detected. Pick the source format manually.")

    output = os.path.join(
        _get_data_root(), "uploaded",
        f"converted_{os.path.basename(os.path.normpath(source or 'dataset'))}",
    )
    st.caption(f"Output: `{output}`")

    if st.button(
        "Convert any format → COCO", type="primary",
        key="convert_any_btn", use_container_width=True,
        disabled=not source,
    ):
        target = _FORMATS.get(fmt_label, "") or detected
        if not target:
            st.error("Choose the source format or point at a convertible folder.")
        elif target == "coco":
            st.session_state["_data_tab_next"] = "Preview"
            st.success("Dataset is already COCO JSON.")
            st.rerun()
        else:
            _convert_to_coco(source, "train", target)


def _render_convert_help():
    st.caption(
        "Supported: COCO, Pascal VOC, YOLO, KITTI, LabelMe, CSV bounding boxes, "
        "and Supervisely."
    )
