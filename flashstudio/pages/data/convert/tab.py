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
        def _looks_like_root(candidate):
            return os.path.isdir(candidate) and any(
                os.path.isdir(os.path.join(candidate, child))
                for child in (
                    "train", "valid", "val", "annotations", "images",
                    "train2017", "val2017",
                )
            )

        for candidate in (
            os.path.dirname(os.path.dirname(train)),
            os.path.dirname(train),
        ):
            if _looks_like_root(candidate):
                return candidate
    return ""


def _detect(source, hint):
    from flashstudio.utils.universal_convert import detect_extra_format

    if not source or not os.path.isdir(source):
        return ""
    if hint != "Auto-detect":
        return _FORMATS.get(hint, "")
    from flashstudio.pages.data.helpers import _find_root

    root = _find_root(source) or source
    extra = detect_extra_format(root)
    if extra:
        return extra
    from flashstudio.utils.dataset_paths import resolve_annotation_file

    for candidate in (
        root,
        os.path.join(root, "train"),
        os.path.join(root, "train2017"),
        os.path.join(root, "images"),
    ):
        if candidate and resolve_annotation_file(
            candidate, split="train", root=root,
        ):
            return "coco"
    if any(
        name.endswith(".coco.json") for name in os.listdir(root)
    ):
        return "coco"
    try:
        from flashdet.data import detect_dataset_format

        detected = detect_dataset_format(root)
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

    if "_convert_source" not in st.session_state:
        st.session_state["_convert_source"] = _guess_source()
    source = st.text_input(
        "Dataset folder",
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

    source_name = os.path.basename(os.path.normpath(source or "dataset")).replace(" ", "_")
    output = os.path.join(
        _get_data_root(), "uploaded", f"train_{source_name}_coco"
    )
    st.caption(f"Output: `{output}`")

    if st.button(
        "Convert any format → COCO", type="primary",
        key="convert_any_btn", use_container_width=True,
        disabled=not source or not os.path.isdir(str(source)),
    ):
        target = _FORMATS.get(fmt_label, "") or detected
        if not target:
            st.error("Choose the source format or point at a convertible folder.")
        elif target == "coco":
            from flashstudio.pages.data.helpers import _use_existing

            _use_existing(
                source,
                {"name": os.path.basename(os.path.normpath(source))},
            )
        else:
            # Convert tab: let the source dataset define its own classes instead
            # of reusing a previous dataset's class list.
            st.session_state.pop("class_names", None)
            st.session_state.pop("num_classes", None)
            _convert_to_coco(source, "train", target)
