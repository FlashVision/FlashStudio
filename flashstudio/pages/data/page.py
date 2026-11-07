"""FlashStudio — Data Page (ultra-compact, no-scroll)."""

import os
import streamlit as st

from flashstudio.constants import IMG_EXTENSIONS_ALL
from flashstudio.pages.data.upload.tab import _render_upload
from flashstudio.pages.data.download.tab import _render_download
from flashstudio.pages.data.preview.tab import _render_preview
from flashstudio.pages.data.verify.tab import _render_verify
from flashstudio.pages.data.fix.tab import _render_fix
from flashstudio.pages.data.edit.tab import _render_edit
from flashstudio.pages.data.quality.tab import _render_quality
from flashstudio.pages.data.helpers import _render_dataset_download_status


LOAD_TAB = "Load data"

# Upload and Download share one tab, so old callers that asked for either
# tab are redirected to the combined loader (and its matching mode).
_LOAD_ALIASES = {"Upload": "Upload", "Download": "Download"}


def _normalize_tab(name):
    """Map legacy/alias tab names to the current tab labels."""
    if name in _LOAD_ALIASES:
        return LOAD_TAB
    return name


def _data_tabs(labels):
    """Render workflow tabs and allow other sections to request the next tab.

    ``st.tabs`` only gained ``key``/``default`` in newer Streamlit versions;
    fall back to plain tabs so the package still works on older releases.
    """
    import inspect

    next_tab = st.session_state.pop("_data_tab_next", None)
    if next_tab in _LOAD_ALIASES:
        st.session_state["_data_source_mode"] = next_tab
        next_tab = LOAD_TAB

    params = inspect.signature(st.tabs).parameters
    if "default" not in params or "on_change" not in params:
        return st.tabs(labels)

    if next_tab in labels:
        st.session_state["_data_tab"] = next_tab
    selected = _normalize_tab(st.session_state.get("_data_tab"))
    if selected not in labels:
        selected = labels[0]
    st.session_state["_data_tab"] = selected
    return st.tabs(
        labels, key="_data_tab", default=selected, on_change="rerun"
    )


def _count_images(path: str) -> int:
    if not path or not os.path.isdir(path):
        return 0
    return sum(
        1
        for _root, _dirs, files in os.walk(path)
        for name in files
        if name.lower().endswith(IMG_EXTENSIONS_ALL)
    )


def _count_labels(path: str) -> int:
    if not path or not os.path.isdir(path):
        return 0
    return sum(
        1
        for _root, _dirs, files in os.walk(path)
        for name in files
        if name.endswith(".txt")
    )


def _clear_dataset():
    from flashstudio.utils import flash

    for key in [
        "dataset_name", "train_img_path", "val_img_path",
        "train_ann_path", "val_ann_path",
        "train_label_path", "val_label_path",
        "dataset_output_path", "dataset_id", "dataset_classes",
        "class_names", "num_classes", "ann_format",
        "detected_format", "_pending_ann_format",
        "_last_verify_ok", "_quality_report",
        "_fixed_coco_download", "_fixed_coco_name",
    ]:
        st.session_state.pop(key, None)
    for key in list(st.session_state):
        if str(key).startswith(("_fix_report_", "_quality_report_",
                                "_train_img_input", "_val_img_input")):
            st.session_state.pop(key, None)
    st.session_state["_data_tab"] = LOAD_TAB
    st.session_state["_data_source_mode"] = "Upload"
    flash("Dataset cleared", "info")
    st.rerun()


def _dataset_health():
    train = st.session_state.get("train_img_path", "")
    val = st.session_state.get("val_img_path", "")
    from flashstudio.utils import get_class_names_list
    from flashstudio.utils.dataset_paths import (
        resolve_annotation_file, summarize_coco_splits,
    )

    summary = summarize_coco_splits(train, val)
    train_images = _count_images(train)
    val_images = _count_images(val)
    train_ann = (
        resolve_annotation_file(train, split="train",
                                hint=st.session_state.get("train_ann_path", ""))
        if train and os.path.isdir(train) else ""
    )
    val_ann = (
        resolve_annotation_file(val, split="val",
                                hint=st.session_state.get("val_ann_path", ""))
        if val and os.path.isdir(val) else ""
    )
    train_labels = _count_labels(st.session_state.get("train_label_path", ""))
    val_labels = _count_labels(st.session_state.get("val_label_path", ""))
    coco_annotations = summary.get("annotations", 0)
    annotations = coco_annotations or train_labels + val_labels
    annotations_label = "Annotations" if coco_annotations else "Label files"
    classes = summary.get("class_names") or get_class_names_list()

    issues = []
    if not train_images:
        issues.append("No training images")
    if not val_images:
        issues.append("No validation split")
    if train_images and not train_ann and not train_labels:
        issues.append("Missing training annotations")
    if val_images and not val_ann and not val_labels:
        issues.append("Missing validation annotations")
    if st.session_state.get("detected_format") in ("txt", "voc"):
        issues.append("Annotations still need conversion to COCO")
    if not classes:
        issues.append("No classes detected")

    return {
        "dataset": st.session_state.get("dataset_name") or "Loaded dataset",
        "train_images": train_images,
        "val_images": val_images,
        "annotations": annotations,
        "annotations_label": annotations_label,
        "classes": classes,
        "issues": issues,
        "train": train,
        "val": val,
    }


def _render_dataset_health():
    health = _dataset_health()
    with st.container(border=True):
        hc1, hc2 = st.columns([8, 1])
        with hc1:
            st.markdown(f"**{health['dataset']}**")
        with hc2:
            if st.button("Clear", key="clear_dataset_btn", use_container_width=True):
                _clear_dataset()

        stats = [
            f"<b>{health['train_images']:,}</b> train",
            f"<b>{health['val_images']:,}</b> validation",
            f"<b>{health['annotations']:,}</b> "
            f"{health['annotations_label'].lower()}",
            f"<b>{len(health['classes'])}</b> classes",
        ]
        if not health["issues"]:
            stats.append("✓ ready")
        st.markdown(
            '<div class="ds-card-stats">'
            + "".join(f"<span>{item}</span>" for item in stats)
            + "</div>",
            unsafe_allow_html=True,
        )

        if health["issues"]:
            st.warning("Needs attention: " + " · ".join(health["issues"]))

        with st.expander("Dataset paths", expanded=False):
            st.caption(f"Train: `{health['train'] or '—'}`")
            st.caption(f"Validation: `{health['val'] or '—'}`")


def _render_loader_controls():
    """Render the Upload/Download choice and only the selected loader."""
    if (
        "_data_source_mode" not in st.session_state
        and st.session_state.get("_dataset_download")
    ):
        st.session_state["_data_source_mode"] = "Download"
    mode = st.radio(
        "How do you want to load your data?",
        ["Upload", "Download"],
        horizontal=True,
        key="_data_source_mode",
        help=(
            "Pick one. Upload a ZIP/TAR or point at folders already on "
            "disk, or download a dataset from the FlashDet registry. "
            "You can switch later if you change your mind."
        ),
    )
    if mode == "Download":
        with st.container(border=True):
            _render_download()
    else:
        _render_upload()


def render_data_page():
    from flashstudio.components.styles import render_page_header
    from flashstudio.utils import show_flashes
    render_page_header("", "Data")
    show_flashes()

    train_path = st.session_state.get("train_img_path", "")
    val_path = st.session_state.get("val_img_path", "")
    loaded = bool(
        (train_path and os.path.isdir(train_path))
        or (val_path and os.path.isdir(val_path))
    )

    _render_dataset_download_status()

    if loaded:
        _render_dataset_health()
    else:
        st.info("No dataset loaded yet — choose **Upload** or **Download** below.")

    (tab_load, tab_preview, tab_verify,
     tab_fix, tab_edit, tab_quality) = _data_tabs(
        ["Load data", "Preview", "Verify", "Fix", "Edit", "Quality"]
    )

    with tab_load:
        if loaded:
            if st.button(
                "Continue to Preview",
                key="data_continue_preview",
                type="primary",
                use_container_width=False,
            ):
                st.session_state["_data_tab_next"] = "Preview"
                st.rerun()
            with st.expander("Replace dataset", expanded=False):
                _render_loader_controls()
        else:
            _render_loader_controls()

    with tab_preview:
        with st.container(border=True):
            _render_preview()

    with tab_verify:
        with st.container(border=True):
            _render_verify()

    with tab_fix:
        with st.container(border=True):
            _render_fix()

    with tab_edit:
        with st.container(border=True):
            _render_edit()

    with tab_quality:
        with st.container(border=True):
            _render_quality()
