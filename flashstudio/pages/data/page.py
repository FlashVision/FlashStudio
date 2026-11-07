"""FlashStudio — Data Page (ultra-compact, no-scroll)."""

import os
import streamlit as st

from flashstudio.constants import IMG_EXTENSIONS
from flashstudio.pages.data.upload.tab import _render_upload
from flashstudio.pages.data.download.tab import _render_download
from flashstudio.pages.data.preview.tab import _render_preview
from flashstudio.pages.data.verify.tab import _render_verify
from flashstudio.pages.data.fix.tab import _render_fix
from flashstudio.pages.data.edit.tab import _render_edit
from flashstudio.pages.data.quality.tab import _render_quality


def render_data_page():
    from flashstudio.components.styles import render_page_header
    from flashstudio.utils import show_flashes
    render_page_header("", "Data", "Upload, repair, edit, and validate datasets")
    show_flashes()

    # Status banner
    dataset = st.session_state.get("dataset_name")
    train_path = st.session_state.get("train_img_path", "")
    val_path = st.session_state.get("val_img_path", "")
    has_paths = bool(
        (train_path and os.path.isdir(train_path))
        or (val_path and os.path.isdir(val_path))
    )
    if dataset and has_paths:
        tc = len([f for f in os.listdir(train_path) if f.lower().endswith(IMG_EXTENSIONS)]) if train_path and os.path.isdir(train_path) else 0
        vc = len([f for f in os.listdir(val_path) if f.lower().endswith(IMG_EXTENSIONS)]) if val_path and os.path.isdir(val_path) else 0
        nc = st.session_state.get("num_classes", 0)
        detail = f"{tc} train · {vc} val"
        if nc:
            detail += f" · {nc} classes"
        dc1, dc2 = st.columns([8, 1])
        with dc1:
            st.success(f"**{dataset}** — {detail}")
        with dc2:
            if st.button("Clear", key="clear_dataset_btn", use_container_width=True):
                from flashstudio.utils import flash
                for k in ["dataset_name", "train_img_path", "val_img_path",
                           "train_label_path", "val_label_path",
                           "dataset_output_path", "dataset_id", "dataset_classes",
                           "class_names", "num_classes",
                           "detected_format", "_last_verify_ok"]:
                    st.session_state.pop(k, None)
                flash("Dataset cleared", "info")
                st.rerun()
        st.caption(f"Train: `{train_path or '—'}` · Validation: `{val_path or '—'}`")
    elif dataset:
        dc1, dc2 = st.columns([8, 1])
        with dc1:
            st.info(
                f"**{dataset}** is named but no image folders are configured. "
                "Add a train or validation folder below."
            )
        with dc2:
            if st.button("Clear", key="clear_dataset_btn", use_container_width=True):
                from flashstudio.utils import flash
                for k in ["dataset_name", "train_img_path", "val_img_path",
                           "train_label_path", "val_label_path",
                           "dataset_output_path", "dataset_id", "dataset_classes",
                           "class_names", "num_classes",
                           "detected_format", "_last_verify_ok"]:
                    st.session_state.pop(k, None)
                flash("Dataset cleared", "info")
                st.rerun()

    tab_upload, tab_download, tab_preview, tab_verify, tab_edit, tab_fix, tab_quality = st.tabs(
        ["Upload", "Download", "Preview", "Verify", "Edit", "Fix", "Quality"]
    )

    with tab_upload:
        st.markdown("#### Upload dataset")
        _render_upload()

    with tab_download:
        with st.container(border=True):
            st.markdown("#### Download dataset")
            _render_download()

    with tab_preview:
        with st.container(border=True):
            st.markdown("#### Preview images")
            _render_preview()

    with tab_verify:
        with st.container(border=True):
            st.markdown("#### Verify dataset")
            _render_verify()

    with tab_edit:
        with st.container(border=True):
            st.markdown("#### Edit annotations")
            _render_edit()

    with tab_fix:
        with st.container(border=True):
            st.markdown("#### Repair annotations")
            _render_fix()

    with tab_quality:
        with st.container(border=True):
            st.markdown("#### Quality report")
            _render_quality()
