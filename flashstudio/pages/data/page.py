"""FlashStudio — Data Page (ultra-compact, no-scroll)."""

import os
import streamlit as st

from flashstudio.constants import IMG_EXTENSIONS
from flashstudio.pages.data.upload.tab import _render_upload
from flashstudio.pages.data.download.tab import _render_download
from flashstudio.pages.data.preview.tab import _render_preview
from flashstudio.pages.data.verify.tab import _render_verify


def render_data_page():
    from flashstudio.components.styles import render_page_header
    from flashstudio.utils import show_flashes
    render_page_header("", "Data")
    show_flashes()

    # Status banner
    dataset = st.session_state.get("dataset_name")
    if dataset:
        train_path = st.session_state.get("train_img_path", "")
        val_path = st.session_state.get("val_img_path", "")
        tc = len([f for f in os.listdir(train_path) if f.lower().endswith(IMG_EXTENSIONS)]) if train_path and os.path.isdir(train_path) else 0
        vc = len([f for f in os.listdir(val_path) if f.lower().endswith(IMG_EXTENSIONS)]) if val_path and os.path.isdir(val_path) else 0
        dc1, dc2 = st.columns([8, 1])
        with dc1:
            st.success(f"**{dataset}** — {tc} train / {vc} val images")
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

    tab_upload, tab_download, tab_preview, tab_verify = st.tabs(["Upload", "Download", "Preview", "Verify"])

    with tab_upload:
        _render_upload()
    with tab_download:
        _render_download()
    with tab_preview:
        _render_preview()
    with tab_verify:
        _render_verify()
