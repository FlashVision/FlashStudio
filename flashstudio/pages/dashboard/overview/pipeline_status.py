"""Dashboard — pipeline status."""

import html
import os

import streamlit as st

from flashstudio.pages.dashboard._common import model_display_name


def render_pipeline_status():
    from flashstudio.utils import get_state

    train_path = st.session_state.get("train_img_path")
    dataset_ready = bool(train_path and os.path.isdir(str(train_path)))
    dataset = st.session_state.get("dataset_name")
    model = model_display_name()
    train_status = st.session_state.get("training_status", "")
    exported = st.session_state.get("exported_files")

    steps = [
        ("Data", dataset_ready, dataset or "Not loaded"),
        ("Model", bool(get_state("model_framework")), model or "Not set"),
        ("Training", train_status in ("Complete", "Running"), train_status or "Not started"),
        ("Export", bool(exported), "Ready" if exported else "Pending"),
    ]
    done = sum(1 for _label, ok, _detail in steps if ok)
    st.caption(f"{done} of {len(steps)} stages ready")

    for label, ok, detail in steps:
        color = "#059669" if ok else "#D1D5DB"
        detail_html = html.escape(str(detail))
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:.6rem;padding:.4rem 0;'
            f'border-bottom:1px solid var(--fs-border-soft);">'
            f'<div style="width:9px;height:9px;border-radius:50%;background:{color};'
            f'flex-shrink:0;"></div>'
            f'<div style="flex:1;">'
            f'<span style="font-size:.84rem;font-weight:700;color:var(--fs-text);">{label}</span>'
            f'<span style="font-size:.75rem;color:var(--fs-muted);margin-left:.45rem;">'
            f'{detail_html}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
