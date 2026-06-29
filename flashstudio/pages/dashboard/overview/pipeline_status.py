"""Dashboard — pipeline status section."""

import streamlit as st
from flashstudio.pages.dashboard._common import (
    COLOR_SUCCESS, COLOR_TEXT_PRIMARY, COLOR_TEXT_MUTED, COLOR_BORDER,
)


def render_pipeline_status():
    from flashstudio.utils import get_state
    ds = st.session_state.get("dataset_name")
    model = get_state("model_arch")
    train_status = st.session_state.get("training_status", "")
    exported = st.session_state.get("exported_files")

    steps = [
        ("Data", ds is not None, ds or "Not loaded"),
        ("Model", model is not None, model.replace("FlashDet-", "") if model else "Not set"),
        ("Training", train_status in ("Complete", "Running"), train_status or "Not started"),
        ("Export", bool(exported), "Ready" if exported else "Pending"),
    ]

    for label, done, detail in steps:
        color = COLOR_SUCCESS if done else "#D1D5DB"
        text_color = COLOR_TEXT_PRIMARY if done else COLOR_TEXT_MUTED
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.35rem 0;'
            f'border-bottom:1px solid {COLOR_BORDER};">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:{color};flex-shrink:0;"></div>'
            f'<div style="flex:1;">'
            f'<span style="font-size:0.85rem;font-weight:600;color:{text_color};">{label}</span>'
            f'<span style="font-size:0.78rem;color:{COLOR_TEXT_MUTED};margin-left:0.5rem;">{detail}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
