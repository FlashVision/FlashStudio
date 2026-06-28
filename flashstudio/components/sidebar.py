"""FlashStudio sidebar — clean icon-based navigation."""

import streamlit as st
from flashstudio.utils.device import get_gpu_info
from flashstudio.components.project_manager import (
    get_active_project, save_project_state,
)

NAV = [
    ("🏠", "Dashboard"),
    ("📦", "Data"),
    ("🧠", "Model"),
    ("🏋️", "Training"),
    ("📤", "Export"),
    ("🔍", "Inference"),
]


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:0.4rem 0 0.3rem;">'
            '<span style="font-size:1.5rem;">⚡</span>'
            '<b style="font-size:1rem;color:#1A1A2E;margin-left:0.3rem;">FlashStudio</b>'
            '</div>',
            unsafe_allow_html=True,
        )

        project = get_active_project()
        if project:
            if st.button(project['name'][:20], key="sb_project", use_container_width=True):
                save_project_state()
                st.session_state["show_project_manager"] = True
                st.rerun()

        current = st.session_state.get("current_step", 0)
        for i, (icon, label) in enumerate(NAV):
            btn_type = "primary" if i == current else "secondary"
            if st.button(f"{icon}  {label}", key=f"sb_{i}", type=btn_type, use_container_width=True):
                st.session_state["current_step"] = i
                st.rerun()

        st.divider()

        gpu = get_gpu_info()
        st.markdown(
            f'<div style="font-size:0.75rem;color:#4B5563;padding:0.1rem 0.5rem;line-height:1.9;">'
            f'GPU: {gpu["name"][:15] if gpu["available"] else "CPU"}<br>'
            f'Data: {(st.session_state.get("dataset_name") or "—")[:15]}<br>'
            f'Model: {st.session_state.get("model_arch", "Pico").replace("FlashDet-", "")}<br>'
            f'Status: {st.session_state.get("training_status", "Idle")[:12]}'
            f'</div>',
            unsafe_allow_html=True,
        )
