"""FlashStudio sidebar — clean icon-based navigation."""

import os
import streamlit as st
from flashstudio.utils.device import get_gpu_info
from flashstudio.components.project_manager import (
    get_active_project, save_project_state,
)
from flashstudio.constants import (
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_BODY, COLOR_TEXT_SECONDARY,
    SIDEBAR_LABEL_TRUNCATE,
)

NAV = [
    ("1", "Dashboard"),
    ("2", "Data"),
    ("3", "Model"),
    ("4", "Training"),
    ("5", "Export"),
    ("6", "Inference"),
]


def _get_logo_b64():
    """Load logo as base64 for inline HTML rendering."""
    import base64
    logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "assets", "logo.png")
    if not os.path.isfile(logo_path):
        logo_path = os.path.join(os.getcwd(), "assets", "logo.png")
    if os.path.isfile(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def render_sidebar():
    with st.sidebar:
        logo_b64 = _get_logo_b64()
        if logo_b64:
            st.markdown(
                f'<div style="display:flex;flex-direction:column;align-items:center;gap:0.3rem;padding:0.6rem 0.2rem 0.5rem;">'
                f'<img src="data:image/png;base64,{logo_b64}" style="width:56px;height:56px;">'
                f'<b style="font-size:1.1rem;color:{COLOR_TEXT_PRIMARY};">FlashStudio</b>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="display:flex;flex-direction:column;align-items:center;gap:0.2rem;padding:0.5rem 0 0.3rem;">'
                f'<b style="font-size:1.5rem;">FS</b>'
                f'<b style="font-size:1rem;color:{COLOR_TEXT_PRIMARY};">FlashStudio</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

        project = get_active_project()
        if project:
            if st.button(project['name'][:SIDEBAR_LABEL_TRUNCATE], key="sb_project", use_container_width=True):
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
        from flashstudio.constants import GPU_NAME_TRUNCATE, DATASET_NAME_TRUNCATE
        train_status = st.session_state.get("training_status", "Idle")
        status_color = COLOR_SUCCESS if train_status == "Complete" else (
            COLOR_WARNING if train_status == "Running" else (
            COLOR_ERROR if train_status == "Failed" else COLOR_TEXT_SECONDARY))
        st.markdown(
            f'<div style="font-size:0.75rem;color:{COLOR_TEXT_BODY};padding:0.1rem 0.5rem;line-height:1.9;">'
            f'GPU: {gpu["name"][:GPU_NAME_TRUNCATE] if gpu["available"] else "CPU"}<br>'
            f'Data: {(st.session_state.get("dataset_name") or "—")[:DATASET_NAME_TRUNCATE]}<br>'
            f'Model: {st.session_state.get("model_arch", "Pico").replace("FlashDet-", "")}<br>'
            f'Status: <span style="color:{status_color};font-weight:600;">{train_status[:12]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
