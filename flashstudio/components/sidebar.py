"""FlashStudio sidebar — navigation + project selector."""

import streamlit as st
from flashstudio.utils.device import get_gpu_info
from flashstudio.components.project_manager import (
    get_active_project, render_project_selector, save_project_state,
)

PAGES = ["🏠 Dashboard", "📦 Data", "🧠 Model", "🏋️ Training", "📤 Export", "🔍 Inference"]


def render_sidebar():
    """Render sidebar with project selector + navigation."""
    with st.sidebar:
        # Logo
        st.markdown(
            "<div style='text-align:center; padding:0.5rem 0;'>"
            "<span style='font-size:1.6rem;'>⚡</span> "
            "<b style='font-size:1.1rem;'>FlashStudio</b>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ─── Active Project ───
        project = get_active_project()
        if project:
            with st.container(border=True):
                st.caption("📋 Active Project")
                st.markdown(f"**{project['name']}**")
                if st.button("📋 Manage Projects", key="sb_manage_projects",
                             use_container_width=True):
                    save_project_state()
                    st.session_state["show_project_manager"] = True
                    st.rerun()

        st.divider()

        # ─── Navigation buttons ───
        current = st.session_state.get("current_step", 0)
        for i, page in enumerate(PAGES):
            btn_type = "primary" if i == current else "secondary"
            if st.button(page, key=f"sb_{i}", type=btn_type, use_container_width=True):
                st.session_state["current_step"] = i
                st.rerun()

        st.divider()

        # ─── Compact status ───
        st.caption(f"Model: {st.session_state.get('model_arch', 'FlashDet-Pico')}")
        st.caption(f"Dataset: {st.session_state.get('dataset_name', '—')}")
        st.caption(f"Status: {st.session_state.get('training_status', 'Not started')}")

        st.divider()

        # ─── Environment ───
        gpu = get_gpu_info()
        if gpu["available"]:
            st.caption(f"🖥️ {gpu['name']}")
        else:
            st.caption("💻 CPU Mode")
