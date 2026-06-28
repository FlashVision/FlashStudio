"""FlashStudio — Main Streamlit Application."""

import streamlit as st

st.set_page_config(
    page_title="FlashStudio",
    page_icon="FS",
    layout="wide",
    initial_sidebar_state="expanded",
)

from flashstudio.components.styles import inject_custom_css  # noqa: E402
from flashstudio.components.sidebar import render_sidebar  # noqa: E402
from flashstudio.components.wizard import render_step_indicator, render_navigation  # noqa: E402
from flashstudio.components.project_manager import (  # noqa: E402
    get_active_project, render_project_manager_page,
    save_project_state, list_projects, create_project,
    load_project_state,
)
from flashstudio.pages.dashboard import render_dashboard  # noqa: E402
from flashstudio.pages.data import render_data_page  # noqa: E402
from flashstudio.pages.model import render_model_page  # noqa: E402
from flashstudio.pages.training import render_training_page  # noqa: E402
from flashstudio.pages.export import render_export_page  # noqa: E402
from flashstudio.pages.inference import render_inference_page  # noqa: E402

STEPS = [
    {"id": "dashboard", "label": "Dashboard", "icon": "1"},
    {"id": "data", "label": "Data", "icon": "2"},
    {"id": "model", "label": "Model", "icon": "3"},
    {"id": "training", "label": "Training", "icon": "4"},
    {"id": "export", "label": "Export", "icon": "5"},
    {"id": "inference", "label": "Inference", "icon": "6"},
]

PAGE_RENDERERS = {
    "dashboard": render_dashboard,
    "data": render_data_page,
    "model": render_model_page,
    "training": render_training_page,
    "export": render_export_page,
    "inference": render_inference_page,
}


def main():
    import traceback as _tb

    inject_custom_css()

    if "current_step" not in st.session_state:
        st.session_state["current_step"] = 0

    # Project system — check if user has an active project
    active_project = get_active_project()

    if active_project is None and not list_projects():
        # First time user — show project creation
        _render_first_time_setup()
        return

    # Load project state once on initial page load
    if active_project and not st.session_state.get("_project_state_loaded"):
        load_project_state(active_project["id"])
        st.session_state["_project_state_loaded"] = True

    if st.session_state.get("show_project_manager"):
        render_project_manager_page()
        if st.button("← Back to Workspace", key="back_from_projects"):
            st.session_state["show_project_manager"] = False
            st.rerun()
        return

    render_sidebar()
    render_step_indicator(STEPS, st.session_state["current_step"])

    current = STEPS[st.session_state["current_step"]]

    try:
        PAGE_RENDERERS[current["id"]]()
    except Exception as _e:
        st.error(f"Error: {_e}")
        st.code(_tb.format_exc())

    render_navigation(STEPS, st.session_state["current_step"])

    # Auto-save project state on each page render
    if active_project:
        save_project_state()


def _render_first_time_setup():
    """Compact first-time project creation."""
    st.markdown(
        "<div style='text-align:center; padding:2rem 0 0.5rem;'>"
        "<b style='font-size:2.5rem;'>FS</b><br>"
        "<b style='font-size:1.3rem;'>Welcome to FlashStudio</b><br>"
        "<span style='color:#6B7280;font-size:0.9rem;'>Create your first project to get started</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        col_form, col_info = st.columns([3, 2])
        with col_form:
            name = st.text_input("Project Name", placeholder="e.g. Traffic Detection", key="first_project_name")
            desc = st.text_input("Description (optional)", placeholder="Detect vehicles in traffic footage", key="first_project_desc")
            if st.button("Create and Start", type="primary", disabled=not name, use_container_width=True):
                create_project(name, desc)
                st.rerun()
        with col_info:
            st.markdown("#### What's a Project?")
            for item in ["Dataset management", "Model configuration", "Training runs", "Model exports", "Inference pipeline"]:
                st.markdown(f'<span style="font-size:0.84rem;color:#4B5563;">{item}</span>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
