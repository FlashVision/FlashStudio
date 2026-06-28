"""FlashStudio — Main Streamlit Application."""

import streamlit as st

st.set_page_config(
    page_title="FlashStudio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from flashstudio.components.styles import inject_custom_css  # noqa: E402
from flashstudio.components.sidebar import render_sidebar  # noqa: E402
from flashstudio.components.wizard import render_step_indicator, render_navigation  # noqa: E402
from flashstudio.components.project_manager import (  # noqa: E402
    get_active_project, render_project_manager_page,
    save_project_state, list_projects, create_project,
)
from flashstudio.pages.dashboard import render_dashboard  # noqa: E402
from flashstudio.pages.data import render_data_page  # noqa: E402
from flashstudio.pages.model import render_model_page  # noqa: E402
from flashstudio.pages.training import render_training_page  # noqa: E402
from flashstudio.pages.export import render_export_page  # noqa: E402
from flashstudio.pages.inference import render_inference_page  # noqa: E402

STEPS = [
    {"id": "dashboard", "label": "Dashboard", "icon": "🏠"},
    {"id": "data", "label": "Data", "icon": "📦"},
    {"id": "model", "label": "Model", "icon": "🧠"},
    {"id": "training", "label": "Training", "icon": "🏋️"},
    {"id": "export", "label": "Export", "icon": "📤"},
    {"id": "inference", "label": "Inference", "icon": "🔍"},
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
    """Show first-time project creation screen."""
    st.markdown(
        "<div style='text-align:center; padding:3rem 0 1rem;'>"
        "<span style='font-size:3rem;'>⚡</span><br>"
        "<b style='font-size:1.8rem;'>Welcome to FlashStudio</b><br>"
        "<span style='color:#6B7280;'>Create your first project to get started</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        col_form, col_info = st.columns([2, 1])

        with col_form:
            st.markdown("### Create Your First Project")
            name = st.text_input("Project Name", placeholder="e.g. Traffic Detection",
                                 key="first_project_name")
            desc = st.text_input("Description (optional)",
                                 placeholder="Detect vehicles and pedestrians in traffic camera footage",
                                 key="first_project_desc")

            if st.button("🚀 Create Project & Start", type="primary",
                         disabled=not name, use_container_width=True):
                create_project(name, desc)
                st.rerun()

        with col_info:
            st.markdown("### What is a Project?")
            st.markdown("""
            A project keeps everything organized:

            - **Dataset** — your training/val data
            - **Model Config** — architecture & hyperparams
            - **Training Runs** — checkpoints & metrics
            - **Exports** — ONNX models
            - **Inference** — test results

            Each project is isolated — switch between
            projects without losing progress.
            """)


if __name__ == "__main__":
    main()
