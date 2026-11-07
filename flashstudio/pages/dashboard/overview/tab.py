"""Dashboard overview — orchestrates the overview sections."""

import streamlit as st

from flashstudio.pages.dashboard.overview.project_banner import render_project_banner
from flashstudio.pages.dashboard.overview.metrics import render_metrics
from flashstudio.pages.dashboard.overview.pipeline_status import render_pipeline_status
from flashstudio.pages.dashboard.overview.quick_actions import render_quick_actions
from flashstudio.pages.dashboard.overview.recent_runs import render_recent_runs


def render_overview(project, stats):
    from flashstudio.utils.device import get_gpu_info

    gpu = get_gpu_info()

    if project:
        render_project_banner(project)

    render_metrics(gpu, stats)

    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown("#### What to do next")
            st.caption("Follow the workflow from data to deployment.")
            render_quick_actions()
    with right:
        with st.container(border=True):
            st.markdown("#### Pipeline status")
            render_pipeline_status()
        with st.container(border=True):
            st.markdown("#### Quick links")
            q1, q2 = st.columns(2)
            with q1:
                if st.button("Projects", key="dash_links_projects", use_container_width=True):
                    st.session_state["show_project_manager"] = True
                    st.rerun()
                if st.button("Datasets", key="dash_links_data", use_container_width=True):
                    st.session_state["current_step"] = 1
                    st.rerun()
            with q2:
                if st.button("Inference", key="dash_links_infer", use_container_width=True):
                    st.session_state["current_step"] = 5
                    st.rerun()
                if st.button("Optimize", key="dash_links_export", use_container_width=True):
                    st.session_state["current_step"] = 4
                    st.rerun()

    with st.container(border=True):
        st.markdown("#### Recent runs")
        render_recent_runs()

    with st.expander("Workspace & environment", expanded=False):
        render_environment(project)


def render_environment(project):
    """Workspace and dependency details."""
    from flashstudio.utils import get_state
    from flashstudio.utils.device import get_gpu_info
    from flashstudio.components.project_manager import get_project_dir

    left, right = st.columns(2)
    with left:
        save_dir = get_state("save_dir") or "—"
        st.caption(f"**Runs** · `{save_dir}`")
        if project:
            st.caption(f"**Project** · `{get_project_dir(project['id'])}`")
        else:
            st.caption("No active project.")

    with right:
        gpu = get_gpu_info()
        device = gpu["name"] if gpu.get("available") else "CPU"
        st.caption(f"**Device** · {device}")
        try:
            import flashdet

            st.caption(f"**FlashDet** · {flashdet.__version__}")
        except Exception:
            st.caption("**FlashDet** · not installed")
        try:
            import torch

            st.caption(f"**PyTorch** · {torch.__version__}")
        except Exception:
            st.caption("**PyTorch** · not installed")
