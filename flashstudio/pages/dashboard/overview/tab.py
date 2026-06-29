"""Dashboard overview — orchestrates all overview sections."""

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

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("#### Pipeline Status")
            render_pipeline_status()

    with right:
        with st.container(border=True):
            st.markdown("#### Quick Actions")
            render_quick_actions()

    with st.container(border=True):
        st.markdown("#### Recent Runs")
        render_recent_runs()
