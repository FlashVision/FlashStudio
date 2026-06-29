"""FlashStudio — Dashboard."""

import streamlit as st
from flashstudio.components.styles import render_page_header
from flashstudio.pages.dashboard.overview.tab import render_overview


def render_dashboard():
    from flashstudio.components.project_manager import get_active_project, get_project_stats
    from flashstudio.utils import show_flashes

    render_page_header("", "Dashboard")
    show_flashes()

    project = get_active_project()
    stats = get_project_stats(project["id"]) if project else None

    render_overview(project, stats)
