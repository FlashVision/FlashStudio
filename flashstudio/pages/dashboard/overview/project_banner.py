"""Dashboard — project info banner section."""

import streamlit as st
from flashstudio.pages.dashboard._common import (
    COLOR_SUCCESS, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED, COLOR_BG_HIGHLIGHT,
)


def render_project_banner(project):
    proj_name = project.get("name", "Untitled")
    proj_desc = project.get("description", "")
    created = project.get("created", "")[:10]
    modified = project.get("last_modified", "")[:10]
    st.markdown(
        f'<div style="background:{COLOR_BG_HIGHLIGHT};border-left:4px solid {COLOR_SUCCESS};padding:0.8rem 1rem;'
        f'border-radius:0 8px 8px 0;margin-bottom:0.8rem;">'
        f'<div style="font-size:1.1rem;font-weight:700;color:{COLOR_TEXT_PRIMARY};">{proj_name}</div>'
        f'{("<div style=" + chr(34) + "font-size:0.85rem;color:" + COLOR_TEXT_SECONDARY + ";margin-top:0.2rem;" + chr(34) + ">" + proj_desc + "</div>") if proj_desc else ""}'
        f'<div style="font-size:0.78rem;color:{COLOR_TEXT_MUTED};margin-top:0.3rem;">Created: {created} · Modified: {modified}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
