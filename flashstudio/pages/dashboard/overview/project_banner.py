"""Dashboard — project hero card."""

import html

import streamlit as st


def render_project_banner(project):
    name = html.escape(str(project.get("name", "Untitled")))
    desc = html.escape(str(project.get("description", "")))
    created = html.escape(str(project.get("created", ""))[:10])
    modified = html.escape(str(project.get("last_modified", ""))[:10])

    with st.container(border=True):
        left, right = st.columns([5, 1])
        with left:
            desc_html = (
                f'<div class="page-subtitle" style="margin-top:.15rem;">{desc}</div>'
                if desc else
                '<div class="page-subtitle" style="margin-top:.15rem;">'
                'Datasets, model configuration, training runs, and exports in one place.</div>'
            )
            st.markdown(
                f'<div style="padding:.15rem 0 .1rem;">'
                f'<div class="page-kicker">Project workspace</div>'
                f'<div style="font-size:1.55rem;font-weight:800;letter-spacing:-.03em;'
                f'color:var(--fs-text);margin:.12rem 0;">{name}</div>'
                f'{desc_html}'
                f'<div style="font-size: var(--fs-text-sm);color:var(--fs-faint);margin-top:.4rem;">'
                f'Created {created} · Updated {modified}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown("<div style='height:.4rem;'></div>", unsafe_allow_html=True)
            if st.button("Manage projects", key="dash_manage_projects",
                         use_container_width=True):
                st.session_state["show_project_manager"] = True
                st.rerun()
