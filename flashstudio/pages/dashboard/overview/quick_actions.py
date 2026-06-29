"""Dashboard — quick action buttons section."""

import streamlit as st


def render_quick_actions():
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        if st.button("Upload Data", use_container_width=True, key="dash_goto_data"):
            st.session_state["current_step"] = 1
            st.rerun()
        if st.button("Configure Model", use_container_width=True, key="dash_goto_model"):
            st.session_state["current_step"] = 2
            st.rerun()
    with ac2:
        if st.button("Start Training", use_container_width=True, type="primary", key="dash_goto_train"):
            st.session_state["current_step"] = 3
            st.rerun()
        if st.button("Export Model", use_container_width=True, key="dash_goto_export"):
            st.session_state["current_step"] = 4
            st.rerun()
    with ac3:
        if st.button("Run Inference", use_container_width=True, key="dash_goto_infer"):
            st.session_state["current_step"] = 5
            st.rerun()
        if st.button("Projects", use_container_width=True, key="dash_goto_proj"):
            st.session_state["show_project_manager"] = True
            st.rerun()
