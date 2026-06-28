"""Step indicator and navigation wizard component."""

import streamlit as st


def render_step_indicator(steps: list, current_step: int):
    """Render horizontal step progress indicator."""
    items_html = ""
    for i, step in enumerate(steps):
        if i < current_step:
            state_class = "step-done"
            icon = "✅"
        elif i == current_step:
            state_class = "step-active"
            icon = step["icon"]
        else:
            state_class = "step-pending"
            icon = step["icon"]

        items_html += f"""
        <div class="step-item {state_class}">
            <span class="step-icon">{icon}</span>
            <div class="step-label">{step['label']}</div>
            <div class="step-bar"></div>
        </div>"""

    st.markdown(f'<div class="step-indicator">{items_html}</div>', unsafe_allow_html=True)


def render_navigation(steps: list, current_step: int):
    """Render back/next navigation buttons."""
    col_back, col_spacer, col_next = st.columns([1, 3, 1])

    with col_back:
        if current_step > 0:
            prev_label = f"← {steps[current_step - 1]['label']}"
            if st.button(prev_label, key="wizard_back", use_container_width=True):
                st.session_state["current_step"] = current_step - 1
                st.rerun()

    with col_next:
        if current_step < len(steps) - 1:
            next_label = f"{steps[current_step + 1]['label']} →"
            if st.button(next_label, key="wizard_next", use_container_width=True, type="primary"):
                st.session_state["current_step"] = current_step + 1
                st.rerun()
