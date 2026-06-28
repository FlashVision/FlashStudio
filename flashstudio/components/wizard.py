"""Step indicator and page navigation buttons."""

import streamlit as st


def render_step_indicator(steps: list, current_step: int):
    """Thin horizontal step indicator bar."""
    items = ""
    for i, step in enumerate(steps):
        cls = "step-done" if i < current_step else ("step-active" if i == current_step else "step-pending")
        icon = step["icon"]
        items += (
            f'<div class="step-item {cls}">'
            f'<span class="step-icon">{icon}</span>'
            f'<div class="step-label">{step["label"]}</div>'
            f'<div class="step-bar"></div></div>'
        )
    st.markdown(f'<div class="step-indicator">{items}</div>', unsafe_allow_html=True)


def render_navigation(steps: list, current_step: int):
    """Previous / Next navigation buttons at the bottom of each page."""
    total = len(steps)
    if total <= 1:
        return

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    is_first = current_step <= 0
    is_last = current_step >= total - 1

    prev_label = f"Previous: {steps[current_step - 1]['label']}" if not is_first else ""
    next_label = f"Next: {steps[current_step + 1]['label']}" if not is_last else ""

    col_prev, col_spacer, col_next = st.columns([1, 3, 1])
    with col_prev:
        if not is_first:
            if st.button(prev_label, key="nav_prev", use_container_width=True):
                st.session_state["current_step"] = current_step - 1
                st.rerun()
    with col_next:
        if not is_last:
            if st.button(next_label, key="nav_next", type="primary", use_container_width=True):
                st.session_state["current_step"] = current_step + 1
                st.rerun()
