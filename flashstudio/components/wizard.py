"""Minimal step indicator — no navigation buttons (sidebar handles nav)."""

import streamlit as st


def render_step_indicator(steps: list, current_step: int):
    """Thin horizontal step indicator bar."""
    items = ""
    for i, step in enumerate(steps):
        cls = "step-done" if i < current_step else ("step-active" if i == current_step else "step-pending")
        icon = "✓" if i < current_step else step["icon"]
        items += (
            f'<div class="step-item {cls}">'
            f'<span class="step-icon">{icon}</span>'
            f'<div class="step-label">{step["label"]}</div>'
            f'<div class="step-bar"></div></div>'
        )
    st.markdown(f'<div class="step-indicator">{items}</div>', unsafe_allow_html=True)


def render_navigation(steps: list, current_step: int):
    """No-op — navigation handled by sidebar."""
    pass
