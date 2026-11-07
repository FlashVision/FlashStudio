"""Workflow navigation footer."""

import streamlit as st


def render_step_indicator(steps: list, current_step: int):
    """Render a compact workflow rail (kept for integrations and tests)."""
    items = ""
    for i, step in enumerate(steps):
        cls = "step-done" if i < current_step else (
            "step-active" if i == current_step else "step-pending"
        )
        items += (
            f'<div class="step-item {cls}">'
            f'<span class="step-icon">{step.get("icon", "")}</span>'
            f'<div class="step-label">{step.get("label", "")}</div>'
            f'<div class="step-bar"></div></div>'
        )
    st.markdown(f'<div class="step-indicator">{items}</div>', unsafe_allow_html=True)


def render_navigation(steps: list, current_step: int):
    """Back / continue controls with workflow context."""
    total = len(steps)
    if total <= 1:
        return

    is_first = current_step <= 0
    is_last = current_step >= total - 1
    prev_step = steps[current_step - 1] if not is_first else None
    next_step = steps[current_step + 1] if not is_last else None

    st.markdown('<div class="workflow-divider"></div>', unsafe_allow_html=True)
    left, mid, right = st.columns([1, 2, 1])

    with left:
        if prev_step:
            if st.button(
                f"←  {prev_step['label']}",
                key="nav_prev", use_container_width=True,
            ):
                st.session_state["current_step"] = current_step - 1
                st.rerun()

    with mid:
        st.markdown(
            f'<div style="text-align:center;padding-top:.25rem;">'
            f'<div class="workflow-step">Step {current_step + 1} of {total}'
            f' · {steps[current_step]["label"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        if next_step:
            if st.button(
                f"Continue: {next_step['label']}  →",
                key="nav_next", type="primary", use_container_width=True,
            ):
                st.session_state["current_step"] = current_step + 1
                st.rerun()
