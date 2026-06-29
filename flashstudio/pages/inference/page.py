"""FlashStudio — Inference Page (ultra-compact, no-scroll)."""

import streamlit as st
from flashstudio.components.styles import render_page_header
from flashstudio.pages.inference._common import SOLUTIONS


def render_inference_page():
    from flashstudio.utils import show_flashes
    from flashstudio.pages.inference.model.tab import _tab_model
    from flashstudio.pages.inference.data.tab import _tab_data
    from flashstudio.pages.inference.solution.tab import _tab_solution
    from flashstudio.pages.inference.run.tab import _tab_run

    render_page_header("", "Inference")
    show_flashes()

    has_model = bool(st.session_state.get("infer_weights_file") or st.session_state.get("infer_weights_path"))
    has_data = bool(st.session_state.get("infer_images") or st.session_state.get("infer_video") or st.session_state.get("infer_stream_url"))
    sol = SOLUTIONS.get(st.session_state.get("selected_solution", "None (Detection Only)"), {})
    has_zone = not sol.get("needs_zone") or bool(st.session_state.get("zone_line_points") or st.session_state.get("zone_polygons"))
    def dot(ok):
        return '<span style="color:#22C55E;">&#9679;</span>' if ok else '<span style="color:#EF4444;">&#9679;</span>'
    st.markdown(
        f'<div class="ds-card-stats" style="padding:0.2rem 0;">'
        f'<span>{dot(has_model)} Model</span>'
        f'<span>{dot(has_data)} Data</span>'
        f'<span>{dot(has_zone)} Zone</span>'
        f'<span style="margin-left:auto;color:#9CA3AF;">Ready: {sum([has_model, has_data, has_zone])}/3</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    tab_model, tab_data, tab_solution, tab_run = st.tabs(["Model", "Data", "Solution", "Run"])
    with tab_model:
        _tab_model()
    with tab_data:
        _tab_data()
    with tab_solution:
        _tab_solution()
    with tab_run:
        _tab_run()
