"""FlashStudio — Inference Page."""

import streamlit as st
from flashstudio.components.styles import render_page_header
from flashstudio.pages.inference._common import (
    INFER_FAMILIES, INFER_NOTES, SOLUTIONS,
)
from flashstudio.pages.inference.run.detection import _resolve_weights_path


def render_inference_page():
    from flashstudio.utils import get_state, show_flashes
    from flashstudio.pages.inference.model.tab import _tab_model
    from flashstudio.pages.inference.data.tab import _tab_data
    from flashstudio.pages.inference.solution.tab import _tab_solution
    from flashstudio.pages.inference.run.tab import _tab_run
    from flashstudio.pages.inference.run.live import _render_live_tab

    render_page_header("", "Inference", "Choose a framework, add input, run detection")
    show_flashes()

    if "infer_family" not in st.session_state:
        family = str(get_state("model_framework") or "FlashDet")
        st.session_state["infer_family"] = (
            family if family in INFER_FAMILIES else "FlashDet"
        )

    family = st.radio(
        "Framework", INFER_FAMILIES, horizontal=True, key="infer_family",
        help="Each framework uses its own checkpoint formats and runtime.",
    )
    previous = st.session_state.get("_infer_family_last")
    if previous and previous != family:
        for key in (
            "infer_weights_path", "infer_weights_file",
            "_infer_adapter", "_infer_adapter_key",
            "_infer_weights_tmp_path", "_infer_weights_tmp_key",
            "infer_img_results", "video_results",
        ):
            st.session_state.pop(key, None)
        st.session_state["infer_framework"] = "Auto"
        st.session_state["weight_source"] = "Training output"
    st.session_state["_infer_family_last"] = family
    st.caption(INFER_NOTES.get(family, ""))
    from flashstudio.components.dependencies import render_dependency_hint

    render_dependency_hint(
        family, "inference",
        key_suffix=f"infer_{family.lower()}",
    )

    has_model = bool(_resolve_weights_path())
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

    tab_model, tab_data, tab_solution, tab_run, tab_live = st.tabs(
        ["Model", "Data", "Solution", "Run", "Live"]
    )
    with tab_model:
        _tab_model(family)
    with tab_data:
        _tab_data()
    with tab_solution:
        _tab_solution()
    with tab_run:
        _tab_run()
    with tab_live:
        _render_live_tab()
