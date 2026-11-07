"""FlashStudio — Inference Page."""

import streamlit as st
from flashstudio.components.styles import render_page_header
from flashstudio.pages.inference._common import (
    INFER_FAMILIES, SOLUTIONS,
)
from flashstudio.pages.inference.run.detection import _resolve_weights_path


def _infer_tabs(labels):
    """Workflow tabs with optional programmatic advance to the next step."""
    import inspect

    next_tab = st.session_state.pop("_infer_tab_next", None)
    if next_tab == "Live":
        st.session_state["_infer_run_mode"] = "Live"
        next_tab = "Run"
    params = inspect.signature(st.tabs).parameters
    if "default" not in params or "on_change" not in params:
        return st.tabs(labels)
    if next_tab in labels:
        st.session_state["_infer_tab"] = next_tab
    selected = st.session_state.get("_infer_tab")
    if selected == "Live":
        selected = "Run"
    if selected not in labels:
        selected = labels[0]
    st.session_state["_infer_tab"] = selected
    return st.tabs(
        labels, key="_infer_tab", default=selected, on_change="rerun"
    )


def render_inference_page():
    from flashstudio.utils import get_state, show_flashes
    from flashstudio.pages.inference.model.tab import (
        _apply_weight_selection, _tab_model,
    )
    from flashstudio.pages.inference.data.tab import _tab_data
    from flashstudio.pages.inference.solution.tab import _tab_solution
    from flashstudio.pages.inference.run.tab import _tab_run
    from flashstudio.pages.inference.run.live import _render_live_tab

    render_page_header("", "Inference")
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
        st.session_state["_infer_tab_next"] = "Model"
    st.session_state["_infer_family_last"] = family
    from flashstudio.components.dependencies import render_dependency_hint

    render_dependency_hint(
        family, "inference",
        key_suffix=f"infer_{family.lower()}",
    )

    # A download button from the Model tab stores the choice and asks for the
    # Data tab next; apply it before readiness badges are computed because the
    # Model tab itself may not run when another tab is selected.
    _apply_weight_selection()

    has_model = bool(_resolve_weights_path())
    input_type = st.session_state.get("infer_input_type", "Images")
    if input_type == "Images":
        has_data = bool(st.session_state.get("infer_images"))
    elif input_type == "Video":
        has_data = bool(st.session_state.get("infer_video"))
    else:
        has_data = bool(st.session_state.get("infer_stream_url"))
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

    tab_model, tab_data, tab_solution, tab_run = _infer_tabs(
        ["Model", "Data", "Solution", "Run"]
    )
    with tab_model:
        _tab_model(family)
    with tab_data:
        _tab_data()
    with tab_solution:
        _tab_solution()
    with tab_run:
        default_mode = (
            "Live" if st.session_state.get("infer_input_type") == "RTSP" else "Batch"
        )
        st.session_state.setdefault("_infer_run_mode", default_mode)
        st.radio(
            "Run mode",
            ["Batch", "Live"],
            horizontal=True,
            key="_infer_run_mode",
            help=(
                "Batch processes selected images or a video file. "
                "Live reads a webcam or RTSP stream continuously."
            ),
        )
        if st.session_state.get("_infer_run_mode") == "Live":
            _render_live_tab()
        else:
            _tab_run()
