"""FlashStudio — Model Config (ultra-compact, no-scroll)."""

import streamlit as st

from flashstudio.pages.model.architecture.tab import _tab_arch
from flashstudio.pages.model.hyperparams.tab import _tab_hyper, _summary_bar
from flashstudio.pages.model.augment.tab import _tab_aug
from flashstudio.pages.model.advanced.tab import _tab_adv


def _sync_runtime_frameworks():
    """Follow the Model framework on Export/Inference unless overridden there.

    A user who switches the framework on the Model page expects Export and
    Inference to use the same framework. Each page keeps its own radio, so an
    explicit choice made on those pages is preserved; only values still
    following the previous Model framework are updated.
    """
    framework = st.session_state.get("model_framework")
    if not framework:
        return
    previous = st.session_state.get("_model_framework_last")
    if previous and previous != framework:
        if st.session_state.get("export_framework", previous) == previous:
            st.session_state["export_framework"] = framework
            st.session_state["_export_framework_last"] = framework
            # Artifacts and optimization state belong to the old framework.
            for key in (
                "exported_files", "opt_fusion", "opt_onnx", "opt_int8",
                "opt_benchmark", "ul_opt_benchmark",
            ):
                st.session_state.pop(key, None)
        if st.session_state.get("infer_family", previous) == previous:
            st.session_state["infer_family"] = framework
    st.session_state["_model_framework_last"] = framework


def render_model_page():
    from flashstudio.components.styles import render_page_header
    from flashstudio.utils import show_flashes, update_config_mirror
    from flashstudio.pages.model.architecture.tab import _sync_framework_from_family

    render_page_header("", "Model")
    show_flashes()

    # Show the current configuration before the tabs so users do not have to
    # scroll through every tab to confirm what will be trained.
    _sync_framework_from_family()
    _summary_bar()

    tab_arch, tab_options = st.tabs(["Architecture", "Training options"])

    with tab_arch:
        _tab_arch()

    _sync_runtime_frameworks()

    with tab_options:
        sub_hyper, sub_aug, sub_adv = st.tabs(
            ["Hyperparameters", "Augmentation", "Advanced"]
        )
        with sub_hyper:
            _tab_hyper()
        with sub_aug:
            _tab_aug()
        with sub_adv:
            _tab_adv()

    update_config_mirror()
