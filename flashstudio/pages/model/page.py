"""FlashStudio — Model Config (ultra-compact, no-scroll)."""

import streamlit as st

from flashstudio.pages.model.architecture.tab import _tab_arch
from flashstudio.pages.model.hyperparams.tab import _tab_hyper, _summary_bar
from flashstudio.pages.model.augment.tab import _tab_aug
from flashstudio.pages.model.advanced.tab import _tab_adv


def render_model_page():
    from flashstudio.components.styles import render_page_header
    from flashstudio.utils import show_flashes, update_config_mirror
    render_page_header("", "Model")
    show_flashes()

    tab_arch, tab_hyper, tab_aug, tab_adv = st.tabs(["Architecture", "Hyperparams", "Augment", "Advanced"])

    with tab_arch:
        _tab_arch()
    with tab_hyper:
        _tab_hyper()
    with tab_aug:
        _tab_aug()
    with tab_adv:
        _tab_adv()

    _summary_bar()

    update_config_mirror()
