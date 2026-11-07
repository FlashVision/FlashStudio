"""FlashStudio — Augmentation tab."""

import streamlit as st

from flashstudio.utils import get_state


def _tab_aug():
    st.markdown("#### Augmentation policy")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Multi-image")
            st.checkbox("Mosaic (4-image)", bool(get_state("aug_mosaic")), key="aug_mosaic")
            st.checkbox("MixUp", bool(get_state("aug_mixup")), key="aug_mixup")
            st.checkbox("CopyPaste", bool(get_state("aug_copypaste")), key="aug_copypaste")
    with col2:
        with st.container(border=True):
            st.markdown("#### Built-in (always on)")
            for a in ["Scale jitter 0.5–1.5×", "Horizontal flip", "Color jitter", "Letterbox resize", "ImageNet norm"]:
                st.markdown(f'<span style="font-size: var(--fs-text-md);color:#4B5563;">• {a}</span>', unsafe_allow_html=True)
