"""FlashStudio — Augmentation tab."""

import streamlit as st


def _tab_aug():
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Multi-Image")
            st.checkbox("Mosaic (4-image)", True, key="aug_mosaic")
            st.checkbox("MixUp", False, key="aug_mixup")
            st.checkbox("CopyPaste", False, key="aug_copypaste")
    with col2:
        with st.container(border=True):
            st.markdown("#### Built-in (always on)")
            for a in ["Scale jitter 0.5–1.5×", "Horizontal flip", "Color jitter", "Letterbox resize", "ImageNet norm"]:
                st.markdown(f'<span style="font-size:0.84rem;color:#4B5563;">• {a}</span>', unsafe_allow_html=True)
