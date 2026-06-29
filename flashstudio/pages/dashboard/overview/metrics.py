"""Dashboard — key metrics section (GPU, dataset, model, best mAP)."""

import os
import streamlit as st
from flashstudio.pages.dashboard._common import (
    IMG_EXTENSIONS,
    COLOR_SUCCESS, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED, GPU_NAME_TRUNCATE, DATASET_NAME_TRUNCATE,
)


def render_metrics(gpu, stats):
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        with st.container(border=True):
            gpu_name = gpu["name"][:GPU_NAME_TRUNCATE] if gpu["available"] else "CPU only"
            vram = f"{gpu['memory_total']:.0f} GB" if gpu["available"] else "\u2014"
            st.markdown(
                f'<div style="text-align:center;padding:0.4rem 0;">'
                f'<div style="font-size:0.78rem;color:{COLOR_TEXT_SECONDARY};font-weight:500;">GPU</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:{COLOR_TEXT_PRIMARY};">{gpu_name}</div>'
                f'<div style="font-size:0.75rem;color:{COLOR_TEXT_MUTED};">VRAM: {vram}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with m2:
        with st.container(border=True):
            ds_name = (st.session_state.get("dataset_name") or "Not loaded")[:DATASET_NAME_TRUNCATE]
            train_path = st.session_state.get("train_img_path", "")
            img_count = 0
            if train_path and os.path.isdir(train_path):
                img_count = len([f for f in os.listdir(train_path) if f.lower().endswith(IMG_EXTENSIONS)])
            st.markdown(
                f'<div style="text-align:center;padding:0.4rem 0;">'
                f'<div style="font-size:0.78rem;color:{COLOR_TEXT_SECONDARY};font-weight:500;">Dataset</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:{COLOR_TEXT_PRIMARY};">{ds_name}</div>'
                f'<div style="font-size:0.75rem;color:{COLOR_TEXT_MUTED};">{img_count} images</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with m3:
        with st.container(border=True):
            from flashstudio.utils import get_state
            model = get_state("model_arch")
            model_short = model.replace("FlashDet-", "")
            epochs = get_state("epochs")
            st.markdown(
                f'<div style="text-align:center;padding:0.4rem 0;">'
                f'<div style="font-size:0.78rem;color:{COLOR_TEXT_SECONDARY};font-weight:500;">Model</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:{COLOR_TEXT_PRIMARY};">{model_short}</div>'
                f'<div style="font-size:0.75rem;color:{COLOR_TEXT_MUTED};">{epochs} epochs</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with m4:
        with st.container(border=True):
            best_map = f"{stats['best_map']:.3f}" if stats and stats["best_map"] is not None else "\u2014"
            runs_count = stats["runs"] if stats else 0
            st.markdown(
                f'<div style="text-align:center;padding:0.4rem 0;">'
                f'<div style="font-size:0.78rem;color:{COLOR_TEXT_SECONDARY};font-weight:500;">Best mAP</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:{COLOR_SUCCESS if best_map != "\u2014" else COLOR_TEXT_PRIMARY};">{best_map}</div>'
                f'<div style="font-size:0.75rem;color:{COLOR_TEXT_MUTED};">{runs_count} runs</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
