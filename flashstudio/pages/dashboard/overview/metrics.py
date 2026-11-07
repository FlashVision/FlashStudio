"""Dashboard — key metrics."""

import os

import streamlit as st

from flashstudio.pages.dashboard._common import (
    IMG_EXTENSIONS, GPU_NAME_TRUNCATE, DATASET_NAME_TRUNCATE,
    model_display_name,
)


def render_metrics(gpu, stats):
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        gpu_name = gpu["name"][:GPU_NAME_TRUNCATE] if gpu.get("available") else "CPU only"
        vram = f"{gpu['memory_total']:.0f} GB VRAM" if gpu.get("available") else "Training on CPU"
        st.metric("Compute", gpu_name, delta=vram, delta_color="off")
    with m2:
        ds_name = (st.session_state.get("dataset_name") or "Not loaded")[:DATASET_NAME_TRUNCATE]
        train_path = st.session_state.get("train_img_path", "")
        img_count = 0
        if train_path and os.path.isdir(train_path):
            img_count = len([
                f for f in os.listdir(train_path)
                if f.lower().endswith(IMG_EXTENSIONS)
            ])
        st.metric("Dataset", ds_name, delta=f"{img_count} training images", delta_color="off")
    with m3:
        from flashstudio.utils import get_state

        st.metric(
            "Model",
            model_display_name(),
            delta=f"{get_state('epochs')} epochs planned",
            delta_color="off",
        )
    with m4:
        best_map = f"{stats['best_map']:.3f}" if stats and stats["best_map"] is not None else "—"
        runs_count = stats["runs"] if stats else 0
        st.metric("Best mAP@0.5", best_map, delta=f"{runs_count} runs", delta_color="off")
