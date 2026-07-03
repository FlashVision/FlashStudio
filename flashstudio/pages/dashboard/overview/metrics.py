"""Dashboard — key metrics section (GPU, dataset, model, best mAP)."""

import os
import streamlit as st
from flashstudio.pages.dashboard._common import (
    IMG_EXTENSIONS,
    GPU_NAME_TRUNCATE, DATASET_NAME_TRUNCATE,
)


def render_metrics(gpu, stats):
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        gpu_name = gpu["name"][:GPU_NAME_TRUNCATE] if gpu["available"] else "CPU only"
        vram = f"{gpu['memory_total']:.0f} GB" if gpu["available"] else "\u2014"
        st.metric("GPU", gpu_name, delta=f"VRAM: {vram}", delta_color="off")
    with m2:
        ds_name = (st.session_state.get("dataset_name") or "Not loaded")[:DATASET_NAME_TRUNCATE]
        train_path = st.session_state.get("train_img_path", "")
        img_count = 0
        if train_path and os.path.isdir(train_path):
            img_count = len([f for f in os.listdir(train_path) if f.lower().endswith(IMG_EXTENSIONS)])
        st.metric("Dataset", ds_name, delta=f"{img_count} images", delta_color="off")
    with m3:
        from flashstudio.utils import get_state
        model = get_state("model_arch")
        model_short = model.replace("FlashDet-", "")
        epochs = get_state("epochs")
        st.metric("Model", model_short, delta=f"{epochs} epochs", delta_color="off")
    with m4:
        best_map = f"{stats['best_map']:.3f}" if stats and stats["best_map"] is not None else "\u2014"
        runs_count = stats["runs"] if stats else 0
        st.metric("Best mAP", best_map, delta=f"{runs_count} runs", delta_color="off")
