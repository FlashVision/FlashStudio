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
            img_count = sum(
                1
                for _root, _dirs, files in os.walk(train_path)
                for name in files
                if name.lower().endswith(IMG_EXTENSIONS)
            )
        img_delta = f"{img_count:,} training image" + ("" if img_count == 1 else "s")
        st.metric("Dataset", ds_name, delta=img_delta, delta_color="off")
    with m3:
        from flashstudio.utils import get_state

        epochs = get_state("epochs")
        st.metric(
            "Model",
            model_display_name(),
            delta=f"{epochs} epoch" + ("" if str(epochs) == "1" else "s") + " planned",
            delta_color="off",
        )
    with m4:
        best_map = f"{stats['best_map']:.3f}" if stats and stats["best_map"] is not None else "—"
        runs_count = stats["runs"] if stats else 0
        runs_delta = f"{runs_count} run" + ("" if runs_count == 1 else "s")
        st.metric("Best mAP@0.5", best_map, delta=runs_delta, delta_color="off")
