"""Training visualizations rendering."""

import os
import streamlit as st

from flashstudio.components.media import render_fixed_image
from flashstudio.constants import IMG_EXTENSIONS, VIS_SKIP_FILE, VIS_QUEUE_SIZE


def _render_visualizations(run_dir):
    """Show GT-vs-Prediction images in a FIFO queue — latest N shown, oldest dropped."""
    vis_dir = os.path.join(run_dir, "visualizations")
    all_images = []
    if os.path.isdir(vis_dir):
        for f in sorted(os.listdir(vis_dir)):
            if f.lower().endswith(IMG_EXTENSIONS) and f != VIS_SKIP_FILE:
                all_images.append(os.path.join(vis_dir, f))

    if not all_images:
        st.info("No visualization images yet. Ground truth vs prediction images appear here during training.")
        return

    queue = all_images[-VIS_QUEUE_SIZE:]

    st.caption(f"Latest {len(queue)} of {len(all_images)} images")

    cols = st.columns(VIS_QUEUE_SIZE)
    for i, img_path in enumerate(queue):
        fname = os.path.basename(img_path)
        label = fname.replace(".jpg", "").replace(".png", "").replace("_", " ").title()
        with cols[i]:
            st.image(img_path, caption=label, use_container_width=True)


def _render_image_grid(img_dir, description, max_images=3):
    """Render images from a directory in a fixed-size grid (max 3 by default)."""
    if not os.path.isdir(img_dir):
        st.info(f"Directory not found: {img_dir}")
        return

    images = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])
    if not images:
        st.info("No images found.")
        return

    shown = images[:max_images]
    extra = f" of {len(images)}" if len(images) > max_images else ""
    st.caption(f"{description} · {len(shown)}{extra}")

    cols = st.columns(len(shown))
    for i, col in enumerate(cols):
        img_path = os.path.join(img_dir, shown[i])
        with col:
            # Fixed window height keeps portrait and landscape images aligned
            # instead of producing ragged rows.
            render_fixed_image(img_path, shown[i][:40])
