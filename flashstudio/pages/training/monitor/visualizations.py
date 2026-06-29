"""Training visualizations rendering."""

import os
import streamlit as st

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
        st.info("No visualization images yet. GT vs Prediction images appear here during training.")
        return

    queue = all_images[-VIS_QUEUE_SIZE:]

    st.caption(f"Showing latest {len(queue)} of {len(all_images)} total  |  Queue drops oldest when new arrives")

    cols = st.columns(VIS_QUEUE_SIZE)
    for i, img_path in enumerate(queue):
        fname = os.path.basename(img_path)
        label = fname.replace(".jpg", "").replace(".png", "").replace("_", " ").title()
        with cols[i]:
            st.image(img_path, caption=label, use_container_width=True)


def _render_image_grid(img_dir, description):
    """Render images from a directory in a grid."""
    if not os.path.isdir(img_dir):
        st.info(f"Directory not found: {img_dir}")
        return

    images = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])
    if not images:
        st.info("No images found.")
        return

    st.caption(f"{description} — {len(images)} images")

    for i in range(0, len(images), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(images):
                img_path = os.path.join(img_dir, images[idx])
                with col:
                    st.image(img_path, caption=images[idx][:30], use_container_width=True)
