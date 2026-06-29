"""FlashStudio — Export settings UI (weights display & config)."""

import os
import streamlit as st
from flashstudio.constants import (
    DEFAULT_SAVE_DIR, EXPORT_IMG_SIZES, EXPORT_OPSET_MIN,
    EXPORT_OPSET_MAX, EXPORT_OPSET_DEFAULT, EXPORT_FORMATS,
    EXPORT_WEIGHT_MAP, CKPT_BEST, MAX_WEIGHTS_DISPLAY,
)


def _get_export_save_dir():
    """Resolve save_dir: _config_mirror > session_state > DEFAULT_SAVE_DIR."""
    from flashstudio.utils import get_state
    sd = get_state("save_dir")
    return sd if sd else DEFAULT_SAVE_DIR


def render_export_settings():
    """Render the weights display and config UI. Returns (weights_path, img_size, export_fmt)."""
    st.markdown("#### Weights")
    save_dir = _get_export_save_dir()
    if os.path.isdir(save_dir):
        wts = []
        for root, _dirs, files in os.walk(save_dir):
            for f in files:
                if f.endswith(".pth"):
                    wts.append(os.path.join(root, f))
        if wts:
            cols = st.columns(min(len(wts), MAX_WEIGHTS_DISPLAY))
            for i, wp_disp in enumerate(sorted(wts)[:MAX_WEIGHTS_DISPLAY]):
                sz = os.path.getsize(wp_disp) / (1024 * 1024)
                bn = os.path.basename(wp_disp)
                lbl = "Best" if "best" in bn else ("Last" if "last" in bn else bn[:6])
                with cols[i]:
                    st.metric(lbl, f"{sz:.1f}MB")
        else:
            st.caption(f"No weights found in `{save_dir}`")
    else:
        st.caption("No save directory set. Train a model first or set `Save Dir` on Model page.")

    st.markdown("#### Config")
    src = st.radio("Source", ["Best (inference)", "Best (FP16)", "Last", "Custom"],
                   key="weights_source", horizontal=True)
    if src == "Custom":
        st.text_input("Path", placeholder="model.pth", key="export_weights_path")

    sd = _get_export_save_dir()
    if src != "Custom":
        targets = EXPORT_WEIGHT_MAP.get(src, [CKPT_BEST])
        wp = ""
        if os.path.isdir(sd):
            for root, _dirs, files in os.walk(sd):
                for target in targets:
                    if target in files:
                        wp = os.path.join(root, target)
                        break
                if wp:
                    break
        if not wp:
            wp = os.path.join(sd, targets[0])
    else:
        wp = st.session_state.get("export_weights_path", "")

    oc = st.columns(4)
    with oc[0]:
        export_fmt = st.selectbox("Format", EXPORT_FORMATS, key="export_format")
    with oc[1]:
        img_sz = st.select_slider("Img", EXPORT_IMG_SIZES, value=EXPORT_IMG_SIZES[0], key="export_img_size")
    with oc[2]:
        st.number_input("Opset", EXPORT_OPSET_MIN, EXPORT_OPSET_MAX, EXPORT_OPSET_DEFAULT, key="export_opset")
    with oc[3]:
        st.checkbox("Dynamic", True, key="export_dynamic")

    return wp, img_sz, export_fmt
