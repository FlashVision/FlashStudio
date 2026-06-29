"""Inference — Model tab."""

import os
import streamlit as st
from flashstudio.constants import (
    INFER_CONF_THRESHOLD, INFER_NMS_THRESHOLD, INFER_IMG_SIZE,
    INFER_NUM_CLASSES, DEFAULT_SAVE_DIR, BEST_WEIGHT_PRIORITY,
    CKPT_BEST_INFERENCE, FLASHDET_MODELS,
)
from flashstudio.pages.inference._common import _get_class_names, _get_device_options


def _tab_model():
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Architecture & Params")
            st.selectbox("Model", list(FLASHDET_MODELS.keys()),
                         key="infer_model_arch")
            pc1, pc2 = st.columns(2)
            with pc1:
                st.slider("Confidence", 0.0, 1.0, INFER_CONF_THRESHOLD, 0.05, key="infer_conf")
                st.number_input("Img Size", 320, 1920, INFER_IMG_SIZE, 32, key="infer_img_size")
            with pc2:
                st.slider("NMS IoU", 0.0, 1.0, INFER_NMS_THRESHOLD, 0.05, key="infer_nms")
                st.selectbox("Device", _get_device_options(), key="infer_device")
            st.number_input("Classes", 1, 1000, INFER_NUM_CLASSES, key="infer_num_classes")
            filter_classes = _get_class_names()
            st.multiselect("Filter (empty=all)", filter_classes, default=[], key="infer_class_filter")

    with col2:
        with st.container(border=True):
            st.markdown("#### Weights")
            ws = st.radio("Source", ["Upload", "Path", "Training output"], key="weight_source", horizontal=True)
            if ws == "Upload":
                up = st.file_uploader("File", type=["pt", "pth", "onnx", "engine"], key="infer_weights_file")
                if up:
                    st.success(f"{up.name} ({up.size / 1e6:.1f}MB)")
            elif ws == "Path":
                st.text_input("Path", placeholder="/path/best.pth", key="infer_weights_path")
            else:
                sd = st.session_state.get("save_dir", DEFAULT_SAVE_DIR)
                best_path = ""
                for candidate in BEST_WEIGHT_PRIORITY:
                    if os.path.isdir(sd):
                        for root, _d, files in os.walk(sd):
                            if candidate in files:
                                best_path = os.path.join(root, candidate)
                                break
                    if best_path:
                        break
                if not best_path:
                    best_path = os.path.join(sd, CKPT_BEST_INFERENCE)
                st.session_state["infer_weights_path"] = best_path
                exists = os.path.isfile(best_path)
                if exists:
                    sz = os.path.getsize(best_path) / (1024 * 1024)
                    st.success(f"`{os.path.basename(best_path)}` ({sz:.1f}MB)")
                else:
                    st.caption(f"Not found: `{best_path}`")
