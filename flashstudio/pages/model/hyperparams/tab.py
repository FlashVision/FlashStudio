"""FlashStudio — Hyperparameters tab, summary bar."""

import yaml
import streamlit as st
from flashstudio.constants import (
    OPTIMIZERS,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_IMG_SIZE,
    TRAIN_WEIGHT_DECAY, TRAIN_WARMUP_EPOCHS, TRAIN_PATIENCE,
    TRAIN_NUM_WORKERS, TRAIN_GRAD_ACCUM, TRAIN_VAL_INTERVAL,
    TRAIN_LR_FINAL_RATIO, BATCH_SIZE_OPTIONS, IMG_SIZE_OPTIONS,
    DEFAULT_MODEL_ARCH,
)


def _tab_hyper():
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Core")
            hc1, hc2 = st.columns(2)
            with hc1:
                st.number_input("Epochs", min_value=1, max_value=10000, value=TRAIN_EPOCHS, step=10, key="epochs")
                st.select_slider("Batch", BATCH_SIZE_OPTIONS, value=TRAIN_BATCH_SIZE, key="batch_size")
                st.select_slider("Img Size", IMG_SIZE_OPTIONS, value=TRAIN_IMG_SIZE, key="img_size")
            with hc2:
                st.slider("LR", 1e-5, 1e-1, TRAIN_LR, format="%.5f", key="lr")
                st.slider("Weight Decay", 0.0, 0.1, TRAIN_WEIGHT_DECAY, format="%.3f", key="weight_decay")
                st.selectbox("Optimizer", OPTIMIZERS, key="optimizer")

    with col2:
        with st.container(border=True):
            st.markdown("#### Schedule & Options")
            sc1, sc2 = st.columns(2)
            with sc1:
                st.slider("Warmup Ep", 0, 20, TRAIN_WARMUP_EPOCHS, key="warmup_epochs")
                st.slider("LR Final ×", 0.01, 0.5, TRAIN_LR_FINAL_RATIO, format="%.2f", key="lr_final_ratio")
                st.slider("Workers", 0, 16, TRAIN_NUM_WORKERS, key="num_workers")
            with sc2:
                st.slider("Patience", 5, 100, TRAIN_PATIENCE, key="patience")
                st.number_input("Grad Accum", 1, 16, TRAIN_GRAD_ACCUM, key="grad_accum")
                st.slider("Val Interval", 1, 20, TRAIN_VAL_INTERVAL, key="val_interval")
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                st.checkbox("AMP FP16", True, key="amp")
            with oc2:
                st.checkbox("Grad Clip", True, key="grad_clip")
            with oc3:
                st.checkbox("Multi-Scale", False, key="multiscale")


def _summary_bar():
    model = st.session_state.get("model_arch", DEFAULT_MODEL_ARCH)
    nc = st.session_state.get("num_classes", 0)
    cls_names = st.session_state.get("class_names", "")
    if isinstance(cls_names, list):
        cls_names = "\n".join(cls_names)
    nc_display = nc if nc else len([c for c in cls_names.strip().split("\n") if c.strip()]) if cls_names.strip() else "—"
    st.markdown(
        f'<div class="info-bar">'
        f'Model: <b>{model.replace("FlashDet-", "")}</b> · '
        f'Classes: <b>{nc_display}</b> · '
        f'Ep: <b>{st.session_state.get("epochs", TRAIN_EPOCHS)}</b> · '
        f'BS: <b>{st.session_state.get("batch_size", TRAIN_BATCH_SIZE)}</b> · '
        f'LR: <b>{st.session_state.get("lr", TRAIN_LR):.1e}</b> · '
        f'Img: <b>{st.session_state.get("img_size", TRAIN_IMG_SIZE)}</b> · '
        f'AMP: <b>{"On" if st.session_state.get("amp") else "Off"}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Save / Load Config", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            from flashstudio.utils.config import build_training_config
            full_cfg = build_training_config()
            st.download_button("Download YAML", yaml.dump(full_cfg, default_flow_style=False, sort_keys=False),
                               file_name="config.yaml", mime="text/yaml", use_container_width=True)
        with c2:
            up = st.file_uploader("Load", type=["yaml", "yml"], key="cfg_upload", label_visibility="collapsed")
            if up:
                try:
                    loaded = yaml.safe_load(up.read().decode("utf-8"))
                    if st.button("Apply", key="apply_cfg", use_container_width=True):
                        from flashstudio.utils import flash
                        from flashstudio.pages.model.utils.config import _apply_cfg
                        _apply_cfg(loaded)
                        flash("Config applied", "success")
                        st.rerun()
                except Exception as e:
                    st.error(str(e)[:40])
