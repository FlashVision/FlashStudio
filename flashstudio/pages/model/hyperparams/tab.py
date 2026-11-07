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
from flashstudio.utils import get_state


def _value(key, fallback):
    """Resolve a persisted value even after Streamlit drops a widget key."""
    val = get_state(key)
    return fallback if val is None else val


def _choice(key, options, fallback):
    val = _value(key, fallback)
    return val if val in options else fallback


def _tab_hyper():
    st.markdown("#### Training schedule")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Core")
            hc1, hc2 = st.columns(2)
            with hc1:
                st.number_input(
                    "Epochs", min_value=1, max_value=10000,
                    value=int(_value("epochs", TRAIN_EPOCHS)), step=10, key="epochs",
                )
                st.select_slider(
                    "Batch size", BATCH_SIZE_OPTIONS,
                    value=_choice("batch_size", BATCH_SIZE_OPTIONS, TRAIN_BATCH_SIZE),
                    key="batch_size",
                )
                st.select_slider(
                    "Image size", IMG_SIZE_OPTIONS,
                    value=_choice("img_size", IMG_SIZE_OPTIONS, TRAIN_IMG_SIZE),
                    key="img_size",
                )
            with hc2:
                st.slider(
                    "Learning rate", 1e-5, 1e-1,
                    float(_value("lr", TRAIN_LR)), format="%.5f", key="lr",
                )
                st.slider(
                    "Weight decay", 0.0, 0.1,
                    float(_value("weight_decay", TRAIN_WEIGHT_DECAY)),
                    format="%.3f", key="weight_decay",
                )
                st.selectbox(
                    "Optimizer", OPTIMIZERS,
                    index=OPTIMIZERS.index(_value("optimizer", "AdamW"))
                    if _value("optimizer", "AdamW") in OPTIMIZERS else 0,
                    key="optimizer",
                )

    with col2:
        with st.container(border=True):
            st.markdown("#### Schedule & Options")
            sc1, sc2 = st.columns(2)
            with sc1:
                st.slider("Warmup epochs", 0, 20,
                          int(_value("warmup_epochs", TRAIN_WARMUP_EPOCHS)),
                          key="warmup_epochs")
                st.slider("LR final ratio", 0.01, 0.5,
                          float(_value("lr_final_ratio", TRAIN_LR_FINAL_RATIO)),
                          format="%.2f", key="lr_final_ratio")
                st.slider("Data workers", 0, 16,
                          int(_value("num_workers", TRAIN_NUM_WORKERS)),
                          key="num_workers")
            with sc2:
                st.slider("Patience", 5, 100,
                          int(_value("patience", TRAIN_PATIENCE)), key="patience")
                st.number_input("Grad accumulation", 1, 16,
                                int(_value("grad_accum", TRAIN_GRAD_ACCUM)),
                                key="grad_accum")
                st.slider("Validation interval", 1, 20,
                          int(_value("val_interval", TRAIN_VAL_INTERVAL)),
                          key="val_interval")
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                st.checkbox("AMP (FP16)", bool(_value("amp", True)), key="amp")
            with oc2:
                st.checkbox("Gradient clipping", bool(_value("grad_clip", True)), key="grad_clip")
            with oc3:
                st.checkbox("Multi-scale", bool(_value("multiscale", False)), key="multiscale",
                            help="Not supported by the bundled Trainer backend; ignored when enabled.")
            if st.session_state.get("multiscale"):
                st.caption("Multi-scale is ignored by the bundled FlashDet Trainer backend.")

    with st.expander("FlashDet schedule, EMA & validation", expanded=False):
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.number_input("Seed", 0, 2**31 - 1, int(_value("seed", 0)), key="seed")
            st.slider("Nominal batch (nbs)", 16, 128, int(_value("nbs", 64)), key="nbs")
        with ec2:
            st.slider("EMA decay", 0.9900, 0.9999,
                      float(_value("ema_decay", 0.9998)),
                      step=0.0001, format="%.4f", key="ema_decay")
            st.number_input("EMA warmup", 0, 20000,
                            int(_value("ema_warmup", 2000)), step=100,
                            key="ema_warmup")
        with ec3:
            st.slider("Warmup momentum", 0.5, 1.0,
                      float(_value("warmup_momentum", 0.8)),
                      step=0.01, key="warmup_momentum")
            st.slider("Base momentum", 0.80, 0.99,
                      float(_value("base_momentum", 0.937)),
                      step=0.001, format="%.3f", key="base_momentum")
        with ec4:
            st.slider("Val confidence", 0.001, 0.5,
                      float(_value("val_score_thr", 0.01)),
                      step=0.001, format="%.3f", key="val_score_thr")
            st.number_input("Max detections", 10, 1000,
                            int(_value("max_det", 300)), step=10,
                            key="max_det")
            st.slider("Grad clip norm", 1.0, 50.0,
                      float(_value("grad_clip_norm", 10.0)),
                      step=0.5, key="grad_clip_norm")


def _summary_bar():
    framework = st.session_state.get("model_framework", "FlashDet")
    if framework == "Ultralytics":
        model = (
            f"{st.session_state.get('yolo_series', 'YOLOv8')}"
            f"-{str(st.session_state.get('yolo_size', 'n')).upper()}"
        )
    elif framework == "YOLOX":
        model = "YOLOX"
    else:
        model = st.session_state.get("model_arch", DEFAULT_MODEL_ARCH).replace("FlashDet-", "")
    nc = st.session_state.get("num_classes", 0)
    cls_names = st.session_state.get("class_names", "")
    if isinstance(cls_names, list):
        cls_names = "\n".join(cls_names)
    nc_display = nc if nc else len([c for c in cls_names.strip().split("\n") if c.strip()]) if cls_names.strip() else "—"
    st.markdown(
        f'<div class="info-bar">'
        f'Model: <b>{model.replace("FlashDet-", "")}</b> · '
        f'Classes: <b>{nc_display}</b> · '
        f'Epochs: <b>{st.session_state.get("epochs", TRAIN_EPOCHS)}</b> · '
        f'Batch: <b>{st.session_state.get("batch_size", TRAIN_BATCH_SIZE)}</b> · '
        f'LR: <b>{st.session_state.get("lr", TRAIN_LR):.1e}</b> · '
        f'Image: <b>{st.session_state.get("img_size", TRAIN_IMG_SIZE)}</b> · '
        f'AMP: <b>{"On" if st.session_state.get("amp", True) else "Off"}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )
    framework = st.session_state.get("model_framework", "FlashDet")
    backend = st.session_state.get("train_backend", "FlashDet")
    if backend == "Ultralytics":
        note = (
            "Native Ultralytics training, ONNX/TorchScript export, "
            "and TensorRT via the Ultralytics runtime."
        )
    elif framework == "Ultralytics":
        note = (
            "Ultralytics architecture with the FlashDet trainer; switch the "
            "backend on Training → Launch for native Ultralytics training."
        )
    elif framework == "YOLOX":
        note = (
            "YOLOX uses the FlashDet deployment pipeline; training is not "
            "implemented by the installed FlashDet build."
        )
    else:
        note = (
            "Pico-Zoo backbones, LoRA/QLoRA, fusion, ONNX, and INT8 are "
            "available from the FlashDet flow."
        )
    st.caption(f"Framework: **{framework}** · Training backend: **{backend}** · {note}")
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
