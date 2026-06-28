"""FlashStudio — Model Config (ultra-compact, no-scroll)."""

import os
import yaml
import streamlit as st
from flashstudio.utils.config import apply_training_config
from flashstudio.constants import (
    FLASHDET_MODELS, OPTIMIZERS, LORA_VARIANTS, ARCH_FAMILIES,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_IMG_SIZE,
    TRAIN_WEIGHT_DECAY, TRAIN_WARMUP_EPOCHS, TRAIN_PATIENCE,
    TRAIN_NUM_WORKERS, TRAIN_GRAD_ACCUM, TRAIN_VAL_INTERVAL,
    TRAIN_LR_FINAL_RATIO, BATCH_SIZE_OPTIONS, IMG_SIZE_OPTIONS,
    DEFAULT_MODEL_ARCH,
)


def render_model_page():
    from flashstudio.components.styles import render_page_header
    from flashstudio.utils import show_flashes
    render_page_header("", "Model")
    show_flashes()

    tab_arch, tab_hyper, tab_aug, tab_adv = st.tabs(["Architecture", "Hyperparams", "Augment", "Advanced"])

    with tab_arch:
        _tab_arch()
    with tab_hyper:
        _tab_hyper()
    with tab_aug:
        _tab_aug()
    with tab_adv:
        _tab_adv()

    # Bottom summary bar
    _summary_bar()


def _tab_arch():
    arch = st.radio("Family", ARCH_FAMILIES, key="arch_family", horizontal=True)

    # Head config — num_classes from dataset
    nc = st.session_state.get("num_classes", 0)
    cls_names = st.session_state.get("class_names", "")
    if isinstance(cls_names, list):
        cls_names = "\n".join(cls_names)
    if not nc and cls_names.strip():
        nc = len([c for c in cls_names.strip().split("\n") if c.strip()])
        st.session_state["num_classes"] = nc

    if arch == "FlashDet (recommended)":
        col_sel, col_info = st.columns([2, 3])
        with col_sel:
            model = st.radio("Size", list(FLASHDET_MODELS.keys()), index=0, key="model_arch")
            info = FLASHDET_MODELS[model]
            if model == "FlashDet-Pico":
                st.selectbox("Backbone", ["LiteBackbone (ShuffleNetV2-0.5x)", "PicoBackbone (RepNeXt)"], key="pico_backbone")
            else:
                st.caption(f"Backbone: {info['backbone']}")
            st.caption(f"{info['neck']} → {info['head']} · CIoU+BCE+L1 · STAL · ProgLoss")

        with col_info:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Params", info["params"])
            with m2:
                st.metric("Speed", info["speed"])
            with m3:
                st.metric("Best For", info["for"])
            with m4:
                st.metric("Head Classes", nc if nc else "—")
            if nc:
                st.caption(f"Model head output → **{nc}** classes (auto-set from dataset)")
            else:
                st.info("No classes detected yet. Go to **Data** → **Upload** to load a dataset and classes will be set automatically.")
            _pretrain_finetune()
    else:
        name = arch.replace(" (recommended)", "")
        col_sel, col_info = st.columns([2, 3])
        with col_sel:
            st.slider("Width ×", 0.25, 1.5, 0.5, 0.25, key="yolo_width_mult")
            st.slider("Depth ×", 0.33, 1.5, 0.33, 0.33, key="yolo_depth_mult")
            if name == "YOLOv9":
                st.checkbox("PGI", True, key="yolo_use_pgi")
            elif name == "YOLOv10":
                st.checkbox("PSA", True, key="yolo_use_psa")
            elif name == "YOLOv11":
                st.checkbox("C2PSA", True, key="yolo_use_c2psa")
        with col_info:
            st.caption(f"{name} — fixed backbone/neck/head. Control via width/depth multipliers.")
            if nc:
                st.metric("Head Classes", nc)
            _pretrain_finetune()


def _pretrain_finetune():
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Pretrained")
            pt = st.radio("Init", ["COCO pretrained", "ImageNet backbone", "Random", "Custom"],
                          key="pretrain_option", label_visibility="collapsed")
            if pt == "Custom":
                st.text_input("Path", placeholder="/path/model.pth", key="custom_weights")
        with c2:
            st.markdown("#### Fine-tune")
            ft = st.radio("Mode", ["Full fine-tune", "Freeze backbone", "Freeze BB+neck", "LoRA"],
                          key="finetune_strategy", label_visibility="collapsed")
            if ft == "LoRA":
                lc1, lc2 = st.columns(2)
                with lc1:
                    st.selectbox("Variant", LORA_VARIANTS, key="lora_variant")
                    st.slider("Rank", 4, 64, 16, key="lora_rank")
                with lc2:
                    st.slider("Alpha", 8, 128, 32, key="lora_alpha")
                    st.slider("Dropout", 0.0, 0.5, 0.0, key="lora_dropout")
                st.multiselect("Targets", ["backbone", "neck", "head"], default=["backbone", "neck"], key="lora_targets")
                st.checkbox("QLoRA", False, key="qlora")
                if st.session_state.get("qlora"):
                    st.selectbox("Quant", ["int8", "nf4"], key="qlora_dtype")


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


def _tab_aug():
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Multi-Image")
            st.checkbox("Mosaic (4-image)", True, key="aug_mosaic")
            st.checkbox("MixUp", False, key="aug_mixup")
            st.checkbox("CopyPaste", False, key="aug_copypaste")
    with col2:
        with st.container(border=True):
            st.markdown("#### Built-in (always on)")
            for a in ["Scale jitter 0.5–1.5×", "Horizontal flip", "Color jitter", "Letterbox resize", "ImageNet norm"]:
                st.markdown(f'<span style="font-size:0.84rem;color:#4B5563;">• {a}</span>', unsafe_allow_html=True)


def _tab_adv():
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Memory")
            ac1, ac2 = st.columns(2)
            with ac1:
                st.checkbox("Act. Checkpoint", False, key="activation_checkpointing")
                st.checkbox("Act. Offload", False, key="activation_offloading")
            with ac2:
                st.checkbox("Optim in Bwd", False, key="optimizer_in_bwd")
                st.checkbox("torch.compile", False, key="compile_model")
            st.checkbox("8-bit Optimizer", False, key="use_8bit_optimizer")
            st.checkbox("Chunked Loss", False, key="chunked_loss")
            if st.session_state.get("chunked_loss"):
                st.number_input("Chunk Size", 256, 4096, 1024, key="chunk_size")

    with col2:
        with st.container(border=True):
            st.markdown("#### Distributed & IO")
            st.checkbox("Multi-GPU (DDP)", False, key="ddp")

            # Class info
            nc = st.session_state.get("num_classes", 0)
            cls_names = st.session_state.get("class_names", "")
            if isinstance(cls_names, list):
                cls_names = "\n".join(cls_names)
            if nc and cls_names.strip():
                names_list = [c.strip() for c in cls_names.strip().split("\n") if c.strip()]
                preview = ", ".join(names_list[:5])
                if len(names_list) > 5:
                    preview += f" ... (+{len(names_list) - 5})"
                st.caption(f"Classes ({nc}): {preview}")
            else:
                st.caption("Classes: not set — go to Data → Upload")

            st.text_input("Class File (.txt)", placeholder="auto-generated from dataset classes",
                          key="class_file", help="Leave empty to auto-generate from dataset classes")

            from flashstudio.utils import DEFAULTS
            if "save_dir" not in st.session_state:
                st.session_state["save_dir"] = DEFAULTS["save_dir"]
            st.text_input("Save Dir", key="save_dir")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.checkbox("Save Best Only", True, key="save_best")
            with rc2:
                st.checkbox("Resume", False, key="resume_training")
            if st.session_state.get("resume_training"):
                st.text_input("Checkpoint", placeholder="checkpoint_last.pth", key="resume_path")


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
                        _apply_cfg(loaded)
                        flash("Config applied", "success")
                        st.rerun()
                except Exception as e:
                    st.error(str(e)[:40])


def _build_cfg():
    return {
        "model": {"family": st.session_state.get("arch_family", ARCH_FAMILIES[0]), "variant": st.session_state.get("model_arch", DEFAULT_MODEL_ARCH)},
        "training": {"epochs": st.session_state.get("epochs", TRAIN_EPOCHS), "batch_size": st.session_state.get("batch_size", TRAIN_BATCH_SIZE),
                     "lr": st.session_state.get("lr", TRAIN_LR), "img_size": st.session_state.get("img_size", TRAIN_IMG_SIZE)},
        "augmentation": {"mosaic": st.session_state.get("aug_mosaic", True), "mixup": st.session_state.get("aug_mixup", False)},
        "advanced": {"amp": st.session_state.get("amp", True), "compile": st.session_state.get("compile_model", False)},
    }


def _apply_cfg(config):
    if "model" in config:
        m = config["model"]
        for k, sk in [("family", "arch_family"), ("variant", "model_arch"), ("pretrained", "pretrain_option")]:
            if k in m:
                st.session_state[sk] = m[k]
    apply_training_config(config)
