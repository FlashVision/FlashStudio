"""FlashStudio — Model Config (ultra-compact, no-scroll)."""

import os
import yaml
import streamlit as st
from flashstudio.utils.config import apply_training_config

FLASHDET_MODELS = {
    "FlashDet-Pico": {"size": "p", "params": "~298K", "speed": "Ultra-fast", "backbone": "LiteBackbone(0.5x)", "neck": "PicoNeck(64ch)", "head": "E2EDualHead", "for": "Edge/Mobile"},
    "FlashDet-Nano": {"size": "n", "params": "~790K", "speed": "Very fast", "backbone": "FlashBB(stem=32)", "neck": "PicoNeck(96ch)", "head": "E2EDualHead", "for": "IoT"},
    "FlashDet-Small": {"size": "s", "params": "~1.8M", "speed": "Fast", "backbone": "FlashBB(stem=48)", "neck": "PicoNeck(128ch)", "head": "E2EDualHead", "for": "General"},
    "FlashDet-Medium": {"size": "m", "params": "~3.6M", "speed": "Balanced", "backbone": "FlashBB(stem=64)", "neck": "PicoNeck(192ch)", "head": "E2EDualHead", "for": "Accuracy"},
    "FlashDet-Large": {"size": "l", "params": "~5.8M", "speed": "Accurate", "backbone": "FlashBB(stem=80)", "neck": "PicoNeck(256ch)", "head": "E2EDualHead", "for": "Accuracy"},
    "FlashDet-X": {"size": "x", "params": "~9.0M", "speed": "Max acc", "backbone": "FlashBB(stem=96)", "neck": "PicoNeck(320ch)", "head": "E2EDualHead", "for": "Server"},
}
OPTIMIZERS = ["AdamW", "SGD", "MuSGD"]
LORA_VARIANTS = ["standard", "dora", "lora_plus", "adalora", "ortho", "lora_fa"]


def render_model_page():
    from flashstudio.components.styles import render_page_header
    render_page_header("🧠", "Model")

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
    arch = st.radio("Family", ["FlashDet (recommended)", "YOLOv8", "YOLOv9", "YOLOv10", "YOLOv11", "YOLOX"],
                    key="arch_family", horizontal=True)

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
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Params", info["params"])
            with m2:
                st.metric("Speed", info["speed"])
            with m3:
                st.metric("Best For", info["for"])
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
                st.multiselect("Targets", ["backbone", "fpn", "neck", "head"], default=["backbone", "fpn"], key="lora_targets")
                st.checkbox("QLoRA", False, key="qlora")
                if st.session_state.get("qlora"):
                    st.selectbox("Quant", ["int8", "int4"], key="qlora_dtype")


def _tab_hyper():
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Core")
            hc1, hc2 = st.columns(2)
            with hc1:
                st.slider("Epochs", 1, 500, 100, key="epochs")
                st.select_slider("Batch", [2, 4, 8, 16, 32, 64, 128], value=16, key="batch_size")
                st.select_slider("Img Size", [320, 416, 640], value=320, key="img_size")
            with hc2:
                st.slider("LR", 1e-5, 1e-1, 1e-3, format="%.5f", key="lr")
                st.slider("Weight Decay", 0.0, 0.1, 0.05, format="%.3f", key="weight_decay")
                st.selectbox("Optimizer", OPTIMIZERS, key="optimizer")

    with col2:
        with st.container(border=True):
            st.markdown("#### Schedule & Options")
            sc1, sc2 = st.columns(2)
            with sc1:
                st.slider("Warmup Ep", 0, 20, 3, key="warmup_epochs")
                st.slider("LR Final ×", 0.01, 0.5, 0.1, format="%.2f", key="lr_final_ratio")
                st.slider("Workers", 0, 16, 4, key="num_workers")
            with sc2:
                st.slider("Patience", 5, 100, 50, key="patience")
                st.number_input("Grad Accum", 1, 16, 1, key="grad_accum")
                st.slider("Val Interval", 1, 20, 5, key="val_interval")
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
            st.text_input("Class File", placeholder="classes.txt", key="class_file")
            _default = os.path.join(os.getcwd(), "flashstudio_runs")
            st.text_input("Save Dir", value=_default, key="save_dir")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.checkbox("Save Best Only", True, key="save_best")
            with rc2:
                st.checkbox("Resume", False, key="resume_training")
            if st.session_state.get("resume_training"):
                st.text_input("Checkpoint", placeholder="checkpoint_last.pth", key="resume_path")


def _summary_bar():
    model = st.session_state.get("model_arch", "FlashDet-Pico")
    st.markdown(
        f'<div class="info-bar">'
        f'Model: <b>{model.replace("FlashDet-", "")}</b> · '
        f'Ep: <b>{st.session_state.get("epochs", 100)}</b> · '
        f'BS: <b>{st.session_state.get("batch_size", 16)}</b> · '
        f'LR: <b>{st.session_state.get("lr", 0.001):.1e}</b> · '
        f'Img: <b>{st.session_state.get("img_size", 320)}</b> · '
        f'AMP: <b>{"On" if st.session_state.get("amp") else "Off"}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Save / Load Config", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download YAML", yaml.dump(_build_cfg(), default_flow_style=False, sort_keys=False),
                               file_name="config.yaml", mime="text/yaml", use_container_width=True)
        with c2:
            up = st.file_uploader("Load", type=["yaml", "yml"], key="cfg_upload", label_visibility="collapsed")
            if up:
                try:
                    loaded = yaml.safe_load(up.read().decode("utf-8"))
                    if st.button("Apply", key="apply_cfg", use_container_width=True):
                        _apply_cfg(loaded); st.rerun()
                except Exception as e:
                    st.error(str(e)[:40])


def _build_cfg():
    return {
        "model": {"family": st.session_state.get("arch_family", "FlashDet"), "variant": st.session_state.get("model_arch", "FlashDet-Pico")},
        "training": {"epochs": st.session_state.get("epochs", 100), "batch_size": st.session_state.get("batch_size", 16),
                     "lr": st.session_state.get("lr", 0.001), "img_size": st.session_state.get("img_size", 320)},
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
