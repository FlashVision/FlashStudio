"""FlashStudio — Model architecture tab."""

import streamlit as st
from flashstudio.constants import (
    FLASHDET_MODELS, LORA_VARIANTS, ARCH_FAMILIES,
)


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
