"""FlashStudio — Config building/applying helpers."""

import streamlit as st
from flashstudio.utils.config import apply_training_config
from flashstudio.constants import (
    ARCH_FAMILIES, DEFAULT_MODEL_ARCH,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_IMG_SIZE,
)


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
