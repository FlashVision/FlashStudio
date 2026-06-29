"""FlashStudio shared utilities."""

import os

from flashstudio.constants import (
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_IMG_SIZE,
    TRAIN_WEIGHT_DECAY, TRAIN_WARMUP_EPOCHS, TRAIN_PATIENCE,
    TRAIN_NUM_WORKERS, TRAIN_GRAD_ACCUM, DEFAULT_MODEL_ARCH,
    DEFAULT_ARCH_FAMILY, DEFAULT_SAVE_DIR,
)

# Central defaults — single source of truth, no hardcoding elsewhere
DEFAULTS = {
    "epochs": TRAIN_EPOCHS,
    "batch_size": TRAIN_BATCH_SIZE,
    "lr": TRAIN_LR,
    "img_size": TRAIN_IMG_SIZE,
    "weight_decay": TRAIN_WEIGHT_DECAY,
    "warmup_epochs": TRAIN_WARMUP_EPOCHS,
    "patience": TRAIN_PATIENCE,
    "num_workers": TRAIN_NUM_WORKERS,
    "grad_accum": TRAIN_GRAD_ACCUM,
    "model_arch": DEFAULT_MODEL_ARCH,
    "arch_family": DEFAULT_ARCH_FAMILY,
    "save_dir": DEFAULT_SAVE_DIR,
}


def get_default(key: str):
    """Get a default value for a session state key."""
    return DEFAULTS.get(key)


def get_state(key: str):
    """Get a session state value with config mirror fallback.

    Priority: _config_mirror > session_state > DEFAULTS.
    Mirror is kept fresh by update_config_mirror() on every Model page render,
    so it always holds the latest user choices.  Mirror wins over session_state
    because Streamlit drops widget keys when navigating away, and
    _ensure_config_in_session may re-seed stale values from the project file.
    """
    import streamlit as st
    mirror = st.session_state.get("_config_mirror", {})
    if key in mirror:
        return mirror[key]
    return st.session_state.get(key, DEFAULTS.get(key))


def update_config_mirror():
    """Sync _config_mirror with current session_state values.

    Call this after Model page renders so the mirror always holds the
    latest user choices, surviving page navigations that clear widget keys.
    """
    import streamlit as st
    mirror_keys = [
        "epochs", "batch_size", "lr", "img_size", "weight_decay",
        "warmup_epochs", "patience", "num_workers", "grad_accum",
        "val_interval", "lr_final_ratio", "optimizer",
        "model_arch", "arch_family", "pico_backbone",
        "amp", "grad_clip", "multiscale",
        "aug_mosaic", "aug_mixup", "aug_copypaste",
        "finetune_strategy", "pretrain_option", "custom_weights",
        "lora_variant", "lora_rank", "lora_alpha", "lora_dropout", "lora_targets",
        "activation_checkpointing", "activation_offloading", "optimizer_in_bwd",
        "compile_model", "use_8bit_optimizer", "ddp", "chunked_loss", "chunk_size",
        "save_dir", "save_best", "resume_training",
    ]
    mirror = st.session_state.get("_config_mirror", {})
    for k in mirror_keys:
        if k in st.session_state:
            mirror[k] = st.session_state[k]
    st.session_state["_config_mirror"] = mirror

from flashstudio.utils.device import (
    is_colab,
    get_device,
    get_gpu_info,
    get_colab_runtime_type,
    has_cuda,
)
from flashstudio.utils.filesystem import (
    dir_size_bytes,
    dir_size_str,
    ensure_dir,
    safe_rmtree,
)
from flashstudio.utils.config import (
    build_training_config,
    apply_training_config,
    save_config_yaml,
    load_config_yaml,
)


def flash(msg: str, kind: str = "success"):
    """Queue a flash message to display on the next render.

    kind: 'success', 'error', 'warning', 'info'
    """
    import streamlit as st
    msgs = st.session_state.setdefault("_flash_messages", [])
    msgs.append((kind, msg))


def show_flashes():
    """Display and clear all queued flash messages at the top of a page."""
    import streamlit as st
    msgs = st.session_state.pop("_flash_messages", [])
    for kind, msg in msgs:
        getattr(st, kind, st.info)(msg)


def get_class_names_str() -> str:
    """Return class names as a newline-separated string, regardless of stored type."""
    import streamlit as st
    raw = st.session_state.get("class_names", "")
    if isinstance(raw, list):
        return "\n".join(raw)
    return str(raw)


def get_class_names_list() -> list:
    """Return class names as a list of strings."""
    raw = get_class_names_str()
    return [c.strip() for c in raw.strip().split("\n") if c.strip()]


__all__ = [
    "DEFAULTS",
    "get_default",
    "get_state",
    "update_config_mirror",
    "is_colab",
    "get_device",
    "get_gpu_info",
    "get_colab_runtime_type",
    "has_cuda",
    "dir_size_bytes",
    "dir_size_str",
    "ensure_dir",
    "safe_rmtree",
    "build_training_config",
    "apply_training_config",
    "save_config_yaml",
    "load_config_yaml",
    "flash",
    "show_flashes",
    "get_class_names_str",
    "get_class_names_list",
]
