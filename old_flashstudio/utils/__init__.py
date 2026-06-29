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
    """Get a session state value with central default fallback."""
    import streamlit as st
    return st.session_state.get(key, DEFAULTS.get(key))

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
