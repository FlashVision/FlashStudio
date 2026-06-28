"""FlashStudio shared utilities."""

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


__all__ = [
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
]
