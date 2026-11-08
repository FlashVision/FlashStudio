"""User-configurable storage locations.

FlashStudio writes three kinds of large files:

* downloaded/uploaded datasets
* training runs (checkpoints, logs, curves)
* pretrained weights (FlashDet registry, Ultralytics zoo)

Datasets and runs default to the active project folder, which keeps projects
isolated.  This module lets the user point each of them at any folder (for
example a second drive) and remembers the choices between sessions.

Dataset and run overrides are stored in the project session state (so every
project can use a different location).  The pretrained-weight cache is shared
by all projects and therefore stored in ``~/.flashstudio/storage.json``.
"""

from __future__ import annotations

import json
import os

BASE_DIR = os.path.join(os.path.expanduser("~"), ".flashstudio")
STORAGE_FILE = os.path.join(BASE_DIR, "storage.json")
DEFAULT_WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")


def _load() -> dict:
    try:
        with open(STORAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(STORAGE_FILE), exist_ok=True)
    tmp = STORAGE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, STORAGE_FILE)


def clean_path(raw: str, base: str = "") -> str:
    """Normalize a user-entered folder path.

    ``~`` is expanded and relative paths resolve against *base* (usually the
    project data folder).  An empty value returns an empty string, meaning
    "use the default".
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    text = os.path.expanduser(text)
    if base and not os.path.isabs(text):
        text = os.path.join(base, text)
    return os.path.abspath(text)


def _project_data_dir() -> str:
    try:
        from flashstudio.components.project_manager import get_active_project_data_dir

        root = get_active_project_data_dir()
        if root:
            return root
    except Exception:
        pass
    from flashstudio.constants import DEFAULT_DATA_DIR

    return DEFAULT_DATA_DIR


def _project_runs_dir() -> str:
    try:
        from flashstudio.components.project_manager import get_active_project_runs_dir

        root = get_active_project_runs_dir()
        if root:
            return root
    except Exception:
        pass
    from flashstudio.constants import DEFAULT_SAVE_DIR

    return DEFAULT_SAVE_DIR


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def default_datasets_root() -> str:
    return os.path.join(_project_data_dir(), "datasets")


def resolve_datasets_root(raw: str) -> str:
    """Turn user input into an absolute dataset folder (no side effects)."""
    return clean_path(raw, base=_project_data_dir()) or default_datasets_root()


def get_datasets_root() -> str:
    """Current dataset download folder (session override or project default)."""
    import streamlit as st

    return resolve_datasets_root(
        str(st.session_state.get("download_output_dir") or "")
    )


def set_datasets_root(raw: str) -> str:
    """Point downloads at *raw* and return the resolved folder.

    The value is kept in ``download_output_dir`` so it is saved with the
    project and picked up by the Download tab and the background worker.
    """
    import streamlit as st

    resolved = resolve_datasets_root(raw)
    st.session_state["download_output_dir"] = resolved
    st.session_state["_download_dir_input"] = resolved
    return resolved


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def default_runs_root() -> str:
    return _project_runs_dir()


def resolve_runs_root(raw: str) -> str:
    """Turn user input into an absolute training-run folder (no side effects)."""
    return clean_path(raw, base=_project_data_dir()) or default_runs_root()


def get_runs_root() -> str:
    """Current training output folder (Model → Advanced → Save Dir)."""
    import streamlit as st

    return resolve_runs_root(str(st.session_state.get("save_dir") or ""))


def set_runs_root(raw: str) -> str:
    """Point training runs at *raw* and return the resolved folder."""
    import streamlit as st

    resolved = resolve_runs_root(raw)
    st.session_state["save_dir"] = resolved
    mirror = st.session_state.get("_config_mirror")
    if isinstance(mirror, dict):
        mirror["save_dir"] = resolved
    return resolved


# ---------------------------------------------------------------------------
# Pretrained weights (shared across projects)
# ---------------------------------------------------------------------------

def resolve_weights_root(raw: str) -> str:
    """Turn user input into an absolute weights folder (no side effects)."""
    return clean_path(raw) or DEFAULT_WEIGHTS_DIR


def get_weights_root() -> str:
    return resolve_weights_root(str(_load().get("weights_root") or ""))


def set_weights_root(raw: str) -> str:
    resolved = resolve_weights_root(raw)
    data = _load()
    if resolved == DEFAULT_WEIGHTS_DIR:
        data.pop("weights_root", None)
    else:
        data["weights_root"] = resolved
    _save(data)
    return resolved


def ensure_weights_root() -> str:
    root = get_weights_root()
    os.makedirs(root, exist_ok=True)
    return root


def ensure_dir(path: str) -> tuple[bool, str]:
    """Create *path* if needed.

    Returns ``(ok, message)``.  A path that exists but is a file is reported
    as invalid instead of being silently replaced.
    """
    try:
        if os.path.exists(path) and not os.path.isdir(path):
            return False, f"`{path}` is a file, not a folder."
        os.makedirs(path, exist_ok=True)
        return True, ""
    except OSError as exc:
        return False, f"Could not create `{path}`: {exc}"
