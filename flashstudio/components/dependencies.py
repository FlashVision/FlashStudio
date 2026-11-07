"""Dependency checks and optional in-app pip installation."""

from __future__ import annotations

import importlib
import subprocess
import sys

import streamlit as st

DEPENDENCIES = {
    "FlashDet": {
        "module": "flashdet",
        "pip": "flashdet[export]",
        "label": "FlashDet",
        "purpose": "FlashDet training, native inference, fusion, and INT8 export",
    },
    "Ultralytics": {
        "module": "ultralytics",
        "pip": "ultralytics",
        "label": "Ultralytics",
        "purpose": "YOLO training, inference, ONNX, TorchScript, and TensorRT export",
    },
    "YOLOX": {
        "module": "flashdet",
        "pip": "flashdet[export]",
        "label": "FlashDet",
        "purpose": "YOLOX inference and export (YOLOX runs through FlashDet)",
    },
    "ONNX Runtime": {
        "module": "onnxruntime",
        "pip": "onnxruntime",
        "label": "ONNX Runtime",
        "purpose": "portable ONNX inference",
    },
    "TorchScript": {
        "module": "torch",
        "pip": "torch",
        "label": "PyTorch / TorchScript",
        "purpose": "TorchScript inference",
    },
    "TensorRT": {
        "module": "tensorrt",
        "pip": "tensorrt",
        "label": "TensorRT",
        "purpose": "TensorRT engine inference and INT8 engine export",
    },
}


def dependency_info(name: str) -> dict:
    """Return module/pip/installed/version information for a dependency."""
    spec = DEPENDENCIES.get(name, {})
    module = spec.get("module", name.lower())
    info = {
        "name": name,
        "module": module,
        "pip": spec.get("pip", module),
        "label": spec.get("label", name),
        "purpose": spec.get("purpose", ""),
        "installed": False,
        "version": "",
    }
    try:
        mod = importlib.import_module(module)
        info["installed"] = True
        info["version"] = str(getattr(mod, "__version__", "") or "")
    except Exception:
        pass
    return info


def install_package(spec: str, dry_run: bool = False) -> tuple[bool, str]:
    """Install a pip spec using the current interpreter.

    Returns ``(success, output)``. Use ``dry_run`` to return the command without
    executing it.
    """
    command = [sys.executable, "-m", "pip", "install", spec]
    if dry_run:
        return True, " ".join(command)
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=1800,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, output.strip()
    except Exception as exc:
        return False, str(exc)


def render_dependency_hint(name: str, context: str = "", key_suffix: str = ""):
    """Show installed status, or an install action when the package is missing."""
    info = dependency_info(name)
    if info["installed"]:
        version = f" {info['version']}" if info["version"] else ""
        st.caption(f"✓ {info['label']}{version} installed")
        return True

    suffix = key_suffix or context or "default"
    st.warning(
        f"**{info['label']} is not installed.** It is required for {info['purpose'] or context}."
    )
    st.code(f"pip install \"{info['pip']}\"", language="bash")
    if st.button(
        f"Install {info['label']}", key=f"install_{name.lower()}_{suffix}",
        type="primary", use_container_width=True,
    ):
        with st.spinner(f"Installing {info['label']}…"):
            ok, output = install_package(info["pip"])
        if ok:
            importlib.invalidate_caches()
            st.success(f"{info['label']} installed. Reloading…")
            st.rerun()
        else:
            st.error(f"Install failed: {output[-1200:]}")
    return False
