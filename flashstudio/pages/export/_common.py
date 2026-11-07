"""Shared Ship/Export helpers — framework-aware checkpoints."""

from __future__ import annotations

import os

EXPORT_FRAMEWORKS = ["FlashDet", "Ultralytics", "YOLOX"]

FRAMEWORK_NOTES = {
    "FlashDet": (
        "FlashDet deployment pipeline: layer fusion, ONNX export with parity, "
        "INT8 quantization, and benchmarking."
    ),
    "Ultralytics": (
        "Ultralytics deployment: ONNX, TorchScript, TensorRT engine export, "
        "and speed benchmarking."
    ),
    "YOLOX": (
        "YOLOX checkpoints use the FlashDet deployment pipeline. Training is not "
        "implemented by the installed FlashDet build, but inference and export work."
    ),
}

_ARCH_CACHE: dict[tuple[str, float], str] = {}


def _cache_dir() -> str:
    path = os.path.join(os.path.expanduser("~"), ".flashstudio", "weights")
    os.makedirs(path, exist_ok=True)
    return path


def checkpoint_architecture(path: str) -> str:
    """Read the architecture label from a checkpoint (cached by mtime)."""
    if not path or not os.path.isfile(path):
        return ""
    if path.lower().endswith(".pt"):
        return "ultralytics"
    try:
        key = (os.path.abspath(path), os.path.getmtime(path))
    except OSError:
        return ""
    if key in _ARCH_CACHE:
        return _ARCH_CACHE[key]

    arch = ""
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        if isinstance(cfg, dict):
            arch = str(cfg.get("architecture") or "flashdet").lower()
    except Exception:
        arch = ""
    _ARCH_CACHE[key] = arch
    return arch


def _scan(root: str) -> list[str]:
    found = []
    if not root or not os.path.isdir(root):
        return found
    for dirpath, dirs, files in os.walk(root):
        if (
            os.path.basename(dirpath) == "weights"
            and os.path.basename(os.path.dirname(dirpath)) == "ultralytics"
        ):
            continue
        dirs[:] = [d for d in dirs if d not in ("yolo_dataset", "optimized")]
        for name in files:
            if name.endswith((".pth", ".pt")):
                found.append(os.path.join(dirpath, name))
    return found


def framework_weights(framework: str, root: str | None = None) -> list[str]:
    """Return checkpoints that belong to *framework*, newest first."""
    framework = str(framework or "FlashDet")
    roots = []
    if root:
        roots.append(root)
    roots.append(_cache_dir())

    paths = []
    for base in roots:
        for path in _scan(base):
            if path not in paths:
                paths.append(path)

    selected = []
    for path in paths:
        if path.lower().endswith(".pt"):
            if framework == "Ultralytics":
                selected.append(path)
            continue
        arch = checkpoint_architecture(path)
        if framework == "YOLOX":
            if arch == "yolox":
                selected.append(path)
        elif framework == "Ultralytics":
            continue
        else:
            if arch != "yolox":
                selected.append(path)

    selected.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return selected


def weights_with_mtime(framework: str, root: str | None = None) -> list[tuple[str, float]]:
    return [
        (path, os.path.getmtime(path))
        for path in framework_weights(framework, root)
        if os.path.isfile(path)
    ]
