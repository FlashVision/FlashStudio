"""Weight discovery and download helpers (FlashDet registry + Ultralytics zoo).

All downloads are cached under ``~/.flashstudio/weights`` so inference,
training, and export share the same checkpoints.
"""

from __future__ import annotations

import os
import shutil

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".flashstudio", "weights")

YOLO_SERIES = ["YOLOv8", "YOLOv9", "YOLOv10", "YOLOv11"]
YOLO_SIZES = ["n", "s", "m", "l", "x"]

# Ultralytics width/depth multipliers for each size.
YOLO_SIZE_MULT = {
    "n": (0.25, 0.33),
    "s": (0.50, 0.33),
    "m": (0.75, 0.67),
    "l": (1.00, 1.00),
    "x": (1.25, 1.33),
}


def cache_dir() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def flashdet_weights() -> list[dict]:
    """Return FlashDet registry entries as display-friendly dicts."""
    try:
        from flashdet.hub import WEIGHTS

        return [
            {
                "name": name,
                "description": item.get("description", ""),
                "num_classes": item.get("num_classes"),
                "exists": os.path.isfile(
                    os.path.join(cache_dir(), f"flashdet_{name}.pth")
                ),
            }
            for name, item in sorted(WEIGHTS.items())
        ]
    except Exception:
        return []


def download_flashdet(name: str, force: bool = False) -> str:
    """Download a FlashDet registry checkpoint into the shared cache."""
    from flashdet.hub import download_weights

    return download_weights(name, save_dir=cache_dir(), force=force)


def ultralytics_weight_name(family: str, size: str = "n") -> str:
    """Map a YOLO series/size to the Ultralytics checkpoint stem."""
    family = str(family or "")
    size = str(size or "n").lower()
    if size not in YOLO_SIZES:
        size = "n"
    if "YOLOv9" in family:
        return "yolov9e" if size == "x" else "yolov9c"
    if "YOLOv10" in family:
        return f"yolov10{size}"
    if "YOLOv11" in family:
        return f"yolo11{size}"
    return f"yolov8{size}"


def ultralytics_weights() -> list[dict]:
    """Return common Ultralytics checkpoints with cache state."""
    entries = []
    for family in YOLO_SERIES:
        for size in YOLO_SIZES:
            stem = ultralytics_weight_name(family, size)
            path = os.path.join(cache_dir(), f"{stem}.pt")
            entries.append({
                "name": stem,
                "label": f"{family} {size.upper()}",
                "exists": os.path.isfile(path),
            })
    return entries


def download_ultralytics(name: str, force: bool = False) -> str:
    """Download an Ultralytics checkpoint into the shared cache."""
    target = os.path.join(cache_dir(), os.path.basename(name))
    if os.path.isfile(target) and not force:
        return target
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install it with `pip install ultralytics`."
        ) from exc

    existing = os.path.isfile(name)
    model = YOLO(name)
    source = str(getattr(model, "ckpt_path", "") or "")
    if not source or not os.path.isfile(source):
        source = name
    if not os.path.isfile(source):
        raise RuntimeError(f"Ultralytics did not produce a checkpoint for {name}.")
    shutil.copy2(source, target)
    if not existing and os.path.abspath(source) == os.path.abspath(name):
        try:
            os.remove(name)
        except OSError:
            pass
    return target


def resolve_training_checkpoint(framework: str, family: str, size: str) -> str:
    """Download the appropriate starter checkpoint for a framework."""
    if framework == "FlashDet":
        entries = flashdet_weights()
        if not entries:
            raise RuntimeError("No FlashDet registry checkpoints are available.")
        return download_flashdet(entries[0]["name"])
    if framework == "Ultralytics":
        return download_ultralytics(ultralytics_weight_name(family, size))
    raise RuntimeError(f"No pretrained checkpoint registry for {framework}.")
