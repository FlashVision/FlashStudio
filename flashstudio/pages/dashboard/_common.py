"""Shared imports for the dashboard page."""

from flashstudio.constants import (  # noqa: F401
    IMG_EXTENSIONS, DEFAULT_SAVE_DIR, MAX_DISPLAY_RUNS,
    CKPT_BEST, CKPT_BEST_INFERENCE, CKPT_FINAL_INFERENCE,
    CKPT_FINAL_FP16, TRAINING_LOG_CSV,
    COLOR_SUCCESS, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED, COLOR_BORDER, COLOR_BG_HIGHLIGHT,
    GPU_NAME_TRUNCATE, DATASET_NAME_TRUNCATE,
)


def model_display_name() -> str:
    """Return the user-facing model name for the active framework."""
    from flashstudio.utils import get_state

    framework = str(get_state("model_framework") or "FlashDet")
    if framework == "Ultralytics":
        series = str(get_state("yolo_series") or "YOLOv8")
        size = str(get_state("yolo_size") or "n").upper()
        return f"{series}-{size}"
    if framework == "YOLOX":
        return "YOLOX"
    arch = str(get_state("model_arch") or "FlashDet-Pico")
    return arch.replace("FlashDet-", "")
