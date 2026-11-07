"""Inference — shared state, constants, and utility functions."""

import os

import streamlit as st
from flashstudio.constants import COCO_CLASSES

INFER_FAMILIES = ["FlashDet", "Ultralytics", "YOLOX"]

INFER_NOTES = {
    "FlashDet": "Native FlashDet `.pth` checkpoints with the FlashDet predictor.",
    "Ultralytics": "YOLO `.pt`, `.onnx`, `.engine`, and TorchScript models.",
    "YOLOX": "YOLOX `.pth` checkpoints through the FlashDet runtime. Training is not supported.",
}

INFER_RUNTIMES = {
    "FlashDet": ["Auto", "FlashDet", "TorchScript"],
    "Ultralytics": ["Auto", "Ultralytics", "ONNX Runtime", "TorchScript", "TensorRT"],
    "YOLOX": ["Auto", "FlashDet", "TorchScript"],
}

INFER_UPLOAD_TYPES = {
    "FlashDet": ["pth", "pt", "ts"],
    "Ultralytics": ["pt", "onnx", "engine", "ts"],
    "YOLOX": ["pth", "pt", "ts"],
}


def inference_checkpoints(family: str = "FlashDet", root: str | None = None) -> list[str]:
    """Return checkpoints for one framework, newest first."""
    from flashstudio.pages.export._common import framework_weights

    paths = framework_weights(family, root)
    if family == "Ultralytics":
        extra_roots = []
        if root:
            extra_roots.append(root)
        extra_roots.append(os.path.join(os.path.expanduser("~"), ".flashstudio", "weights"))
        for base in extra_roots:
            if not os.path.isdir(base):
                continue
            for dirpath, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in ("yolo_dataset", "ultralytics")]
                for name in files:
                    if name.endswith((".onnx", ".engine")):
                        path = os.path.join(dirpath, name)
                        if path not in paths:
                            paths.append(path)
        paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths


def latest_inference_weight(family: str = "FlashDet", root: str | None = None) -> str:
    paths = inference_checkpoints(family, root)
    return paths[0] if paths else ""



try:
    from flashstudio.components.zone_drawer import zone_drawer  # noqa: F401
    _HAS_ZONE_DRAWER = True
except ImportError:
    _HAS_ZONE_DRAWER = False

try:
    from flashdet import Predictor, FlashTracker  # noqa: F401
    _HAS_PREDICTOR = True
except ImportError:
    _HAS_PREDICTOR = False

_SOLUTIONS_AVAILABLE = {}
_solution_classes = [
    "ObjectCounter", "RegionCounter", "SpeedEstimator", "Heatmap",
    "SecurityAlarm", "QueueManager", "ParkingManager", "TrafficFlow",
    "DwellTimeAnalyzer", "DistanceCalculator", "TrajectoryVisualizer",
    "ObjectBlurrer", "CrowdDensity", "WorkoutMonitor", "ObjectCropper",
    "AnalyticsDashboard",
]
for _cls_name in _solution_classes:
    try:
        _mod = __import__("flashdet.solutions", fromlist=[_cls_name])
        _SOLUTIONS_AVAILABLE[_cls_name] = getattr(_mod, _cls_name)
    except (ImportError, AttributeError):
        pass


SOLUTIONS = {
    "None (Detection Only)": {"desc": "Standard object detection", "needs_zone": False},
    "Object Counter (Line)": {"desc": "Count objects crossing line", "needs_zone": True, "zone_type": "line"},
    "Region Counter (Polygon)": {"desc": "Count in polygon zones", "needs_zone": True, "zone_type": "polygon"},
    "Speed Estimator": {"desc": "Estimate speed", "needs_zone": True, "zone_type": "line"},
    "Heatmap": {"desc": "Activity heatmap", "needs_zone": False},
    "Security Alarm (Zone)": {"desc": "Alert on zone entry", "needs_zone": True, "zone_type": "polygon"},
    "Trajectory Visualizer": {"desc": "Motion trails", "needs_zone": False},
    "Object Blurrer": {"desc": "Blur detections", "needs_zone": False},
    "Queue Manager": {"desc": "Queue lengths", "needs_zone": True, "zone_type": "polygon"},
    "Crowd Density": {"desc": "Grid density", "needs_zone": False},
    "Parking Manager": {"desc": "Parking occupancy", "needs_zone": True, "zone_type": "polygon"},
    "Traffic Flow": {"desc": "Traffic analysis", "needs_zone": True, "zone_type": "line"},
    "Dwell Time Analyzer": {"desc": "Time in zones", "needs_zone": True, "zone_type": "polygon"},
    "Distance Calculator": {"desc": "Pairwise distances", "needs_zone": False},
    "Workout Monitor": {"desc": "Exercise reps", "needs_zone": False},
    "Object Cropper": {"desc": "Crop detections", "needs_zone": False},
    "Analytics Dashboard": {"desc": "Real-time stats", "needs_zone": False},
}


def _get_class_names():
    """Get class names from session state, falling back to COCO."""
    raw = st.session_state.get("class_names", "")
    if isinstance(raw, list):
        names = [c.strip() for c in raw if c.strip()]
    elif isinstance(raw, str) and raw.strip():
        names = [c.strip() for c in raw.strip().split("\n") if c.strip()]
    else:
        names = []
    return names if names else COCO_CLASSES


def _get_device_options():
    devices = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i} ({torch.cuda.get_device_name(i)})")
    except ImportError:
        pass
    return devices
