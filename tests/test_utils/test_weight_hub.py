"""Weight hub tests."""

from flashstudio.utils.weight_hub import (
    YOLO_SIZE_MULT,
    flashdet_weights,
    ultralytics_weight_name,
    ultralytics_weights,
)


def test_ultralytics_weight_names():
    assert ultralytics_weight_name("YOLOv8", "n") == "yolov8n"
    assert ultralytics_weight_name("YOLOv10", "m") == "yolov10m"
    assert ultralytics_weight_name("YOLOv11", "x") == "yolo11x"
    assert ultralytics_weight_name("YOLOv9", "x") == "yolov9e"
    assert ultralytics_weight_name("YOLOv9", "s") == "yolov9c"


def test_yolo_size_multipliers_cover_all_sizes():
    assert set(YOLO_SIZE_MULT) == {"n", "s", "m", "l", "x"}
    assert YOLO_SIZE_MULT["n"][0] < YOLO_SIZE_MULT["x"][0]


def test_registries_return_entries():
    assert isinstance(flashdet_weights(), list)
    ul = ultralytics_weights()
    assert len(ul) == 20
    assert {entry["name"] for entry in ul} >= {"yolov8n", "yolo11s", "yolov10m"}
    assert all("exists" in entry for entry in ul)
