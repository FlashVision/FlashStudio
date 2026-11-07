"""Model runtime adapter tests."""

import numpy as np

from flashstudio.utils.model_loader import (
    _OnnxLikeAdapter,
    _patch_torchvision_nms,
    available_frameworks,
    detect_framework,
)


class _DummyOnnx(_OnnxLikeAdapter):
    num_classes = 2
    conf = 0.25
    nms = 0.45


def test_available_frameworks_has_ids():
    items = available_frameworks()
    assert {item["id"] for item in items} >= {"FlashDet", "Ultralytics", "ONNX Runtime", "TorchScript"}
    assert all("installed" in item for item in items)


def test_detect_framework_onnx():
    assert detect_framework("model.onnx") in ("Ultralytics", "ONNX Runtime")
    assert detect_framework("model.pth") == "FlashDet"


def test_decode_six_column_boxes_and_nms():
    adapter = _DummyOnnx()
    adapter.num_classes = 3  # disambiguates 6 columns as xyxy+score+cls
    # [x1, y1, x2, y2, score, class]
    raw = np.array([[
        [10, 10, 20, 20, 0.90, 0],
        [11, 11, 20, 20, 0.80, 0],  # suppressed
        [30, 30, 40, 40, 0.85, 1],
    ]], dtype=np.float32)
    out = adapter._decode(raw, scale=1.0, pad_x=0.0, pad_y=0.0)
    assert len(out) == 2
    assert {int(item[2]) for item in out} == {0, 1}


def test_decode_yolo_class_scores():
    adapter = _DummyOnnx()
    # cx, cy, w, h, then two class scores
    raw = np.array([
        [10, 10, 4, 4, 0.9, 0.1],
        [30, 30, 4, 4, 0.1, 0.8],
    ], dtype=np.float32)
    out = adapter._decode(raw, scale=1.0, pad_x=0.0, pad_y=0.0)
    assert len(out) == 2
    assert sorted(int(item[2]) for item in out) == [0, 1]


def test_nms_patch_does_not_crash():
    _patch_torchvision_nms()
