"""Framework-aware inference checkpoint tests."""

import torch

from flashstudio.pages.inference._common import (
    INFER_FAMILIES,
    INFER_RUNTIMES,
    INFER_UPLOAD_TYPES,
    inference_checkpoints,
    latest_inference_weight,
)


def _save_pth(path, architecture):
    torch.save(
        {"model_state_dict": {}, "config": {"architecture": architecture}},
        path,
    )


def test_runtime_and_upload_options_cover_all_frameworks():
    for family in INFER_FAMILIES:
        assert family in INFER_RUNTIMES
        assert "Auto" in INFER_RUNTIMES[family]
        assert family.lower() not in ("",)
        assert INFER_UPLOAD_TYPES[family]


def test_inference_checkpoints_are_separated(tmp_path):
    _save_pth(tmp_path / "flashdet.pth", "flashdet")
    _save_pth(tmp_path / "yolox.pth", "yolox")
    (tmp_path / "yolo.pt").write_bytes(b"pt")
    (tmp_path / "yolo.onnx").write_bytes(b"onnx")

    flashdet = inference_checkpoints("FlashDet", str(tmp_path))
    yolox = inference_checkpoints("YOLOX", str(tmp_path))
    ultra = inference_checkpoints("Ultralytics", str(tmp_path))

    assert str(tmp_path / "flashdet.pth") in flashdet
    assert str(tmp_path / "yolox.pth") not in flashdet
    assert str(tmp_path / "yolox.pth") in yolox
    assert str(tmp_path / "flashdet.pth") not in yolox
    assert str(tmp_path / "yolo.pt") in ultra
    assert str(tmp_path / "yolo.onnx") in ultra


def test_latest_inference_weight_returns_empty_when_missing(tmp_path):
    assert latest_inference_weight("YOLOX", str(tmp_path)) == ""
