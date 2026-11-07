"""Framework-aware export checkpoint tests."""

import os

import torch

from flashstudio.pages.export._common import (
    checkpoint_architecture,
    framework_weights,
)


def _save_pth(path, architecture):
    torch.save(
        {
            "model_state_dict": {},
            "config": {"architecture": architecture, "num_classes": 2},
        },
        path,
    )


def test_checkpoint_architecture_reads_config(tmp_path):
    yolox = tmp_path / "yolox.pth"
    _save_pth(yolox, "yolox")
    assert checkpoint_architecture(str(yolox)) == "yolox"

    fd = tmp_path / "flashdet.pth"
    _save_pth(fd, "flashdet")
    assert checkpoint_architecture(str(fd)) == "flashdet"

    pt = tmp_path / "model.pt"
    pt.write_bytes(b"not-a-real-checkpoint")
    assert checkpoint_architecture(str(pt)) == "ultralytics"


def test_framework_weights_are_separated(tmp_path):
    _save_pth(tmp_path / "flashdet.pth", "flashdet")
    _save_pth(tmp_path / "yolox.pth", "yolox")
    (tmp_path / "model.pt").write_bytes(b"pt")

    flashdet = framework_weights("FlashDet", str(tmp_path))
    yolox = framework_weights("YOLOX", str(tmp_path))
    ultra = framework_weights("Ultralytics", str(tmp_path))

    assert str(tmp_path / "flashdet.pth") in flashdet
    assert str(tmp_path / "yolox.pth") not in flashdet
    assert str(tmp_path / "yolox.pth") in yolox
    assert str(tmp_path / "flashdet.pth") not in yolox
    assert str(tmp_path / "model.pt") in ultra
    assert not any(path.endswith(".pth") for path in ultra)


def test_unknown_pth_defaults_to_flashdet(tmp_path):
    path = tmp_path / "unknown.pth"
    path.write_bytes(b"not-a-checkpoint")
    assert checkpoint_architecture(str(path)) == ""
    assert str(path) in framework_weights("FlashDet", str(tmp_path))
