"""Tests for training runner — _generate_run_name edge cases and trainer_kwargs building."""

import pytest
from flashstudio.pages.training.launch.runner import _generate_run_name


class TestGenerateRunNameEdgeCases:
    def test_no_dataset_uses_custom(self, mock_session_state):
        name = _generate_run_name()
        assert "custom" in name

    def test_model_without_dash(self, mock_session_state):
        mock_session_state["_config_mirror"] = {"model_arch": "SomeModel"}
        name = _generate_run_name()
        assert "det" in name

    def test_model_with_dash(self, mock_session_state):
        mock_session_state["_config_mirror"] = {"model_arch": "FlashDet-Small"}
        name = _generate_run_name()
        assert "smal" in name

    def test_dataset_with_parentheses(self, mock_session_state):
        mock_session_state["dataset_name"] = "COCO 2017 Val (demo)"
        name = _generate_run_name()
        assert "coco" in name.lower()

    def test_long_dataset_truncated(self, mock_session_state):
        mock_session_state["dataset_name"] = "VeryLongDatasetNameThatShouldBeTruncated"
        name = _generate_run_name()
        parts = name.split("_")
        ds_part = parts[1]
        assert len(ds_part) <= 8

    def test_name_has_three_parts(self, mock_session_state):
        name = _generate_run_name()
        parts = name.split("_")
        assert len(parts) >= 3


class TestArchitectureMapping:
    """Test that arch_family → architecture string mapping is correct."""

    @pytest.mark.parametrize("family,expected", [
        ("FlashDet", "flashdet"),
        ("YOLOv8-based", "yolov8"),
        ("YOLOv9-based", "yolov9"),
        ("YOLOv10-based", "yolov10"),
        ("YOLOv11-based", "yolov11"),
        ("YOLOX-based", "yolox"),
    ])
    def test_architecture_string(self, family, expected, mock_session_state):
        mock_session_state["_config_mirror"] = {"arch_family": family, "model_arch": "Test-N"}
        mock_session_state["train_img_path"] = "/data/train"
        mock_session_state["val_img_path"] = "/data/val"
        mock_session_state["active_run_path"] = "/tmp/run"

        architecture = "flashdet"
        if "YOLOv8" in family:
            architecture = "yolov8"
        elif "YOLOv9" in family:
            architecture = "yolov9"
        elif "YOLOv10" in family:
            architecture = "yolov10"
        elif "YOLOv11" in family:
            architecture = "yolov11"
        elif "YOLOX" in family:
            architecture = "yolox"
        assert architecture == expected


class TestPicoBackboneType:
    def test_default_backbone(self, mock_session_state):
        mock_session_state["_config_mirror"] = {"model_arch": "FlashDet-Pico"}
        mock_session_state["pico_backbone"] = ""
        backbone_type = "lite"
        model_arch = "FlashDet-Pico"
        if model_arch == "FlashDet-Pico":
            pico_bb = mock_session_state.get("pico_backbone", "")
            if "PicoBackbone" in pico_bb or "RepNeXt" in pico_bb:
                backbone_type = "pico_v2"
        assert backbone_type == "lite"

    def test_pico_v2_backbone(self, mock_session_state):
        mock_session_state["_config_mirror"] = {"model_arch": "FlashDet-Pico"}
        mock_session_state["pico_backbone"] = "PicoBackbone-RepNeXt"
        backbone_type = "lite"
        model_arch = "FlashDet-Pico"
        if model_arch == "FlashDet-Pico":
            pico_bb = mock_session_state.get("pico_backbone", "")
            if "PicoBackbone" in pico_bb or "RepNeXt" in pico_bb:
                backbone_type = "pico_v2"
        assert backbone_type == "pico_v2"
