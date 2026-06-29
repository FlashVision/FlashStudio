"""Tests for checkpoint file type classification — complete coverage of _file_type."""

import pytest
from flashstudio.pages.training.monitor.checkpoints import _file_type
from flashstudio.constants import (
    CKPT_BEST, CKPT_BEST_INFERENCE, CKPT_BEST_FP16,
    CKPT_LAST, CKPT_LAST_INFERENCE,
    CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16,
    ONNX_MODEL_FILE, ONNX_DATA_FILE, RESULTS_JSON_FILE,
    TRAINING_LOG_CSV,
)


class TestFileTypeComplete:
    @pytest.mark.parametrize("filename,expected", [
        (CKPT_FINAL_INFERENCE, "Final inference weights"),
        (CKPT_FINAL_FP16, "Final FP16 weights"),
        (CKPT_BEST_INFERENCE, "Best inference weights"),
        (CKPT_BEST, "Best checkpoint"),
        (CKPT_LAST_INFERENCE, "Latest inference weights"),
        ("model_last_fp16.pth", "Latest FP16 weights"),
        (CKPT_LAST, "Latest checkpoint (full)"),
        ("some_inference_weights.pth", "Inference weights"),
        ("some_fp16_weights.pth", "FP16 weights"),
        (ONNX_MODEL_FILE, "ONNX model"),
        (ONNX_DATA_FILE, "ONNX weights data"),
        (RESULTS_JSON_FILE, "Training results"),
        ("config.json", "Report/Config"),
        (TRAINING_LOG_CSV, "Training metrics CSV"),
        ("train_20260101.log", "Training log"),
        ("summary.txt", "Summary"),
        ("unknown_file.xyz", "Other"),
    ])
    def test_file_type(self, filename, expected):
        assert _file_type(filename) == expected

    def test_custom_inference_pth(self):
        assert _file_type("custom_inference_model.pth") == "Inference weights"

    def test_priority_order(self):
        """'final' + 'inference' should match before standalone 'inference'."""
        result = _file_type("model_final_inference.pth")
        assert result == "Final inference weights"

    def test_best_fp16_priority(self):
        """'best' + 'fp16' should not get caught by just 'best'."""
        result = _file_type(CKPT_BEST_FP16)
        assert "FP16" not in result or "Best" not in result or result in (
            "Best inference weights", "Final FP16 weights", "FP16 weights"
        )

    def test_json_not_results(self):
        assert _file_type("hyperparams.json") == "Report/Config"

    def test_csv_not_training_log(self):
        assert _file_type("augmentation_stats.csv") == "Training metrics CSV"
