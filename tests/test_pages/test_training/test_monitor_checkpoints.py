"""Tests for flashstudio.pages.training.monitor.checkpoints — file type classification."""

import pytest


class TestFileType:
    def test_best_inference(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert "Best" in _file_type("model_best_inference.pth")
        assert "inference" in _file_type("model_best_inference.pth").lower()

    def test_best_checkpoint(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert "Best" in _file_type("checkpoint_best.pth")

    def test_final_fp16(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        result = _file_type("model_final_fp16.pth")
        assert "FP16" in result or "Final" in result

    def test_final_inference(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        result = _file_type("model_final_inference.pth")
        assert "Final" in result
        assert "inference" in result.lower()

    def test_last_checkpoint(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        result = _file_type("checkpoint_last.pth")
        assert "Latest" in result or "last" in result.lower()

    def test_last_inference(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        result = _file_type("model_last_inference.pth")
        assert "Latest" in result or "inference" in result.lower()

    def test_last_fp16(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        result = _file_type("model_last_fp16.pth")
        assert "FP16" in result or "Latest" in result

    def test_csv(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert "CSV" in _file_type("training_log.csv")

    def test_log(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert "log" in _file_type("train_20240101.log").lower()

    def test_onnx(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert "ONNX" in _file_type("model.onnx")

    def test_onnx_data(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        result = _file_type("model.onnx.data")
        assert "ONNX" in result

    def test_results_json(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        result = _file_type("results.json")
        assert "result" in result.lower() or "Report" in result

    def test_config_json(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        result = _file_type("config.json")
        assert "Report" in result or "Config" in result

    def test_summary_txt(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert "Summary" in _file_type("summary.txt")

    def test_unknown(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        result = _file_type("random_file.xyz")
        assert result == "Other"

    def test_plain_inference(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        result = _file_type("model_inference.pth")
        assert "Inference" in result or "inference" in result.lower()

    def test_plain_fp16(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        result = _file_type("model_fp16.pth")
        assert "FP16" in result
