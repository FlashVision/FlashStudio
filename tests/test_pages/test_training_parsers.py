"""Tests for training page parsers and utilities — CSV/log parsing, file type detection."""

import os
import csv
import pytest


class TestParseTrainingCSV:
    def test_parse_valid_csv(self, sample_training_csv, tmp_dir):
        from flashstudio.pages.training.monitor.parsers import _parse_training_csv
        history = _parse_training_csv(tmp_dir)
        assert history is not None
        assert len(history["epochs"]) == 5
        assert len(history["train_loss"]) == 5
        assert history["train_loss"][0] == pytest.approx(5.4321, abs=0.001)

    def test_parse_val_epochs(self, sample_training_csv, tmp_dir):
        from flashstudio.pages.training.monitor.parsers import _parse_training_csv
        history = _parse_training_csv(tmp_dir)
        assert history is not None
        assert len(history["val_epochs"]) == 2
        assert len(history["mAP50"]) == 2

    def test_nonexistent_csv(self, tmp_dir):
        from flashstudio.pages.training.monitor.parsers import _parse_training_csv
        result = _parse_training_csv(tmp_dir)
        assert result is None

    def test_empty_csv(self, tmp_dir):
        from flashstudio.pages.training.monitor.parsers import _parse_training_csv
        csv_path = os.path.join(tmp_dir, "training_log.csv")
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,lr\n")
        result = _parse_training_csv(tmp_dir)
        assert result is None


class TestParseTrainingLog:
    def test_parse_valid_log(self, sample_training_log):
        from flashstudio.pages.training.monitor.parsers import _parse_training_log
        history = _parse_training_log(sample_training_log)
        assert history is not None
        assert history["model_info"] == "FlashDetN (320x320)"
        assert history["device"] == "cuda"
        assert history["total_epochs"] == 100
        assert len(history["val_loss"]) == 2
        assert len(history["mAP50"]) == 2

    def test_parse_classes(self, sample_training_log):
        from flashstudio.pages.training.monitor.parsers import _parse_training_log
        history = _parse_training_log(sample_training_log)
        assert history is not None
        assert len(history["classes"]) == 3
        assert "cat" in history["classes"]

    def test_nonexistent_log(self):
        from flashstudio.pages.training.monitor.parsers import _parse_training_log
        result = _parse_training_log("/nonexistent/log.log")
        assert result is None


class TestFileType:
    def test_best_inference(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert "Best" in _file_type("model_best_inference.pth")

    def test_final_fp16(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert "FP16" in _file_type("model_final_fp16.pth")

    def test_csv(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert "CSV" in _file_type("training_log.csv")

    def test_log(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert "log" in _file_type("train_20240101.log").lower()

    def test_onnx(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert "ONNX" in _file_type("model.onnx")

    def test_unknown(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        result = _file_type("random_file.xyz")
        assert isinstance(result, str)


class TestRunMeta:
    def test_empty_dir(self, tmp_dir):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta
        meta = _get_run_meta(tmp_dir)
        assert meta["status"] == "Empty"
        assert meta["epochs"] == "?"

    def test_dir_with_log(self, tmp_dir, sample_training_log):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta
        meta = _get_run_meta(tmp_dir)
        assert meta["status"] in ("Started", "In Progress", "Empty")
        assert meta["model"] == "FlashDetN (320x320)"

    def test_dir_with_csv(self, tmp_dir, sample_training_csv):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta
        meta = _get_run_meta(tmp_dir)
        assert meta["epochs"] == 5
        assert meta["mAP"] == pytest.approx(0.25, abs=0.01)
