"""Tests for flashstudio.pages.training.monitor.parsers — CSV and log parsing."""

import os
import pytest


class TestFindLogFile:
    def test_finds_largest(self, tmp_dir):
        from flashstudio.pages.training.monitor.parsers import _find_log_file

        log1 = os.path.join(tmp_dir, "train_20240101.log")
        log2 = os.path.join(tmp_dir, "train_20240102.log")
        with open(log1, "w") as f:
            f.write("short")
        with open(log2, "w") as f:
            f.write("much longer content here for testing purposes")

        assert _find_log_file(tmp_dir) == log2

    def test_no_logs(self, tmp_dir):
        from flashstudio.pages.training.monitor.parsers import _find_log_file
        assert _find_log_file(tmp_dir) is None

    def test_ignores_non_training_logs(self, tmp_dir):
        from flashstudio.pages.training.monitor.parsers import _find_log_file

        with open(os.path.join(tmp_dir, "app.log"), "w") as f:
            f.write("app log")
        assert _find_log_file(tmp_dir) is None


class TestParseTrainingCSV:
    def test_valid_csv(self, sample_training_csv, tmp_dir):
        from flashstudio.pages.training.monitor.parsers import _parse_training_csv

        history = _parse_training_csv(tmp_dir)
        assert history is not None
        assert len(history["epochs"]) == 5
        assert len(history["train_loss"]) == 5
        assert history["train_loss"][0] == pytest.approx(5.4321, abs=0.001)

    def test_val_epochs_paired(self, sample_training_csv, tmp_dir):
        from flashstudio.pages.training.monitor.parsers import _parse_training_csv

        history = _parse_training_csv(tmp_dir)
        assert len(history["val_epochs"]) == 2
        assert 3 in history["val_epochs"]
        assert 5 in history["val_epochs"]
        assert len(history["mAP50"]) == 2

    def test_nonexistent_csv(self, tmp_dir):
        from flashstudio.pages.training.monitor.parsers import _parse_training_csv
        assert _parse_training_csv(tmp_dir) is None

    def test_header_only_csv(self, tmp_dir):
        from flashstudio.pages.training.monitor.parsers import _parse_training_csv

        csv_path = os.path.join(tmp_dir, "training_log.csv")
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,lr\n")
        assert _parse_training_csv(tmp_dir) is None

    def test_lr_parsed(self, sample_training_csv, tmp_dir):
        from flashstudio.pages.training.monitor.parsers import _parse_training_csv

        history = _parse_training_csv(tmp_dir)
        assert len(history["lr"]) == 5
        assert history["lr"][0] == pytest.approx(0.001, abs=0.0001)


class TestParseTrainingLog:
    def test_valid_log(self, sample_training_log):
        from flashstudio.pages.training.monitor.parsers import _parse_training_log

        history = _parse_training_log(sample_training_log)
        assert history is not None
        assert history["model_info"] == "FlashDetN (320x320)"
        assert history["device"] == "cuda"
        assert history["total_epochs"] == 100

    def test_classes_parsed(self, sample_training_log):
        from flashstudio.pages.training.monitor.parsers import _parse_training_log

        history = _parse_training_log(sample_training_log)
        assert len(history["classes"]) == 3
        assert "cat" in history["classes"]
        assert "dog" in history["classes"]
        assert "bird" in history["classes"]

    def test_val_metrics(self, sample_training_log):
        from flashstudio.pages.training.monitor.parsers import _parse_training_log

        history = _parse_training_log(sample_training_log)
        assert len(history["val_loss"]) == 2
        assert len(history["mAP50"]) == 2
        assert history["mAP50"][1] == pytest.approx(0.045, abs=0.001)

    def test_nonexistent_log(self):
        from flashstudio.pages.training.monitor.parsers import _parse_training_log
        assert _parse_training_log("/nonexistent/log.log") is None

    def test_none_path(self):
        from flashstudio.pages.training.monitor.parsers import _parse_training_log
        assert _parse_training_log(None) is None

    def test_batch_size_parsed(self, sample_training_log):
        from flashstudio.pages.training.monitor.parsers import _parse_training_log

        history = _parse_training_log(sample_training_log)
        assert history["batch_size"] == 16
