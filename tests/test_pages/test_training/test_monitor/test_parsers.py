"""Tests for training monitor parsers."""

import os

from flashstudio.pages.training.monitor.parsers import (
    _find_log_file,
    _parse_training_csv,
    _parse_training_log,
)


def test_find_log_file_returns_largest_or_none(tmp_dir):
    # No logs → None
    assert _find_log_file(tmp_dir) is None

    # Create two log files with different sizes
    log1 = os.path.join(tmp_dir, "train_20250101_000000.log")
    log2 = os.path.join(tmp_dir, "train_20250102_000000.log")
    with open(log1, "w") as f:
        f.write("short")
    with open(log2, "w") as f:
        f.write("this is a much longer log file content for testing")

    result = _find_log_file(tmp_dir)
    assert result == log2


def test_parse_training_csv_with_fixture(sample_training_csv, tmp_dir):
    history = _parse_training_csv(tmp_dir)
    assert history is not None
    assert "epochs" in history
    assert len(history["epochs"]) == 5
    assert "train_loss" in history
    assert len(history["train_loss"]) == 5
    # mAP50 should have values from validation epochs
    assert "mAP50" in history
    mAP_vals = [v for v in history["mAP50"] if v is not None]
    assert len(mAP_vals) >= 2


def test_parse_training_log_with_fixture(sample_training_log):
    history = _parse_training_log(sample_training_log)
    assert history is not None
    assert history["model_info"] != ""
    assert "device" in history
    assert history["device"] == "cuda"
    assert len(history["classes"]) == 3


def test_parse_training_csv_nonexistent(tmp_dir):
    result = _parse_training_csv(os.path.join(tmp_dir, "nonexistent"))
    assert result is None


def test_parse_training_log_nonexistent():
    result = _parse_training_log("/nonexistent/path/train.log")
    assert result is None
