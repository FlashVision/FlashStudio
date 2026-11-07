"""Tests for live training progress estimation."""

import os

from flashstudio.pages.training.progress import (
    _training_progress,
    format_eta,
    format_percent,
)


def _write_log(run_dir, body):
    path = os.path.join(run_dir, "train_1.log")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def test_progress_uses_in_flight_batch(tmp_path):
    run_dir = str(tmp_path)
    _write_log(
        run_dir,
        "Epochs: 10, Batch: 16, LR: 0.001\n"
        "Epoch 1/10 (lr=0.1)\n"
        "Epoch [1] Batch [10/100] Loss: 5\n"
        "Epoch 2/10 (lr=0.1)\n"
        "Epoch [2] Batch [50/100] Loss: 4\n"
        "Epoch 3/10 (lr=0.1)\n"
        "Epoch [3] Batch [25/100] Loss: 4\n",
    )
    with open(os.path.join(run_dir, "training_log.csv"), "w") as fh:
        fh.write("epoch,lr,train_loss\n1,0.1,5\n2,0.1,4\n")

    info = _training_progress(run_dir)
    assert info is not None
    # (2 completed epochs + 0.25 of epoch 3) / 10 epochs
    assert info["fraction"] == 0.225
    assert info["percent"] == "22%"
    assert info["epoch"] == 3
    assert info["total_epochs"] == 10
    assert info["batches_per_epoch"] == 100
    assert "batch 25/100" in info["detail"]


def test_progress_uses_compact_batch_lines(tmp_path):
    run_dir = str(tmp_path)
    _write_log(
        run_dir,
        "Epoch 1/10 (lr=0.001)\n"
        "[2026-11-07 21:21:47] [INFO]   [50/100] Loss: 4.0\n",
    )
    info = _training_progress(run_dir)
    assert info is not None
    assert info["fraction"] == 0.05
    assert info["epoch"] == 1
    assert info["batches_per_epoch"] == 100
    assert "batch 50/100" in info["detail"]


def test_progress_reports_complete(tmp_path):
    run_dir = str(tmp_path)
    _write_log(
        run_dir,
        "Epochs: 4, Batch: 8, LR: 0.001\n"
        "Epoch 4/4 (lr=0.1)\n"
        "Val Loss: 1.0 | mAP@0.5: 0.5\n"
        "Training Complete!\n",
    )
    info = _training_progress(run_dir)
    assert info is not None
    assert info["complete"] is True
    assert info["fraction"] == 1.0
    assert info["percent"] == "100%"


def test_progress_none_without_artifacts(tmp_path):
    assert _training_progress(str(tmp_path)) is None


def test_percent_and_eta_formatting():
    assert format_percent(0.005) == "0.5%"
    assert format_percent(0.125) == "12%"
    assert format_percent(1.0) == "100%"
    assert format_eta(30) == "~30s left"
    assert format_eta(600) == "~10m left"
    assert format_eta(7200) == "~2.0h left"
    assert format_eta(None) == ""
