"""Edge case tests for training monitor parsers."""

import os
import csv
import pytest
from flashstudio.pages.training.monitor.parsers import (
    _find_log_file, _parse_training_csv, _parse_training_log,
)
from flashstudio.constants import TRAINING_LOG_CSV


class TestParseCsvEdgeCases:
    def test_csv_with_missing_val_columns(self, tmp_path):
        csv_path = tmp_path / TRAINING_LOG_CSV
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "lr"])
            w.writeheader()
            w.writerow({"epoch": 1, "train_loss": "5.0", "lr": "0.001"})
            w.writerow({"epoch": 2, "train_loss": "4.0", "lr": "0.0009"})
        history = _parse_training_csv(str(tmp_path))
        assert history is not None
        assert len(history["epochs"]) == 2
        assert all(v is None for v in history["mAP50"])

    def test_csv_with_val_mAP_column(self, tmp_path):
        """FlashDet uses 'val_mAP' instead of 'mAP@0.5' in some versions."""
        csv_path = tmp_path / TRAINING_LOG_CSV
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "lr", "val_mAP"])
            w.writeheader()
            w.writerow({"epoch": 1, "train_loss": "5.0", "lr": "0.001", "val_mAP": "0.1"})
        history = _parse_training_csv(str(tmp_path))
        assert history is not None
        assert history["mAP50"][0] == pytest.approx(0.1, abs=0.01)

    def test_csv_with_sub_losses(self, tmp_path):
        csv_path = tmp_path / TRAINING_LOG_CSV
        fields = ["epoch", "train_loss", "lr", "train_box", "train_cls", "train_l1"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerow({"epoch": 1, "train_loss": "5.0", "lr": "0.001",
                         "train_box": "2.0", "train_cls": "1.5", "train_l1": "1.5"})
        history = _parse_training_csv(str(tmp_path))
        assert history is not None
        assert history["train_box"][0] == pytest.approx(2.0, abs=0.01)
        assert history["train_cls"][0] == pytest.approx(1.5, abs=0.01)

    def test_csv_empty_file(self, tmp_path):
        csv_path = tmp_path / TRAINING_LOG_CSV
        csv_path.write_text("epoch,train_loss,lr\n")
        history = _parse_training_csv(str(tmp_path))
        assert history is None

    def test_csv_with_non_numeric_epoch(self, tmp_path):
        csv_path = tmp_path / TRAINING_LOG_CSV
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "lr"])
            w.writeheader()
            w.writerow({"epoch": "bad", "train_loss": "5.0", "lr": "0.001"})
            w.writerow({"epoch": "2", "train_loss": "4.0", "lr": "0.0009"})
        history = _parse_training_csv(str(tmp_path))
        assert history is not None
        assert len(history["epochs"]) == 1

    def test_csv_reads_header_from_log(self, tmp_path):
        csv_path = tmp_path / TRAINING_LOG_CSV
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "lr"])
            w.writeheader()
            w.writerow({"epoch": 1, "train_loss": "5.0", "lr": "0.001"})

        log_path = tmp_path / "train_20260101.log"
        log_path.write_text(
            "Model Size: FlashDet-Nano\n"
            "Device: cuda:0\n"
            "Epochs: 100\n"
            "Batch Size: 32\n"
            "Learning Rate: 0.001\n"
        )
        history = _parse_training_csv(str(tmp_path))
        assert history["model_info"]
        assert history["device"] == "cuda:0"
        assert history["total_epochs"] == 100
        assert history["batch_size"] == 32


class TestParseLogEdgeCases:
    def test_log_with_simple_val_format(self, tmp_path):
        log_path = tmp_path / "train_20260101.log"
        log_path.write_text(
            "Model: p, Input: (320, 320)\n"
            "Device: cpu\n"
            "Epochs: 5, Batch: 8, LR: 0.001\n"
            "Classes (2): ['cat', 'dog']\n"
            "Epoch 1/5 (lr=0.001000)\n"
            "  Val Loss: 5.00 | mAP@0.5: 0.05\n"
        )
        history = _parse_training_log(str(log_path))
        assert history is not None
        assert len(history["val_loss"]) >= 1
        assert history["val_loss"][0] == pytest.approx(5.0, abs=0.1)
        assert history["mAP50"][0] == pytest.approx(0.05, abs=0.01)

    def test_log_with_ema_decay(self, tmp_path):
        log_path = tmp_path / "train_20260101.log"
        log_path.write_text(
            "Model: n, Input: (640, 640)\n"
            "Device: cuda\n"
            "Epochs: 2, Batch: 16, LR: 0.001\n"
            "Classes (1): ['obj']\n"
            "Epoch 1/2 (lr=0.001000, ema_decay=0.99900)\n"
            "  Val Loss: 3.00 | mAP@0.5: 0.10\n"
        )
        history = _parse_training_log(str(log_path))
        assert history is not None
        assert len(history["ema_decay"]) >= 1
        assert history["ema_decay"][0] == pytest.approx(0.999, abs=0.001)

    def test_log_with_batch_level_loss(self, tmp_path):
        log_path = tmp_path / "train_20260101.log"
        log_path.write_text(
            "Model: p, Input: (320, 320)\n"
            "Device: cpu\n"
            "Epochs: 1, Batch: 4, LR: 0.001\n"
            "Classes (1): ['obj']\n"
            "Epoch 1/1 (lr=0.001000)\n"
            "Epoch [1] Batch [4/10] Loss: 5.00 (loss_total: 4.80, o2m_cls: 1.2, o2m_box: 1.5)\n"
            "  Val Loss: 3.00 | mAP@0.5: 0.10\n"
        )
        history = _parse_training_log(str(log_path))
        assert history is not None
        assert history["train_loss"][0] == pytest.approx(4.80, abs=0.01)
        assert history["o2m_cls"][0] == pytest.approx(1.2, abs=0.1)

    def test_log_with_epoch_time(self, tmp_path):
        log_path = tmp_path / "train_20260101.log"
        log_path.write_text(
            "Model: p, Input: (320, 320)\n"
            "Device: cpu\n"
            "Epochs: 1, Batch: 4, LR: 0.001\n"
            "Classes (1): ['obj']\n"
            "Epoch 1/1 (lr=0.001000)\n"
            "Epoch time: 12.5s\n"
            "  Val Loss: 3.00 | mAP@0.5: 0.10\n"
        )
        history = _parse_training_log(str(log_path))
        assert len(history["epoch_time"]) >= 1
        assert history["epoch_time"][0] == pytest.approx(12.5, abs=0.1)

    def test_empty_log(self, tmp_path):
        log_path = tmp_path / "train_20260101.log"
        log_path.write_text("")
        history = _parse_training_log(str(log_path))
        assert history is None


class TestFindLogFileEdgeCases:
    def test_prefers_larger_file(self, tmp_path):
        small = tmp_path / "train_20260101.log"
        small.write_text("short")
        large = tmp_path / "train_20260102.log"
        large.write_text("this is a much longer log" * 10)
        result = _find_log_file(str(tmp_path))
        assert result == str(large)

    def test_ignores_non_train_logs(self, tmp_path):
        (tmp_path / "debug.log").write_text("debug info")
        (tmp_path / "train_20260101.log").write_text("train info")
        result = _find_log_file(str(tmp_path))
        assert "train_" in os.path.basename(result)

    def test_empty_dir(self, tmp_path):
        assert _find_log_file(str(tmp_path)) is None
