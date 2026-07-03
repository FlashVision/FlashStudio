"""Tests for run_meta — edge cases for status detection, metadata extraction."""

import csv
import pytest
from flashstudio.constants import (
    CKPT_BEST, CKPT_LAST, CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16, TRAINING_LOG_CSV,
)
from flashstudio.pages.training.monitor.run_meta import _get_run_meta


class TestRunMetaStatusDetection:
    def test_complete_with_final(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        (rd / CKPT_FINAL_INFERENCE).write_bytes(b"\x00")
        meta = _get_run_meta(str(rd))
        assert meta["status"] == "Complete"

    def test_complete_with_final_fp16(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        (rd / CKPT_FINAL_FP16).write_bytes(b"\x00")
        meta = _get_run_meta(str(rd))
        assert meta["status"] == "Complete"

    def test_complete_with_best_and_last_and_log(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        (rd / CKPT_BEST).write_bytes(b"\x00")
        (rd / CKPT_LAST).write_bytes(b"\x00")
        log = rd / "train_20260101.log"
        log.write_text("Training Complete!\n")
        meta = _get_run_meta(str(rd))
        assert meta["status"] == "Complete"

    def test_in_progress_with_best_and_last(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        (rd / CKPT_BEST).write_bytes(b"\x00")
        (rd / CKPT_LAST).write_bytes(b"\x00")
        log = rd / "train_20260101.log"
        log.write_text("Epoch 50/100\n")
        meta = _get_run_meta(str(rd))
        assert meta["status"] == "In Progress"

    def test_in_progress_with_last_and_log(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        (rd / CKPT_LAST).write_bytes(b"\x00")
        (rd / "train_20260101.log").write_text("Epoch 1/10\n")
        meta = _get_run_meta(str(rd))
        assert meta["status"] == "In Progress"

    def test_started_with_csv_only(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        (rd / TRAINING_LOG_CSV).write_text("epoch,train_loss\n1,5.0\n")
        meta = _get_run_meta(str(rd))
        assert meta["status"] == "Started"

    def test_started_with_log_only(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        (rd / "train_20260101.log").write_text("Starting...\n")
        meta = _get_run_meta(str(rd))
        assert meta["status"] == "Started"


class TestRunMetaMetadataExtraction:
    def test_extracts_model_from_model_size(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        (rd / "train_20260101.log").write_text("Model Size: FlashDet-Nano\n")
        meta = _get_run_meta(str(rd))
        assert "FlashDet-Nano" in meta["model"]

    def test_extracts_model_from_old_format(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        (rd / "train_20260101.log").write_text("Model: p, Input: (320, 320)\n")
        meta = _get_run_meta(str(rd))
        assert "p" in meta["model"] and "320" in meta["model"]

    def test_extracts_epochs(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        (rd / "train_20260101.log").write_text("Epochs: 200\n")
        meta = _get_run_meta(str(rd))
        assert meta["epochs"] == "200"

    def test_extracts_batch_size(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        (rd / "train_20260101.log").write_text("Batch Size: 64\n")
        meta = _get_run_meta(str(rd))
        assert meta.get("batch_size") == "64"

    def test_mAP_from_csv_fallback(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        csv_path = rd / TRAINING_LOG_CSV
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "mAP@0.5"])
            w.writeheader()
            w.writerow({"epoch": 1, "train_loss": "5.0", "mAP@0.5": "0.250"})
        meta = _get_run_meta(str(rd))
        assert meta["mAP"] == pytest.approx(0.25, abs=0.01)

    def test_mAP_from_log_tail_fallback(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        log = rd / "train_20260101.log"
        log.write_text("Epoch 1/10\n" * 5 + "Best mAP@0.5: 0.420\n")
        meta = _get_run_meta(str(rd))
        assert meta["mAP"] == pytest.approx(0.42, abs=0.01)

    def test_size_calculated(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        (rd / "weights.pth").write_bytes(b"\x00" * 1024)
        meta = _get_run_meta(str(rd))
        assert meta["size"] != ""

    def test_date_populated(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        meta = _get_run_meta(str(rd))
        assert meta["date"] != ""


class TestRunMetaDisplayName:
    def test_enriched_name_with_model(self, tmp_path):
        rd = tmp_path / "run_001"
        rd.mkdir()
        (rd / "train_20260101.log").write_text("Model Size: FlashDet-N\n")
        meta = _get_run_meta(str(rd))
        assert "|" in meta["display_name"]
        assert "run_001" in meta["display_name"]

    def test_plain_name_without_model(self, tmp_path):
        rd = tmp_path / "run_002"
        rd.mkdir()
        meta = _get_run_meta(str(rd))
        assert meta["display_name"] == "run_002"
