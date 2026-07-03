"""Tests for training_hooks — run_gt_verification edge cases, CSV logger multi-epoch."""

import csv
import json
import pytest
from flashstudio.utils.training_hooks import run_gt_verification, _make_palette
from flashstudio.constants import (
    GT_VERIFICATION_DIR, GT_REPORT_FILE, GT_SUMMARY_FILE, TRAINING_LOG_CSV,
)

try:
    from flashstudio.utils.training_hooks import StudioCSVLogger
    _HAS_HOOKS = True
except ImportError:
    _HAS_HOOKS = False


class TestGtVerificationEdgeCases:
    def test_no_annotation_file(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        save = tmp_path / "output"
        save.mkdir()
        run_gt_verification(str(train), "", str(save), ["cat"])
        gt_dir = save / GT_VERIFICATION_DIR
        assert gt_dir.exists()
        report = json.loads((gt_dir / GT_REPORT_FILE).read_text())
        assert report["passed"] is True
        assert report["splits"] == {}

    def test_invalid_json(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        (train / "_annotations.coco.json").write_text("not json")
        save = tmp_path / "output"
        save.mkdir()
        run_gt_verification(str(train), "", str(save), ["cat"])
        gt_dir = save / GT_VERIFICATION_DIR
        report = json.loads((gt_dir / GT_REPORT_FILE).read_text())
        assert report["passed"] is False

    def test_val_only_no_train_ann(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        val = tmp_path / "val"
        val.mkdir()
        ann = {"categories": [{"id": 0, "name": "obj"}],
               "images": [{"id": 1, "file_name": "img.jpg"}],
               "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]}]}
        (val / "_annotations.coco.json").write_text(json.dumps(ann))
        save = tmp_path / "output"
        save.mkdir()
        run_gt_verification(str(train), str(val), str(save), ["obj"])
        gt_dir = save / GT_VERIFICATION_DIR
        report = json.loads((gt_dir / GT_REPORT_FILE).read_text())
        assert "val" in report["splits"]

    def test_creates_raw_images_dir(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        ann = {"categories": [{"id": 0, "name": "cat"}],
               "images": [{"id": 1, "file_name": "img.jpg"}],
               "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 50, 50]}]}
        (train / "_annotations.coco.json").write_text(json.dumps(ann))
        save = tmp_path / "output"
        save.mkdir()
        run_gt_verification(str(train), "", str(save), ["cat"])
        raw_dir = save / GT_VERIFICATION_DIR / "images" / "raw"
        assert raw_dir.exists()

    def test_summary_file_content(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        ann = {"categories": [{"id": 0, "name": "cat"}],
               "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
               "annotations": [
                   {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]},
                   {"id": 2, "image_id": 2, "category_id": 0, "bbox": [0, 0, 20, 20]},
               ]}
        (train / "_annotations.coco.json").write_text(json.dumps(ann))
        save = tmp_path / "output"
        save.mkdir()
        run_gt_verification(str(train), "", str(save), ["cat"])
        summary = (save / GT_VERIFICATION_DIR / GT_SUMMARY_FILE).read_text()
        assert "train" in summary
        assert "2 images" in summary
        assert "2 annotations" in summary


class TestMakePaletteEdgeCases:
    def test_large_palette(self):
        palette = _make_palette(100)
        assert len(palette) == 100

    def test_one_class(self):
        palette = _make_palette(1)
        assert len(palette) == 1
        assert isinstance(palette[0], tuple)

    def test_colors_distinct(self):
        palette = _make_palette(10)
        colors = list(palette.values())
        assert len(set(colors)) > 5


@pytest.mark.skipif(not _HAS_HOOKS, reason="flashdet not installed")
class TestCSVLoggerMultiEpoch:
    def test_multiple_epochs(self, tmp_path):
        from unittest.mock import MagicMock
        logger = StudioCSVLogger(save_dir=str(tmp_path))
        trainer = MagicMock()
        for ep in range(1, 6):
            metrics = {"train_loss": 10.0 - ep, "lr": 0.001 * (0.95 ** ep)}
            if ep % 2 == 0:
                metrics["val_loss"] = 8.0 - ep
                metrics["val_mAP"] = ep * 0.1
            logger.on_epoch_end(trainer, epoch=ep, metrics=metrics)
        with open(logger.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5
        assert "epoch" in rows[0]
        assert "train_loss" in rows[0]

    def test_csv_path_uses_constant(self, tmp_path):
        logger = StudioCSVLogger(save_dir=str(tmp_path))
        assert logger.csv_path.endswith(TRAINING_LOG_CSV)

    def test_val_mAP_column_name(self, tmp_path):
        from unittest.mock import MagicMock
        logger = StudioCSVLogger(save_dir=str(tmp_path))
        trainer = MagicMock()
        metrics = {"train_loss": 5.0, "lr": 0.001, "val_mAP": 0.42}
        logger.on_epoch_end(trainer, epoch=1, metrics=metrics)
        with open(logger.csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert "mAP@0.5" in row
        assert float(row["mAP@0.5"]) == pytest.approx(0.42, abs=0.01)
