"""Tests for flashstudio.utils.training_hooks."""

import os
import json
import pytest
import tempfile

from flashstudio.utils.training_hooks import _make_palette, run_gt_verification

try:
    from flashstudio.utils.training_hooks import StudioCSVLogger, StudioVisualizationCallback
    _HAS_HOOKS = True
except ImportError:
    _HAS_HOOKS = False


class TestMakePalette:
    def test_returns_dict_with_n_keys(self):
        palette = _make_palette(3)
        assert isinstance(palette, dict)
        assert len(palette) == 3

    def test_values_are_bgr_tuples(self):
        palette = _make_palette(4)
        for key, color in palette.items():
            assert isinstance(color, tuple)
            assert len(color) == 3
            for channel in color:
                assert 0 <= channel <= 255

    def test_zero_returns_at_least_one_key(self):
        palette = _make_palette(0)
        assert isinstance(palette, dict)
        assert len(palette) >= 1

    def test_five_has_five_keys(self):
        palette = _make_palette(5)
        assert len(palette) == 5


class TestRunGtVerification:
    def test_creates_verification_outputs(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        val_dir = tmp_path / "val"
        val_dir.mkdir()
        save_dir = tmp_path / "output"
        save_dir.mkdir()

        coco_json = {
            "categories": [{"id": 1, "name": "cat"}],
            "images": [{"id": 1, "file_name": "img.jpg"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]}],
        }
        ann_path = train_dir / "_annotations.coco.json"
        ann_path.write_text(json.dumps(coco_json))

        run_gt_verification(
            train_images=str(train_dir),
            val_images=str(val_dir),
            save_dir=str(save_dir),
            class_names=["cat"],
        )

        gt_dir = save_dir / "gt_verification"
        assert gt_dir.exists()
        assert (gt_dir / "verification_report.json").exists()
        assert (gt_dir / "verification_summary.txt").exists()

        with open(gt_dir / "verification_report.json") as f:
            report = json.load(f)
        assert report["passed"] is True
        assert report["num_classes"] == 1
        assert "train" in report["splits"]


@pytest.mark.skipif(not _HAS_HOOKS, reason="flashdet not installed")
class TestStudioCSVLogger:
    def test_creates_csv_on_epoch_end(self, tmp_path):
        from unittest.mock import MagicMock

        logger = StudioCSVLogger(save_dir=str(tmp_path))
        trainer = MagicMock()
        metrics = {"train_loss": 5.0, "lr": 0.001}

        logger.on_epoch_end(trainer, epoch=1, metrics=metrics)

        assert os.path.isfile(logger.csv_path)
        with open(logger.csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        header = lines[0].strip()
        assert "epoch" in header
        assert "train_loss" in header
        assert "lr" in header


@pytest.mark.skipif(not _HAS_HOOKS, reason="flashdet not installed")
class TestStudioVisualizationCallback:
    def test_init_creates_vis_dir(self, tmp_path):
        cb = StudioVisualizationCallback(save_dir=str(tmp_path), max_kept=5)
        assert os.path.isdir(cb.vis_dir)

    def test_cleanup_keeps_only_max_kept(self, tmp_path):
        cb = StudioVisualizationCallback(save_dir=str(tmp_path), max_kept=3)

        for i in range(6):
            fpath = os.path.join(cb.vis_dir, f"epoch{i:04d}.jpg")
            with open(fpath, "w") as f:
                f.write("fake")

        cb._cleanup()

        remaining = [f for f in os.listdir(cb.vis_dir) if f.endswith(".jpg")]
        assert len(remaining) <= 3
