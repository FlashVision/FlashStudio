"""Tests for flashstudio.pages.training.monitor.run_meta — run metadata extraction."""

import os
import pytest


class TestGetRunMeta:
    def test_empty_dir(self, tmp_dir):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta

        meta = _get_run_meta(tmp_dir)
        assert meta["status"] == "Empty"
        assert meta["epochs"] == "?"
        assert meta["mAP"] is None

    def test_has_log(self, tmp_dir, sample_training_log):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta

        meta = _get_run_meta(tmp_dir)
        assert meta["status"] in ("Started", "In Progress", "Empty")
        assert meta["model"] == "FlashDetN (320x320)"

    def test_has_csv(self, tmp_dir, sample_training_csv):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta

        meta = _get_run_meta(tmp_dir)
        assert meta["epochs"] == 5
        assert meta["mAP"] == pytest.approx(0.25, abs=0.01)

    def test_complete_with_final(self, tmp_dir):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta

        with open(os.path.join(tmp_dir, "model_final_inference.pth"), "w") as f:
            f.write("final")
        meta = _get_run_meta(tmp_dir)
        assert meta["status"] == "Complete"

    def test_complete_with_best_and_last(self, tmp_dir):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta

        with open(os.path.join(tmp_dir, "checkpoint_best.pth"), "w") as f:
            f.write("best")
        with open(os.path.join(tmp_dir, "checkpoint_last.pth"), "w") as f:
            f.write("last")
        import time
        log = os.path.join(tmp_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
        with open(log, "w") as f:
            f.write("line1\nline2\nline3\nTraining Complete!\n")
        meta = _get_run_meta(tmp_dir)
        assert meta["status"] == "Complete"

    def test_in_progress(self, tmp_dir):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta

        with open(os.path.join(tmp_dir, "checkpoint_last.pth"), "w") as f:
            f.write("last")
        with open(os.path.join(tmp_dir, "training_log.csv"), "w") as f:
            f.write("epoch,train_loss\n1,5.0\n")
        meta = _get_run_meta(tmp_dir)
        assert meta["status"] == "In Progress"

    def test_display_name(self, tmp_dir, sample_training_csv):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta

        meta = _get_run_meta(tmp_dir)
        assert meta["display_name"]
        assert os.path.basename(tmp_dir) in meta["display_name"]

    def test_size_formatted(self, tmp_dir):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta

        with open(os.path.join(tmp_dir, "data.bin"), "w") as f:
            f.write("x" * 1000)
        meta = _get_run_meta(tmp_dir)
        assert meta["size"]

    def test_date_populated(self, tmp_dir):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta

        meta = _get_run_meta(tmp_dir)
        assert meta["date"]
