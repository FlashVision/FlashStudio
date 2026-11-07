"""Tests for training launch dialogs — _cleanup_run_keep_best edge cases."""

import os
import pytest
from flashstudio.constants import (
    CKPT_BEST, CKPT_BEST_INFERENCE, CKPT_BEST_FP16,
    CKPT_LAST, CKPT_LAST_INFERENCE,
    CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16,
    TRAINING_LOG_CSV, ONNX_MODEL_FILE, ONNX_DATA_FILE,
    VIS_DIR_NAMES, GT_VERIFICATION_DIR, TRAINING_LOG_GLOB,
)
from flashstudio.pages.training.launch.dialogs import _cleanup_run_keep_best


class TestCleanupKeepBestEdgeCases:
    @pytest.fixture()
    def populated_run(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        for ckpt in (CKPT_BEST, CKPT_BEST_INFERENCE, CKPT_BEST_FP16,
                     CKPT_LAST, CKPT_LAST_INFERENCE,
                     CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16):
            (rd / ckpt).write_bytes(b"\x00")
        (rd / TRAINING_LOG_CSV).write_text("epoch,loss\n1,5.0\n")
        (rd / ONNX_MODEL_FILE).write_bytes(b"\x00")
        (rd / ONNX_DATA_FILE).write_bytes(b"\x00")
        (rd / "train_20260101.log").write_text("log")
        (rd / "config.yaml").write_text("epochs: 10")
        (rd / "results.json").write_text("{}")
        (rd / "checkpoint_epoch_50.pth").write_bytes(b"\x00")

        vis = rd / VIS_DIR_NAMES[0]
        vis.mkdir()
        (vis / "epoch0001.jpg").write_bytes(b"\xff")

        gt = rd / GT_VERIFICATION_DIR
        gt.mkdir()
        (gt / "report.json").write_text("{}")

        plots = rd / VIS_DIR_NAMES[1]
        plots.mkdir()
        (plots / "curve.png").write_bytes(b"\xff")

        return str(rd)

    def test_keeps_best_checkpoint(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        assert os.path.isfile(os.path.join(populated_run, CKPT_BEST))

    def test_keeps_best_inference(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        assert os.path.isfile(os.path.join(populated_run, CKPT_BEST_INFERENCE))

    def test_keeps_best_fp16(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        assert os.path.isfile(os.path.join(populated_run, CKPT_BEST_FP16))

    def test_keeps_final_inference(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        assert os.path.isfile(os.path.join(populated_run, CKPT_FINAL_INFERENCE))

    def test_keeps_final_fp16(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        assert os.path.isfile(os.path.join(populated_run, CKPT_FINAL_FP16))

    def test_keeps_csv(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        assert os.path.isfile(os.path.join(populated_run, TRAINING_LOG_CSV))

    def test_keeps_onnx(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        assert os.path.isfile(os.path.join(populated_run, ONNX_MODEL_FILE))
        assert os.path.isfile(os.path.join(populated_run, ONNX_DATA_FILE))

    def test_keeps_log_files(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        import glob
        logs = glob.glob(os.path.join(populated_run, TRAINING_LOG_GLOB))
        assert len(logs) >= 1

    def test_removes_last_checkpoint(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        assert not os.path.isfile(os.path.join(populated_run, CKPT_LAST))
        assert not os.path.isfile(os.path.join(populated_run, CKPT_LAST_INFERENCE))

    def test_removes_intermediate_checkpoints(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        assert not os.path.isfile(os.path.join(populated_run, "checkpoint_epoch_50.pth"))

    def test_removes_config_yaml(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        assert not os.path.isfile(os.path.join(populated_run, "config.yaml"))

    def test_removes_results_json(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        assert not os.path.isfile(os.path.join(populated_run, "results.json"))

    def test_removes_visualizations_dir(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        assert not os.path.isdir(os.path.join(populated_run, VIS_DIR_NAMES[0]))

    def test_removes_gt_verification_dir(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        assert not os.path.isdir(os.path.join(populated_run, GT_VERIFICATION_DIR))

    def test_removes_plots_dir(self, populated_run):
        _cleanup_run_keep_best(populated_run)
        assert not os.path.isdir(os.path.join(populated_run, VIS_DIR_NAMES[1]))

    def test_returns_removed_count(self, populated_run):
        removed = _cleanup_run_keep_best(populated_run)
        assert removed > 0

    def test_empty_run_dir(self, tmp_path):
        rd = tmp_path / "empty"
        rd.mkdir()
        removed = _cleanup_run_keep_best(str(rd))
        assert removed == 0

    def test_only_best_present(self, tmp_path):
        rd = tmp_path / "minimal"
        rd.mkdir()
        (rd / CKPT_BEST).write_bytes(b"\x00")
        (rd / TRAINING_LOG_CSV).write_text("epoch\n")
        removed = _cleanup_run_keep_best(str(rd))
        assert removed == 0
        assert os.path.isfile(os.path.join(str(rd), CKPT_BEST))
