"""Tests for training launch dialogs."""

import os

from flashstudio.pages.training.launch.dialogs import _cleanup_run_keep_best


def test_cleanup_run_keep_best_importable():
    assert _cleanup_run_keep_best is not None


def test_cleanup_run_keep_best_callable():
    assert callable(_cleanup_run_keep_best)


def test_cleanup_keeps_best_and_csv_and_log(tmp_dir):
    open(os.path.join(tmp_dir, "checkpoint_best.pth"), "w").close()
    open(os.path.join(tmp_dir, "training_log.csv"), "w").close()
    open(os.path.join(tmp_dir, "checkpoint_last.pth"), "w").close()
    open(os.path.join(tmp_dir, "random.pth"), "w").close()
    open(os.path.join(tmp_dir, "train_20250101.log"), "w").close()

    removed = _cleanup_run_keep_best(tmp_dir)

    assert os.path.isfile(os.path.join(tmp_dir, "checkpoint_best.pth"))
    assert os.path.isfile(os.path.join(tmp_dir, "training_log.csv"))
    assert os.path.isfile(os.path.join(tmp_dir, "train_20250101.log"))
    assert not os.path.isfile(os.path.join(tmp_dir, "checkpoint_last.pth"))
    assert not os.path.isfile(os.path.join(tmp_dir, "random.pth"))
    assert removed >= 2


def test_cleanup_keeps_model_onnx(tmp_dir):
    open(os.path.join(tmp_dir, "model.onnx"), "w").close()
    open(os.path.join(tmp_dir, "checkpoint_last.pth"), "w").close()

    _cleanup_run_keep_best(tmp_dir)

    assert os.path.isfile(os.path.join(tmp_dir, "model.onnx"))


def test_cleanup_removes_visualizations_dir(tmp_dir):
    viz_dir = os.path.join(tmp_dir, "visualizations")
    os.makedirs(viz_dir)
    open(os.path.join(viz_dir, "img.png"), "w").close()

    removed = _cleanup_run_keep_best(tmp_dir)

    assert not os.path.isdir(viz_dir)
    assert removed >= 1


def test_cleanup_empty_dir_returns_zero(tmp_dir):
    removed = _cleanup_run_keep_best(tmp_dir)
    assert removed == 0
