"""Tests for flashstudio.pages.training.launch.dialogs — cleanup logic."""

import os


class TestCleanupRunKeepBest:
    def test_removes_non_essential(self, tmp_dir):
        from flashstudio.pages.training.launch.dialogs import _cleanup_run_keep_best

        with open(os.path.join(tmp_dir, "checkpoint_best.pth"), "w") as f:
            f.write("best")
        with open(os.path.join(tmp_dir, "training_log.csv"), "w") as f:
            f.write("csv")
        with open(os.path.join(tmp_dir, "checkpoint_last.pth"), "w") as f:
            f.write("last")
        with open(os.path.join(tmp_dir, "random_file.pth"), "w") as f:
            f.write("random")
        with open(os.path.join(tmp_dir, "train_20240101.log"), "w") as f:
            f.write("log")

        removed = _cleanup_run_keep_best(tmp_dir)
        assert removed >= 2
        assert os.path.isfile(os.path.join(tmp_dir, "checkpoint_best.pth"))
        assert os.path.isfile(os.path.join(tmp_dir, "training_log.csv"))
        assert os.path.isfile(os.path.join(tmp_dir, "train_20240101.log"))
        assert not os.path.isfile(os.path.join(tmp_dir, "checkpoint_last.pth"))
        assert not os.path.isfile(os.path.join(tmp_dir, "random_file.pth"))

    def test_removes_vis_directories(self, tmp_dir):
        from flashstudio.pages.training.launch.dialogs import _cleanup_run_keep_best

        vis_dir = os.path.join(tmp_dir, "visualizations")
        os.makedirs(vis_dir)
        with open(os.path.join(vis_dir, "vis.jpg"), "w") as f:
            f.write("x")

        removed = _cleanup_run_keep_best(tmp_dir)
        assert removed >= 1
        assert not os.path.isdir(vis_dir)

    def test_keeps_onnx_files(self, tmp_dir):
        from flashstudio.pages.training.launch.dialogs import _cleanup_run_keep_best

        onnx = os.path.join(tmp_dir, "model.onnx")
        with open(onnx, "w") as f:
            f.write("model")

        _cleanup_run_keep_best(tmp_dir)
        assert os.path.isfile(onnx)

    def test_empty_dir(self, tmp_dir):
        from flashstudio.pages.training.launch.dialogs import _cleanup_run_keep_best

        removed = _cleanup_run_keep_best(tmp_dir)
        assert removed == 0
