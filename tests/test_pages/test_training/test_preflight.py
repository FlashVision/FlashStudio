"""Tests for flashstudio.pages.training.launch.preflight — pre-flight checks."""

import os
import json
import pytest


class TestPreflightChecks:
    def test_no_data_fails(self, mock_session_state):
        from flashstudio.pages.training.launch.preflight import _run_preflight_checks

        checks = _run_preflight_checks()
        labels = {c[0]: c[1] for c in checks}
        assert labels["Dataset"] is False
        assert labels["Annotations"] is False

    def test_no_classes_fails(self, mock_session_state):
        from flashstudio.pages.training.launch.preflight import _run_preflight_checks

        checks = _run_preflight_checks()
        labels = {c[0]: c[1] for c in checks}
        assert labels["Classes"] is False

    def test_default_model_passes(self, mock_session_state):
        """When user hasn't explicitly set model_arch, the default
        (FlashDet-Pico) is used via get_state(), so Model check passes."""
        from flashstudio.pages.training.launch.preflight import _run_preflight_checks

        checks = _run_preflight_checks()
        labels = {c[0]: c[1] for c in checks}
        assert labels["Model"] is True

    def test_full_config_passes(self, mock_session_state, tmp_dir):
        from flashstudio.pages.training.launch.preflight import _run_preflight_checks

        train_dir = os.path.join(tmp_dir, "train")
        os.makedirs(train_dir)
        ann = {"categories": [{"id": 1, "name": "cat"}], "images": [], "annotations": []}
        with open(os.path.join(train_dir, "_annotations.coco.json"), "w") as f:
            json.dump(ann, f)

        mock_session_state["train_img_path"] = train_dir
        mock_session_state["class_names"] = "cat"
        mock_session_state["num_classes"] = 1
        mock_session_state["model_arch"] = "FlashDet-N"

        checks = _run_preflight_checks()
        labels = {c[0]: c[1] for c in checks}
        assert labels["Dataset"] is True
        assert labels["Annotations"] is True
        assert labels["Classes"] is True
        assert labels["Model"] is True

    def test_class_names_as_list(self, mock_session_state, tmp_dir):
        from flashstudio.pages.training.launch.preflight import _run_preflight_checks

        train_dir = os.path.join(tmp_dir, "train")
        os.makedirs(train_dir)
        with open(os.path.join(train_dir, "ann.json"), "w") as f:
            json.dump({}, f)

        mock_session_state["train_img_path"] = train_dir
        mock_session_state["class_names"] = ["cat", "dog"]
        mock_session_state["model_arch"] = "FlashDet-P"

        checks = _run_preflight_checks()
        labels = {c[0]: c[1] for c in checks}
        assert labels["Classes"] is True

    def test_gpu_check_always_present(self, mock_session_state):
        from flashstudio.pages.training.launch.preflight import _run_preflight_checks

        checks = _run_preflight_checks()
        labels = {c[0] for c in checks}
        assert "GPU" in labels

    def test_disk_check_always_present(self, mock_session_state):
        from flashstudio.pages.training.launch.preflight import _run_preflight_checks

        checks = _run_preflight_checks()
        labels = {c[0] for c in checks}
        assert "Disk" in labels

    def test_returns_list_of_tuples(self, mock_session_state):
        from flashstudio.pages.training.launch.preflight import _run_preflight_checks

        checks = _run_preflight_checks()
        assert isinstance(checks, list)
        for check in checks:
            assert len(check) == 3
            assert isinstance(check[0], str)
            assert isinstance(check[1], bool)
