"""Tests for flashstudio.pages.training — import and structure checks."""

import pytest


class TestTrainingPageImports:
    def test_render_training_page(self):
        from flashstudio.pages.training import render_training_page
        assert callable(render_training_page)

    def test_page_module(self):
        from flashstudio.pages.training.page import render_training_page
        assert callable(render_training_page)

    def test_launch_tab(self):
        from flashstudio.pages.training.launch.tab import _render_start_tab
        assert callable(_render_start_tab)

    def test_monitor_tab(self):
        from flashstudio.pages.training.monitor.tab import _render_monitor_tab
        assert callable(_render_monitor_tab)

    def test_preflight(self):
        from flashstudio.pages.training.launch.preflight import _run_preflight_checks
        assert callable(_run_preflight_checks)

    def test_runner(self):
        from flashstudio.pages.training.launch.runner import _generate_run_name
        assert callable(_generate_run_name)

    def test_controls(self):
        from flashstudio.pages.training.launch.controls import (
            _stop_training, _pause_training, _resume_active_training,
        )
        assert all(callable(f) for f in [_stop_training, _pause_training, _resume_active_training])

    def test_dialogs(self):
        from flashstudio.pages.training.launch.dialogs import _cleanup_run_keep_best
        assert callable(_cleanup_run_keep_best)

    def test_parsers(self):
        from flashstudio.pages.training.monitor.parsers import (
            _find_log_file, _parse_training_csv, _parse_training_log,
        )
        assert all(callable(f) for f in [_find_log_file, _parse_training_csv, _parse_training_log])

    def test_checkpoints(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert callable(_file_type)

    def test_run_meta(self):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta
        assert callable(_get_run_meta)
