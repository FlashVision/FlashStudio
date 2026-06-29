"""Tests for export weights exporter."""

from flashstudio.pages.export.weights.exporter import _run_export


def test_run_export_importable():
    assert _run_export is not None


def test_run_export_callable():
    assert callable(_run_export)
