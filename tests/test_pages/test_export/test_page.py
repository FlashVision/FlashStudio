"""Tests for export page."""

from flashstudio.pages.export.page import render_export_page


def test_render_export_page_importable():
    assert render_export_page is not None


def test_render_export_page_callable():
    assert callable(render_export_page)
