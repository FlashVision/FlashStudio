"""Tests for flashstudio.pages.data.page."""

from flashstudio.pages.data.page import render_data_page


def test_render_data_page_callable():
    assert callable(render_data_page)
