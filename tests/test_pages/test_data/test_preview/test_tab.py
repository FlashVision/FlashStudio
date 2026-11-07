"""Tests for flashstudio.pages.data.preview.tab."""

from flashstudio.pages.data.preview.tab import _render_preview


def test_render_preview_callable():
    assert callable(_render_preview)
