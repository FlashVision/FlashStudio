"""Tests for flashstudio.pages.data.download.tab."""

from flashstudio.pages.data.download.tab import _render_download


def test_render_download_callable():
    assert callable(_render_download)
