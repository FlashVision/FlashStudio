"""Tests for flashstudio.components.styles — style helpers."""

from unittest.mock import patch


class TestRenderInfoBar:
    def test_produces_html(self):
        with patch("streamlit.markdown") as mock_md:
            from flashstudio.components.styles import render_info_bar
            render_info_bar({"Model": "FlashDet-N", "Status": "Ready"})
            mock_md.assert_called_once()
            html = mock_md.call_args[0][0]
            assert "Model" in html
            assert "FlashDet-N" in html
            assert "Status" in html
            assert "info-bar" in html

    def test_empty_items(self):
        with patch("streamlit.markdown") as mock_md:
            from flashstudio.components.styles import render_info_bar
            render_info_bar({})
            mock_md.assert_called_once()


class TestRenderPageHeader:
    def test_produces_html(self):
        with patch("streamlit.markdown") as mock_md:
            from flashstudio.components.styles import render_page_header
            render_page_header("icon", "Page Title", "subtitle")
            mock_md.assert_called_once()
            html = mock_md.call_args[0][0]
            assert "Page Title" in html

    def test_without_subtitle(self):
        with patch("streamlit.markdown") as mock_md:
            from flashstudio.components.styles import render_page_header
            render_page_header("icon", "Title Only")
            mock_md.assert_called_once()


class TestInjectCustomCss:
    def test_callable(self):
        from flashstudio.components.styles import inject_custom_css
        assert callable(inject_custom_css)
