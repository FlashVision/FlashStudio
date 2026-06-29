"""Tests for flashstudio.pages.data — import and structure checks."""



class TestDataPageImports:
    def test_render_data_page_importable(self):
        from flashstudio.pages.data import render_data_page
        assert callable(render_data_page)

    def test_page_module_importable(self):
        from flashstudio.pages.data.page import render_data_page
        assert callable(render_data_page)

    def test_common_importable(self):
        from flashstudio.pages.data._common import (
            _extract_classes_from_coco_json,
            _dataset_already_downloaded,
        )
        assert callable(_extract_classes_from_coco_json)
        assert callable(_dataset_already_downloaded)

    def test_helpers_importable(self):
        from flashstudio.pages.data.helpers import (
            _find_root, _find_images, _find_ann, _find_labels,
        )
        assert callable(_find_root)
        assert callable(_find_images)
        assert callable(_find_ann)
        assert callable(_find_labels)
