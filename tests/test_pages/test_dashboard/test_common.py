"""Tests for flashstudio.pages.dashboard._common — shared imports."""



class TestDashboardCommonImports:
    def test_all_constants_importable(self):
        from flashstudio.pages.dashboard._common import (
            IMG_EXTENSIONS, MAX_DISPLAY_RUNS,
            COLOR_SUCCESS, GPU_NAME_TRUNCATE, DATASET_NAME_TRUNCATE,
        )
        assert isinstance(IMG_EXTENSIONS, tuple)
        assert isinstance(MAX_DISPLAY_RUNS, int)
        assert isinstance(GPU_NAME_TRUNCATE, int)
        assert isinstance(DATASET_NAME_TRUNCATE, int)
        assert COLOR_SUCCESS.startswith("#")

    def test_constants_match_canonical(self):
        from flashstudio.pages.dashboard._common import IMG_EXTENSIONS as dash_ext
        from flashstudio.constants import IMG_EXTENSIONS as const_ext
        assert dash_ext is const_ext
