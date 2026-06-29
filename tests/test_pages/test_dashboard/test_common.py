"""Tests for flashstudio.pages.dashboard._common — shared imports."""

import pytest


class TestDashboardCommonImports:
    def test_all_constants_importable(self):
        from flashstudio.pages.dashboard._common import (
            IMG_EXTENSIONS, DEFAULT_SAVE_DIR, MAX_DISPLAY_RUNS,
            CKPT_BEST, CKPT_BEST_INFERENCE, CKPT_FINAL_INFERENCE,
            CKPT_FINAL_FP16, TRAINING_LOG_CSV,
            COLOR_SUCCESS, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY,
            COLOR_TEXT_MUTED, COLOR_BORDER, COLOR_BG_HIGHLIGHT,
            GPU_NAME_TRUNCATE, DATASET_NAME_TRUNCATE,
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
