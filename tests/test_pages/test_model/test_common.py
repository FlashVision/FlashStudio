"""Tests for flashstudio.pages.model._common — re-exports."""

import pytest


class TestModelCommonImports:
    def test_imports_available(self):
        from flashstudio.pages.model._common import ARCH_FAMILIES, DEFAULT_MODEL_ARCH
        assert isinstance(ARCH_FAMILIES, list)
        assert isinstance(DEFAULT_MODEL_ARCH, str)

    def test_arch_families_matches_constants(self):
        from flashstudio.pages.model._common import ARCH_FAMILIES
        from flashstudio.constants import ARCH_FAMILIES as AF2
        assert ARCH_FAMILIES is AF2

    def test_default_model_arch_in_flashdet_models(self):
        from flashstudio.pages.model._common import DEFAULT_MODEL_ARCH
        from flashstudio.constants import FLASHDET_MODELS
        assert DEFAULT_MODEL_ARCH in FLASHDET_MODELS
