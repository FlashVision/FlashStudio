"""Tests for flashstudio.core.constants — verify re-export from canonical location."""

import pytest


class TestCoreReExport:
    """flashstudio.core.constants should mirror flashstudio.constants exactly."""

    def test_import_from_core(self):
        from flashstudio.core.constants import FLASHDET_MODELS, COCO_CLASSES
        assert isinstance(FLASHDET_MODELS, dict)
        assert isinstance(COCO_CLASSES, list)

    def test_matches_canonical(self):
        import flashstudio.constants as canonical
        import flashstudio.core.constants as core

        for name in [
            "IMG_EXTENSIONS", "VIDEO_EXTENSIONS", "WEIGHT_EXTENSIONS",
            "CKPT_BEST", "CKPT_LAST", "FLASHDET_MODELS", "COCO_CLASSES",
            "COLOR_PRIMARY", "TRAIN_EPOCHS", "EXPORT_FORMATS",
        ]:
            assert getattr(core, name) is getattr(canonical, name), (
                f"core.{name} is not the same object as constants.{name}"
            )

    def test_core_init_reexports(self):
        from flashstudio.core import FLASHDET_MODELS, COCO_CLASSES
        from flashstudio.constants import FLASHDET_MODELS as FM2, COCO_CLASSES as CC2
        assert FLASHDET_MODELS is FM2
        assert COCO_CLASSES is CC2
