"""Tests for flashstudio.pages.model.augment.tab."""


def test_tab_aug_importable():
    from flashstudio.pages.model.augment.tab import _tab_aug
    assert _tab_aug is not None


def test_tab_aug_callable():
    from flashstudio.pages.model.augment.tab import _tab_aug
    assert callable(_tab_aug)
