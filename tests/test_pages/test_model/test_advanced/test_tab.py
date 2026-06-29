"""Tests for flashstudio.pages.model.advanced.tab."""


def test_tab_adv_importable():
    from flashstudio.pages.model.advanced.tab import _tab_adv
    assert _tab_adv is not None


def test_tab_adv_callable():
    from flashstudio.pages.model.advanced.tab import _tab_adv
    assert callable(_tab_adv)
