"""Tests for flashstudio.pages.model.hyperparams.tab."""


def test_tab_hyper_importable():
    from flashstudio.pages.model.hyperparams.tab import _tab_hyper
    assert _tab_hyper is not None


def test_tab_hyper_callable():
    from flashstudio.pages.model.hyperparams.tab import _tab_hyper
    assert callable(_tab_hyper)


def test_summary_bar_importable():
    from flashstudio.pages.model.hyperparams.tab import _summary_bar
    assert _summary_bar is not None


def test_summary_bar_callable():
    from flashstudio.pages.model.hyperparams.tab import _summary_bar
    assert callable(_summary_bar)
