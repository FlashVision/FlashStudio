"""Tests for flashstudio.pages.model.architecture.tab."""


def test_tab_arch_importable():
    from flashstudio.pages.model.architecture.tab import _tab_arch
    assert _tab_arch is not None


def test_tab_arch_callable():
    from flashstudio.pages.model.architecture.tab import _tab_arch
    assert callable(_tab_arch)


def test_pretrain_finetune_importable():
    from flashstudio.pages.model.architecture.tab import _pretrain_finetune
    assert _pretrain_finetune is not None


def test_pretrain_finetune_callable():
    from flashstudio.pages.model.architecture.tab import _pretrain_finetune
    assert callable(_pretrain_finetune)


def test_render_architecture_tab_from_init():
    from flashstudio.pages.model.architecture import _tab_arch
    assert callable(_tab_arch)


def test_pretrain_finetune_from_init():
    from flashstudio.pages.model.architecture import _pretrain_finetune
    assert callable(_pretrain_finetune)
