"""Tests for flashstudio.pages.inference.model.tab — model configuration."""



class TestImports:
    def test_tab_model_importable(self):
        from flashstudio.pages.inference.model.tab import _tab_model
        assert _tab_model is not None

    def test_package_reexports_tab_model(self):
        from flashstudio.pages.inference.model import _tab_model
        assert _tab_model is not None


class TestCallable:
    def test_tab_model_callable(self):
        from flashstudio.pages.inference.model.tab import _tab_model
        assert callable(_tab_model)
