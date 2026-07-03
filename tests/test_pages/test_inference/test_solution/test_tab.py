"""Tests for flashstudio.pages.inference.solution.tab — solution selection & zone drawing."""



class TestImports:
    def test_tab_solution_importable(self):
        from flashstudio.pages.inference.solution.tab import _tab_solution
        assert _tab_solution is not None

    def test_zone_draw_ui_importable(self):
        from flashstudio.pages.inference.solution.tab import _zone_draw_ui
        assert _zone_draw_ui is not None

    def test_manual_zone_input_importable(self):
        from flashstudio.pages.inference.solution.tab import _manual_zone_input
        assert _manual_zone_input is not None

    def test_store_zone_coords_importable(self):
        from flashstudio.pages.inference.solution.tab import _store_zone_coords
        assert _store_zone_coords is not None

    def test_package_reexports_tab_solution(self):
        from flashstudio.pages.inference.solution import _tab_solution
        assert _tab_solution is not None


class TestCallable:
    def test_tab_solution_callable(self):
        from flashstudio.pages.inference.solution.tab import _tab_solution
        assert callable(_tab_solution)

    def test_zone_draw_ui_callable(self):
        from flashstudio.pages.inference.solution.tab import _zone_draw_ui
        assert callable(_zone_draw_ui)

    def test_manual_zone_input_callable(self):
        from flashstudio.pages.inference.solution.tab import _manual_zone_input
        assert callable(_manual_zone_input)

    def test_store_zone_coords_callable(self):
        from flashstudio.pages.inference.solution.tab import _store_zone_coords
        assert callable(_store_zone_coords)
