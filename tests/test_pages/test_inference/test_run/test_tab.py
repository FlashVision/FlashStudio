"""Tests for flashstudio.pages.inference.run.tab — run orchestration."""



class TestImports:
    def test_tab_run_importable(self):
        from flashstudio.pages.inference.run.tab import _tab_run
        assert _tab_run is not None

    def test_run_images_importable(self):
        from flashstudio.pages.inference.run.tab import _run_images
        assert _run_images is not None

    def test_show_image_results_importable(self):
        from flashstudio.pages.inference.run.tab import _show_image_results
        assert _show_image_results is not None

    def test_show_video_results_importable(self):
        from flashstudio.pages.inference.run.tab import _show_video_results
        assert _show_video_results is not None

    def test_package_reexports_tab_run(self):
        from flashstudio.pages.inference.run import _tab_run
        assert _tab_run is not None


class TestCallable:
    def test_tab_run_callable(self):
        from flashstudio.pages.inference.run.tab import _tab_run
        assert callable(_tab_run)

    def test_run_images_callable(self):
        from flashstudio.pages.inference.run.tab import _run_images
        assert callable(_run_images)

    def test_show_image_results_callable(self):
        from flashstudio.pages.inference.run.tab import _show_image_results
        assert callable(_show_image_results)

    def test_show_video_results_callable(self):
        from flashstudio.pages.inference.run.tab import _show_video_results
        assert callable(_show_video_results)
