"""Tests for flashstudio.pages.inference.data.tab — data input handling."""



class TestImports:
    def test_tab_data_importable(self):
        from flashstudio.pages.inference.data.tab import _tab_data
        assert _tab_data is not None

    def test_get_video_metadata_importable(self):
        from flashstudio.pages.inference.data.tab import _get_video_metadata
        assert _get_video_metadata is not None

    def test_get_first_frame_importable(self):
        from flashstudio.pages.inference.data.tab import _get_first_frame
        assert _get_first_frame is not None

    def test_test_stream_importable(self):
        from flashstudio.pages.inference.data.tab import _test_stream
        assert _test_stream is not None

    def test_package_reexports_tab_data(self):
        from flashstudio.pages.inference.data import _tab_data
        assert _tab_data is not None


class TestCallable:
    def test_tab_data_callable(self):
        from flashstudio.pages.inference.data.tab import _tab_data
        assert callable(_tab_data)

    def test_get_video_metadata_callable(self):
        from flashstudio.pages.inference.data.tab import _get_video_metadata
        assert callable(_get_video_metadata)

    def test_get_first_frame_callable(self):
        from flashstudio.pages.inference.data.tab import _get_first_frame
        assert callable(_get_first_frame)

    def test_test_stream_callable(self):
        from flashstudio.pages.inference.data.tab import _test_stream
        assert callable(_test_stream)


class TestGetVideoMetadata:
    def test_none_input_returns_none(self):
        from flashstudio.pages.inference.data.tab import _get_video_metadata
        result = _get_video_metadata(None)
        assert result is None

    def test_invalid_file_returns_none(self):
        from flashstudio.pages.inference.data.tab import _get_video_metadata
        import io
        fake = io.BytesIO(b"not a video")
        result = _get_video_metadata(fake)
        assert result is None

    def test_returns_dict_or_none(self):
        from flashstudio.pages.inference.data.tab import _get_video_metadata
        import io
        fake = io.BytesIO(b"\x00" * 128)
        result = _get_video_metadata(fake)
        assert result is None or isinstance(result, dict)

    def test_dict_has_expected_keys_if_valid(self):
        from flashstudio.pages.inference.data.tab import _get_video_metadata
        import io
        fake = io.BytesIO(b"\x00" * 128)
        result = _get_video_metadata(fake)
        if result is not None:
            for key in ("width", "height", "fps", "total_frames"):
                assert key in result
