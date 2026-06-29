"""Tests for flashstudio.pages.inference._common — solutions registry, device options."""



class TestSolutionsRegistry:
    def test_solutions_not_empty(self):
        from flashstudio.pages.inference._common import SOLUTIONS
        assert len(SOLUTIONS) > 0

    def test_detection_only_present(self):
        from flashstudio.pages.inference._common import SOLUTIONS
        assert "None (Detection Only)" in SOLUTIONS

    def test_solution_has_desc(self):
        from flashstudio.pages.inference._common import SOLUTIONS
        for name, info in SOLUTIONS.items():
            assert "desc" in info, f"Solution '{name}' missing 'desc'"
            assert "needs_zone" in info, f"Solution '{name}' missing 'needs_zone'"

    def test_zone_solutions_have_zone_type(self):
        from flashstudio.pages.inference._common import SOLUTIONS
        for name, info in SOLUTIONS.items():
            if info["needs_zone"]:
                assert "zone_type" in info, f"Zone solution '{name}' missing 'zone_type'"
                assert info["zone_type"] in ("line", "polygon")

    def test_counter_needs_zone(self):
        from flashstudio.pages.inference._common import SOLUTIONS
        assert SOLUTIONS["Object Counter (Line)"]["needs_zone"] is True
        assert SOLUTIONS["Object Counter (Line)"]["zone_type"] == "line"

    def test_heatmap_no_zone(self):
        from flashstudio.pages.inference._common import SOLUTIONS
        assert SOLUTIONS["Heatmap"]["needs_zone"] is False


class TestGetClassNames:
    def test_from_string(self, mock_session_state):
        from flashstudio.pages.inference._common import _get_class_names

        mock_session_state["class_names"] = "cat\ndog\nbird"
        result = _get_class_names()
        assert result == ["cat", "dog", "bird"]

    def test_from_list(self, mock_session_state):
        from flashstudio.pages.inference._common import _get_class_names

        mock_session_state["class_names"] = ["cat", "dog", "bird"]
        result = _get_class_names()
        assert result == ["cat", "dog", "bird"]

    def test_empty_falls_back_to_coco(self, mock_session_state):
        from flashstudio.pages.inference._common import _get_class_names
        from flashstudio.constants import COCO_CLASSES

        result = _get_class_names()
        assert result == COCO_CLASSES

    def test_whitespace_stripped(self, mock_session_state):
        from flashstudio.pages.inference._common import _get_class_names

        mock_session_state["class_names"] = " cat \n dog \n  "
        result = _get_class_names()
        assert result == ["cat", "dog"]


class TestGetDeviceOptions:
    def test_always_includes_cpu(self):
        from flashstudio.pages.inference._common import _get_device_options

        devices = _get_device_options()
        assert "cpu" in devices
        assert isinstance(devices, list)

    def test_returns_list(self):
        from flashstudio.pages.inference._common import _get_device_options

        devices = _get_device_options()
        assert isinstance(devices, list)
        assert len(devices) >= 1
