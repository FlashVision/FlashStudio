"""Tests for flashstudio.utils — top-level helpers."""

from flashstudio.utils import (
    get_default, get_state, DEFAULTS,
    get_class_names_str, get_class_names_list,
)


class TestDefaults:
    def test_defaults_is_dict(self):
        assert isinstance(DEFAULTS, dict)
        assert len(DEFAULTS) > 0

    def test_get_default_known(self):
        for key, val in DEFAULTS.items():
            assert get_default(key) == val

    def test_get_default_unknown(self):
        result = get_default("__nonexistent_key__")
        assert result is None

    def test_get_default_unknown_is_none(self):
        result = get_default("__nonexistent_key__")
        assert result is None


class TestGetState:
    def test_get_state_returns_default(self, mock_session_state):
        val = get_state("epochs")
        assert val == DEFAULTS["epochs"]

    def test_get_state_from_session(self, mock_session_state):
        mock_session_state["epochs"] = 42
        val = get_state("epochs")
        assert val == 42

    def test_get_state_unknown_returns_none(self, mock_session_state):
        val = get_state("__nonexistent__")
        assert val is None


class TestClassNames:
    def test_get_class_names_str(self, mock_session_state):
        mock_session_state["class_names"] = "cat,dog,bird"
        result = get_class_names_str()
        assert isinstance(result, str)
        assert "cat" in result

    def test_get_class_names_list(self, mock_session_state):
        mock_session_state["class_names"] = "cat\ndog\nbird"
        result = get_class_names_list()
        assert isinstance(result, list)
        assert len(result) == 3
        assert "cat" in result

    def test_list_strips_whitespace(self, mock_session_state):
        mock_session_state["class_names"] = " cat \n dog \n bird "
        result = get_class_names_list()
        assert result == ["cat", "dog", "bird"]

    def test_empty(self, mock_session_state):
        result = get_class_names_list()
        assert isinstance(result, list)
