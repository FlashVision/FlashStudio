"""Tests for flash messages, update_config_mirror, and class name helpers."""

from flashstudio.utils import (
    flash, show_flashes, update_config_mirror,
    get_state, get_class_names_str, get_class_names_list, DEFAULTS,
)


class TestFlashMessages:
    def test_flash_queues_message(self, mock_session_state):
        flash("Hello", "success")
        msgs = mock_session_state.get("_flash_messages", [])
        assert len(msgs) == 1
        assert msgs[0] == ("success", "Hello")

    def test_flash_multiple(self, mock_session_state):
        flash("A", "info")
        flash("B", "error")
        flash("C", "warning")
        msgs = mock_session_state["_flash_messages"]
        assert len(msgs) == 3

    def test_show_flashes_clears_queue(self, mock_session_state):
        flash("Test", "success")
        show_flashes()
        assert "_flash_messages" not in mock_session_state

    def test_show_flashes_empty_no_error(self, mock_session_state):
        show_flashes()


class TestUpdateConfigMirror:
    def test_creates_mirror(self, mock_session_state):
        mock_session_state["epochs"] = 200
        mock_session_state["batch_size"] = 32
        update_config_mirror()
        mirror = mock_session_state["_config_mirror"]
        assert mirror["epochs"] == 200
        assert mirror["batch_size"] == 32

    def test_mirror_survives_key_deletion(self, mock_session_state):
        mock_session_state["epochs"] = 500
        update_config_mirror()
        del mock_session_state["epochs"]
        assert get_state("epochs") == 500

    def test_mirror_updates_existing(self, mock_session_state):
        mock_session_state["lr"] = 0.01
        update_config_mirror()
        mock_session_state["lr"] = 0.001
        update_config_mirror()
        assert mock_session_state["_config_mirror"]["lr"] == 0.001

    def test_mirror_only_copies_known_keys(self, mock_session_state):
        mock_session_state["random_unknown_key_xyz"] = 999
        update_config_mirror()
        mirror = mock_session_state["_config_mirror"]
        assert "random_unknown_key_xyz" not in mirror

    def test_mirror_includes_all_training_keys(self, mock_session_state):
        keys = ["epochs", "batch_size", "lr", "img_size", "warmup_epochs",
                "patience", "num_workers", "grad_accum"]
        for k in keys:
            mock_session_state[k] = 42
        update_config_mirror()
        mirror = mock_session_state["_config_mirror"]
        for k in keys:
            assert k in mirror

    def test_mirror_includes_model_keys(self, mock_session_state):
        mock_session_state["model_arch"] = "FlashDet-Large"
        mock_session_state["arch_family"] = "FlashDet"
        update_config_mirror()
        mirror = mock_session_state["_config_mirror"]
        assert mirror["model_arch"] == "FlashDet-Large"
        assert mirror["arch_family"] == "FlashDet"

    def test_mirror_includes_augmentation_keys(self, mock_session_state):
        mock_session_state["aug_mosaic"] = True
        mock_session_state["aug_mixup"] = False
        update_config_mirror()
        mirror = mock_session_state["_config_mirror"]
        assert mirror["aug_mosaic"] is True
        assert mirror["aug_mixup"] is False


class TestGetStateWithMirror:
    def test_mirror_priority_over_session(self, mock_session_state):
        mock_session_state["_config_mirror"] = {"epochs": 999}
        mock_session_state["epochs"] = 100
        assert get_state("epochs") == 999

    def test_session_fallback_when_not_in_mirror(self, mock_session_state):
        mock_session_state["_config_mirror"] = {}
        mock_session_state["epochs"] = 100
        assert get_state("epochs") == 100

    def test_defaults_fallback(self, mock_session_state):
        mock_session_state["_config_mirror"] = {}
        val = get_state("epochs")
        assert val == DEFAULTS["epochs"]

    def test_none_for_unknown_key(self, mock_session_state):
        assert get_state("nonexistent_key_abc") is None


class TestClassNamesEdgeCases:
    def test_list_input_converted_to_str(self, mock_session_state):
        mock_session_state["class_names"] = ["cat", "dog", "bird"]
        result = get_class_names_str()
        assert result == "cat\ndog\nbird"

    def test_list_input_to_list(self, mock_session_state):
        mock_session_state["class_names"] = ["cat", "dog"]
        result = get_class_names_list()
        assert result == ["cat", "dog"]

    def test_empty_string_returns_empty_list(self, mock_session_state):
        mock_session_state["class_names"] = ""
        assert get_class_names_list() == []

    def test_whitespace_only_returns_empty_list(self, mock_session_state):
        mock_session_state["class_names"] = "   \n   \n   "
        assert get_class_names_list() == []

    def test_single_class(self, mock_session_state):
        mock_session_state["class_names"] = "person"
        assert get_class_names_list() == ["person"]

    def test_trailing_newlines_ignored(self, mock_session_state):
        mock_session_state["class_names"] = "cat\ndog\n\n\n"
        assert get_class_names_list() == ["cat", "dog"]
