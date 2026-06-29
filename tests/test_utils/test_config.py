"""Tests for flashstudio.utils.config."""

import os
import pytest
import yaml
from flashstudio.utils.config import (
    CONFIG_SCHEMA, save_config_yaml, load_config_yaml,
    config_to_yaml_str, get_pipeline_steps,
)


class TestConfigSchema:
    def test_has_required_sections(self):
        expected = ["project", "dataset", "model", "training",
                     "optimizer", "augmentation", "finetune", "advanced", "output"]
        for section in expected:
            assert section in CONFIG_SCHEMA

    def test_each_section_has_keys(self):
        for section, keys in CONFIG_SCHEMA.items():
            assert isinstance(keys, list)
            assert len(keys) > 0


class TestSaveLoadConfigYaml:
    def test_roundtrip(self, tmp_dir):
        config = {
            "model": {"arch": "FlashDet-Pico"},
            "training": {"epochs": 50, "lr": 0.001},
        }
        path = os.path.join(tmp_dir, "config.yaml")
        save_config_yaml(config, path)

        assert os.path.isfile(path)
        loaded = load_config_yaml(path)
        assert loaded == config

    def test_load_nonexistent_returns_empty(self):
        result = load_config_yaml("/nonexistent/config.yaml")
        assert result == {}

    def test_load_invalid_yaml(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.yaml")
        with open(path, "w") as f:
            f.write(":::invalid yaml:::")
        result = load_config_yaml(path)
        assert isinstance(result, dict)


class TestConfigToYamlStr:
    def test_produces_valid_yaml(self):
        config = {"model": {"size": "n"}, "epochs": 100}
        result = config_to_yaml_str(config)
        assert isinstance(result, str)
        parsed = yaml.safe_load(result)
        assert parsed == config
