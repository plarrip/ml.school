import json
import os
from unittest.mock import Mock

import pytest

from inference.model import Model


@pytest.fixture
def context():
    """Return a mock context."""
    mock_context = Mock()
    mock_context.artifacts = {
        "model": "model",
        "features_transformer": "features_transformer",
        "target_transformer": "target_transformer",
    }

    return mock_context


@pytest.fixture
def model(monkeypatch):
    """Return a model instance."""
    model = Model()

    def mock_load(path):
        return Mock(artifact=path)

    monkeypatch.setattr("joblib.load", mock_load)
    monkeypatch.setattr("keras.saving.load_model", lambda _: Mock(artifact="model"))

    return model


def test_load_artifacts_loads_keras_model(model, context):
    model.load_context(context)
    assert model.model.artifact == "model"


def test_load_artifacts_loads_features_transformers(model, context):
    model.load_context(context)
    artifact = model.features_transformer.artifact
    assert artifact == context.artifacts["features_transformer"]


def test_load_artifacts_loads_target_transformer(model, context):
    model.load_context(context)
    artifact = model.target_transformer.artifact
    assert artifact == context.artifacts["target_transformer"]


def test_keras_backend_is_set_to_tensorflow_by_default(model, context, monkeypatch):
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    model.load_context(context)
    assert os.getenv("KERAS_BACKEND") == "tensorflow"


def test_keras_backend_is_unchanged_if_present(model, context, monkeypatch):
    monkeypatch.setenv("KERAS_BACKEND", "jax")
    model.load_context(context)
    assert os.getenv("KERAS_BACKEND") == "jax"


class TestVersionLoading:
    """Default version is registered from context; extra versions load from config."""

    def test_default_version_registered_after_load_context(self, model, context):
        """load_context should register the default version in self._versions."""
        model.load_context(context)
        assert "1" in model._versions

    def test_default_version_components_match_instance_attributes(self, model, context):
        """The default version's components must be the same objects as self.model etc."""
        model.load_context(context)
        v = model._versions["1"]
        assert v["model"] is model.model
        assert v["features_transformer"] is model.features_transformer
        assert v["target_transformer"] is model.target_transformer

    def test_default_version_label_read_from_env(self, model, context, monkeypatch):
        """MODEL_DEFAULT_VERSION env var should control the default version label."""
        monkeypatch.setenv("MODEL_DEFAULT_VERSION", "stable")
        model.load_context(context)
        assert "stable" in model._versions
        assert model._default_version == "stable"

    def test_additional_versions_loaded_from_config(
        self, model, context, monkeypatch, tmp_path
    ):
        """MODEL_VERSIONS_CONFIG should load named additional versions."""
        v2_model = Mock(artifact="model_v2")
        v2_ft = Mock(artifact="ft_v2")
        v2_tt = Mock(artifact="tt_v2")

        config = {
            "2": {
                "model": "model_v2",
                "features_transformer": "ft_v2",
                "target_transformer": "tt_v2",
            }
        }
        config_file = tmp_path / "versions.json"
        config_file.write_text(json.dumps(config))
        monkeypatch.setenv("MODEL_VERSIONS_CONFIG", str(config_file))

        def mock_load(path):
            mapping = {
                "features_transformer": Mock(artifact="ft"),
                "target_transformer": Mock(artifact="tt"),
                "ft_v2": v2_ft,
                "tt_v2": v2_tt,
            }
            return mapping.get(path, Mock(artifact=path))

        monkeypatch.setattr("joblib.load", mock_load)
        monkeypatch.setattr(
            "keras.saving.load_model",
            lambda p: v2_model if p == "model_v2" else Mock(artifact=p),
        )

        model.load_context(context)

        assert "2" in model._versions
        assert model._versions["2"]["model"] is v2_model

    def test_missing_versions_config_logs_warning(
        self, model, context, monkeypatch, caplog
    ):
        """A non-existent MODEL_VERSIONS_CONFIG path should log a warning."""
        import logging

        monkeypatch.setenv("MODEL_VERSIONS_CONFIG", "/nonexistent/versions.json")

        with caplog.at_level(logging.WARNING, logger="model"):
            model.load_context(context)

        assert any("not found" in m for m in caplog.messages)

    def test_invalid_versions_config_logs_exception(
        self, model, context, monkeypatch, tmp_path, caplog
    ):
        """Malformed JSON in MODEL_VERSIONS_CONFIG should log an exception."""
        import logging

        config_file = tmp_path / "versions.json"
        config_file.write_text("{not valid json")
        monkeypatch.setenv("MODEL_VERSIONS_CONFIG", str(config_file))

        with caplog.at_level(logging.ERROR, logger="model"):
            model.load_context(context)

        assert any("Failed to parse" in m for m in caplog.messages)

    def test_default_version_label_in_config_is_skipped(
        self, model, context, monkeypatch, tmp_path, caplog
    ):
        """A config entry whose label matches the default version should be skipped."""
        import logging

        config = {"1": {"model": "x", "features_transformer": "x", "target_transformer": "x"}}
        config_file = tmp_path / "versions.json"
        config_file.write_text(json.dumps(config))
        monkeypatch.setenv("MODEL_VERSIONS_CONFIG", str(config_file))

        with caplog.at_level(logging.WARNING, logger="model"):
            model.load_context(context)

        assert any("shadows the default" in m for m in caplog.messages)
