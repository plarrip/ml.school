import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest


def test_predict_returns_empty_list_if_input_is_empty(model):
    assert model.predict(None, model_input=[]) == []


def test_predict_return_empty_list_on_invalid_input(model, monkeypatch):
    mock_process_input = Mock(return_value=None)
    monkeypatch.setattr(model, "process_input", mock_process_input)

    input_data = [{"island": "Torgersen"}]
    result = model.predict(context=None, model_input=input_data)
    assert result == []


def test_predict_returns_empty_list_on_invalid_prediction(model, monkeypatch):
    mock_process_input = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    model.model.predict = Mock(return_value=None)
    monkeypatch.setattr(model, "process_input", mock_process_input)

    input_data = [{"island": "Torgersen", "culmen_length_mm": 39.1}]
    result = model.predict(context=None, model_input=input_data)
    assert result == []


def test_predict(model):
    model_input = [{"island": "Torgersen", "culmen_length_mm": 39.1}]
    mock_process_input = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    mock_process_output = Mock(
        return_value=[{"prediction": "Adelie", "confidence": 0.6}],
    )
    model.model.predict = Mock(return_value=np.array([[0.6, 0.3, 0.1]]))
    model.process_input = mock_process_input
    model.process_output = mock_process_output

    result = model.predict(None, model_input)

    assert result == [{"prediction": "Adelie", "confidence": 0.6}]
    mock_process_input.assert_called_once()
    mock_process_output.assert_called_once()
    model.model.predict.assert_called_once()


def test_predict_backend_is_called(model):
    model.backend = Mock()
    model.predict(None, [{"island": "Torgersen"}])
    model.backend.save.assert_called_once()


def test_predict_backend_receives_model_input(model):
    model.backend = Mock()
    model_input = [{"island": "Torgersen"}, {"island": "Biscoe"}]
    model.predict(context=None, model_input=model_input)

    backend_input_arg = model.backend.save.call_args[0][0]
    assert backend_input_arg.island.iloc[0] == "Torgersen"
    assert backend_input_arg.island.iloc[1] == "Biscoe"


def test_predict_backend_receives_prediction(model):
    model.backend = Mock()
    model_input = [{"island": "Torgersen"}]
    model.predict(context=None, model_input=model_input)

    backend_output_arg = model.backend.save.call_args[0][1]
    assert backend_output_arg == [
        {"prediction": "Adelie", "confidence": 0.6},
    ]


def test_predict_backend_receives_prediction_none(model):
    model.backend = Mock()
    model.process_output = Mock(return_value=None)
    model.predict(context=None, model_input=[{"island": "Torgersen"}])

    backend_output_arg = model.backend.save.call_args[0][1]
    assert backend_output_arg is None


def test_process_input_should_transform_input_data(model):
    model.features_transformer.transform = Mock(
        return_value=np.array([[0.1, 0.2]]),
    )
    model_input = [{"island": ["Torgersen"]}]
    result = model.process_input(model_input)

    model.features_transformer.transform.assert_called_once_with(model_input)
    assert np.array_equal(result, np.array([[0.1, 0.2]]))


def test_process_input_returns_none_on_exception(model):
    model.features_transformer.transform = Mock(side_effect=Exception("Invalid input"))
    input_data = pd.DataFrame({"island": ["Torgersen"]})
    result = model.process_input(input_data)

    model.features_transformer.transform.assert_called_once_with(input_data)

    assert result is None, (
        "Since there was an exception, the function should return None."
    )


def test_process_output_returns_json(model):
    output = np.array([[0.6, 0.3, 0.1]])
    result = model.process_output(output)
    assert isinstance(result[0], dict)


def test_process_output_returns_prediction_and_confidence(model):
    output = np.array([[0.6, 0.3, 0.1]])
    result = model.process_output(output)

    assert result[0].keys() == {"prediction", "confidence"}


def test_process_output_returns_species(model):
    output = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    result = model.process_output(output)

    assert result[0]["prediction"] == "Adelie"
    assert result[1]["prediction"] == "Chinstrap"
    assert result[2]["prediction"] == "Gentoo"


def test_process_output_returns_empty_list_if_it_receives_none(model):
    assert model.process_output(None) == []


class TestPredictionLatency:
    """Latency is logged after every successful prediction."""

    def test_latency_is_logged_on_successful_prediction(self, model, caplog):
        """A completed prediction should emit a latency log line."""
        with caplog.at_level(logging.INFO, logger="model"):
            model.predict(None, [{"island": "Torgersen"}])

        assert any("Prediction completed in" in m for m in caplog.messages)

    def test_latency_log_includes_sample_count(self, model, caplog):
        """The latency log line should mention how many samples were processed."""
        model.model.predict = Mock(return_value=np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]]))
        model.backend = None

        with caplog.at_level(logging.INFO, logger="model"):
            model.predict(None, [{"island": "Torgersen"}, {"island": "Biscoe"}])

        latency_messages = [m for m in caplog.messages if "Prediction completed in" in m]
        assert latency_messages
        assert "2 samples" in latency_messages[0]

    def test_latency_is_not_logged_for_empty_input(self, model, caplog):
        """An empty input should return early without logging a latency line."""
        with caplog.at_level(logging.INFO, logger="model"):
            model.predict(None, [])

        assert not any("Prediction completed in" in m for m in caplog.messages)


class TestPredictionErrors:
    """Errors during inference are caught and logged."""

    def test_inference_error_is_logged(self, model, caplog):
        """A crash inside model.predict should be logged as an exception."""
        model.model.predict = Mock(side_effect=RuntimeError("CUDA OOM"))

        with caplog.at_level(logging.ERROR, logger="model"):
            result = model.predict(None, [{"island": "Torgersen"}])

        assert result == []
        assert any("error during model inference" in m.lower() for m in caplog.messages)

    def test_inference_error_returns_empty_list(self, model):
        """A crash inside model.predict should return an empty list, not propagate."""
        model.model.predict = Mock(side_effect=RuntimeError("CUDA OOM"))
        result = model.predict(None, [{"island": "Torgersen"}])
        assert result == []

    def test_input_processing_failure_is_logged_as_error(self, model, caplog):
        """When process_input returns None, an error should be logged."""
        model.process_input = Mock(return_value=None)

        with caplog.at_level(logging.ERROR, logger="model"):
            model.predict(None, [{"island": "Torgersen"}])

        assert any("Prediction skipped" in m for m in caplog.messages)


class TestPredictionCache:
    """Identical inputs are served from the in-memory cache."""

    def test_cache_hit_skips_model_inference(self, model):
        """A repeated input should not call model.predict a second time."""
        payload = [{"island": "Torgersen"}]
        model.predict(None, payload)
        call_count_after_first = model.model.predict.call_count

        model.predict(None, payload)

        assert model.model.predict.call_count == call_count_after_first

    def test_cache_hit_returns_same_result(self, model):
        """A repeated input should return the same prediction as the first call."""
        payload = [{"island": "Torgersen"}]
        first = model.predict(None, payload)
        second = model.predict(None, payload)

        assert first == second

    def test_cache_miss_calls_model(self, model):
        """Different inputs should each trigger a real model.predict call."""
        model.predict(None, [{"island": "Torgersen"}])
        model.predict(None, [{"island": "Biscoe"}])

        assert model.model.predict.call_count == 2

    def test_cache_hit_is_logged(self, model, caplog):
        """A cache hit should emit an INFO log."""
        payload = [{"island": "Torgersen"}]
        model.predict(None, payload)

        with caplog.at_level(logging.INFO, logger="model"):
            model.predict(None, payload)

        assert any("Cache hit" in m for m in caplog.messages)

    def test_cache_evicts_oldest_entry_when_full(self, model):
        """When the cache is full, the oldest entry is evicted to stay within max size."""
        model._cache_max_size = 2

        model.predict(None, [{"island": "Torgersen"}])
        model.predict(None, [{"island": "Biscoe"}])
        model.predict(None, [{"island": "Dream"}])

        assert len(model._cache) == 2


class TestConfidenceThreshold:
    """Predictions below the confidence threshold return an uncertain response."""

    def test_low_confidence_returns_none_prediction(self, model):
        """A confidence below the threshold should set prediction to None."""
        model._confidence_threshold = 0.9
        model.model.predict = Mock(return_value=np.array([[0.5, 0.3, 0.2]]))

        result = model.predict(None, [{"island": "Torgersen"}])

        assert result[0]["prediction"] is None
        assert result[0]["confidence"] == pytest.approx(0.5)

    def test_high_confidence_returns_species(self, model):
        """A confidence above the threshold should return the predicted species."""
        model._confidence_threshold = 0.4
        model.model.predict = Mock(return_value=np.array([[0.6, 0.3, 0.1]]))

        result = model.predict(None, [{"island": "Torgersen"}])

        assert result[0]["prediction"] == "Adelie"

    def test_confidence_exactly_at_threshold_is_not_uncertain(self, model):
        """A confidence equal to the threshold is not considered uncertain (strict <)."""
        model._confidence_threshold = 0.6
        model.model.predict = Mock(return_value=np.array([[0.6, 0.3, 0.1]]))

        result = model.predict(None, [{"island": "Torgersen"}])

        assert result[0]["prediction"] == "Adelie"

    def test_zero_threshold_never_returns_uncertain(self, model):
        """The default threshold of 0.0 should never suppress any prediction."""
        model._confidence_threshold = 0.0
        model.model.predict = Mock(return_value=np.array([[0.1, 0.6, 0.3]]))

        result = model.predict(None, [{"island": "Torgersen"}])

        assert result[0]["prediction"] is not None

    def test_low_confidence_is_logged_as_warning(self, model, caplog):
        """A below-threshold prediction should emit a WARNING log."""
        model._confidence_threshold = 0.9
        model.model.predict = Mock(return_value=np.array([[0.5, 0.3, 0.2]]))

        with caplog.at_level(logging.WARNING, logger="model"):
            model.predict(None, [{"island": "Torgersen"}])

        assert any("Low-confidence" in m for m in caplog.messages)

    def test_mixed_confidences_in_batch(self, model):
        """In a batch, each prediction is evaluated independently against the threshold."""
        model._confidence_threshold = 0.6
        model.model.predict = Mock(
            return_value=np.array([[0.8, 0.1, 0.1], [0.4, 0.4, 0.2]])
        )
        model.backend = None

        result = model.predict(None, [{"island": "Torgersen"}, {"island": "Biscoe"}])

        assert result[0]["prediction"] == "Adelie"
        assert result[1]["prediction"] is None


class TestModelVersioning:
    """Version is resolved from params; unknown versions fall back to default."""

    def _make_version_components(self, prediction_return):
        """Build a mock version dict matching the components structure."""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=prediction_return)
        mock_ft = Mock()
        mock_ft.transform = Mock(side_effect=lambda x: x)
        mock_tt = Mock()
        mock_species = Mock()
        mock_species.categories_ = [["Adelie", "Chinstrap", "Gentoo"]]
        mock_tt.named_transformers_ = {"species": mock_species}
        return {"model": mock_model, "features_transformer": mock_ft, "target_transformer": mock_tt}

    def test_default_version_used_when_no_params(self, model):
        """Passing params=None should use the default version."""
        result = model.predict(None, [{"island": "Torgersen"}])
        assert result == [{"prediction": "Adelie", "confidence": 0.6}]

    def test_default_version_used_when_version_key_absent(self, model):
        """params without a 'version' key should use the default version."""
        result = model.predict(None, [{"island": "Torgersen"}], params={"other": "x"})
        assert result == [{"prediction": "Adelie", "confidence": 0.6}]

    def test_named_version_routes_to_correct_model(self, model):
        """params={'version': '2'} should use the v2 model, not the default."""
        v2 = self._make_version_components(np.array([[0.1, 0.8, 0.1]]))
        model._versions["2"] = v2

        result = model.predict(None, [{"island": "Torgersen"}], params={"version": "2"})

        v2["model"].predict.assert_called_once()
        model.model.predict.assert_not_called()
        assert result[0]["prediction"] == "Chinstrap"

    def test_unknown_version_falls_back_to_default(self, model):
        """Requesting a version that doesn't exist should return the default version's prediction."""
        result = model.predict(
            None, [{"island": "Torgersen"}], params={"version": "99"}
        )
        assert result == [{"prediction": "Adelie", "confidence": 0.6}]

    def test_unknown_version_logs_warning(self, model, caplog):
        """An unknown version request should emit a WARNING log."""
        with caplog.at_level(logging.WARNING, logger="model"):
            model.predict(None, [{"island": "Torgersen"}], params={"version": "99"})

        assert any("not found" in m for m in caplog.messages)

    def test_version_is_logged_on_each_request(self, model, caplog):
        """The serving version label should appear in the INFO logs for every request."""
        with caplog.at_level(logging.INFO, logger="model"):
            model.predict(None, [{"island": "Torgersen"}])

        assert any("Serving model version '1'" in m for m in caplog.messages)

    def test_different_versions_have_separate_cache_entries(self, model):
        """The same input served by two different versions must produce two cache entries."""
        v2 = self._make_version_components(np.array([[0.1, 0.8, 0.1]]))
        model._versions["2"] = v2
        payload = [{"island": "Torgersen"}]

        model.predict(None, payload, params={"version": "1"})
        model.predict(None, payload, params={"version": "2"})

        assert len(model._cache) == 2

    def test_cache_hit_respects_version(self, model):
        """A repeat request for version 2 should hit the cache and skip v2 model.predict."""
        v2 = self._make_version_components(np.array([[0.1, 0.8, 0.1]]))
        model._versions["2"] = v2
        payload = [{"island": "Torgersen"}]

        model.predict(None, payload, params={"version": "2"})
        model.predict(None, payload, params={"version": "2"})

        assert v2["model"].predict.call_count == 1
