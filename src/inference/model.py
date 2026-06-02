import hashlib
import importlib
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import pydantic
from mlflow.models import set_model
from mlflow.pyfunc.model import PythonModelContext


class Input(pydantic.BaseModel):
    """Prediction input that will be received from the client.

    This class is responsible for defining the structure of the input data that the
    model will receive from the client. The input data will be automatically validated
    by MLflow against this schema before making a prediction.
    """

    island: str | None = None
    culmen_length_mm: float | None = None
    culmen_depth_mm: float | None = None
    flipper_length_mm: float | None = None
    body_mass_g: float | None = None
    sex: str | None = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def unwrap_numpy(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: v.item() if hasattr(v, "item") else v for k, v in data.items()}
        return data


class Output(pydantic.BaseModel):
    """Prediction output that will be returned to the client.

    This class is responsible for defining the structure of the output data that the
    model will return to the client.
    """

    prediction: str | None = None
    confidence: float | None = None


class Model(mlflow.pyfunc.PythonModel):
    """A custom model implementing an inference pipeline to classify penguins.

    This inference pipeline has three phases: processing the input data, prediction, and
    processing the output before generating the response to the client. The pipeline
    will optionally store the input requests and predictions.

    The [Custom MLflow Models with mlflow.pyfunc](https://mlflow.org/blog/custom-pyfunc)
    blog post is a great reference to understand how to use custom Python models in
    MLflow.
    """

    def __init__(self) -> None:
        """Initialize the model."""
        self.backend = None
        self._cache: OrderedDict = OrderedDict()
        self._cache_max_size = 128
        self._confidence_threshold = 0.0
        self._versions: dict = {}
        self._default_version: str = "1"

    def load_context(self, context: PythonModelContext | None) -> None:
        """Load and prepare the model context to make predictions.

        This function is called only once as soon as the model is constructed. It loads
        the transformers and the Keras model specified as artifacts.
        """
        self._configure_logging()
        self._configure_threshold()
        self._initialize_backend()
        self._load_artifacts(context)

        self.logger.info("Model is ready to receive requests")

    def predict(
        self,
        context,  # noqa: ARG002
        model_input: list[Input],
        params: dict[str, Any] | None = None,
    ) -> Output:
        """Handle the request received from the client.

        This method is responsible for processing the input data received from the
        client, making a prediction using the model, and returning a readable response
        to the client.
        """
        # Let's convert the input data into a DataFrame so we can process it
        # using the Scikit-Learn transformers.
        model_input = pd.DataFrame([sample.model_dump() for sample in model_input])

        if model_input.empty:
            self.logger.warning("Received an empty request.")
            return []

        n_samples = len(model_input)
        sample_word = "samples" if n_samples > 1 else "sample"
        self.logger.info("Received prediction request with %d %s", n_samples, sample_word)

        version_label, components = self._resolve_version(params)
        self.logger.info("Serving model version '%s'", version_label)

        cache_key = (version_label, self._cache_key(model_input))
        if cache_key in self._cache:
            self.logger.info("Cache hit — returning cached prediction for %d %s", n_samples, sample_word)
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        model_output = []
        start = time.perf_counter()

        transformed_payload = self.process_input(
            model_input,
            features_transformer=components["features_transformer"],
        )
        if transformed_payload is not None:
            self.logger.info("Making a prediction using the transformed payload...")
            try:
                predictions = components["model"].predict(transformed_payload, verbose=0)
            except Exception:
                self.logger.exception("There was an error during model inference.")
                return []

            model_output = self.process_output(
                predictions,
                target_transformer=components["target_transformer"],
            )
        else:
            self.logger.error(
                "Prediction skipped: input processing failed for %d %s.",
                n_samples,
                sample_word,
            )

        latency = time.perf_counter() - start
        self.logger.info(
            "Prediction completed in %.3fs for %d %s",
            latency,
            n_samples,
            sample_word,
        )

        if len(self._cache) >= self._cache_max_size:
            self._cache.popitem(last=False)
        self._cache[cache_key] = model_output

        if self.backend is not None:
            self.backend.save(model_input, model_output)

        self.logger.debug("%s", model_output)

        return model_output

    def process_input(
        self,
        payload: pd.DataFrame,
        features_transformer=None,
    ) -> pd.DataFrame | None:
        """Process the input data received from the client.

        This method is responsible for transforming the input data received from the
        client into a format that can be used by the model.
        """
        self.logger.info("Transforming payload...")

        ft = features_transformer if features_transformer is not None else self.features_transformer

        # We need to transform the payload using the transformer. This can raise an
        # exception if the payload is not valid, in which case we should return None
        # to indicate that the prediction should not be made.
        try:
            result = ft.transform(payload)
        except Exception:
            self.logger.exception("There was an error processing the payload.")
            return None

        return result

    def process_output(self, output: np.ndarray, target_transformer=None) -> list[dict[str, Any]]:
        """Process the prediction received from the model.

        This method is responsible for transforming the prediction received from the
        model into a readable format that will be returned to the client.
        """
        self.logger.info("Processing prediction received from the model...")

        result = []
        if output is not None:
            prediction = np.argmax(output, axis=1)
            confidence = np.max(output, axis=1)

            tt = target_transformer if target_transformer is not None else self.target_transformer

            # Let's transform the prediction index back to the
            # original species. We can use the target transformer
            # to access the list of classes.
            classes = tt.named_transformers_["species"].categories_[0]
            prediction = np.vectorize(lambda x: classes[x])(prediction)

            # We can now return the prediction and the confidence from the model.
            # Notice that we need to unwrap the numpy values so we can serialize the
            # output as JSON. Predictions below the confidence threshold are returned
            # as None to signal that the model is uncertain.
            result = []
            for p, c in zip(prediction, confidence, strict=True):
                if c.item() < self._confidence_threshold:
                    self.logger.warning(
                        "Low-confidence prediction (%.3f < threshold %.3f) — returning uncertain response.",
                        c.item(),
                        self._confidence_threshold,
                    )
                    result.append({"prediction": None, "confidence": c.item()})
                else:
                    result.append({"prediction": p.item(), "confidence": c.item()})

        return result

    def _resolve_version(self, params: dict | None) -> tuple[str, dict]:
        """Return the version label and components for the requested version.

        Checks params["version"] first, then falls back to the default version.
        Logs a warning and uses the default when an unknown version is requested.
        """
        requested = (params or {}).get("version", self._default_version)
        if requested not in self._versions:
            self.logger.warning(
                "Version '%s' not found, falling back to default '%s'.",
                requested,
                self._default_version,
            )
            requested = self._default_version
        return requested, self._versions[requested]

    def _load_additional_versions(self, config_path: str) -> None:
        """Load extra model versions from a JSON config file.

        The file must map version labels to artifact paths:
        {
            "2": {
                "model": "/path/to/model.keras",
                "features_transformer": "/path/to/ft.pkl",
                "target_transformer": "/path/to/tt.pkl"
            }
        }
        """
        import keras

        path = Path(config_path)
        if not path.exists():
            self.logger.warning("Versions config file not found: %s", config_path)
            return

        try:
            config = json.loads(path.read_text())
        except Exception:
            self.logger.exception("Failed to parse versions config file.")
            return

        for label, paths in config.items():
            if label == self._default_version:
                self.logger.warning(
                    "Version '%s' in config shadows the default version, skipping.",
                    label,
                )
                continue
            try:
                self._versions[label] = {
                    "model": keras.saving.load_model(paths["model"]),
                    "features_transformer": joblib.load(paths["features_transformer"]),
                    "target_transformer": joblib.load(paths["target_transformer"]),
                }
                self.logger.info("Loaded model version '%s'", label)
            except Exception:
                self.logger.exception("Failed to load model version '%s'.", label)

    def _configure_threshold(self):
        """Read and validate the confidence threshold from the environment."""
        raw = os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.0")
        try:
            self._confidence_threshold = float(raw)
        except ValueError:
            self.logger.warning(
                "Invalid MODEL_CONFIDENCE_THRESHOLD '%s', using 0.0.",
                raw,
            )
            self._confidence_threshold = 0.0
        self.logger.info("Confidence threshold: %.2f", self._confidence_threshold)

    def _cache_key(self, df: pd.DataFrame) -> str:
        """Return a stable hex digest that uniquely identifies the DataFrame contents."""
        return hashlib.md5(
            pd.util.hash_pandas_object(df, index=False).values.tobytes()
        ).hexdigest()

    def _initialize_backend(self):
        """Initialize the model backend that the pipeline will use to store the data.

        The backend is responsible for storing the input requests and the predictions
        from the model. The inference pipeline will dynamically create an instance of
        the specified backend and use it to store the data.
        """
        # For the configuration to remain clean and easy to remember, we want to
        # reference backend classes as "backend.<class_name>" without having to include
        # their full class path. To accomplish this, we need to import the
        # inference.backend module so it's available to the `import_module` call.
        with suppress(ImportError):
            import inference.backend  # noqa: F401

        self.logger.info("Initializing model backend...")
        backend_class = os.getenv("MODEL_BACKEND", "backend.Local")

        if backend_class is not None:
            # We can optionally load a JSON configuration file and use it to initialize
            # the backend instance.
            backend_config = os.getenv("MODEL_BACKEND_CONFIG", None)

            try:
                if backend_config is not None:
                    backend_config = Path(backend_config)
                    backend_config = (
                        json.loads(backend_config.read_text())
                        if backend_config.exists()
                        else None
                    )

                module, cls = backend_class.rsplit(".", 1)
                module = importlib.import_module(module)
                self.backend = getattr(module, cls)(config=backend_config)
            except Exception:
                self.logger.exception(
                    'There was an error initializing backend "%s".',
                    backend_class,
                )

        self.logger.info("Backend: %s", backend_class if self.backend else None)

    def _load_artifacts(self, context: PythonModelContext | None):
        if context is None:
            self.logger.warning("No model context was provided.")
            return

        self._default_version = os.getenv("MODEL_DEFAULT_VERSION", "1")

        # By default, we want to use the TensorFlow backend for Keras.
        if not os.getenv("KERAS_BACKEND"):
            os.environ["KERAS_BACKEND"] = "tensorflow"

        import keras

        self.logger.info("Keras backend: %s", os.environ.get("KERAS_BACKEND"))

        # First, we need to load the transformation pipelines from the model artifacts.
        # These will help us transform the input data and the output predictions.
        self.features_transformer = joblib.load(
            context.artifacts["features_transformer"],
        )
        self.target_transformer = joblib.load(context.artifacts["target_transformer"])

        # Then, we can load the Keras model we trained.
        self.model = keras.saving.load_model(context.artifacts["model"])

        # Register the default version. The dict stores references to the same
        # objects, so test mocks applied to self.model etc. propagate automatically.
        self._versions[self._default_version] = {
            "model": self.model,
            "features_transformer": self.features_transformer,
            "target_transformer": self.target_transformer,
        }
        self.logger.info("Loaded default model version '%s'", self._default_version)

        versions_config_path = os.getenv("MODEL_VERSIONS_CONFIG")
        if versions_config_path:
            self._load_additional_versions(versions_config_path)

    def _configure_logging(self):
        """Configure how the logging system will behave."""
        import sys

        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )

        self.logger = logging.getLogger("model")


set_model(Model())
