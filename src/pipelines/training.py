import os
from pathlib import Path

from metaflow import (
    Parameter,
    card,
    current,
    environment,
    step,
)

# The `register` step needs the actual `inference/backend.py` and `inference/model.py`
# files on disk (via `mlflow.pyfunc.log_model`). Metaflow only auto-packages modules
# that are already imported by the time it builds the code package, so we import this
# at the top level rather than lazily inside the step.
import inference
from common.pipeline import Pipeline, dataset

environment_variables = {
    "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "tensorflow"),
}


class FeatureEngineer:
    """Sklearn-compatible transformer that adds domain-specific engineered features.

    New columns added:
    - culmen_ratio: bill length / bill depth (captures bill shape, a key species trait)
    - body_mass_index: body mass / flipper length (captures body density proxy)
    """

    def fit(self, X, y=None):  # noqa: ARG002, N803
        """No fitting required; this transformer is stateless."""
        return self

    def transform(self, X):  # noqa: N803
        """Append the engineered features and return the augmented DataFrame."""
        result = X.copy()
        result["culmen_ratio"] = result["culmen_length_mm"] / result["culmen_depth_mm"]
        result["body_mass_index"] = result["body_mass_g"] / result["flipper_length_mm"]
        return result

    def get_params(self, deep=True):  # noqa: ARG002, FBT002
        """Return empty params dict (required by sklearn clone/Pipeline)."""
        return {}

    def set_params(self, **params: object):  # noqa: ARG002
        """No-op (required by sklearn Pipeline)."""
        return self


def build_features_transformer(feature_engineering=False):  # noqa: FBT002
    """Build a Scikit-Learn transformer to preprocess the feature columns."""
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        # We can use the `handle_unknown="ignore"` parameter to ignore unseen categories
        # during inference. When encoding an unknown category, the transformer will
        # return an all-zero vector.
        OneHotEncoder(handle_unknown="ignore"),
    )

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                # We'll apply the numeric transformer to all columns that are not
                # categorical (object).
                make_column_selector(dtype_exclude="object"),
            ),
            (
                "categorical",
                categorical_transformer,
                # We want to make sure we ignore the target column which is also a
                # categorical column. To accomplish this, we can specify the column
                # names we only want to encode.
                ["island", "sex"],
            ),
        ],
    )

    if feature_engineering:
        return Pipeline([
            ("engineer", FeatureEngineer()),
            ("transform", column_transformer),
        ])

    return column_transformer


def build_target_transformer():
    """Build a Scikit-Learn transformer to preprocess the target column."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    return ColumnTransformer(
        transformers=[("species", OrdinalEncoder(), ["species"])],
    )


def build_model(input_shape, learning_rate=0.01, learning_rate_decay=False):  # noqa: FBT002
    """Build and compile the neural network to predict the species of a penguin."""
    from keras import Input, layers, models, optimizers

    if learning_rate_decay:
        # Decay the learning rate by 4% every 100 steps so early training takes
        # large steps and late training fine-tunes near the optimum.
        lr = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=100,
            decay_rate=0.96,
        )
    else:
        lr = learning_rate

    model = models.Sequential(
        [
            Input(shape=(input_shape,)),
            layers.Dense(10, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ],
    )

    model.compile(
        optimizer=optimizers.SGD(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


class Training(Pipeline):
    """Training pipeline.

    This pipeline trains, evaluates, and registers a model to predict the species of
    a given penguin.
    """

    feature_engineering = Parameter(
        "feature-engineering",
        help="Enable domain-specific feature engineering (culmen ratio, body mass index).",  # noqa: E501
        default=False,
    )

    learning_rate_decay = Parameter(
        "learning-rate-decay",
        help="Enable exponential decay of the learning rate during training.",
        default=False,
    )

    training_epochs = Parameter(
        "training-epochs",
        help="Maximum epochs to train; early stopping may halt sooner.",
        default=50,
    )

    early_stopping_patience = Parameter(
        "early-stopping-patience",
        help="Epochs without val_loss improvement before stopping training.",
        default=5,
    )

    training_batch_size = Parameter(
        "training-batch-size",
        help="Batch size that will be used to train the model.",
        default=32,
    )

    accuracy_threshold = Parameter(
        "accuracy-threshold",
        help="Minimum accuracy threshold required to register the model.",
        default=0.7,
    )

    @dataset
    @card
    @step
    def start(self):
        """Start and prepare the Training pipeline."""
        import mlflow

        self.logger.info("MLflow tracking server: %s", self.mlflow_tracking_uri)

        self.mode = "production" if current.is_production else "development"
        self.logger.info("Running flow in %s mode.", self.mode)

        try:
            # Let's start a new MLflow run to track the execution of this flow. We want
            # to set the name of the MLflow run to the Metaflow run ID so we can easily
            # recognize how they relate to each other.
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            raise RuntimeError(message) from e

        mlflow.log_param("feature_engineering", self.feature_engineering)
        mlflow.log_param("learning_rate_decay", self.learning_rate_decay)

        # Now that everything is set up, we want to run a cross-validation process
        # to evaluate the model and train a final model on the entire dataset. Since
        # these two steps are independent, we can run them in parallel.
        self.next(self.cross_validation, self.transform)

    @card
    @step
    def cross_validation(self):
        """Generate the indices to split the data for the cross-validation process."""
        from sklearn.model_selection import KFold

        # We are going to use a 5-fold cross-validation process. We'll shuffle the data
        # before splitting it into batches.
        kfold = KFold(n_splits=5, shuffle=True)

        # We can now generate the indices to split the dataset into training and test
        # sets. This will return a tuple with the fold number and the training and test
        # indices for each of 5 folds.
        self.folds = list(enumerate(kfold.split(self.data)))

        # We can use a `foreach` to run every fold on a separate branch. Notice how we
        # pass the tuple with the fold number and the indices to next step.
        self.next(self.transform_fold, foreach="folds")

    @step
    def transform_fold(self):
        """Transform the data to build a model during the cross-validation process.

        This step will run for each fold in the cross-validation process. It uses
        a SciKit-Learn pipeline to preprocess the dataset before training a model.
        """
        # Let's start by unpacking the indices representing the training and test data
        # for the current fold.
        self.fold, (self.train_indices, self.test_indices) = self.input
        self.logger.info("Transforming fold %d...", self.fold)

        # We can use the indices to split the data into training and test sets.
        train_data = self.data.iloc[self.train_indices]
        test_data = self.data.iloc[self.test_indices]

        # Let's build the SciKit-Learn pipeline to process the feature columns,
        # fit it to the training data and transform both the training and test data.
        features_transformer = build_features_transformer(self.feature_engineering)
        self.x_train = features_transformer.fit_transform(train_data)
        self.x_test = features_transformer.transform(test_data)

        # Finally, we can build the SciKit-Learn pipeline to process the target column,
        # fit it to the training data and transform both the training and test data.
        target_transformer = build_target_transformer()
        self.y_train = target_transformer.fit_transform(train_data)
        self.y_test = target_transformer.transform(test_data)

        # After processing the data and storing it as artifacts in the flow, we can move
        # to the training step.
        self.next(self.train_fold)

    @card
    @environment(vars=environment_variables)
    @step
    def train_fold(self):
        """Train a model as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It trains the
        model using the data we processed in the previous step.
        """
        import mlflow

        self.logger.info("Training fold %d...", self.fold)

        # We want to track the training process under the same MLflow run we started at
        # the beginning of the flow. Since we are running cross-validation, we will
        # create a nested run for each fold to keep track of each model individually.
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(
                run_name=f"cross-validation-fold-{self.fold}",
                nested=True,
            ) as run,
        ):
            # Let's store the identifier of the nested run in an artifact so we can
            # reuse it later when we evaluate the model.
            self.mlflow_fold_run_id = run.info.run_id

            # We are currently training a model corresponding to an individual fold,
            # so we don't want to log that model because it's useless.
            mlflow.autolog(log_models=False)

            # Use the fold's test split as validation data so early stopping can monitor
            # generalisation without touching the final evaluation metrics.
            from keras.callbacks import EarlyStopping

            self.model = build_model(self.x_train.shape[1], learning_rate_decay=self.learning_rate_decay)
            history = self.model.fit(
                self.x_train,
                self.y_train,
                epochs=self.training_epochs,
                batch_size=self.training_batch_size,
                validation_data=(self.x_test, self.y_test),
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss",
                        patience=self.early_stopping_patience,
                        restore_best_weights=True,
                    ),
                ],
                verbose=0,
            )

            self.epochs_trained = len(history.history["loss"])
            mlflow.log_metrics(
                {
                    "epochs_trained": self.epochs_trained,
                    "val_loss": history.history["val_loss"][-1],
                    "val_accuracy": history.history["val_accuracy"][-1],
                },
                run_id=self.mlflow_fold_run_id,
            )

        self.logger.info(
            "Fold %d - epochs: %d - train_loss: %f - val_loss: %f",
            self.fold,
            self.epochs_trained,
            history.history["loss"][-1],
            history.history["val_loss"][-1],
        )

        # After training a model for this fold, we want to evaluate it.
        self.next(self.evaluate_fold)

    @card(type="confusion_matrix")
    @environment(vars=environment_variables)
    @step
    def evaluate_fold(self):
        """Evaluate the model we created as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It evaluates
        the model using the test data associated with the current fold.
        """
        import mlflow
        import numpy as np
        from sklearn.metrics import confusion_matrix, precision_score, recall_score

        self.logger.info("Evaluating fold %d...", self.fold)

        # Let's evaluate the model using the test data we processed before.
        self.test_loss, self.test_accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=0,
        )

        # Compute predictions once and reuse them for all metrics.
        y_pred = np.argmax(self.model.predict(self.x_test, verbose=0), axis=1)
        y_true = self.y_test.flatten().astype(int)

        self.test_precision = precision_score(y_true, y_pred, average="weighted")
        self.test_recall = recall_score(y_true, y_pred, average="weighted")

        # Compute the confusion matrix for the card visualization.
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        self.confusion_matrix_labels = ["Adelie", "Chinstrap", "Gentoo"]

        self.logger.info(
            "Fold %d - loss: %f - accuracy: %f - precision: %f - recall: %f",
            self.fold,
            self.test_loss,
            self.test_accuracy,
            self.test_precision,
            self.test_recall,
        )

        # Let's track the evaluation metrics under the nested MLflow run corresponding
        # to the current fold.
        mlflow.log_metrics(
            {
                "test_loss": self.test_loss,
                "test_accuracy": self.test_accuracy,
                "test_precision": self.test_precision,
                "test_recall": self.test_recall,
            },
            run_id=self.mlflow_fold_run_id,
        )

        # When we finish evaluating the models in the cross-validation process, we want
        # to average the scores to determine the overall model performance.
        self.next(self.average_scores)

    @card
    @step
    def average_scores(self, inputs):
        """Averages the scores computed for each individual model."""
        import mlflow
        import numpy as np

        # We need access to the `mlflow_run_id` artifact that we set at the start of
        # the flow, but since we are in a join step, we need to merge the artifacts
        # from the incoming branches to make `mlflow_run_id` available. This merge will
        # discard every artifact that was created in the previous branches and keep only
        # the `mlflow_run_id` artifact.
        self.merge_artifacts(inputs, include=["mlflow_run_id"])

        # Let's calculate the mean and standard deviation of the accuracy and loss from
        # all the cross-validation folds.
        metrics = [[i.test_accuracy, i.test_loss] for i in inputs]

        self.test_accuracy, self.test_loss = np.mean(metrics, axis=0)
        self.test_accuracy_std, self.test_loss_std = np.std(metrics, axis=0)
        self.mean_epochs_trained = int(np.mean([i.epochs_trained for i in inputs]))

        self.logger.info("Accuracy: %f ±%f", self.test_accuracy, self.test_accuracy_std)
        self.logger.info("Loss: %f ±%f", self.test_loss, self.test_loss_std)
        self.logger.info("Mean epochs trained: %d", self.mean_epochs_trained)

        # Let's log the model metrics on the parent run.
        mlflow.log_metrics(
            {
                "test_accuracy": self.test_accuracy,
                "test_accuracy_std": self.test_accuracy_std,
                "test_loss": self.test_loss,
                "test_loss_std": self.test_loss_std,
                "mean_epochs_trained": self.mean_epochs_trained,
            },
            run_id=self.mlflow_run_id,
        )

        # After we finish evaluating the cross-validation process, we can send the flow
        # to the registration step to register the final version of the model.
        self.next(self.register)

    @card
    @step
    def transform(self):
        """Apply the transformation pipeline to the entire dataset.

        We'll use the entire dataset to build the final model, so we need to transform
        the dataset before training.

        We want to store the transformers as artifacts so we can later use them
        to transform the input data during inference.
        """
        # Let's build the SciKit-Learn pipeline and transform the dataset features.
        self.features_transformer = build_features_transformer(self.feature_engineering)
        self.x = self.features_transformer.fit_transform(self.data)

        # Let's build the SciKit-Learn pipeline and transform the target column.
        self.target_transformer = build_target_transformer()
        self.y = self.target_transformer.fit_transform(self.data)

        # Now that we have transformed the data, we can train the final model.
        self.next(self.train)

    @card
    @environment(vars=environment_variables)
    @step
    def train(self):
        """Train the final model using the entire dataset."""
        import mlflow

        self.logger.info("Training final model...")

        # Let's log the training process under the current MLflow run.
        with mlflow.start_run(run_id=self.mlflow_run_id, log_system_metrics=True):
            # We want to log the model manually, so let's disable automatic logging.
            mlflow.autolog(log_models=False)

            from keras.callbacks import EarlyStopping

            self.model = build_model(self.x.shape[1], learning_rate_decay=self.learning_rate_decay)
            history = self.model.fit(
                self.x,
                self.y,
                epochs=self.training_epochs,
                batch_size=self.training_batch_size,
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss",
                        patience=self.early_stopping_patience,
                        restore_best_weights=True,
                    ),
                ],
                verbose=2,
            )

            epochs_trained = len(history.history["loss"])
            self.logger.info("Trained for %d epochs.", epochs_trained)
            mlflow.log_metrics(
                {
                    "epochs_trained": epochs_trained,
                    "val_loss": history.history["val_loss"][-1],
                    "val_accuracy": history.history["val_accuracy"][-1],
                },
            )

        # After we finish training the model, we want to register it.
        self.next(self.register)

    @environment(vars=environment_variables)
    @step
    def register(self, inputs):
        """Register the model in the model registry.

        This function will prepare and register the final model in the model registry
        if its accuracy is above a predefined threshold.
        """
        import tempfile

        import mlflow

        # Since this is a join step, we need to merge the artifacts from the incoming
        # branches to make them available here.
        self.merge_artifacts(inputs)

        # We only want to register the model if its accuracy is above the
        # `accuracy_threshold` parameter.
        if self.test_accuracy >= self.accuracy_threshold:
            self.registered = True
            self.logger.info("Registering model...")

            # We'll register the model under the current MLflow run. We also need to
            # create a temporary directory to store the model artifacts.
            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                self.artifacts = self._get_model_artifacts(directory)
                self.pip_requirements = self._get_model_pip_requirements()

                # Let's locate the `inference` package. We can't derive this from
                # `__file__` because Metaflow doesn't preserve the local `src` layout
                # when it packages and unpacks the code on a remote compute platform.
                root = Path(inference.__file__).parent
                self.code_paths = [(root / "backend.py").as_posix()]

                # We can now register the model in the model registry. This will
                # automatically create a new version of the model.
                mlflow.pyfunc.log_model(
                    name="model",
                    python_model=root / "model.py",
                    registered_model_name="penguins",
                    code_paths=self.code_paths,
                    artifacts=self.artifacts,
                    pip_requirements=self.pip_requirements,
                )

        else:
            self.registered = False
            self.logger.info(
                "The accuracy of the model (%.2f) is lower than the accuracy threshold "
                "(%.2f). Skipping model registration.",
                self.test_accuracy,
                self.accuracy_threshold,
            )

        # Let's now move to the final step of the pipeline.
        self.next(self.end)

    @step
    def end(self):
        """End the Training pipeline."""
        self.logger.info("The pipeline finished successfully.")

    def _get_model_artifacts(self, directory: str):
        """Return the list of artifacts that will be included with model.

        The model must preprocess the raw input data before making a prediction, so we
        need to include the Scikit-Learn transformers as part of the model package.
        """
        import joblib

        # Let's start by saving the model inside the supplied directory.
        model_path = (Path(directory) / "model.keras").as_posix()
        self.model.save(model_path)

        # We also want to save the Scikit-Learn transformers so we can package them
        # with the model and use them during inference.
        features_transformer_path = (Path(directory) / "features.joblib").as_posix()
        target_transformer_path = (Path(directory) / "target.joblib").as_posix()
        joblib.dump(self.features_transformer, features_transformer_path)
        joblib.dump(self.target_transformer, target_transformer_path)

        return {
            "model": model_path,
            "features_transformer": features_transformer_path,
            "target_transformer": target_transformer_path,
        }

    def _get_model_pip_requirements(self):
        """Return the list of required packages to run the model in production."""
        import keras
        import numpy as np
        import pandas as pd
        import sklearn
        import tensorflow as tf

        return [
            f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}",
            f"keras=={keras.__version__}",
            f"tensorflow=={tf.__version__}",
        ]

if __name__ == "__main__":
    Training()
