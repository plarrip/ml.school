import json

from metaflow import Config, FlowSpec, Parameter, step


class Assignment10(FlowSpec):
    """Load a JSON config and allow runtime overrides via Parameters.

    Flow: start -> train -> end
    Config provides the base settings. Parameters override specific values,
    demonstrating how the flow behaves differently depending on the inputs.
    """

    config = Config(
        "config",
        default=".guide/introduction-to-metaflow/data/assignment10.json",
        parser=json.loads,
    )

    # These parameters override individual config values when provided.
    # Default to None so we can detect when the user hasn't set them.
    learning_rate = Parameter(
        "learning_rate",
        help="Override the learning rate from the config file",
        default=None,
        type=float,
        required=False,
    )

    max_epochs = Parameter(
        "max_epochs",
        help="Override the max epochs from the config file",
        default=None,
        type=int,
        required=False,
    )

    @step
    def start(self):
        """Resolve final settings: Parameters take priority over Config."""
        self.final_learning_rate = (
            self.learning_rate
            if self.learning_rate is not None
            else self.config.learning_rate
        )
        self.final_max_epochs = (
            self.max_epochs
            if self.max_epochs is not None
            else self.config.max_epochs
        )

        print("--- Configuration ---")
        print(f"  model:         {self.config.model}")
        print(f"  batch_size:    {self.config.batch_size}  (config only, no override)")
        print(f"  learning_rate: {self.final_learning_rate}"
              f"{'  <- overridden' if self.learning_rate is not None else '  (from config)'}")
        print(f"  max_epochs:    {self.final_max_epochs}"
              f"{'  <- overridden' if self.max_epochs is not None else '  (from config)'}")

        self.next(self.train)

    @step
    def train(self):
        """Simulate training with the resolved settings."""
        print(f"Training {self.config.model} for {self.final_max_epochs} epochs "
              f"with lr={self.final_learning_rate} and batch_size={self.config.batch_size}")
        self.next(self.end)

    @step
    def end(self):
        """Print a summary of what was used."""
        print("\n--- Final settings used ---")
        print(f"  model:         {self.config.model}")
        print(f"  learning_rate: {self.final_learning_rate}")
        print(f"  max_epochs:    {self.final_max_epochs}")
        print(f"  batch_size:    {self.config.batch_size}")


if __name__ == "__main__":
    Assignment10()
