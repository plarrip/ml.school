import os

from dotenv import load_dotenv
from metaflow import FlowSpec, environment, step


class Assignment7(FlowSpec):
    """Compare @environment decorator vs python-dotenv for env variables.

    - 'via_decorator' step: uses @environment to inject a variable at
      step execution time. This approach works both locally and on
      remote compute (e.g. AWS Batch), because Metaflow forwards the
      value when it launches the step.

    - 'via_dotenv' step: uses python-dotenv to load variables from a
      .env file at runtime. This only works if the .env file exists on
      the machine running the step — it won't work on remote compute
      unless the file is shipped there separately.
    """

    @environment(vars={"DECORATOR_GREETING": "Hello from @environment!"})
    @step
    def start(self):
        """Read an env variable injected by @environment."""
        value = os.getenv("DECORATOR_GREETING")
        print(f"[@environment] DECORATOR_GREETING = {value!r}")
        self.decorator_value = value
        self.next(self.via_dotenv)

    @step
    def via_dotenv(self):
        """Load env variables from the .env file using python-dotenv."""
        load_dotenv()  # reads the nearest .env file up the directory tree
        value = os.getenv("DOTENV_GREETING")
        print(f"[python-dotenv] DOTENV_GREETING = {value!r}")
        self.dotenv_value = value
        self.next(self.end)

    @step
    def end(self):
        """Summarise both approaches side by side."""
        print("\n--- Comparison ---")
        print(f"@environment  -> {self.decorator_value!r}")
        print(f"python-dotenv -> {self.dotenv_value!r}")
        print("\nKey difference:")
        print("  @environment forwards values to remote compute automatically.")
        print("  python-dotenv requires the .env file to exist on every machine.")


if __name__ == "__main__":
    Assignment7()
