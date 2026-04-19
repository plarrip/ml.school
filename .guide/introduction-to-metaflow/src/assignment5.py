import random

from metaflow import FlowSpec, retry, step


class Assignment5(FlowSpec):
    """Demonstrate the @retry decorator with a flaky step.

    The 'call_service' step simulates an external service that fails
    50% of the time. The @retry decorator retries it up to 3 times
    before the flow gives up and raises an error.
    """

    @step
    def start(self):
        """Initialize the flow."""
        print("Starting flow — will call a flaky external service.")
        self.next(self.call_service)

    @retry(times=3)
    @step
    def call_service(self):
        """Simulate a flaky external service call (50% failure rate)."""
        if random.random() < 0.5:
            raise RuntimeError("Service unavailable! Retrying...")
        self.response = "Success! Service responded with data."
        print(self.response)
        self.next(self.end)

    @step
    def end(self):
        """Print the final response from the service."""
        print(f"Final response: {self.response}")


if __name__ == "__main__":
    Assignment5()
