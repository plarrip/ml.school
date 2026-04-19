from metaflow import FlowSpec, step


class Assignment2(FlowSpec):
    """Track a sequence of numerical operations on a single artifact.

    Flow: start -> add -> subtract -> multiply -> end
    Each step applies an arithmetic operation and records the new value.
    The final step prints the history, sum, and average.
    """

    @step
    def start(self):
        """Initialize the artifact and history list."""
        self.value = 10
        self.history = [self.value]
        print(f"Initial value: {self.value}")
        self.next(self.add)

    @step
    def add(self):
        """Add 5 to the current value."""
        self.value += 5
        self.history.append(self.value)
        print(f"After addition (+5): {self.value}")
        self.next(self.subtract)

    @step
    def subtract(self):
        """Subtract 3 from the current value."""
        self.value -= 3
        self.history.append(self.value)
        print(f"After subtraction (-3): {self.value}")
        self.next(self.multiply)

    @step
    def multiply(self):
        """Multiply the current value by 2."""
        self.value *= 2
        self.history.append(self.value)
        print(f"After multiplication (*2): {self.value}")
        self.next(self.end)

    @step
    def end(self):
        """Print the full history, sum, and average."""
        total = sum(self.history)
        average = total / len(self.history)
        print(f"\nValue history: {self.history}")
        print(f"Sum:           {total}")
        print(f"Average:       {average:.2f}")


if __name__ == "__main__":
    Assignment2()
