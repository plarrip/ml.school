from metaflow import FlowSpec, step


class Assignment3(FlowSpec):
    """Parallel branches that each transform a shared starting value.

    Flow: start -> (branch_add | branch_multiply) -> join -> end
    branch_add adds a constant; branch_multiply multiplies by a constant.
    The join step prints both outcomes and computes their sum.
    """

    @step
    def start(self):
        """Initialize the starting value."""
        self.value = 10
        print(f"Starting value: {self.value}")
        self.next(self.branch_add, self.branch_multiply)

    @step
    def branch_add(self):
        """Add 7 to the starting value."""
        self.result = self.value + 7
        print(f"branch_add result: {self.value} + 7 = {self.result}")
        self.next(self.join)

    @step
    def branch_multiply(self):
        """Multiply the starting value by 3."""
        self.result = self.value * 3
        print(f"branch_multiply result: {self.value} * 3 = {self.result}")
        self.next(self.join)

    @step
    def join(self, inputs):
        """Merge branches, print outcomes, and compute their sum."""
        add_result = inputs.branch_add.result
        multiply_result = inputs.branch_multiply.result

        print(f"branch_add outcome:      {add_result}")
        print(f"branch_multiply outcome: {multiply_result}")

        self.total = add_result + multiply_result
        print(f"Sum of both outcomes:    {self.total}")

        self.merge_artifacts(inputs, exclude=["result"])
        self.next(self.end)

    @step
    def end(self):
        """Print the final aggregated result."""
        print(f"Starting value: {self.value}")
        print(f"Final total:    {self.total}")


if __name__ == "__main__":
    Assignment3()
