from metaflow import FlowSpec, step


class Assignment11(FlowSpec):
    """Demonstrate merge_artifacts() handling of conflicting artifact names.

    Flow: start -> branch_square, branch_double -> join -> end
    Both branches set self.result, which conflicts at the join step.
    We resolve it by reading branch-specific values before calling
    merge_artifacts(inputs, exclude=["result"]).
    """

    @step
    def start(self):
        """Initialize the base value and fan out to two branches."""
        self.value = 10
        self.next(self.branch_square, self.branch_double)

    @step
    def branch_square(self):
        """Square the value and store as self.result (conflicting name)."""
        self.result = self.value ** 2
        print(f"[branch_square] result = {self.result}")
        self.next(self.join)

    @step
    def branch_double(self):
        """Double the value and store as self.result (conflicting name)."""
        self.result = self.value * 2
        print(f"[branch_double] result = {self.result}")
        self.next(self.join)

    @step
    def join(self, inputs):
        """Merge branches, resolving the conflicting 'result' artifact.

        Without exclude=["result"], merge_artifacts() would raise an error
        because both branches have different values for self.result.
        We read branch-specific values first, then exclude the conflict.
        """
        # Access branch-specific results before merging
        squared = inputs.branch_square.result
        doubled = inputs.branch_double.result

        print(f"[join] branch_square.result = {squared}")
        print(f"[join] branch_double.result = {doubled}")

        # Resolve the conflict by excluding 'result' from auto-merge
        self.merge_artifacts(inputs, exclude=["result"])

        # Store our own aggregated artifact instead
        self.squared = squared
        self.doubled = doubled
        self.combined = squared + doubled

        self.next(self.end)

    @step
    def end(self):
        """Print the final summary."""
        print(f"\nBase value:    {self.value}")
        print(f"Squared:       {self.squared}")
        print(f"Doubled:       {self.doubled}")
        print(f"Combined:      {self.combined}")


if __name__ == "__main__":
    Assignment11()
