import statistics

from metaflow import FlowSpec, step


class Assignment12(FlowSpec):
    """Three parallel branches compute stats on the same dataset.

    Flow: start -> branch_mean, branch_median, branch_std -> join -> end
    All branches set self.result (overlapping name).
    merge_artifacts(exclude=["result"]) resolves the conflict;
    branch-specific values are accessed via inputs.<branch_name>.result.
    """

    @step
    def start(self):
        """Define the dataset and fan out to three stat branches."""
        self.data = [4, 8, 15, 16, 23, 42, 7, 3, 11, 19]
        print(f"Dataset: {self.data}")
        self.next(self.branch_mean, self.branch_median, self.branch_std)

    @step
    def branch_mean(self):
        """Calculate the mean and store as self.result."""
        self.result = statistics.mean(self.data)
        print(f"[branch_mean]   result = {self.result}")
        self.next(self.join)

    @step
    def branch_median(self):
        """Calculate the median and store as self.result."""
        self.result = statistics.median(self.data)
        print(f"[branch_median] result = {self.result}")
        self.next(self.join)

    @step
    def branch_std(self):
        """Calculate the standard deviation and store as self.result."""
        self.result = statistics.stdev(self.data)
        print(f"[branch_std]    result = {self.result:.4f}")
        self.next(self.join)

    @step
    def join(self, inputs):
        """Merge branches and resolve the conflicting 'result' artifact.

        Access each branch's result via inputs.<branch_name>.result before
        calling merge_artifacts so the shared artifact name doesn't cause
        an error.
        """
        # Preserve branch-specific computations under named artifacts
        self.mean   = inputs.branch_mean.result
        self.median = inputs.branch_median.result
        self.std    = inputs.branch_std.result

        # Exclude 'result' — different values across branches, we handle it above
        self.merge_artifacts(inputs, exclude=["result"])

        # Aggregate all branch results into a single summary artifact
        self.summary = {
            "mean":   self.mean,
            "median": self.median,
            "std":    self.std,
        }

        self.next(self.end)

    @step
    def end(self):
        """Print branch-specific results and the aggregated summary."""
        print(f"\nDataset: {self.data}")
        print("\nBranch-specific results (preserved):")
        print(f"  branch_mean   -> mean:   {self.mean:.4f}")
        print(f"  branch_median -> median: {self.median}")
        print(f"  branch_std    -> std:    {self.std:.4f}")
        print(f"\nAggregated summary artifact: {self.summary}")


if __name__ == "__main__":
    Assignment12()
