import json

from metaflow import FlowSpec, Parameter, step


class Assignment4(FlowSpec):
    """Square each number in a list using a foreach loop.

    Flow: start -> square (foreach) -> join -> end
    Each number is squared in its own branch. The join step
    collects all results and prints the list and total sum.
    """

    numbers = Parameter(
        "numbers",
        help="JSON list of numbers to square (e.g. '[1,2,3,4,5]')",
        default="[1, 2, 3, 4, 5]",
    )

    @step
    def start(self):
        """Parse the numbers parameter and fan out via foreach."""
        self.nums = json.loads(self.numbers)
        print(f"Input numbers: {self.nums}")
        self.next(self.square, foreach="nums")

    @step
    def square(self):
        """Square the current number."""
        self.squared = self.input ** 2
        print(f"{self.input}² = {self.squared}")
        self.next(self.join)

    @step
    def join(self, inputs):
        """Collect squared results, print list and total sum."""
        self.squared_numbers = [i.squared for i in inputs]
        self.total = sum(self.squared_numbers)
        print(f"Squared numbers: {self.squared_numbers}")
        print(f"Total sum:       {self.total}")
        self.merge_artifacts(inputs, exclude=["squared"])
        self.next(self.end)

    @step
    def end(self):
        """Print the final results."""
        print(f"Input:   {self.nums}")
        print(f"Squared: {self.squared_numbers}")
        print(f"Sum:     {self.total}")


if __name__ == "__main__":
    Assignment4()
