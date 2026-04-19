import csv
import io

from metaflow import FlowSpec, IncludeFile, step


class Assignment8(FlowSpec):
    """Load a CSV file with IncludeFile, parse it, and handle errors.

    Flow: start -> process -> end
    The 'process' step parses the CSV content and reports row/column counts.
    Error handling covers empty files and malformed content.
    """

    data_file = IncludeFile(
        "data_file",
        is_text=True,
        help="Path to the CSV file to process",
        default=".guide/introduction-to-metaflow/data/sample.csv",
    )

    @step
    def start(self):
        """Load the file and pass it along."""
        print("File loaded — moving to processing step.")
        self.next(self.process)

    @step
    def process(self):
        """Parse the CSV content and report row/column counts."""
        content = self.data_file.strip()

        if not content:
            print("Error: the file is empty.")
            self.rows = 0
            self.columns = 0
            self.next(self.end)
            return

        try:
            reader = csv.reader(io.StringIO(content))
            all_rows = list(reader)

            if len(all_rows) < 2:
                print("Error: file has no data rows (only a header or nothing).")
                self.rows = 0
                self.columns = 0
                self.next(self.end)
                return

            header = all_rows[0]
            data_rows = all_rows[1:]

            # Check for malformed rows (inconsistent column count)
            expected_cols = len(header)
            malformed = [
                i + 2
                for i, row in enumerate(data_rows)
                if len(row) != expected_cols
            ]
            if malformed:
                print(f"Error: malformed rows at line(s) {malformed}.")
                self.rows = 0
                self.columns = 0
                self.next(self.end)
                return

            self.rows = len(data_rows)
            self.columns = expected_cols
            print(f"Columns ({self.columns}): {header}")
            print(f"Data rows: {self.rows}")

        except csv.Error as e:
            print(f"Error: could not parse CSV content — {e}")
            self.rows = 0
            self.columns = 0

        self.next(self.end)

    @step
    def end(self):
        """Print the final summary."""
        if self.rows == 0:
            print("Processing failed — see errors above.")
        else:
            print(f"\nSummary: {self.rows} rows x {self.columns} columns")


if __name__ == "__main__":
    Assignment8()
