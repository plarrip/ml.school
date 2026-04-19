import random

from metaflow import FlowSpec, card, step


class Assignment6(FlowSpec):
    """Generate a random dataset and visualize it with a custom card.

    Flow: start -> visualize -> end
    The 'visualize' step generates random data and renders a bar chart
    and a summary table using a custom HTML card with Chart.js.
    """

    @step
    def start(self):
        """Generate a random dataset of 10 values."""
        self.labels = [f"Item {i}" for i in range(1, 11)]
        self.values = [random.randint(10, 100) for _ in range(10)]
        print(f"Generated dataset: {list(zip(self.labels, self.values))}")
        self.next(self.visualize)

    @card(type="html")
    @step
    def visualize(self):
        """Build an HTML card with a bar chart and a summary table."""
        total = sum(self.values)
        average = total / len(self.values)
        minimum = min(self.values)
        maximum = max(self.values)

        rows = "".join(
            f"<tr><td>{label}</td><td>{value}</td></tr>"
            for label, value in zip(self.labels, self.values)
        )

        labels_js = str(self.labels)
        values_js = str(self.values)

        self.html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ font-family: sans-serif; padding: 24px; max-width: 800px; }}
                h1 {{ color: #333; }}
                canvas {{ margin-bottom: 32px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
                th {{ background: #f0f0f0; }}
                .stats {{ display: flex; gap: 24px; margin-bottom: 24px; }}
                .stat {{ background: #e8f4fd; padding: 12px 20px; border-radius: 8px; }}
                .stat span {{ font-size: 1.5em; font-weight: bold; color: #1a73e8; }}
            </style>
        </head>
        <body>
            <h1>Random Dataset Report</h1>

            <div class="stats">
                <div class="stat">Total<br><span>{total}</span></div>
                <div class="stat">Average<br><span>{average:.1f}</span></div>
                <div class="stat">Min<br><span>{minimum}</span></div>
                <div class="stat">Max<br><span>{maximum}</span></div>
            </div>

            <canvas id="chart" height="100"></canvas>
            <script>
                new Chart(document.getElementById('chart'), {{
                    type: 'bar',
                    data: {{
                        labels: {labels_js},
                        datasets: [{{
                            label: 'Value',
                            data: {values_js},
                            backgroundColor: 'rgba(26, 115, 232, 0.7)',
                            borderColor: 'rgba(26, 115, 232, 1)',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }} }}
                }});
            </script>

            <h2>Data Table</h2>
            <table>
                <tr><th>Label</th><th>Value</th></tr>
                {rows}
            </table>
        </body>
        </html>
        """
        self.next(self.end)

    @step
    def end(self):
        """Print a summary of the dataset."""
        print(f"Total:   {sum(self.values)}")
        print(f"Average: {sum(self.values) / len(self.values):.1f}")
        print(f"Min:     {min(self.values)}")
        print(f"Max:     {max(self.values)}")


if __name__ == "__main__":
    Assignment6()
