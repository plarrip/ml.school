from metaflow import (
    Parameter,
    card,
    step,
)

from common.pipeline import Pipeline, backend, dataset


class Monitoring(Pipeline):
    """A monitoring pipeline to monitor the performance of a hosted model.

    This pipeline runs a series of tests and generates several reports using the
    data captured by the hosted model and a reference dataset.
    """

    limit = Parameter(
        "samples",
        help=(
            "The maximum number of samples that will be loaded from the production "
            "datastore to run the monitoring tests and reports. The flow will load "
            "the most recent samples."
        ),
        default=500,
    )

    freshness_days = Parameter(
        "freshness-days",
        help="Maximum age in days of the most recent production sample. The pipeline will fail if data is staler than this.",
        default=7,
        required=False,
    )

    accuracy_threshold = Parameter(
        "accuracy-threshold",
        help="Minimum accuracy the model must achieve. The pipeline will fail if accuracy drops below this value.",
        default=0.8,
        required=False,
    )

    days = Parameter(
        "days",
        help=(
            "If specified, only production samples from the last N days will be used. "
            "If not specified, all loaded samples will be used."
        ),
        default=None,
        required=False,
    )

    @dataset
    @backend
    @card
    @step
    def start(self):
        """Start the monitoring pipeline."""
        import pandas as pd
        from evidently import DataDefinition, Dataset, MulticlassClassification

        # Let's load the reference data. When running some of the tests and reports,
        # we need to have a prediction column in the reference data to match the
        # production dataset.
        reference_data = self.data
        reference_data["prediction"] = reference_data["species"]
        reference_data = reference_data.rename(
            columns={"species": "target"},
        )

        data_definition = DataDefinition(
            classification=[
                MulticlassClassification(
                    target="target",
                    prediction_labels="prediction",
                )
            ]
        )

        self.reference_dataset = Dataset.from_pandas(
            reference_data, data_definition=data_definition
        )

        # Let's now load the production data. We need to filter out the samples that
        # don't have ground truth labels.
        current_data = self.backend_impl.load(self.limit)
        if current_data is not None and not current_data.empty:
            from datetime import UTC, datetime, timedelta

            dates = pd.to_datetime(current_data["date"], utc=True)
            most_recent = dates.max()
            freshness_cutoff = datetime.now(UTC) - timedelta(days=int(self.freshness_days))
            if most_recent < freshness_cutoff:
                raise RuntimeError(
                    f"Production data is stale. Most recent sample: {most_recent.date()}."
                )

            if self.days:
                cutoff = datetime.now(UTC) - timedelta(days=int(self.days))
                current_data = current_data[dates >= cutoff]
            current_data = current_data.drop(columns=["date"])
            numeric_cols = [
                "culmen_length_mm",
                "culmen_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
            ]
            current_data[numeric_cols] = current_data[numeric_cols].apply(
                pd.to_numeric, errors="coerce"
            )
            current_data = (
                current_data[current_data["target"].notna()]
                if not current_data.empty
                else None
            )
        else:
            current_data = None

        # We want to make sure there's production data available to run the reports.
        # If there's no production data, we'll skip the reports that need it.
        self.current_dataset = None
        if current_data is not None and not current_data.empty:
            self.current_dataset = Dataset.from_pandas(
                current_data, data_definition=data_definition
            )

        self.next(self.data_summary_report)

    @card(type="html")
    @step
    def data_summary_report(self):
        """Generate a report with descriptive statistics for each column.

        This report will provide detailed feature statistics, and will run a few tests
        to check for missing values, duplicated rows, and other data quality issues.
        """
        from evidently import Report
        from evidently.metrics import DuplicatedRowCount
        from evidently.presets import ValueStats

        report = Report(
            [
                # These will generate statistics for each individual column.
                ValueStats(column="island", row_count_tests=[]),
                ValueStats(column="sex", row_count_tests=[]),
                ValueStats(column="culmen_length_mm", row_count_tests=[]),
                ValueStats(column="culmen_depth_mm", row_count_tests=[]),
                ValueStats(column="flipper_length_mm", row_count_tests=[]),
                ValueStats(column="body_mass_g", row_count_tests=[]),
                # This will check for duplicated rows in the dataset. Having duplicated
                # rows is not a problem for the model, but it might indicate an issue
                # with the data pipeline.
                DuplicatedRowCount(),
            ],
            include_tests=True,
        )

        # We only want to run the report if there's production data available.
        if self.current_dataset:
            result = report.run(
                current_data=self.current_dataset,
                reference_data=self.reference_dataset,
            )
            self.html = result.get_html_str(as_iframe=False)
        else:
            self._message("No production data.")

        self.next(self.data_drift_report)

    @card(type="html")
    @step
    def data_drift_report(self):
        """Generate a report visualizing data distribution and drift.

        This report will generate a visualization of the data distribution of every
        column and determine if there's any drift in the data.
        """
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(
            [
                # We want to report dataset drift as long as one of the columns has
                # drifted. We can accomplish this by specifying that the share of
                # drifting columns in the production dataset must stay under 10% (one
                # column drifting out of 6 columns represents 16.66%).
                DataDriftPreset(
                    columns=[
                        "island",
                        "sex",
                        "culmen_length_mm",
                        "culmen_depth_mm",
                        "flipper_length_mm",
                        "body_mass_g",
                        "target",
                    ],
                    drift_share=0.1,
                ),
            ],
            include_tests=True,
        )

        # We only want to run the report if there's production data available.
        if self.current_dataset:
            result = report.run(
                reference_data=self.reference_dataset,
                current_data=self.current_dataset,
            )
            self.html = result.get_html_str(as_iframe=False)
        else:
            self._message("No production data.")

        self.next(self.classification_report)

    @card(type="html")
    @step
    def classification_report(self):
        """Generate a Classification report.

        This report will evaluate the quality of the multi-class classification model.
        """
        from evidently import Report
        from evidently.presets import ClassificationPreset

        report = Report(
            [
                # This preset evaluates the quality of the classification model.
                ClassificationPreset(),
            ],
            include_tests=True,
        )

        # We only want to run the report if there's production data available.
        if self.current_dataset:
            result = report.run(
                # The reference data uses the same target as prediction, so skip
                # reference metrics to avoid comparing against itself.
                current_data=self.current_dataset,
                reference_data=self.reference_dataset,
            )
            self.html = result.get_html_str(as_iframe=False)
        else:
            self._message("No production data.")

        self.next(self.comprehensive_report)

    @card(type="html")
    @step
    def comprehensive_report(self):
        """Generate a combined dashboard with data quality and classification metrics."""
        from evidently import Report
        from evidently.presets import ClassificationPreset, DataSummaryPreset

        report = Report(
            [DataSummaryPreset(), ClassificationPreset()],
            include_tests=True,
        )

        if self.current_dataset:
            result = report.run(
                current_data=self.current_dataset,
                reference_data=self.reference_dataset,
            )
            self.html = result.get_html_str(as_iframe=False)
        else:
            self._message("No production data.")

        self.next(self.accuracy_check)

    @card(type="html")
    @step
    def accuracy_check(self):
        """Fail the pipeline if model accuracy drops below the configured threshold."""
        from evidently import Report
        from evidently.metrics.classification import Accuracy
        from evidently.tests import gte

        report = Report(
            [Accuracy(tests=[gte(float(self.accuracy_threshold))])],
        )

        if self.current_dataset:
            result = report.run(
                current_data=self.current_dataset,
                reference_data=self.reference_dataset,
            )
            self.html = result.get_html_str(as_iframe=False)

            if any(t.status != "SUCCESS" for t in result.tests_results):
                raise RuntimeError(
                    f"Model accuracy is below the threshold of {self.accuracy_threshold}."
                )
        else:
            self._message("No production data.")

        self.next(self.llm_summary)

    @card(type="html")
    @step
    def llm_summary(self):
        """Use an LLM to generate a plain-language summary of the monitoring metrics."""
        import os

        from dotenv import load_dotenv
        from evidently import Report
        from evidently.metrics import DriftedColumnsCount
        from evidently.metrics.classification import Accuracy, F1Score
        from google import genai

        if not self.current_dataset:
            self._message("No production data.")
            self.next(self.end)
            return

        report = Report(
            [Accuracy(), F1Score(), DriftedColumnsCount()],
            include_tests=True,
        )
        result = report.run(
            current_data=self.current_dataset,
            reference_data=self.reference_dataset,
        )

        metric_rows = "".join(
            f"""<tr>
                <td>{t.description}</td>
                <td style="color: {'#34a853' if t.status == 'SUCCESS' else '#ea4335'}; font-weight: bold;">
                    {'✅ Pass' if t.status == 'SUCCESS' else '❌ Fail'}
                </td>
            </tr>"""
            for t in result.tests_results
        )
        metrics_text = "\n".join(f"- {t.description} [{t.status}]" for t in result.tests_results)

        prompt = (
            "You are an ML monitoring assistant for a penguin species classifier "
            "running in production. Below are the latest monitoring metrics:\n\n"
            f"{metrics_text}\n\n"
            "Write a concise 3-5 sentence summary of the model's current health. "
            "Highlight any concerns and state clearly whether immediate action is needed."
        )

        load_dotenv()
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        self.html = f"""<!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 24px; color: #333; }}
                h1 {{ color: #1a1a2e; margin-bottom: 8px; }}
                h2 {{ color: #444; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.1em; margin: 32px 0 12px; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 8px; }}
                th {{ text-align: left; font-size: 0.75em; text-transform: uppercase; color: #888; padding: 6px 12px; border-bottom: 2px solid #eee; }}
                td {{ padding: 10px 12px; border-bottom: 1px solid #f0f0f0; font-size: 0.95em; }}
                tr:hover td {{ background: #fafafa; }}
                .summary {{ background: #f6fef6; border-left: 4px solid #34a853; padding: 18px 22px; border-radius: 4px; line-height: 1.8; font-size: 1.05em; white-space: pre-wrap; }}
            </style>
        </head>
        <body>
            <h1>Monitoring LLM Summary</h1>

            <h2>Metrics</h2>
            <table>
                <thead><tr><th>Metric</th><th>Status</th></tr></thead>
                <tbody>{metric_rows}</tbody>
            </table>

            <h2>Summary</h2>
            <div class="summary">{response.text}</div>
        </body>
        </html>"""

        self.next(self.end)

    @step
    def end(self):
        """Finish the monitoring flow."""
        self.logger.info("Finishing monitoring flow.")

    def _message(self, message):
        """Display a message in the HTML card associated to a step."""
        self.html = message
        self.logger.info(message)


if __name__ == "__main__":
    Monitoring()
