"""Custom Metaflow card that renders permutation-based feature importance."""

import base64
import io

import matplotlib.pyplot as plt
from metaflow.plugins.cards.card_modules.card import MetaflowCard


class FeatureImportanceCard(MetaflowCard):
    """Renders a horizontal bar chart of permutation-based feature importances."""

    type = "feature_importance"

    def render(self, task):
        """Return an HTML page with an embedded feature importance chart."""
        importances = task["feature_importances"].data
        feature_names = task["feature_names"].data
        accuracy = task["feature_importances_baseline_acc"].data if "feature_importances_baseline_acc" in task else None

        # Sort by absolute importance descending.
        pairs = sorted(zip(importances, feature_names, strict=True), reverse=True)
        importances_sorted, names_sorted = zip(*pairs, strict=False)

        fig, ax = plt.subplots(figsize=(8, max(4, len(names_sorted) * 0.45)))
        colors = ["#d73027" if v > 0 else "#4575b4" for v in importances_sorted]
        bars = ax.barh(range(len(names_sorted)), importances_sorted, color=colors)
        ax.set_yticks(range(len(names_sorted)))
        ax.set_yticklabels(names_sorted, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("Mean accuracy drop when feature is permuted", fontsize=11)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_title("Permutation Feature Importance", fontsize=13, pad=12)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        subtitle = f"Baseline accuracy: {accuracy:.4f}" if accuracy is not None else ""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Feature Importance</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 32px;
      background: #fafafa;
      color: #333;
    }}
    h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
    .subtitle {{ color: #666; font-size: 0.95rem; margin-bottom: 24px; }}
    .card {{
      background: #fff;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,.12);
      display: inline-block;
      padding: 24px;
    }}
    .legend {{ margin-top: 16px; font-size: 0.85rem; color: #555; }}
    .dot {{ display: inline-block; width: 12px; height: 12px;
            border-radius: 2px; margin-right: 4px; vertical-align: middle; }}
    img {{ max-width: 100%; display: block; }}
  </style>
</head>
<body>
  <h1>Feature Importance</h1>
  <p class="subtitle">{subtitle}</p>
  <div class="card">
    <img src="data:image/png;base64,{img_b64}" alt="Feature Importance" />
    <p class="legend">
      <span class="dot" style="background:#d73027"></span>
      Positive bar: shuffling this feature hurts accuracy (important).
      &nbsp;
      <span class="dot" style="background:#4575b4"></span>
      Negative bar: shuffling slightly helps (noisy/redundant).
    </p>
  </div>
</body>
</html>"""


CARDS = [FeatureImportanceCard]
