"""Custom Metaflow card that renders a confusion matrix for a classification fold."""

import base64
import io

import matplotlib.pyplot as plt
from metaflow.plugins.cards.card_modules.card import MetaflowCard

LABELS = ["Adelie", "Chinstrap", "Gentoo"]


class ConfusionMatrixCard(MetaflowCard):
    """A Metaflow card that renders a confusion matrix for a classification fold."""

    type = "confusion_matrix"

    def render(self, task):
        """Return an HTML page with an embedded confusion matrix heatmap."""
        cm = task["confusion_matrix"].data
        labels = (
            task["confusion_matrix_labels"].data
            if "confusion_matrix_labels" in task
            else LABELS
        )
        fold = task["fold"].data if "fold" in task else None
        accuracy = task["test_accuracy"].data if "test_accuracy" in task else None
        loss = task["test_loss"].data if "test_loss" in task else None

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yticklabels(labels, fontsize=12)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="white" if cm[i, j] > thresh else "black",
                )

        ax.set_ylabel("Actual", fontsize=13)
        ax.set_xlabel("Predicted", fontsize=13)
        ax.xaxis.set_label_position("top")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        subtitle_parts = []
        if fold is not None:
            subtitle_parts.append(f"Fold {fold}")
        if accuracy is not None:
            subtitle_parts.append(f"Accuracy: {accuracy:.4f}")
        if loss is not None:
            subtitle_parts.append(f"Loss: {loss:.4f}")
        subtitle = " &nbsp;|&nbsp; ".join(subtitle_parts)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Confusion Matrix</title>
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
    img {{ max-width: 100%; display: block; }}
  </style>
</head>
<body>
  <h1>Confusion Matrix</h1>
  <p class="subtitle">{subtitle}</p>
  <div class="card">
    <img src="data:image/png;base64,{img_b64}" alt="Confusion Matrix" />
  </div>
</body>
</html>"""


CARDS = [ConfusionMatrixCard]
