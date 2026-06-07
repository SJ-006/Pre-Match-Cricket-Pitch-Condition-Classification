"""Evaluation and reporting for cricket pitch classification."""

from __future__ import annotations

import json
import logging

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import FIGURES_DIR, METRICS_PATH, MODEL_PATH, PITCH_LABELS
from src.preprocess import PreprocessedData, preprocess_dataset, transform_splits

logger = logging.getLogger(__name__)


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    """Save confusion matrix heatmap."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[PITCH_LABELS[i] for i in [0, 1, 2]],
        yticklabels=[PITCH_LABELS[i] for i in [0, 1, 2]],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=180)
    plt.close()


def plot_feature_importance(model, feature_names: list[str]) -> None:
    """Save RandomForest feature importance chart."""
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    top = importances.tail(20)
    colors = ["#3b82f6"] * len(top)
    for index in range(max(0, len(top) - 5), len(top)):
        colors[index] = "#f97316"
    plt.figure(figsize=(9, 7))
    top.plot(kind="barh", color=colors)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=180)
    plt.close()


def evaluate_model(data: PreprocessedData | None = None) -> dict:
    """Evaluate persisted RandomForest model and save plots/metrics."""
    data = preprocess_dataset() if data is None else data
    model = joblib.load(MODEL_PATH)
    _, _, x_test = transform_splits(data)
    predictions = model.predict(x_test)

    report = classification_report(
        data.y_test,
        predictions,
        target_names=[PITCH_LABELS[i] for i in [0, 1, 2]],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(data.y_test, predictions)),
        "macro_f1": float(f1_score(data.y_test, predictions, average="macro")),
        "classification_report": report,
    }
    plot_confusion_matrix(data.y_test, predictions)
    plot_feature_importance(model, data.feature_names)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved evaluation metrics to %s.", METRICS_PATH)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    evaluate_model()
