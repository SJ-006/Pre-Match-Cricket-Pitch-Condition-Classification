"""SHAP explainability utilities."""

from __future__ import annotations

import logging
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MODEL_PATH, SHAP_DIR
from src.preprocess import PreprocessedData, preprocess_dataset, transform_splits

logger = logging.getLogger(__name__)


def _safe_feature_names(feature_names: list[str]) -> list[str]:
    """Shorten sklearn feature names for readable plots."""
    return [
        name.replace("numeric__", "").replace("categorical__", "").replace("encoder__", "")
        for name in feature_names
    ]


def generate_shap_plots(data: PreprocessedData | None = None) -> None:
    """Generate global and local SHAP plots for the saved RandomForest model."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP is not installed; skipping explainability plots.")
        return

    data = preprocess_dataset() if data is None else data
    model = joblib.load(MODEL_PATH)
    _, _, x_test = transform_splits(data)
    feature_names = _safe_feature_names(data.feature_names)
    sample = x_test[: min(250, x_test.shape[0])]
    sample_df = pd.DataFrame(sample, columns=feature_names)

    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

    class_values = shap_values[1] if isinstance(shap_values, list) else shap_values
    if isinstance(class_values, np.ndarray) and class_values.ndim == 3:
        class_values = class_values[:, :, 1]

    shap.summary_plot(class_values, sample_df, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "summary_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    shap.summary_plot(class_values, sample_df, plot_type="bar", max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "summary_bar.png", dpi=180, bbox_inches="tight")
    plt.close()

    explanation = shap.Explanation(
        values=class_values[0],
        base_values=np.mean(model.predict_proba(sample)[:, 1]),
        data=sample_df.iloc[0],
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "local_waterfall.png", dpi=180, bbox_inches="tight")
    plt.close()

    logger.info("Saved SHAP plots to %s.", SHAP_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    generate_shap_plots()
