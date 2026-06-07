"""Run the full cricket pitch classification pipeline."""

from __future__ import annotations

import logging

from src.evaluate import evaluate_model
from src.explain import generate_shap_plots
from src.generate_data import build_dataset
from src.preprocess import preprocess_dataset
from src.train import train_models


def run_pipeline() -> dict:
    """Generate data, preprocess, train, evaluate, and explain the model."""
    build_dataset()
    data = preprocess_dataset()
    train_models(data)
    metrics = evaluate_model(data)
    generate_shap_plots(data)
    logging.info(
        "Final accuracy: %.4f | Macro F1: %.4f",
        metrics["accuracy"],
        metrics["macro_f1"],
    )
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_pipeline()
