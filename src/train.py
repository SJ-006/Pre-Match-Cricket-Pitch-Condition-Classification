"""Model training for cricket pitch classification."""

from __future__ import annotations

import logging

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import MODEL_PATH, RANDOM_STATE
from src.preprocess import PreprocessedData, preprocess_dataset, transform_splits

logger = logging.getLogger(__name__)


def train_models(data: PreprocessedData | None = None) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Train model candidates and persist the best RandomForest model."""
    data = preprocess_dataset() if data is None else data
    x_train, x_val, _ = transform_splits(data)

    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        rf,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
    )
    search.fit(x_train, data.y_train)
    best_rf = search.best_estimator_

    candidates = {
        "RandomForest": best_rf,
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    scores: dict[str, float] = {}
    for name, model in candidates.items():
        if name != "RandomForest":
            model.fit(x_train, data.y_train)
        predictions = model.predict(x_val)
        scores[name] = float(f1_score(data.y_val, predictions, average="macro"))
        logger.info("%s validation macro F1: %.4f", name, scores[name])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_rf, MODEL_PATH)
    logger.info("Saved best RandomForest model to %s. Best params: %s", MODEL_PATH, search.best_params_)
    return best_rf, scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    train_models()
