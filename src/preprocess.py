"""Preprocessing pipeline for model-ready cricket pitch data."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATASET_PATH, PREPROCESSOR_PATH, RANDOM_STATE, TARGET_COLUMN
from src.features import add_derived_features

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedData:
    """Container for train, validation, test splits and preprocessing metadata."""

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: list[str]


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a version-compatible OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing transformer for categorical and numeric features."""
    categorical_features = [
        "venue",
        "city",
        "country",
        "match_type",
        "season",
        "day_night",
        "is_subcontinental",
        "is_coastal",
        "high_dew",
        "pitch_freshness",
        "soil_composition",
    ]
    numeric_features = [column for column in x.columns if column not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def preprocess_dataset(dataset_path: str | None = None) -> PreprocessedData:
    """Load, feature-engineer, split, fit preprocessing, and persist transformer."""
    path = DATASET_PATH if dataset_path is None else dataset_path
    df = pd.read_csv(path)
    df = add_derived_features(df)
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    preprocessor = build_preprocessor(x_train)
    preprocessor.fit(x_train)
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    feature_names = list(preprocessor.get_feature_names_out())
    logger.info("Saved preprocessor to %s.", PREPROCESSOR_PATH)
    return PreprocessedData(x_train, x_val, x_test, y_train, y_val, y_test, preprocessor, feature_names)


def transform_splits(data: PreprocessedData) -> tuple:
    """Transform train, validation, and test feature matrices."""
    return (
        data.preprocessor.transform(data.x_train),
        data.preprocessor.transform(data.x_val),
        data.preprocessor.transform(data.x_test),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    preprocess_dataset()
