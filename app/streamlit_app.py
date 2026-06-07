"""Streamlit application for pre-match cricket pitch classification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import (  # noqa: E402
    DATASET_PATH,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    PITCH_LABELS,
    PREPROCESSOR_PATH,
)
from src.features import add_derived_features  # noqa: E402
from src.generate_data import VENUE_CITY  # noqa: E402

st.set_page_config(
    page_title="Cricket Pitch Classifier",
    page_icon="🏏",
    layout="wide",
)

PITCH_STYLE = {
    0: ("🏏", "#16a34a"),
    1: ("💨", "#2563eb"),
    2: ("🌀", "#f97316"),
}


def clean_feature_name(name: str) -> str:
    """Convert transformed sklearn feature names into readable labels."""
    return (
        name.replace("numeric__", "")
        .replace("categorical__", "")
        .replace("venue_", "venue: ")
        .replace("city_", "city: ")
        .replace("match_type_", "match type: ")
        .replace("season_", "season: ")
    )


@st.cache_resource
def load_artifacts() -> tuple:
    """Load trained model and preprocessing artifacts."""
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load generated training dataset for defaults and summaries."""
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame()


def predict_page() -> None:
    """Render prediction page."""
    st.title("Pre-Match Pitch Prediction")
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        st.warning("Run `python run_pipeline.py` first to train the model.")
        return

    model, preprocessor = load_artifacts()
    dataset = load_dataset()
    venues = sorted(VENUE_CITY)

    with st.sidebar:
        venue = st.selectbox("Venue", venues)
        default_city = VENUE_CITY[venue][0]
        city = st.text_input("City", value=default_city)
        match_type = st.selectbox("Match type", ["T20", "ODI", "Test"])
        temperature = st.slider("Temperature (C)", 10.0, 45.0, 30.0, 0.5)
        humidity = st.slider("Humidity (%)", 20.0, 100.0, 62.0, 1.0)
        wind_speed = st.slider("Wind speed", 0.0, 40.0, 12.0, 0.5)
        dew_point = st.slider("Dew point", 0.0, 30.0, 18.0, 0.5)
        cloud_cover = st.slider("Cloud cover (%)", 0.0, 100.0, 35.0, 1.0)
        pitch_age_days = st.slider("Pitch age (days)", 1, 8, 3)
        day_night = st.toggle("Day/night match", value=True)
        submitted = st.button("Predict", type="primary")

    if not submitted:
        st.info("Set the pre-match conditions in the sidebar and run a prediction.")
        return

    venue_subset = dataset[dataset["venue"] == venue] if not dataset.empty else pd.DataFrame()
    ground_avg = (
        float(venue_subset["ground_avg_1st_innings"].median())
        if not venue_subset.empty
        else {"T20": 165.0, "ODI": 270.0, "Test": 330.0}[match_type]
    )
    pace_pct = float(venue_subset["ground_pace_wickets_pct"].median()) if not venue_subset.empty else 50.0
    spin_pct = 100.0 - pace_pct
    row = pd.DataFrame(
        [
            {
                "venue": venue,
                "city": city,
                "country": "India",
                "match_type": match_type,
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "dew_point": dew_point,
                "cloud_cover": cloud_cover,
                "pitch_age_days": pitch_age_days,
                "ground_avg_1st_innings": ground_avg,
                "ground_pace_wickets_pct": pace_pct,
                "ground_spin_wickets_pct": spin_pct,
                "season": "Summer",
                "day_night": int(day_night),
            }
        ]
    )
    engineered = add_derived_features(row)
    transformed = preprocessor.transform(engineered)
    probabilities = model.predict_proba(transformed)[0]
    prediction = int(probabilities.argmax())
    emoji, color = PITCH_STYLE[prediction]

    left, right = st.columns([0.9, 1.1])
    with left:
        st.markdown(
            f"""
            <div style="border-left: 8px solid {color}; padding: 1rem 1.2rem; background: #f8fafc;">
                <div style="font-size: 3rem;">{emoji}</div>
                <h2 style="margin: 0;">{PITCH_LABELS[prediction]}</h2>
                <p style="margin: 0.35rem 0 0;">Confidence: {probabilities[prediction]:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        proba_df = pd.DataFrame(
            {
                "Pitch type": [PITCH_LABELS[index] for index in range(3)],
                "Probability": probabilities,
            }
        )
        fig = px.bar(proba_df, x="Pitch type", y="Probability", color="Pitch type")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    importances = pd.Series(model.feature_importances_, index=preprocessor.get_feature_names_out())
    transformed_values = pd.Series(transformed[0], index=preprocessor.get_feature_names_out())
    local_signal = (importances * transformed_values.abs()).sort_values(ascending=False).head(3)
    local_signal.index = [clean_feature_name(name) for name in local_signal.index]
    st.subheader("Top local model signals")
    st.dataframe(local_signal.rename("Signal strength"), use_container_width=True)


def performance_page() -> None:
    """Render model performance page."""
    st.title("Model Performance")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Macro F1", f"{metrics['macro_f1']:.2%}")
        st.dataframe(pd.DataFrame(metrics["classification_report"]).T, use_container_width=True)
    else:
        st.warning("Metrics are not available yet. Run `python run_pipeline.py`.")

    for image_name in ["confusion_matrix.png", "feature_importance.png"]:
        path = FIGURES_DIR / image_name
        if path.exists():
            st.image(str(path), use_container_width=True)


def shap_page() -> None:
    """Render SHAP explorer page."""
    st.title("SHAP Explorer")
    st.selectbox("Predicted class filter", ["All", "Batting-Friendly", "Pace-Friendly", "Spin-Friendly"])
    for path in [
        FIGURES_DIR / "shap" / "summary_beeswarm.png",
        FIGURES_DIR / "shap" / "summary_bar.png",
        FIGURES_DIR / "shap" / "local_waterfall.png",
    ]:
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info(f"Missing `{path.name}`. Run the pipeline to generate SHAP plots.")


def about_page() -> None:
    """Render about page."""
    st.title("About")
    dataset = load_dataset()
    st.write(
        "This app predicts Indian cricket pitch behavior before a match using venue, "
        "weather, match type, ground history, and pitch preparation signals."
    )
    if not dataset.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(dataset):,}")
        col2.metric("Venues", dataset["venue"].nunique())
        col3.metric("Formats", dataset["match_type"].nunique())
        st.dataframe(dataset["pitch_type"].value_counts().rename("count"), use_container_width=True)
    st.code("Add downloaded Cricsheet/IPL JSON or ZIP files to: data/raw/", language="text")


def main() -> None:
    """Run Streamlit multipage navigation."""
    page = st.sidebar.radio(
        "Section",
        ["Predict", "Model Performance", "SHAP Explorer", "About"],
    )
    if page == "Predict":
        predict_page()
    elif page == "Model Performance":
        performance_page()
    elif page == "SHAP Explorer":
        shap_page()
    else:
        about_page()


if __name__ == "__main__":
    main()
