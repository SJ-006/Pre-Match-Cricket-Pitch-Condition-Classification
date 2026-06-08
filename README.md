# PitchSense AI — Pre-Match Cricket Pitch Condition Classification

PitchSense AI is a machine learning-driven analytics and tactical advisory platform. It predicts whether a cricket pitch is likely to be **Batting-Friendly**, **Pace-Friendly**, or **Spin-Friendly** before the match starts, helping coaches, captains, and curators make data-backed tactical decisions (e.g., team selection, toss strategy).

---

## Key Features

1. **AI Pitch Classification:** Categorizes pitch behavior into three distinct classes using random forest classification.
2. **Interactive Control Panel & Presets:** Quickly load conditions using preset profiles (e.g., *Dry Dust Bowl*, *Green Seamer*, *Balanced Surface*, *Batting Paradise*) or customize parameters on the fly.
3. **Domain Feature Engineering:** Computes complex indicators including `soil_spin_factor`, `compaction_density`, `grass_friction_ratio`, and `spin_degradation_index`.
4. **Anomalous Prediction Validation Lab:** A rule-based physics safety layer that validates predictions against logical cricket turf physics rules (e.g., highlighting anomalies in dry, cracked pitches).
5. **Toss Strategy Advisor:** Uses weather (like dew point thresholds) and predicted surface behavior to output strategic guidance ("Bat First" vs. "Bowl First").
6. **Explainable AI (XAI) with SHAP:** Provides global summary beeswarm/bar charts and local prediction waterfall explanations to show exact feature attributions.
7. **Premium Styling:** Custom dark-themed layout built with Outfit/Inter typography, harmonious gradients, and optimized header positioning.

---

## Model Performance

The final **Random Forest Classifier** (optimized via `RandomizedSearchCV`) achieved the following results on the hold-out test set:

- **Overall Test Accuracy:** `73.72%`
- **Macro F1-Score:** `72.29%`

### Classification Metrics

| Pitch Condition Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Batting-Friendly** (Class 0) | 68.75% | 59.46% | **63.77%** | 74 |
| **Pace-Friendly** (Class 1) | 75.63% | 77.59% | **76.60%** | 116 |
| **Spin-Friendly** (Class 2) | 74.42% | 78.69% | **76.49%** | 122 |

*Note: Batting-friendly surfaces show higher variance due to sensitivity to minor temperature/humidity shifts, whereas Pace and Spin classes exhibit stable turf indicators (high grass or low compaction).*

---

## Dataset & Feature Engineering

The pipeline is designed to combine synthetic India-focused turf data with real-world historical records from **Cricsheet/IPL match data**. 

* **Inputs:** 
  - *Weather:* Temperature, Humidity, Wind Speed, Dew Point, Cloud Cover.
  - *Turf Telemetry:* Soil Composition (Red/Black/Mixed), Grass Coverage (mm), Pitch Age (days), Compaction (kPa), Strip Number.
  - *Metadata:* Venue, City, Match Type (T20, ODI, Test), Season, Day/Night match flag.
* **Adding Raw Files:** Place raw IPL/T20/ODI Cricsheet `.json` files or `.zip` archives into `data/raw/`. The pipeline canonicalizes venue names and extracts historical match-level signals to enrich the training set.

---

## How to Run

### 1. Installation
Ensure Python 3.8+ is installed and run:
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline (Train & Evaluate)
To process the dataset, train model candidates, optimize parameters, and generate SHAP explainability plots, execute:
```bash
python run_pipeline.py
```
This script populates:
- `data/cricket_pitch_dataset.csv` (Enriched dataset)
- `models/preprocessor.pkl` & `models/rf_model.pkl` (Saved models)
- `reports/metrics.json` (Test evaluation metrics)
- `reports/figures/` (Confusion matrix, feature importance, and SHAP plots)

### 3. Launch UI App
Launch the interactive web dashboard:
```bash
streamlit run app/streamlit_app.py
```
*(By default, the app initializes on the **About** page to introduce the platform architecture before proceeding to the Predictor column).*

---

## Project Structure

```text
├── data/
│   ├── raw/                 # Put Cricsheet/IPL raw matches here (.json / .zip)
│   └── cricket_pitch_dataset.csv
├── models/
│   ├── preprocessor.pkl     # Imputer, Scaler, and One-Hot encoders
│   └── rf_model.pkl         # Optimized RandomForest Classifier
├── src/
│   ├── config.py            # Venue names, paths, constants
│   ├── generate_data.py     # Generates synthetic data and processes raw matches
│   ├── features.py          # Derived features (Soil, Grass, Compaction indexes)
│   ├── preprocess.py        # Train-val-test splitting and transformers
│   ├── train.py             # Hyperparameter tuning and model training
│   ├── evaluate.py          # Metrics logging and confusion matrix plotting
│   └── explain.py           # SHAP beeswarm, bar, and waterfall visualizers
├── app/
│   └── streamlit_app.py     # Streamlit web app layout and CSS design
├── reports/
│   ├── metrics.json         # Current model evaluation scores
│   └── figures/             # Confusion matrix, importances, and SHAP plots
├── notebooks/               # Research notebooks
├── run_pipeline.py          # Pipeline coordinator
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```
