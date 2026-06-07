"""Project configuration and shared constants."""

from pathlib import Path

RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SHAP_DIR = FIGURES_DIR / "shap"

DATASET_PATH = DATA_DIR / "cricket_pitch_dataset.csv"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_PATH = MODELS_DIR / "rf_model.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"

TARGET_COLUMN = "pitch_type"

PITCH_LABELS = {
    0: "Batting-Friendly",
    1: "Pace-Friendly",
    2: "Spin-Friendly",
}

INDIAN_VENUES = [
    "Wankhede Stadium",
    "M. A. Chidambaram Stadium",
    "MA Chidambaram Stadium",
    "Eden Gardens",
    "M. Chinnaswamy Stadium",
    "M Chinnaswamy Stadium",
    "Narendra Modi Stadium",
    "Sardar Patel Stadium",
    "Arun Jaitley Stadium",
    "Feroz Shah Kotla",
    "Rajiv Gandhi International Stadium",
    "Maharashtra Cricket Association Stadium",
    "Sawai Mansingh Stadium",
    "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association Stadium",
    "Green Park",
    "Holkar Cricket Stadium",
    "Barabati Stadium",
    "JSCA International Stadium Complex",
    "Vidarbha Cricket Association Stadium",
    "Brabourne Stadium",
    "Dr DY Patil Sports Academy",
    "Himachal Pradesh Cricket Association Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
]

SUBCONTINENTAL_CITIES = [
    "Mumbai",
    "Chennai",
    "Delhi",
    "Kolkata",
    "Bengaluru",
    "Bangalore",
    "Ahmedabad",
    "Hyderabad",
    "Pune",
    "Jaipur",
    "Mohali",
    "Kanpur",
    "Indore",
    "Cuttack",
    "Ranchi",
    "Nagpur",
    "Dharamsala",
    "Lucknow",
]

COASTAL_VENUES = [
    "Wankhede Stadium",
    "M. A. Chidambaram Stadium",
    "MA Chidambaram Stadium",
    "Brabourne Stadium",
    "Dr DY Patil Sports Academy",
]
