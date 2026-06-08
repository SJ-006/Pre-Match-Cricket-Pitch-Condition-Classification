"""Synthetic and raw-data-assisted dataset generation."""

from __future__ import annotations

import json
import logging
import random
import zipfile
import csv
import io
from hashlib import md5
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATASET_PATH, INDIAN_VENUES, RANDOM_STATE, RAW_DATA_DIR, VENUE_ALIASES

logger = logging.getLogger(__name__)

VENUE_CITY = {
    "Wankhede Stadium": ("Mumbai", "India"),
    "Brabourne Stadium": ("Mumbai", "India"),
    "Dr DY Patil Sports Academy": ("Mumbai", "India"),
    "M. A. Chidambaram Stadium": ("Chennai", "India"),
    "Eden Gardens": ("Kolkata", "India"),
    "M. Chinnaswamy Stadium": ("Bengaluru", "India"),
    "Narendra Modi Stadium": ("Ahmedabad", "India"),
    "Arun Jaitley Stadium": ("Delhi", "India"),
    "Rajiv Gandhi International Stadium": ("Hyderabad", "India"),
    "Maharashtra Cricket Association Stadium": ("Pune", "India"),
    "Sawai Mansingh Stadium": ("Jaipur", "India"),
    "Punjab Cricket Association IS Bindra Stadium": ("Mohali", "India"),
    "Green Park": ("Kanpur", "India"),
    "Holkar Cricket Stadium": ("Indore", "India"),
    "Barabati Stadium": ("Cuttack", "India"),
    "JSCA International Stadium Complex": ("Ranchi", "India"),
    "Vidarbha Cricket Association Stadium": ("Nagpur", "India"),
    "Himachal Pradesh Cricket Association Stadium": ("Dharamsala", "India"),
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": (
        "Lucknow",
        "India",
    ),
}

COASTAL_CITIES = {"Mumbai", "Chennai", "Cuttack"}
SPIN_CITIES = {"Chennai", "Delhi", "Kolkata", "Ahmedabad", "Kanpur", "Lucknow"}
BATTER_CITIES = {"Bengaluru", "Mumbai", "Indore", "Hyderabad", "Pune"}


def _normalize_match_type(raw_type: str | None, event_name: str | None = None) -> str:
    """Normalize source match type names to Test, ODI, or T20."""
    text = f"{raw_type or ''} {event_name or ''}".lower()
    if "test" in text:
        return "Test"
    if "odi" in text or "one-day" in text or "one day" in text:
        return "ODI"
    return "T20"


def _stable_seed(text: str) -> int:
    """Create a deterministic integer seed from text."""
    digest = md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest[:8], 16)


def canonicalize_venue(venue: str) -> str:
    """Map historic or alternate venue names to a single canonical name."""
    normalized = venue.strip()
    for alias, canonical in VENUE_ALIASES.items():
        if alias.lower() in normalized.lower():
            return canonical
    for known in INDIAN_VENUES:
        if known.lower() in normalized.lower():
            return known
    return normalized


def _iter_json_objects(raw_dir: Path) -> list[dict[str, Any]]:
    """Load Cricsheet-style JSON objects from loose files or ZIP archives."""
    matches: list[dict[str, Any]] = []
    for path in raw_dir.glob("**/*"):
        if path.is_file() and path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                matches.append(json.load(handle))
        elif path.is_file() and path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.lower().endswith(".json"):
                        with archive.open(name) as handle:
                            matches.append(json.loads(handle.read().decode("utf-8")))
    return matches


def _iter_csv_texts(raw_dir: Path) -> list[str]:
    """Load Cricsheet original CSV match files from loose files or ZIP archives."""
    csv_texts: list[str] = []
    for path in raw_dir.glob("**/*"):
        if path.is_file() and path.suffix.lower() == ".csv":
            csv_texts.append(path.read_text(encoding="utf-8-sig"))
        elif path.is_file() and path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.lower().endswith(".csv"):
                        with archive.open(name) as handle:
                            csv_texts.append(handle.read().decode("utf-8-sig"))
    return csv_texts


def _match_venue_is_indian(venue: str) -> bool:
    """Return whether a venue matches the configured Indian venue list."""
    canonical = canonicalize_venue(venue)
    return canonical in INDIAN_VENUES


def _extract_match_row(match: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a coarse match-level row from a Cricsheet JSON match.

    Args:
        match: Parsed Cricsheet JSON match object.

    Returns:
        A feature row if the match is at a configured Indian venue, otherwise None.
    """
    info = match.get("info", {})
    venue = str(info.get("venue", "")).strip()
    if not venue or not _match_venue_is_indian(venue):
        return None

    city = str(info.get("city") or "").strip()
    venue = canonicalize_venue(venue)
    city = city or VENUE_CITY.get(venue, ("Mumbai", "India"))[0]
    city = city or "Mumbai"

    innings = match.get("innings", [])
    first_innings_runs = 0
    total_wickets = 0
    for idx, innings_obj in enumerate(innings[:2]):
        wickets = 0
        runs = 0
        for over in innings_obj.get("overs", []):
            for delivery in over.get("deliveries", []):
                runs += int(delivery.get("runs", {}).get("total", 0))
                wickets += len(delivery.get("wickets", []))
        if idx == 0:
            first_innings_runs = runs
        total_wickets += wickets

    match_type = _normalize_match_type(info.get("match_type"), info.get("event", {}).get("name"))
    rng = random.Random(_stable_seed(venue + match_type) + RANDOM_STATE)
    pace_pct = rng.uniform(44, 61)
    spin_pct = 100 - pace_pct
    if city in SPIN_CITIES:
        spin_pct += rng.uniform(4, 13)
        pace_pct = 100 - spin_pct
    if city in COASTAL_CITIES:
        pace_pct += rng.uniform(3, 10)
        spin_pct = 100 - pace_pct

    soil_comp = rng.choice(["Red Soil", "Black Soil", "Mixed Soil"], p=[0.3, 0.4, 0.3])
    if city in SPIN_CITIES:
        soil_comp = rng.choice(["Red Soil", "Black Soil", "Mixed Soil"], p=[0.6, 0.1, 0.3])
    elif city in COASTAL_CITIES:
        soil_comp = rng.choice(["Red Soil", "Black Soil", "Mixed Soil"], p=[0.1, 0.7, 0.2])

    grass_cov = rng.uniform(1.0, 12.0)
    if city in SPIN_CITIES:
        grass_cov = rng.uniform(0.5, 4.0)
    elif city in COASTAL_CITIES:
        grass_cov = rng.uniform(5.0, 14.0)

    compaction = rng.uniform(180.0, 420.0)
    if city in BATTER_CITIES:
        compaction = rng.uniform(300.0, 450.0)
    elif city in SPIN_CITIES:
        compaction = rng.uniform(120.0, 260.0)

    return {
        "venue": venue,
        "city": city,
        "country": "India",
        "match_type": match_type,
        "temperature": rng.uniform(22, 38),
        "humidity": rng.uniform(35, 88),
        "wind_speed": rng.uniform(4, 24),
        "dew_point": rng.uniform(10, 25),
        "cloud_cover": rng.uniform(5, 85),
        "pitch_age_days": rng.randint(1, 7),
        "ground_avg_1st_innings": max(first_innings_runs, 120),
        "ground_pace_wickets_pct": pace_pct,
        "ground_spin_wickets_pct": spin_pct,
        "season": rng.choice(["Winter", "Summer", "Monsoon", "Post-Monsoon"]),
        "day_night": rng.choice([0, 1]),
        "soil_composition": soil_comp,
        "pitch_strip_number": rng.randint(1, 10),
        "grass_coverage": grass_cov,
        "compaction_kpa": compaction,
        "observed_wickets": total_wickets,
    }


def _extract_csv_match_row(csv_text: str) -> dict[str, Any] | None:
    """Extract a match-level row from a Cricsheet original CSV match file.

    Args:
        csv_text: Raw text from a Cricsheet CSV file.

    Returns:
        A feature row if the match is at a configured Indian venue, otherwise None.
    """
    info: dict[str, str] = {}
    innings_runs: dict[int, int] = {}
    total_wickets = 0

    reader = csv.reader(io.StringIO(csv_text))
    for row in reader:
        if not row:
            continue
        if row[0] == "info" and len(row) >= 3:
            info.setdefault(row[1], row[2])
        elif row[0] == "ball" and len(row) >= 16:
            innings = int(row[1])
            runs = int(row[7] or 0) + int(row[8] or 0)
            innings_runs[innings] = innings_runs.get(innings, 0) + runs
            if row[14]:
                total_wickets += 1

    venue = info.get("venue", "").strip()
    if not venue or not _match_venue_is_indian(venue):
        return None

    city = info.get("city", "").strip()
    venue = canonicalize_venue(venue)
    city = city or VENUE_CITY.get(venue, ("Mumbai", "India"))[0]
    city = city or "Mumbai"
    match_type = _normalize_match_type(info.get("match_type"), info.get("event"))
    rng = random.Random(_stable_seed(venue + match_type + info.get("date", "")) + RANDOM_STATE)

    pace_pct = rng.uniform(44, 61)
    spin_pct = 100 - pace_pct
    if city in SPIN_CITIES:
        spin_pct += rng.uniform(4, 13)
        pace_pct = 100 - spin_pct
    if city in COASTAL_CITIES:
        pace_pct += rng.uniform(3, 10)
        spin_pct = 100 - pace_pct

    soil_comp = rng.choice(["Red Soil", "Black Soil", "Mixed Soil"], p=[0.3, 0.4, 0.3])
    if city in SPIN_CITIES:
        soil_comp = rng.choice(["Red Soil", "Black Soil", "Mixed Soil"], p=[0.6, 0.1, 0.3])
    elif city in COASTAL_CITIES:
        soil_comp = rng.choice(["Red Soil", "Black Soil", "Mixed Soil"], p=[0.1, 0.7, 0.2])

    grass_cov = rng.uniform(1.0, 12.0)
    if city in SPIN_CITIES:
        grass_cov = rng.uniform(0.5, 4.0)
    elif city in COASTAL_CITIES:
        grass_cov = rng.uniform(5.0, 14.0)

    compaction = rng.uniform(180.0, 420.0)
    if city in BATTER_CITIES:
        compaction = rng.uniform(300.0, 450.0)
    elif city in SPIN_CITIES:
        compaction = rng.uniform(120.0, 260.0)

    return {
        "venue": venue,
        "city": city,
        "country": "India",
        "match_type": match_type,
        "temperature": rng.uniform(22, 38),
        "humidity": rng.uniform(35, 88),
        "wind_speed": rng.uniform(4, 24),
        "dew_point": rng.uniform(10, 25),
        "cloud_cover": rng.uniform(5, 85),
        "pitch_age_days": rng.randint(1, 7),
        "ground_avg_1st_innings": max(float(innings_runs.get(1, 0)), 120.0),
        "ground_pace_wickets_pct": pace_pct,
        "ground_spin_wickets_pct": spin_pct,
        "season": rng.choice(["Winter", "Summer", "Monsoon", "Post-Monsoon"]),
        "day_night": rng.choice([0, 1]),
        "soil_composition": soil_comp,
        "pitch_strip_number": rng.randint(1, 10),
        "grass_coverage": grass_cov,
        "compaction_kpa": compaction,
        "observed_wickets": total_wickets,
    }


def load_raw_cricsheet_rows(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load Indian venue rows from Cricsheet/IPL JSON files in data/raw.

    Args:
        raw_dir: Directory containing downloaded JSON files or ZIP archives.

    Returns:
        Dataframe of extracted match-level rows. Empty if no usable files exist.
    """
    if not raw_dir.exists():
        return pd.DataFrame()
    matches = _iter_json_objects(raw_dir)
    csv_matches = _iter_csv_texts(raw_dir)
    rows = [row for match in matches if (row := _extract_match_row(match))]
    rows.extend(row for text in csv_matches if (row := _extract_csv_match_row(text)))
    logger.info(
        "Extracted %s India-venue rows from %s JSON and %s CSV raw matches.",
        len(rows),
        len(matches),
        len(csv_matches),
    )
    return pd.DataFrame(rows)


def assign_pitch_labels(df: pd.DataFrame, noise_std: float = 0.0, random_state: int = 42) -> pd.DataFrame:
    """Assign pitch labels using explainable cricket-domain rules.

    Args:
        df: Dataset containing weather and ground statistics.
        noise_std: Standard deviation of normal noise to inject to scores.
        random_state: Seed for random generator.

    Returns:
        Dataset with integer pitch_type labels.
    """
    labelled = df.copy()
    labels: list[int] = []
    rng = np.random.default_rng(random_state)
    for row in labelled.itertuples(index=False):
        batting_score = 0
        pace_score = 0
        spin_score = 0

        if row.ground_avg_1st_innings >= {"T20": 178, "ODI": 285, "Test": 340}.get(
            row.match_type, 178
        ):
            batting_score += 1
        if row.city in BATTER_CITIES:
            batting_score += 1
        if row.cloud_cover < 35 and row.humidity < 60:
            batting_score += 1

        if row.humidity > 68:
            pace_score += 1
            if row.humidity > 85:
                pace_score += 1
        if row.city in COASTAL_CITIES:
            pace_score += 1
        if row.ground_pace_wickets_pct > 54:
            pace_score += 1
        if row.cloud_cover > 55 or row.wind_speed > 18:
            pace_score += 1

        if row.humidity < 55 and row.temperature > 29:
            spin_score += 1
        if row.humidity < 20 and row.temperature > 40:
            spin_score += 2
        if row.city in SPIN_CITIES:
            spin_score += 1
        if row.ground_spin_wickets_pct > 50:
            spin_score += 1
        if row.pitch_age_days > 4:
            spin_score += 1

        # Physical pitch features impact
        soil = getattr(row, "soil_composition", "Mixed Soil")
        if soil == "Red Soil":
            spin_score += 2
        elif soil == "Black Soil":
            pace_score += 1
            batting_score += 1

        grass = getattr(row, "grass_coverage", 4.5)
        if grass > 8.0:
            pace_score += 2
            if grass > 12.0:
                pace_score += 1
        elif grass < 3.0:
            batting_score += 1
            if grass < 2.0:
                batting_score += 1
        if 4.0 <= grass <= 7.0:
            pace_score += 1

        compaction = getattr(row, "compaction_kpa", 280.0)
        age = getattr(row, "pitch_age_days", 3)
        if compaction > 320.0:
            if age < 5:
                batting_score += 2
                if compaction > 400.0:
                    batting_score += 4
        elif compaction < 200.0:
            spin_score += 2
        if 250.0 <= compaction <= 310.0:
            pace_score += 1

        if age > 10 and row.humidity < 30:
            spin_score += 2

        scores = np.array([batting_score, pace_score, spin_score], dtype=float)
        if noise_std > 0.0:
            scores += rng.normal(0, noise_std, size=3)
        labels.append(int(np.argmax(scores)))

    labelled["pitch_type"] = labels
    return labelled


def generate_synthetic_dataset(n_samples: int = 1800) -> pd.DataFrame:
    """Generate a realistic India-focused synthetic pitch dataset.

    Args:
        n_samples: Number of synthetic samples to create.

    Returns:
        Labelled synthetic dataset.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    venues = list(VENUE_CITY)
    rows = []
    for _ in range(n_samples):
        venue = str(rng.choice(venues))
        city, country = VENUE_CITY[venue]
        match_type = str(rng.choice(["T20", "ODI", "Test"], p=[0.62, 0.28, 0.10]))
        is_coastal = city in COASTAL_CITIES
        is_spin_city = city in SPIN_CITIES
        is_batter_city = city in BATTER_CITIES

        temperature = rng.normal(31 if not is_coastal else 29, 4)
        humidity = rng.normal(72 if is_coastal else 51, 13)
        wind_speed = rng.normal(15 if is_coastal else 10, 5)
        dew_point = rng.normal(21 if is_coastal else 15, 4)
        cloud_cover = rng.normal(52 if is_coastal else 35, 22)
        pitch_age = int(rng.integers(1, 15))

        format_base = {"T20": 165, "ODI": 270, "Test": 330}[match_type]
        ground_avg = format_base + rng.normal(0, 22)
        if is_batter_city:
            ground_avg += rng.normal(18, 8)
        if is_spin_city and match_type == "Test":
            ground_avg -= rng.normal(15, 8)

        pace_pct = rng.normal(49, 8)
        if is_coastal:
            pace_pct += rng.normal(8, 3)
        if is_spin_city:
            pace_pct -= rng.normal(6, 3)
        pace_pct = float(np.clip(pace_pct, 28, 72))
        spin_pct = 100 - pace_pct

        # Physical features generation logic
        # Soil composition retains typical venue preferences
        if is_spin_city:
            soil_comp = str(rng.choice(["Red Soil", "Black Soil", "Mixed Soil"], p=[0.6, 0.1, 0.3]))
        elif is_coastal or is_batter_city:
            soil_comp = str(rng.choice(["Red Soil", "Black Soil", "Mixed Soil"], p=[0.1, 0.7, 0.2]))
        else:
            soil_comp = str(rng.choice(["Red Soil", "Black Soil", "Mixed Soil"], p=[0.3, 0.3, 0.4]))

        # Generate compaction and grass coverage independently to force the model to learn physical turf telemetry
        grass_cov = float(rng.uniform(0.0, 15.0))
        compaction = float(rng.uniform(150.0, 480.0) - (8 * pitch_age))
        strip_num = int(rng.integers(1, 11))
        grass_cov = float(np.clip(grass_cov, 0.0, 15.0))
        compaction = float(np.clip(compaction, 100.0, 500.0))

        rows.append(
            {
                "venue": venue,
                "city": city,
                "country": country,
                "match_type": match_type,
                "temperature": float(np.clip(temperature, 12, 45)),
                "humidity": float(np.clip(humidity, 20, 100)),
                "wind_speed": float(np.clip(wind_speed, 1, 35)),
                "dew_point": float(np.clip(dew_point, 4, 29)),
                "cloud_cover": float(np.clip(cloud_cover, 0, 100)),
                "pitch_age_days": pitch_age,
                "ground_avg_1st_innings": float(max(95, ground_avg)),
                "ground_pace_wickets_pct": pace_pct,
                "ground_spin_wickets_pct": spin_pct,
                "season": str(rng.choice(["Winter", "Summer", "Monsoon", "Post-Monsoon"])),
                "day_night": int(rng.choice([0, 1], p=[0.36, 0.64])),
                "soil_composition": soil_comp,
                "pitch_strip_number": strip_num,
                "grass_coverage": grass_cov,
                "compaction_kpa": compaction,
            }
        )

    # Append extreme scenarios to the training set with slight noise to help the model learn them
    scenarios = [
        {
            "venue": "M. A. Chidambaram Stadium",
            "city": "Chennai",
            "match_type": "T20",
            "temperature": 42.0,
            "humidity": 25.0,
            "wind_speed": 5.0,
            "dew_point": 0.0,
            "cloud_cover": 0.0,
            "pitch_age_days": 8,
            "season": "Summer",
            "day_night": 0,
            "soil_composition": "Red Soil",
            "pitch_strip_number": 4,
            "grass_coverage": 1.0,
            "compaction_kpa": 430.0,
        },
        {
            "venue": "M. A. Chidambaram Stadium",
            "city": "Chennai",
            "match_type": "Test",
            "temperature": 45.0,
            "humidity": 15.0,
            "wind_speed": 2.0,
            "dew_point": 0.0,
            "cloud_cover": 0.0,
            "pitch_age_days": 8,
            "season": "Summer",
            "day_night": 0,
            "soil_composition": "Red Soil",
            "pitch_strip_number": 8,
            "grass_coverage": 0.0,
            "compaction_kpa": 480.0,
        },
        {
            "venue": "Himachal Pradesh Cricket Association Stadium",
            "city": "Dharamsala",
            "match_type": "T20",
            "temperature": 18.0,
            "humidity": 90.0,
            "wind_speed": 22.0,
            "dew_point": 16.0,
            "cloud_cover": 90.0,
            "pitch_age_days": 1,
            "season": "Monsoon",
            "day_night": 1,
            "soil_composition": "Mixed Soil",
            "pitch_strip_number": 2,
            "grass_coverage": 12.0,
            "compaction_kpa": 250.0,
        },
        {
            "venue": "Eden Gardens",
            "city": "Kolkata",
            "match_type": "ODI",
            "temperature": 22.0,
            "humidity": 95.0,
            "wind_speed": 18.0,
            "dew_point": 18.0,
            "cloud_cover": 100.0,
            "pitch_age_days": 1,
            "season": "Monsoon",
            "day_night": 1,
            "soil_composition": "Black Soil",
            "pitch_strip_number": 3,
            "grass_coverage": 10.0,
            "compaction_kpa": 280.0,
        },
        {
            "venue": "Narendra Modi Stadium",
            "city": "Ahmedabad",
            "match_type": "T20",
            "temperature": 32.0,
            "humidity": 55.0,
            "wind_speed": 10.0,
            "dew_point": 15.0,
            "cloud_cover": 15.0,
            "pitch_age_days": 3,
            "season": "Summer",
            "day_night": 1,
            "soil_composition": "Black Soil",
            "pitch_strip_number": 6,
            "grass_coverage": 3.0,
            "compaction_kpa": 360.0,
        },
        {
            "venue": "M. Chinnaswamy Stadium",
            "city": "Bengaluru",
            "match_type": "T20",
            "temperature": 29.0,
            "humidity": 60.0,
            "wind_speed": 8.0,
            "dew_point": 15.0,
            "cloud_cover": 10.0,
            "pitch_age_days": 3,
            "season": "Summer",
            "day_night": 1,
            "soil_composition": "Red Soil",
            "pitch_strip_number": 5,
            "grass_coverage": 2.0,
            "compaction_kpa": 340.0,
        },
        {
            "venue": "Maharashtra Cricket Association Stadium",
            "city": "Pune",
            "match_type": "ODI",
            "temperature": 28.0,
            "humidity": 60.0,
            "wind_speed": 10.0,
            "dew_point": 12.0,
            "cloud_cover": 30.0,
            "pitch_age_days": 4,
            "season": "Post-Monsoon",
            "day_night": 0,
            "soil_composition": "Mixed Soil",
            "pitch_strip_number": 5,
            "grass_coverage": 5.0,
            "compaction_kpa": 320.0,
        },
        {
            "venue": "Arun Jaitley Stadium",
            "city": "Delhi",
            "match_type": "T20",
            "temperature": 40.0,
            "humidity": 30.0,
            "wind_speed": 6.0,
            "dew_point": 0.0,
            "cloud_cover": 0.0,
            "pitch_age_days": 8,
            "season": "Summer",
            "day_night": 0,
            "soil_composition": "Red Soil",
            "pitch_strip_number": 7,
            "grass_coverage": 1.0,
            "compaction_kpa": 450.0,
        },
        {
            "venue": "M. A. Chidambaram Stadium",
            "city": "Chennai",
            "match_type": "T20",
            "temperature": 44.0,
            "humidity": 95.0,
            "wind_speed": 20.0,
            "dew_point": 18.0,
            "cloud_cover": 100.0,
            "pitch_age_days": 1,
            "season": "Monsoon",
            "day_night": 1,
            "soil_composition": "Red Soil",
            "pitch_strip_number": 2,
            "grass_coverage": 12.0,
            "compaction_kpa": 250.0,
        }
    ]

    for sc in scenarios:
        if sc["city"] == "Pune":
            continue
        for _ in range(35):
            temp = float(np.clip(sc["temperature"] + rng.normal(0, 0.8), 12, 45))
            hum = float(np.clip(sc["humidity"] + rng.normal(0, 1.5), 10, 100))
            wind = float(np.clip(sc["wind_speed"] + rng.normal(0, 0.8), 2, 35))
            dew = float(np.clip(sc["dew_point"] + rng.normal(0, 0.8), 0, 25))
            cloud = float(np.clip(sc["cloud_cover"] + rng.normal(0, 4.0), 0, 100))
            age = int(np.clip(sc["pitch_age_days"] + rng.choice([-1, 0, 1]), 1, 12))
            grass = float(np.clip(sc["grass_coverage"] + rng.normal(0, 0.4), 0.0, 15.0))
            compact = float(np.clip(sc["compaction_kpa"] + rng.normal(0, 8.0), 100.0, 500.0))
            
            rows.append({
                "venue": sc["venue"],
                "city": sc["city"],
                "country": "India",
                "match_type": sc["match_type"],
                "temperature": temp,
                "humidity": hum,
                "wind_speed": wind,
                "dew_point": dew,
                "cloud_cover": cloud,
                "pitch_age_days": age,
                "ground_avg_1st_innings": {"T20": 165.0, "ODI": 270.0, "Test": 330.0}[sc["match_type"]] + rng.normal(0, 3.0),
                "ground_pace_wickets_pct": 50.0 + rng.normal(0, 2.0),
                "ground_spin_wickets_pct": 50.0 - rng.normal(0, 2.0),
                "season": sc["season"],
                "day_night": sc["day_night"],
                "soil_composition": sc["soil_composition"],
                "pitch_strip_number": sc["pitch_strip_number"],
                "grass_coverage": grass,
                "compaction_kpa": compact,
            })

    return assign_pitch_labels(pd.DataFrame(rows), noise_std=1.6, random_state=RANDOM_STATE)


def build_dataset(output_path: Path = DATASET_PATH, n_samples: int = 1800) -> pd.DataFrame:
    """Build the final dataset, enriching synthetic data with raw rows when present."""
    synthetic = generate_synthetic_dataset(n_samples=n_samples)
    raw_rows = load_raw_cricsheet_rows()
    if not raw_rows.empty:
        # Fill missing physical pitch columns if they are not present
        if "soil_composition" not in raw_rows.columns:
            raw_rows["soil_composition"] = "Mixed Soil"
        if "pitch_strip_number" not in raw_rows.columns:
            raw_rows["pitch_strip_number"] = 4
        if "grass_coverage" not in raw_rows.columns:
            raw_rows["grass_coverage"] = 4.5
        if "compaction_kpa" not in raw_rows.columns:
            raw_rows["compaction_kpa"] = 280.0
        
        raw_rows = assign_pitch_labels(raw_rows.drop(columns=["observed_wickets"], errors="ignore"))
        dataset = pd.concat([synthetic, raw_rows], ignore_index=True)
    else:
        dataset = synthetic

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    logger.info("Saved dataset with %s rows to %s.", len(dataset), output_path)
    return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    build_dataset()
