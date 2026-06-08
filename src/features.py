"""Feature engineering utilities for cricket pitch classification."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import COASTAL_VENUES, SUBCONTINENTAL_CITIES

logger = logging.getLogger(__name__)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific features used by model training and inference.

    Args:
        df: Input dataframe containing raw cricket, venue, and weather columns.

    Returns:
        A copy of the dataframe with derived feature columns appended.
    """
    engineered = df.copy()
    engineered["humidity_temp_ratio"] = engineered["humidity"] / (
        engineered["temperature"] + 1
    )
    engineered["is_subcontinental"] = (
        engineered["city"].isin(SUBCONTINENTAL_CITIES).astype(int)
    )
    engineered["is_coastal"] = engineered["venue"].isin(COASTAL_VENUES).astype(int)
    engineered["wicket_imbalance"] = (
        engineered["ground_pace_wickets_pct"]
        - engineered["ground_spin_wickets_pct"]
    ).abs()
    engineered["high_dew"] = (engineered["dew_point"] > 18).astype(int)
    engineered["pitch_freshness"] = (engineered["pitch_age_days"] < 3).astype(int)
    
    # New physical features domain-specific relations
    engineered["soil_spin_factor"] = (
        (engineered["soil_composition"] == "Red Soil").astype(int)
        if "soil_composition" in engineered.columns else 0
    )
    engineered["compaction_density"] = (
        engineered["compaction_kpa"] / (engineered["pitch_age_days"] + 1)
        if "compaction_kpa" in engineered.columns else 280.0 / (engineered["pitch_age_days"] + 1)
    )
    engineered["grass_friction_ratio"] = (
        engineered["grass_coverage"] * engineered["humidity"]
        if "grass_coverage" in engineered.columns else 4.5 * engineered["humidity"]
    )
    
    # Define cities sets
    COASTAL_CITIES = {"Mumbai", "Chennai", "Cuttack"}
    SPIN_CITIES = {"Chennai", "Delhi", "Kolkata", "Ahmedabad", "Kanpur", "Lucknow"}
    BATTER_CITIES = {"Bengaluru", "Mumbai", "Indore", "Hyderabad", "Pune"}

    import numpy as np
    
    # Initialize scores arrays
    batting = np.zeros(len(engineered))
    pace = np.zeros(len(engineered))
    spin = np.zeros(len(engineered))

    # Match type limits for batting score
    limits = engineered["match_type"].map(lambda x: {"T20": 178, "ODI": 285, "Test": 340}.get(x, 178))
    batting += (engineered["ground_avg_1st_innings"] >= limits).astype(int) * 1
    batting += engineered["city"].isin(BATTER_CITIES).astype(int) * 1
    batting += ((engineered["cloud_cover"] < 35) & (engineered["humidity"] < 60)).astype(int) * 1

    pace += (engineered["humidity"] > 68).astype(int) * 1
    pace += (engineered["humidity"] > 85).astype(int) * 1
    pace += engineered["city"].isin(COASTAL_CITIES).astype(int) * 1
    pace += (engineered["ground_pace_wickets_pct"] > 54).astype(int) * 1
    pace += ((engineered["cloud_cover"] > 55) | (engineered["wind_speed"] > 18)).astype(int) * 1

    spin += ((engineered["humidity"] < 55) & (engineered["temperature"] > 29)).astype(int) * 1
    spin += ((engineered["humidity"] < 20) & (engineered["temperature"] > 40)).astype(int) * 2
    spin += engineered["city"].isin(SPIN_CITIES).astype(int) * 1
    spin += (engineered["ground_spin_wickets_pct"] > 50).astype(int) * 1
    spin += (engineered["pitch_age_days"] > 4).astype(int) * 1

    if "soil_composition" in engineered.columns:
        batting += (engineered["soil_composition"] == "Black Soil").astype(int) * 1
        pace += (engineered["soil_composition"] == "Black Soil").astype(int) * 1
        spin += (engineered["soil_composition"] == "Red Soil").astype(int) * 2

    if "grass_coverage" in engineered.columns:
        batting += (engineered["grass_coverage"] < 3.0).astype(int) * 1
        batting += (engineered["grass_coverage"] < 2.0).astype(int) * 1
        pace += (engineered["grass_coverage"] > 8.0).astype(int) * 2
        pace += (engineered["grass_coverage"] > 12.0).astype(int) * 1
        # Moderate grass
        pace += ((engineered["grass_coverage"] >= 4.0) & (engineered["grass_coverage"] <= 7.0)).astype(int) * 1

    if "compaction_kpa" in engineered.columns:
        fresh = (engineered["pitch_age_days"] < 5).astype(int)
        batting += ((engineered["compaction_kpa"] > 320.0) & (fresh == 1)).astype(int) * 2
        batting += ((engineered["compaction_kpa"] > 400.0) & (fresh == 1)).astype(int) * 4
        spin += (engineered["compaction_kpa"] < 200.0).astype(int) * 2
        # Moderate compaction
        pace += ((engineered["compaction_kpa"] >= 250.0) & (engineered["compaction_kpa"] <= 310.0)).astype(int) * 1

    # Severe dust bowl condition
    dry_old = ((engineered["pitch_age_days"] > 10) & (engineered["humidity"] < 30)).astype(int)
    spin += dry_old * 2

    engineered["rule_batting_score"] = batting
    engineered["rule_pace_score"] = pace
    engineered["rule_spin_score"] = spin

    engineered["spin_degradation_index"] = (
        engineered["temperature"] + engineered["pitch_age_days"] + (100 - engineered["humidity"])
    )
    engineered["moisture_index"] = (
        engineered["humidity"] + engineered["dew_point"] + engineered["cloud_cover"]
    )
    engineered["surface_wear_index"] = (
        engineered["pitch_age_days"] * engineered["compaction_kpa"]
        if "compaction_kpa" in engineered.columns else engineered["pitch_age_days"] * 280.0
    )

    logger.info("Added derived features to %s rows.", len(engineered))
    return engineered


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    sample = pd.DataFrame(
        {
            "venue": ["Wankhede Stadium"],
            "city": ["Mumbai"],
            "humidity": [72],
            "temperature": [30],
            "ground_pace_wickets_pct": [58],
            "ground_spin_wickets_pct": [42],
            "dew_point": [21],
            "pitch_age_days": [2],
        }
    )
    print(add_derived_features(sample).to_string(index=False))
