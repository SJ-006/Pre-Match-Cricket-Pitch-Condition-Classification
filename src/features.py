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
