import pandas as pd
from typing import Dict


def load_event_data(file_path: str) -> pd.DataFrame:
    """
    Load raw event-level football data.
    Supports CSV files.
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower().str.strip()
    return df


def preprocess_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize event dataset.
    """
    required_cols = ["team", "player", "event_type", "minute"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=required_cols)
    df["minute"] = df["minute"].astype(int)

    return df


def compute_performance_metrics(df: pd.DataFrame) -> Dict:
    """
    Generate core performance metrics.
    """

    total_shots = len(df[df["event_type"] == "shot"])
    goals = len(df[df["event_type"] == "goal"])

    scoring_efficiency = goals / total_shots if total_shots > 0 else 0

    defensive_actions = len(
        df[df["event_type"].isin(["tackle", "interception"])]
    )

    return {
        "total_shots": total_shots,
        "goals": goals,
        "scoring_efficiency": round(scoring_efficiency, 3),
        "defensive_actions": defensive_actions,
    }