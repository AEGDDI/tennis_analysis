"""Clean and standardize raw ATP match data."""

import pandas as pd

SURFACE_MAP = {"Hard": "Hard", "Clay": "Clay", "Grass": "Grass", "Carpet": "Carpet"}

ROUND_ORDER = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7, "RR": 3, "BR": 6,
}

LEVEL_LABELS = {
    "G": "Grand Slam",
    "M": "Masters 1000",
    "A": "ATP 500/250",
    "F": "ATP Finals",
    "D": "Davis Cup",
    "C": "Challenger",
}


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw match data:
    - Parse tourney_date to datetime
    - Standardize surface names
    - Fill missing ranks / rank points
    - Flag incomplete matches (retirements, walkovers)
    - Sort chronologically and add a global match_idx
    """
    df = df.copy()

    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )

    df["surface"] = df["surface"].map(SURFACE_MAP).fillna("Unknown")
    df["tourney_level_label"] = df["tourney_level"].map(LEVEL_LABELS).fillna("Other")

    df["winner_rank"] = pd.to_numeric(df["winner_rank"], errors="coerce").fillna(9999).astype(int)
    df["loser_rank"] = pd.to_numeric(df["loser_rank"], errors="coerce").fillna(9999).astype(int)
    df["winner_rank_points"] = pd.to_numeric(df["winner_rank_points"], errors="coerce").fillna(0.0)
    df["loser_rank_points"] = pd.to_numeric(df["loser_rank_points"], errors="coerce").fillna(0.0)
    df["winner_age"] = pd.to_numeric(df["winner_age"], errors="coerce")
    df["loser_age"] = pd.to_numeric(df["loser_age"], errors="coerce")

    df["round_num"] = df["round"].map(ROUND_ORDER).fillna(0).astype(int)

    score_str = df["score"].fillna("")
    df["is_complete"] = ~score_str.str.contains(r"W/O|RET|DEF", regex=True)

    # Drop rows with invalid dates or missing player IDs
    df = df.dropna(subset=["tourney_date", "winner_id", "loser_id"])
    df["winner_id"] = df["winner_id"].astype(int)
    df["loser_id"] = df["loser_id"].astype(int)

    df = df.sort_values(["tourney_date", "tourney_id", "round_num"]).reset_index(drop=True)
    df["match_idx"] = df.index

    return df
