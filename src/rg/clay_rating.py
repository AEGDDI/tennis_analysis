"""
Clay court strength rating — composite metric for Roland Garros seeding.

  1. Clay Elo        (45%) — long-term clay performance
  2. Clay win rate   (30%) — rolling 12-month results on clay
  3. Overall form    (25%) — last 20 matches across all surfaces
"""

from typing import List

import numpy as np
import pandas as pd

from src.models.predict import PlayerStateTracker

SURFACE = "Clay"

WEIGHTS = {
    "clay_elo": 0.45,
    "clay_winrate": 0.30,
    "form": 0.25,
}

MIN_CLAY_MATCHES = 5


def compute_clay_ratings(
    tracker: PlayerStateTracker,
    player_ids: List[int],
) -> pd.DataFrame:
    """
    Compute clay strength scores for every player in player_ids.

    Returns DataFrame sorted by clay_strength with columns:
      player_id, name, rank, clay_elo, clay_winrate, clay_n, form,
      clay_elo_z, clay_winrate_z, form_z, clay_strength
    """
    records = []
    for pid in player_ids:
        s = tracker.get_player_state(pid, SURFACE)
        clay_wr = s["surf_winrate"] if s["surf_n"] >= MIN_CLAY_MATCHES else s["form"]
        records.append({
            "player_id": pid,
            "name": s["name"],
            "rank": s["rank"],
            "clay_elo": s["surface_elo"],
            "clay_winrate": clay_wr,
            "clay_n": s["surf_n"],
            "form": s["form"],
            "elo": s["elo"],
        })

    df = pd.DataFrame(records)

    for col in ["clay_elo", "clay_winrate", "form"]:
        mu, sigma = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / sigma if sigma > 0 else 0.0

    df["clay_strength"] = (
        WEIGHTS["clay_elo"] * df["clay_elo_z"]
        + WEIGHTS["clay_winrate"] * df["clay_winrate_z"]
        + WEIGHTS["form"] * df["form_z"]
    )

    return df.sort_values("clay_strength", ascending=False).reset_index(drop=True)


def get_seeding_order(ratings: pd.DataFrame) -> List[int]:
    """Return player IDs ordered by clay strength (seed 1 first)."""
    return ratings["player_id"].tolist()
