"""
Clay court strength rating — a composite metric for Roland Garros.

Analogous to team attack/defense parameters in Poisson football models,
this score captures each player's clay court profile in a single
interpretable number built from three dimensions:

  1. Clay Elo        (45%) — long-term clay performance, the most stable signal
  2. Clay win rate   (30%) — rolling 12-month results on clay
  3. Overall form    (25%) — last 20 matches across all surfaces

Weights favour the long-run signal (clay Elo) while acknowledging that
current form and recent clay results carry meaningful information about
a player's state heading into Roland Garros.
"""

from typing import List

import numpy as np
import pandas as pd

# Importable from the atp-predictor root (notebook working directory)
from src.models.predict import PlayerStateTracker

SURFACE = "Clay"

WEIGHTS = {
    "clay_elo": 0.45,
    "clay_winrate": 0.30,
    "form": 0.25,
}

# Minimum clay matches in the rolling window to treat clay_winrate as reliable
MIN_CLAY_MATCHES = 5


def compute_clay_ratings(
    tracker: PlayerStateTracker,
    player_ids: List[int],
) -> pd.DataFrame:
    """
    Compute clay strength scores for every player in *player_ids*.

    Parameters
    ----------
    tracker    : fitted PlayerStateTracker (call .process_all() first)
    player_ids : ordered list of player IDs to score

    Returns
    -------
    DataFrame sorted descending by *clay_strength*, with columns:
      player_id, name, rank, clay_elo, clay_winrate, clay_n, form,
      clay_elo_z, clay_winrate_z, form_z, clay_strength
    """
    records = []
    for pid in player_ids:
        s = tracker.get_player_state(pid, SURFACE)

        # If a player has too few clay matches, fall back to overall form for
        # the clay_winrate component so we don't penalise them unfairly.
        clay_wr = s["surf_winrate"] if s["surf_n"] >= MIN_CLAY_MATCHES else s["form"]

        records.append(
            {
                "player_id": pid,
                "name": s["name"],
                "rank": s["rank"],
                "clay_elo": s["surface_elo"],
                "clay_winrate": clay_wr,
                "clay_n": s["surf_n"],
                "form": s["form"],
                "elo": s["elo"],
            }
        )

    df = pd.DataFrame(records)

    # Z-score normalisation relative to the field (mean=0, std=1)
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
    """
    Return player IDs ordered by clay strength (seed 1 first).
    Used to determine who occupies each seeded draw slot.
    """
    return ratings["player_id"].tolist()
