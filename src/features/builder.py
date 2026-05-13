"""Build the full feature dataset and prepare it for modelling."""

import os
import pickle

import numpy as np
import pandas as pd

from .elo import compute_elo_ratings
from .h2h import compute_h2h
from .surface import compute_surface_winrates
from .form import compute_recent_form

CACHE_PATH = "data/processed/features.pkl"

MODEL_FEATURES = [
    "elo_diff",
    "surface_elo_diff",
    "rank_diff",
    "rank_points_diff",
    "p1_h2h_winrate",
    "h2h_total",
    "surf_winrate_diff",
    "form_diff",
    "age_diff",
    "p1_surf_n",
    "p2_surf_n",
]


def build_features(
    matches: pd.DataFrame,
    use_cache: bool = True,
    cache_path: str = CACHE_PATH,
) -> pd.DataFrame:
    """Run all feature modules in order and return the enriched DataFrame."""
    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    matches = compute_elo_ratings(matches)
    matches = compute_h2h(matches)
    matches = compute_surface_winrates(matches)
    matches = compute_recent_form(matches)

    matches["rank_diff_raw"] = matches["winner_rank"] - matches["loser_rank"]
    matches["rank_points_diff_raw"] = matches["winner_rank_points"] - matches["loser_rank_points"]
    matches["age_diff_raw"] = (
        matches["winner_age"].fillna(26) - matches["loser_age"].fillna(26)
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(matches, f)
    print(f"Features cached to {cache_path}")
    return matches


def symmetrize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Double the dataset by swapping winner/loser so the model sees both orientations.
    Returns a DataFrame with MODEL_FEATURES columns plus 'target' and metadata.
    """
    def _make_row(row, p1_is_winner: bool):
        if p1_is_winner:
            return {
                "p1_id": row.winner_id, "p2_id": row.loser_id,
                "p1_name": getattr(row, "winner_name", ""),
                "p2_name": getattr(row, "loser_name", ""),
                "p1_elo": row.winner_elo, "p2_elo": row.loser_elo,
                "p1_surface_elo": row.winner_surface_elo,
                "p2_surface_elo": row.loser_surface_elo,
                "p1_rank": row.winner_rank, "p2_rank": row.loser_rank,
                "p1_rank_points": row.winner_rank_points,
                "p2_rank_points": row.loser_rank_points,
                "p1_surf_winrate": row.winner_surf_winrate if not np.isnan(row.winner_surf_winrate) else 0.5,
                "p2_surf_winrate": row.loser_surf_winrate if not np.isnan(row.loser_surf_winrate) else 0.5,
                "p1_form": row.winner_form if not np.isnan(row.winner_form) else 0.5,
                "p2_form": row.loser_form if not np.isnan(row.loser_form) else 0.5,
                "p1_h2h_winrate": row.winner_h2h_winrate,
                "p1_surf_n": row.winner_surf_n, "p2_surf_n": row.loser_surf_n,
                "h2h_total": row.h2h_total,
                "age_diff": row.age_diff_raw,
                "target": 1,
            }
        else:
            return {
                "p1_id": row.loser_id, "p2_id": row.winner_id,
                "p1_name": getattr(row, "loser_name", ""),
                "p2_name": getattr(row, "winner_name", ""),
                "p1_elo": row.loser_elo, "p2_elo": row.winner_elo,
                "p1_surface_elo": row.loser_surface_elo,
                "p2_surface_elo": row.winner_surface_elo,
                "p1_rank": row.loser_rank, "p2_rank": row.winner_rank,
                "p1_rank_points": row.loser_rank_points,
                "p2_rank_points": row.winner_rank_points,
                "p1_surf_winrate": row.loser_surf_winrate if not np.isnan(row.loser_surf_winrate) else 0.5,
                "p2_surf_winrate": row.winner_surf_winrate if not np.isnan(row.winner_surf_winrate) else 0.5,
                "p1_form": row.loser_form if not np.isnan(row.loser_form) else 0.5,
                "p2_form": row.winner_form if not np.isnan(row.winner_form) else 0.5,
                "p1_h2h_winrate": 1.0 - row.winner_h2h_winrate,
                "p1_surf_n": row.loser_surf_n, "p2_surf_n": row.winner_surf_n,
                "h2h_total": row.h2h_total,
                "age_diff": -row.age_diff_raw,
                "target": 0,
            }

    rows_w = [_make_row(r, True) for r in df.itertuples(index=False)]
    rows_l = [_make_row(r, False) for r in df.itertuples(index=False)]

    out = pd.DataFrame(rows_w + rows_l)
    out["tourney_date"] = list(df["tourney_date"]) * 2
    out["year"] = list(df["year"]) * 2
    out["surface"] = list(df["surface"]) * 2
    out["match_idx"] = list(df["match_idx"]) * 2
    out["is_complete"] = list(df["is_complete"]) * 2

    out["elo_diff"] = out["p1_elo"] - out["p2_elo"]
    out["surface_elo_diff"] = out["p1_surface_elo"] - out["p2_surface_elo"]
    out["rank_diff"] = out["p1_rank"] - out["p2_rank"]
    out["rank_points_diff"] = out["p1_rank_points"] - out["p2_rank_points"]
    out["surf_winrate_diff"] = out["p1_surf_winrate"] - out["p2_surf_winrate"]
    out["form_diff"] = out["p1_form"] - out["p2_form"]

    return out.reset_index(drop=True)
