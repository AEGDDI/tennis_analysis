"""Elo rating system — global and surface-specific."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_ELO = 1500.0
K = 32.0
SURFACES = ["Hard", "Clay", "Grass", "Carpet"]


def win_prob(rating_a: float, rating_b: float) -> float:
    """Expected win probability for player A given Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    r_winner: float, r_loser: float, k: float = K
) -> Tuple[float, float]:
    """Return updated (winner_elo, loser_elo) after a match."""
    p = win_prob(r_winner, r_loser)
    return r_winner + k * (1 - p), r_loser + k * (0 - (1 - p))


def compute_elo_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Walk forward through all matches in chronological order.
    Records each player's Elo BEFORE the match is played, then updates.

    Adds columns:
      winner_elo, loser_elo           – global Elo before match
      winner_surface_elo, loser_surface_elo – surface Elo before match
      elo_diff, surface_elo_diff      – winner minus loser
    """
    global_elo: Dict[int, float] = {}
    surf_elo: Dict[Tuple[int, str], float] = {}

    w_elo_pre, l_elo_pre = [], []
    w_surf_pre, l_surf_pre = [], []

    for row in tqdm(matches.itertuples(index=False), total=len(matches), desc="Elo"):
        wid, lid = row.winner_id, row.loser_id
        surface = row.surface

        wg = global_elo.get(wid, DEFAULT_ELO)
        lg = global_elo.get(lid, DEFAULT_ELO)
        ws = surf_elo.get((wid, surface), DEFAULT_ELO)
        ls = surf_elo.get((lid, surface), DEFAULT_ELO)

        w_elo_pre.append(wg)
        l_elo_pre.append(lg)
        w_surf_pre.append(ws)
        l_surf_pre.append(ls)

        # Update global
        new_wg, new_lg = update_elo(wg, lg)
        global_elo[wid] = new_wg
        global_elo[lid] = new_lg

        # Update surface
        new_ws, new_ls = update_elo(ws, ls)
        surf_elo[(wid, surface)] = new_ws
        surf_elo[(lid, surface)] = new_ls

    out = matches.copy()
    out["winner_elo"] = w_elo_pre
    out["loser_elo"] = l_elo_pre
    out["winner_surface_elo"] = w_surf_pre
    out["loser_surface_elo"] = l_surf_pre
    out["elo_diff"] = out["winner_elo"] - out["loser_elo"]
    out["surface_elo_diff"] = out["winner_surface_elo"] - out["loser_surface_elo"]
    return out


def get_current_elo(matches: pd.DataFrame) -> Tuple[Dict[int, float], Dict[Tuple[int, str], float]]:
    """
    Replay all matches and return the final Elo state dictionaries.
    Useful for seeding live predictions.
    """
    global_elo: Dict[int, float] = {}
    surf_elo: Dict[Tuple[int, str], float] = {}

    for row in matches.itertuples(index=False):
        wid, lid = row.winner_id, row.loser_id
        surface = row.surface

        wg = global_elo.get(wid, DEFAULT_ELO)
        lg = global_elo.get(lid, DEFAULT_ELO)
        ws = surf_elo.get((wid, surface), DEFAULT_ELO)
        ls = surf_elo.get((lid, surface), DEFAULT_ELO)

        global_elo[wid], global_elo[lid] = update_elo(wg, lg)
        surf_elo[(wid, surface)], surf_elo[(lid, surface)] = update_elo(ws, ls)

    return global_elo, surf_elo
