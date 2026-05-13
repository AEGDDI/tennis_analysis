"""Rolling surface-specific win rates with a sliding time window."""

from collections import defaultdict, deque

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_surface_winrates(
    matches: pd.DataFrame,
    window_days: int = 365,
) -> pd.DataFrame:
    """
    For each match, compute each player's win rate on the match surface
    over the preceding `window_days` days.

    Adds columns:
      winner_surf_winrate, loser_surf_winrate   – NaN if no history
      winner_surf_n, loser_surf_n               – number of matches in window
      surf_winrate_diff                          – winner minus loser (NaN->0.5)
    """
    window = pd.Timedelta(days=window_days)
    history: dict = defaultdict(deque)

    w_wr, l_wr, w_n, l_n = [], [], [], []

    for row in tqdm(
        matches.itertuples(index=False), total=len(matches), desc="Surface WR"
    ):
        wid, lid = row.winner_id, row.loser_id
        surface = row.surface
        date = row.tourney_date
        cutoff = date - window

        def _stats(key):
            dq = history[key]
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            n = len(dq)
            wins = sum(r for _, r in dq)
            return (wins / n if n > 0 else np.nan), n

        wwr, wn = _stats((wid, surface))
        lwr, ln = _stats((lid, surface))

        w_wr.append(wwr)
        l_wr.append(lwr)
        w_n.append(wn)
        l_n.append(ln)

        history[(wid, surface)].append((date, 1))
        history[(lid, surface)].append((date, 0))

    out = matches.copy()
    out["winner_surf_winrate"] = w_wr
    out["loser_surf_winrate"] = l_wr
    out["winner_surf_n"] = w_n
    out["loser_surf_n"] = l_n
    out["surf_winrate_diff"] = (
        out["winner_surf_winrate"].fillna(0.5) - out["loser_surf_winrate"].fillna(0.5)
    )
    return out
