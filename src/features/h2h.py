"""Head-to-head statistics (pre-match, no look-ahead)."""

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_h2h(matches: pd.DataFrame) -> pd.DataFrame:
    """
    For every match record the winner's H2H win rate against the loser
    using only matches played BEFORE this one.

    Adds columns:
      winner_h2h_wins, winner_h2h_losses, h2h_total, winner_h2h_winrate
    """
    h2h_counts: Dict[Tuple[int, int], list] = defaultdict(lambda: [0, 0])

    w_h2h_wins, w_h2h_losses, totals = [], [], []

    for row in tqdm(matches.itertuples(index=False), total=len(matches), desc="H2H"):
        wid, lid = row.winner_id, row.loser_id
        key = (min(wid, lid), max(wid, lid))
        wins_min, wins_max = h2h_counts[key]

        if wid == key[0]:
            w_wins, w_losses = wins_min, wins_max
        else:
            w_wins, w_losses = wins_max, wins_min

        w_h2h_wins.append(w_wins)
        w_h2h_losses.append(w_losses)
        totals.append(w_wins + w_losses)

        if wid == key[0]:
            h2h_counts[key][0] += 1
        else:
            h2h_counts[key][1] += 1

    out = matches.copy()
    out["winner_h2h_wins"] = w_h2h_wins
    out["winner_h2h_losses"] = w_h2h_losses
    out["h2h_total"] = totals
    out["winner_h2h_winrate"] = np.where(
        out["h2h_total"] > 0,
        out["winner_h2h_wins"] / out["h2h_total"],
        0.5,
    )
    return out
