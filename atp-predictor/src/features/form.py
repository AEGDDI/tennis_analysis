"""Recent match form (last N matches win rate)."""

from collections import defaultdict, deque

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_recent_form(
    matches: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    For each match, compute each player's win rate over their last `window` matches.
    Uses fixed-length deques for O(n) performance.

    Adds columns:
      winner_form, loser_form   – win rate in last N matches (NaN if none)
      winner_form_n, loser_form_n
      form_diff                 – winner minus loser (NaN→0.5)
    """
    # player_id -> deque of 0/1 results (1=win), max length = window
    results: dict = defaultdict(lambda: deque(maxlen=window))

    w_form, l_form, w_fn, l_fn = [], [], [], []

    for row in tqdm(
        matches.itertuples(index=False), total=len(matches), desc="Form"
    ):
        wid, lid = row.winner_id, row.loser_id

        wdq = results[wid]
        ldq = results[lid]
        wn, ln = len(wdq), len(ldq)

        w_form.append(sum(wdq) / wn if wn > 0 else np.nan)
        l_form.append(sum(ldq) / ln if ln > 0 else np.nan)
        w_fn.append(wn)
        l_fn.append(ln)

        wdq.append(1)
        ldq.append(0)

    out = matches.copy()
    out["winner_form"] = w_form
    out["loser_form"] = l_form
    out["winner_form_n"] = w_fn
    out["loser_form_n"] = l_fn
    out["form_diff"] = (
        out["winner_form"].fillna(0.5) - out["loser_form"].fillna(0.5)
    )
    return out
