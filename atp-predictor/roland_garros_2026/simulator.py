"""
Monte Carlo tournament simulator for Roland Garros.

Extends the base TournamentSimulator to track each player's probability
of reaching *every* round — not just of winning the title.  The result
is a players × rounds probability matrix, directly analogous to the
team × position matrix produced by Premier League season simulators.

Round indices
-------------
  0  R128  entered the draw (always 100 %)
  1  R64   won Round 1
  2  R32   won Round 2
  3  R16   won Round 3  (last 16)
  4  QF    won Round 4  (quarter-finals)
  5  SF    won Round 5  (semi-finals)
  6  F     won Round 6  (final)
  7  W     won Round 7  (champion)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.predict import MatchPredictor, PlayerStateTracker

SURFACE = "Clay"
ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "W"]
N_ROUNDS = len(ROUNDS)  # 8


class RGSimulator:
    """
    Monte Carlo simulation of a 128-player Roland Garros bracket.

    Tracks each player's probability of reaching every round from R64
    through to Champion.  Run 20 000+ simulations for stable estimates.

    Parameters
    ----------
    predictor : MatchPredictor wrapping the trained XGBoost model
    tracker   : PlayerStateTracker with up-to-date player states
    """

    def __init__(
        self,
        predictor: MatchPredictor,
        tracker: PlayerStateTracker,
    ) -> None:
        self.predictor = predictor
        self.tracker = tracker
        # Cache match probabilities — each unique pair is computed once
        self._prob_cache: Dict[Tuple[int, int], float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_prob(self, p1_id: int, p2_id: int) -> float:
        """P(p1 beats p2) on clay, with symmetric caching."""
        key = (p1_id, p2_id)
        if key not in self._prob_cache:
            prob, _ = self.predictor.predict_from_tracker(
                self.tracker, p1_id, p2_id, SURFACE
            )
            self._prob_cache[key] = prob
            self._prob_cache[(p2_id, p1_id)] = 1.0 - prob
        return self._prob_cache[key]

    def _simulate_once(
        self,
        draw: List[int],
        rng: np.random.Generator,
        counts: Dict[int, np.ndarray],
    ) -> None:
        """Run one full tournament and increment round counts in-place."""
        bracket = list(draw)
        round_idx = 1  # index 0 = R128 (draw entry) was pre-filled

        while len(bracket) > 1:
            next_round: List[int] = []
            for i in range(0, len(bracket), 2):
                p1, p2 = bracket[i], bracket[i + 1]
                prob = self._match_prob(p1, p2)
                winner = p1 if rng.random() < prob else p2
                counts[winner][round_idx] += 1
                next_round.append(winner)
            bracket = next_round
            round_idx += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        draw: List[int],
        n_sim: int = 20_000,
        seed: int = 42,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Simulate Roland Garros *n_sim* times and return a probability matrix.

        Parameters
        ----------
        draw   : 128 player IDs in bracket order (output of draw.create_rg_draw)
        n_sim  : number of Monte Carlo iterations (≥10 000 recommended)
        seed   : random seed for reproducibility
        verbose: show progress bar

        Returns
        -------
        DataFrame (128 rows × columns below), sorted by champion probability:
          player_id, name, draw_pos, rank, clay_elo, elo,
          clay_winrate, form,
          R128, R64, R32, R16, QF, SF, F, W
        """
        if len(draw) != 128:
            raise ValueError(f"draw must have 128 players, got {len(draw)}")

        rng = np.random.default_rng(seed)

        # Initialise counts: every player starts at R128 = n_sim
        counts: Dict[int, np.ndarray] = {}
        for pid in draw:
            counts[pid] = np.zeros(N_ROUNDS, dtype=np.int64)
            counts[pid][0] = n_sim  # every player enters the draw

        iterator = range(n_sim)
        if verbose:
            iterator = tqdm(iterator, desc="Simulating RG 2026")

        for _ in iterator:
            self._simulate_once(draw, rng, counts)

        # Build result DataFrame
        rows = []
        for pos, pid in enumerate(draw):
            s = self.tracker.get_player_state(pid, SURFACE)
            row: dict = {
                "player_id": pid,
                "name": s["name"],
                "draw_pos": pos + 1,
                "rank": s["rank"],
                "clay_elo": round(s["surface_elo"], 1),
                "elo": round(s["elo"], 1),
                "clay_winrate": round(s["surf_winrate"], 3),
                "form": round(s["form"], 3),
            }
            for i, rnd in enumerate(ROUNDS):
                row[rnd] = round(counts[pid][i] / n_sim, 4)
            rows.append(row)

        df = pd.DataFrame(rows)
        return df.sort_values("W", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Draw sensitivity analysis
    # ------------------------------------------------------------------

    def draw_sensitivity(
        self,
        player_ids: List[int],
        n_draws: int = 100,
        n_sim_per_draw: int = 5_000,
        rng_seed: int = 0,
    ) -> pd.DataFrame:
        """
        Assess how much champion probabilities vary across different random draws.

        Generates *n_draws* independent bracket realisations, simulates each
        *n_sim_per_draw* times, and reports the mean ± std of championship
        probability for each player.

        This quantifies 'draw luck' — how much the bracket lottery matters
        relative to underlying player quality.

        Parameters
        ----------
        player_ids     : 128 player IDs ordered by seeding (seed 1 first)
        n_draws        : number of random draw realisations
        n_sim_per_draw : simulations per draw (lower = faster, less precise)

        Returns
        -------
        DataFrame: player_id, name, win_prob_mean, win_prob_std, win_prob_cv
                   sorted by win_prob_mean descending
        """
        from .draw import create_rg_draw

        win_probs: Dict[int, List[float]] = defaultdict(list)

        for draw_seed in tqdm(range(n_draws), desc="Draw sensitivity"):
            draw = create_rg_draw(player_ids, rng_seed=rng_seed + draw_seed)
            result = self.simulate(draw, n_sim=n_sim_per_draw, seed=draw_seed, verbose=False)
            for _, row in result.iterrows():
                win_probs[row["player_id"]].append(row["W"])

        rows = []
        for pid, probs in win_probs.items():
            s = self.tracker.get_player_state(pid, SURFACE)
            mu = np.mean(probs)
            sigma = np.std(probs)
            rows.append(
                {
                    "player_id": pid,
                    "name": s["name"],
                    "win_prob_mean": round(mu, 4),
                    "win_prob_std": round(sigma, 4),
                    "win_prob_cv": round(sigma / mu if mu > 0 else 0, 3),
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values("win_prob_mean", ascending=False)
            .reset_index(drop=True)
        )

    def matchup_analysis(
        self,
        player_ids: List[int],
        surface: str = SURFACE,
    ) -> pd.DataFrame:
        """
        Head-to-head probability matrix for a list of players.

        Returns a DataFrame where entry [i, j] = P(player i beats player j).
        Useful for previewing likely semi-final and final matchups.
        """
        n = len(player_ids)
        names = [self.tracker.get_player_state(pid, surface)["name"] for pid in player_ids]
        matrix = np.zeros((n, n))

        for i, p1 in enumerate(player_ids):
            for j, p2 in enumerate(player_ids):
                if i == j:
                    matrix[i, j] = 0.5
                else:
                    matrix[i, j] = self._match_prob(p1, p2)

        return pd.DataFrame(matrix, index=names, columns=names).round(3)
