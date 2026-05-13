"""
Monte Carlo tournament simulator for Roland Garros.

Tracks each player's probability of reaching every round — R128 through Champion.
Analogous to the team × position matrix in league football simulators.

Round indices:
  0  R128  entered the draw (always 100%)
  1  R64   won Round 1
  2  R32   won Round 2
  3  R16   won Round 3
  4  QF    won Round 4
  5  SF    won Round 5
  6  F     won Round 6
  7  W     Champion
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.predict import MatchPredictor, PlayerStateTracker

SURFACE = "Clay"
ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "W"]
N_ROUNDS = len(ROUNDS)


class RGSimulator:
    """Monte Carlo simulation of a 128-player Roland Garros bracket."""

    def __init__(self, predictor: MatchPredictor, tracker: PlayerStateTracker) -> None:
        self.predictor = predictor
        self.tracker = tracker
        self._prob_cache: Dict[Tuple[int, int], float] = {}

    def _match_prob(self, p1_id: int, p2_id: int) -> float:
        """P(p1 beats p2) on clay, with symmetric caching."""
        key = (p1_id, p2_id)
        if key not in self._prob_cache:
            prob, _ = self.predictor.predict_from_tracker(self.tracker, p1_id, p2_id, SURFACE)
            self._prob_cache[key] = prob
            self._prob_cache[(p2_id, p1_id)] = 1.0 - prob
        return self._prob_cache[key]

    def _simulate_once(
        self,
        draw: List[int],
        rng: np.random.Generator,
        counts: Dict[int, np.ndarray],
    ) -> None:
        bracket = list(draw)
        round_idx = 1

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

    def simulate(
        self,
        draw: List[int],
        n_sim: int = 20_000,
        seed: int = 42,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Simulate Roland Garros n_sim times and return a probability matrix.

        Returns DataFrame (128 rows) sorted by champion probability with columns:
          player_id, name, draw_pos, rank, clay_elo, elo, clay_winrate, form,
          R128, R64, R32, R16, QF, SF, F, W
        """
        if len(draw) != 128:
            raise ValueError(f"draw must have 128 players, got {len(draw)}")

        rng = np.random.default_rng(seed)

        counts: Dict[int, np.ndarray] = {}
        for pid in draw:
            counts[pid] = np.zeros(N_ROUNDS, dtype=np.int64)
            counts[pid][0] = n_sim

        iterator = tqdm(range(n_sim), desc="Simulating RG 2026") if verbose else range(n_sim)
        for _ in iterator:
            self._simulate_once(draw, rng, counts)

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

        return pd.DataFrame(rows).sort_values("W", ascending=False).reset_index(drop=True)

    def matchup_matrix(self, player_ids: List[int]) -> pd.DataFrame:
        """Head-to-head probability matrix for a list of players."""
        n = len(player_ids)
        names = [self.tracker.get_player_state(pid, SURFACE)["name"] for pid in player_ids]
        matrix = np.zeros((n, n))

        for i, p1 in enumerate(player_ids):
            for j, p2 in enumerate(player_ids):
                matrix[i, j] = 0.5 if i == j else self._match_prob(p1, p2)

        return pd.DataFrame(matrix, index=names, columns=names).round(3)

    def draw_sensitivity(
        self,
        player_ids: List[int],
        n_draws: int = 100,
        n_sim_per_draw: int = 5_000,
        rng_seed: int = 0,
    ) -> pd.DataFrame:
        """
        Assess how much champion probabilities vary across different random draws.
        Quantifies bracket luck vs underlying player quality.
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
            rows.append({
                "player_id": pid,
                "name": s["name"],
                "win_prob_mean": round(mu, 4),
                "win_prob_std": round(sigma, 4),
                "win_prob_cv": round(sigma / mu if mu > 0 else 0, 3),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("win_prob_mean", ascending=False)
            .reset_index(drop=True)
        )
