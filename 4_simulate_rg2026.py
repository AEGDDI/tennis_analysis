# 4_simulate_rg2026.py
# Selects the top 128 clay players, creates the RG draw, runs 20 000 Monte Carlo
# simulations, and saves precomputed results for the Streamlit app.
# Output: data/rg2026_results.pkl

import os
import pickle

import joblib

from src.models.predict import MatchPredictor, PlayerStateTracker
from src.rg.clay_rating import compute_clay_ratings, get_seeding_order
from src.rg.draw import create_rg_draw
from src.rg.simulator import RGSimulator

DRAW_SIZE = 128
N_SIM     = 20_000
SEED      = 42

if __name__ == "__main__":
    print("Loading model and player states...")
    predictor = MatchPredictor(model_path="models/xgb_atp.joblib")

    with open("data/processed/player_states.pkl", "rb") as f:
        tracker: PlayerStateTracker = pickle.load(f)

    print(f"  {len(tracker.name):,} players loaded")

    print("\nSelecting top 128 clay players by ATP rank...")
    ranked_ids = sorted(
        [pid for pid, r in tracker.rank.items() if r < 9999],
        key=lambda pid: tracker.rank[pid],
    )[:DRAW_SIZE]

    print("Computing clay strength scores...")
    ratings = compute_clay_ratings(tracker, ranked_ids)
    seeded_ids = get_seeding_order(ratings)

    print("\nCreating draw (official Grand Slam seeding rules)...")
    draw = create_rg_draw(seeded_ids, rng_seed=SEED)

    print(f"\nRunning {N_SIM:,} Monte Carlo simulations...")
    simulator = RGSimulator(predictor, tracker)
    results = simulator.simulate(draw, n_sim=N_SIM, seed=SEED, verbose=True)

    os.makedirs("data", exist_ok=True)
    with open("data/rg2026_results.pkl", "wb") as f:
        pickle.dump({"results": results, "ratings": ratings, "draw": draw}, f)

    print("\n=== Top 10 Championship Odds ===")
    for _, row in results.head(10).iterrows():
        print(f"  {row['name']:<25}  {row['W']:.1%}")

    print("\nDone — results saved to data/rg2026_results.pkl")
