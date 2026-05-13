# 2_build_features.py
# Cleans the combined match dataset and engineers all model features.
# Reads from data/raw/matches_combined.pkl (output of 1_fetch_data.py).
# Output: data/processed/features.pkl

import pickle

from src.data.process import clean_matches
from src.features.builder import build_features

if __name__ == "__main__":
    print("Loading combined match dataset...")
    with open("data/raw/matches_combined.pkl", "rb") as f:
        matches = pickle.load(f)
    print(f"  {len(matches):,} raw matches loaded")

    print("\nCleaning...")
    matches = clean_matches(matches)
    print(f"  {len(matches):,} matches after cleaning")
    print(f"  {matches['is_complete'].mean():.1%} complete (no retirements/walkovers)")

    print("\nEngineering features (Elo, H2H, surface win rate, form)...")
    print("  This step takes ~5 minutes on first run — result is cached afterwards.")
    features = build_features(
        matches,
        use_cache=False,
        cache_path="data/processed/features.pkl",
    )

    print(f"\nDone — {len(features):,} rows saved to data/processed/features.pkl")
