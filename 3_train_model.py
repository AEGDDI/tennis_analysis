# 3_train_model.py
# Trains the XGBoost model on 2000-2022 data, evaluates on 2023-2025,
# and builds a PlayerStateTracker with live player states.
# Output: models/xgb_atp.joblib, data/processed/player_states.pkl

import os
import pickle

import joblib

from src.data.process import clean_matches
from src.features.builder import build_features, symmetrize, MODEL_FEATURES
from src.models.train import train_model, evaluate
from src.models.predict import PlayerStateTracker

TRAIN_CUTOFF = 2022
VAL_START    = 2023
COMBINED_PATH = "data/raw/matches_combined.pkl"
FEATURES_CACHE = "data/processed/features.pkl"

if __name__ == "__main__":
    print("Loading combined match data...")
    with open(COMBINED_PATH, "rb") as f:
        raw_matches = pickle.load(f)
    print(f"  {len(raw_matches):,} matches loaded from {COMBINED_PATH}")

    clean = clean_matches(raw_matches)

    print("Building features...")
    features = build_features(
        clean,
        use_cache=True,
        cache_path=FEATURES_CACHE,
    )

    print("Symmetrising dataset...")
    sym = symmetrize(features[features["is_complete"]])

    train = sym[sym["year"] <= TRAIN_CUTOFF]
    val   = sym[sym["year"] >= VAL_START]
    print(f"  Train: {len(train):,} rows ({2000}–{TRAIN_CUTOFF})")
    print(f"  Val  : {len(val):,} rows  ({VAL_START}–{sym['year'].max()})")

    print("\nTraining XGBoost...")
    model = train_model(train, model_path="models/xgb_atp.joblib", complete_only=False)

    train_metrics = evaluate(model, train)
    val_metrics   = evaluate(model, val)
    print(f"  Train accuracy : {train_metrics['accuracy']:.3f}")
    print(f"  Val   accuracy : {val_metrics['accuracy']:.3f}")
    print(f"  Val   log-loss : {val_metrics['log_loss']:.3f}")
    print(f"  Val   Brier    : {val_metrics['brier_score']:.3f}")

    print("\nBuilding live player states (replaying all matches)...")
    tracker = PlayerStateTracker()
    tracker.process_all(clean)

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/player_states.pkl", "wb") as f:
        pickle.dump(tracker, f)

    print(f"\nDone — {len(tracker.name):,} player states saved to data/processed/player_states.pkl")
    print(f"  Last match date: {tracker._last_date}")
