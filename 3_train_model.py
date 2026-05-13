# 3_train_model.py
# Trains the XGBoost model on 2000-2021 data, evaluates on 2022-2024,
# and builds a PlayerStateTracker with live player states.
# Output: models/xgb_atp.joblib, data/processed/player_states.pkl

import pickle

import joblib

from src.data.fetch import fetch_matches
from src.data.process import clean_matches
from src.features.builder import build_features, symmetrize, MODEL_FEATURES
from src.models.train import train_model, evaluate
from src.models.predict import PlayerStateTracker

TRAIN_CUTOFF = 2021
VAL_START    = 2022

if __name__ == "__main__":
    print("Loading features...")
    features = build_features(
        clean_matches(fetch_matches(save_dir="data/raw")),
        use_cache=True,
        cache_path="data/processed/features.pkl",
    )

    print("Symmetrising dataset...")
    sym = symmetrize(features[features["is_complete"]])

    train = sym[sym["year"] <= TRAIN_CUTOFF]
    val   = sym[sym["year"] >= VAL_START]
    print(f"  Train: {len(train):,} rows ({2000}–{TRAIN_CUTOFF})")
    print(f"  Val  : {len(val):,} rows  ({VAL_START}–2024)")

    print("\nTraining XGBoost...")
    model = train_model(train, model_path="models/xgb_atp.joblib", complete_only=False)

    train_metrics = evaluate(model, train)
    val_metrics   = evaluate(model, val)
    print(f"  Train accuracy : {train_metrics['accuracy']:.3f}")
    print(f"  Val   accuracy : {val_metrics['accuracy']:.3f}")
    print(f"  Val   log-loss : {val_metrics['log_loss']:.3f}")
    print(f"  Val   Brier    : {val_metrics['brier_score']:.3f}")

    print("\nBuilding live player states (replaying all matches)...")
    raw_matches = fetch_matches(save_dir="data/raw")
    clean = clean_matches(raw_matches)
    tracker = PlayerStateTracker()
    tracker.process_all(clean)

    import os
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/player_states.pkl", "wb") as f:
        pickle.dump(tracker, f)

    print(f"\nDone — {len(tracker.name):,} player states saved to data/processed/player_states.pkl")
