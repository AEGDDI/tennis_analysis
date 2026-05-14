# run_pipeline.py
# Runs all four pipeline steps in order:
#   Step 1 — 1_fetch_data.py        download ATP match data
#   Step 2 — 2_build_features.py    engineer player statistics
#   Step 3 — 3_train_model.py       train XGBoost + build player states
#   Step 4 — 4_simulate_rg2026.py   run Monte Carlo tournament simulation
#
# Usage:
#   python run_pipeline.py            — run all steps
#   python run_pipeline.py --from 3   — restart from step 3
#   python run_pipeline.py --only 4   — run step 4 only

import argparse
import os
import subprocess
import sys
import time

STEPS = [
    (1, "1_fetch_data.py",       "Download ATP match data"),
    (2, "2_build_features.py",   "Engineer player statistics (Elo, H2H, form, surface)"),
    (3, "3_train_model.py",      "Train XGBoost model + build live player states"),
    (4, "4_simulate_rg2026.py",  "Run 20,000-simulation Monte Carlo tournament"),
]

STALE_CACHE = "data/processed/features.pkl"


def banner(text: str) -> None:
    width = 60
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def run_step(script: str, label: str, step_num: int) -> bool:
    banner(f"STEP {step_num}  —  {label}")
    print(f"  Running: python {script}\n")
    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}). Pipeline stopped.")
        return False
    print(f"\n  Done in {elapsed:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the Roland Garros prediction pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="from_step", type=int, metavar="N",
                       help="Start from step N (e.g. --from 3)")
    group.add_argument("--only", dest="only_step", type=int, metavar="N",
                       help="Run only step N (e.g. --only 4)")
    args = parser.parse_args()

    if args.only_step:
        steps_to_run = [s for s in STEPS if s[0] == args.only_step]
        if not steps_to_run:
            print(f"Unknown step {args.only_step}. Valid steps: 1–4.")
            sys.exit(1)
    elif args.from_step:
        steps_to_run = [s for s in STEPS if s[0] >= args.from_step]
        if not steps_to_run:
            print(f"Step {args.from_step} out of range. Valid steps: 1–4.")
            sys.exit(1)
        # If re-running from step 2 or earlier, clear the features cache so it regenerates
        if args.from_step <= 2 and os.path.exists(STALE_CACHE):
            print(f"  Clearing stale features cache: {STALE_CACHE}")
            os.remove(STALE_CACHE)
    else:
        steps_to_run = STEPS
        # Full re-run always clears the features cache
        if os.path.exists(STALE_CACHE):
            print(f"  Clearing stale features cache: {STALE_CACHE}")
            os.remove(STALE_CACHE)

    pipeline_start = time.time()
    print("\nRoland Garros 2026 — Prediction Pipeline")
    print(f"Running {len(steps_to_run)} step(s): {[s[0] for s in steps_to_run]}")

    for num, script, label in steps_to_run:
        ok = run_step(script, label, num)
        if not ok:
            sys.exit(1)

    total = time.time() - pipeline_start
    banner(f"ALL STEPS COMPLETE  —  total time: {total/60:.1f} min")
    print("  Outputs:")
    print("    data/raw/matches_combined.pkl")
    print("    data/processed/features.pkl")
    print("    models/xgb_atp.joblib")
    print("    data/processed/player_states.pkl")
    print("    data/rg2026_results.pkl\n")


if __name__ == "__main__":
    main()
