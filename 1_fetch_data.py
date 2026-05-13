# 1_fetch_data.py
# Downloads ATP match data from two sources:
#   - JeffSackmann/tennis_atp (2000-2026): the primary historical archive
#   - tennis-data.co.uk (2026):            supplements recent clay matches
#                                          that JeffSackmann has not yet published
#
# Output:
#   data/raw/atp_matches_YYYY.csv      (one per year, from JeffSackmann)
#   data/raw/tennisdata_2026.xlsx      (cached tennis-data file)
#   data/raw/atp_players.csv           (player master table)
#   data/raw/matches_combined.pkl      (merged dataset ready for step 2)

import os
import pickle

import pandas as pd

from src.data.fetch import fetch_matches, fetch_players
from src.data.fetch_recent import fetch_supplement_clay

SACKMANN_START = 2000
SACKMANN_END   = 2026   # JeffSackmann is kept up to date; fetch the latest year available

if __name__ == "__main__":
    # ── 1. Primary source: JeffSackmann ──────────────────────────────────────
    print("=" * 60)
    print("SOURCE 1 — JeffSackmann/tennis_atp (GitHub)")
    print("=" * 60)
    matches = fetch_matches(
        start_year=SACKMANN_START,
        end_year=SACKMANN_END,
        save_dir="data/raw",
    )
    players = fetch_players(save_dir="data/raw")

    clay_2026_sackmann = matches[
        (matches["year"] == 2026) & (matches.get("surface", "") == "Clay")
    ] if "surface" in matches.columns else pd.DataFrame()
    print(f"  JeffSackmann 2026 clay matches: {len(clay_2026_sackmann)}")

    # ── 2. Supplement: tennis-data.co.uk 2026 clay ───────────────────────────
    print("\n" + "=" * 60)
    print("SOURCE 2 — tennis-data.co.uk (2026 clay supplement)")
    print("=" * 60)
    supplement = fetch_supplement_clay(
        players_df=players,
        year=2026,
        existing_matches=matches,
        save_dir="data/raw",
    )

    # ── 3. Merge ──────────────────────────────────────────────────────────────
    if not supplement.empty:
        combined = pd.concat([matches, supplement], ignore_index=True)
        print(f"\nMerged: {len(matches):,} + {len(supplement):,} = {len(combined):,} total matches")
    else:
        combined = matches
        print("\nNo supplemental matches added — JeffSackmann is already up to date.")

    # ── 4. Save combined dataset ──────────────────────────────────────────────
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/matches_combined.pkl", "wb") as f:
        pickle.dump(combined, f)

    print(f"\nDone.")
    print(f"  Total matches : {len(combined):,}")
    print(f"  Clay matches  : {len(combined[combined.get('surface','') == 'Clay']) if 'surface' in combined.columns else 'N/A'}")
    print(f"  Players       : {len(players):,}")
    print(f"  Saved to      : data/raw/matches_combined.pkl")
