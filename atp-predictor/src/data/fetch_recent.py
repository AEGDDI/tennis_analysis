"""
Fetch recent ATP match data from BSD Tennis API (bzzoiro.com).

Supplements JeffSackmann's repo (which lags ~6 months) with current season data.
Provides 2025 and 2026 matches with serve/return stats.

To use:
  1. Register for free API key at https://tennis.bzzoiro.com/register/
  2. Store in environment variable BZZOIRO_API_KEY or pass directly
"""

import os
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

BZZOIRO_BASE = "https://tennis.bzzoiro.com/api"
BATCH_SIZE = 100  # pagination limit per request


def fetch_bzzoiro_api_key() -> str:
    """Get API key from environment or prompt user."""
    key = os.getenv("BZZOIRO_API_KEY")
    if not key:
        raise ValueError(
            "BZZOIRO_API_KEY not set. Register at https://tennis.bzzoiro.com/register/ "
            "then set: export BZZOIRO_API_KEY=your_key"
        )
    return key


def fetch_bzzoiro_matches(
    years: list = None,
    api_key: Optional[str] = None,
    surface: str = "Clay",  # Focus on clay for Roland Garros
) -> pd.DataFrame:
    """
    Fetch ATP matches from BSD Tennis API for specified years.

    Parameters
    ----------
    years      : list of years to fetch (default [2025, 2026])
    api_key    : bzzoiro API key (uses BZZOIRO_API_KEY env var if not provided)
    surface    : filter by surface ('Clay', 'Hard', 'Grass', or None for all)

    Returns
    -------
    DataFrame with columns matching ATP match format:
      winner_id, winner_name, loser_id, loser_name, surface, tourney_date, etc.
    """
    if years is None:
        years = [2025, 2026]

    if api_key is None:
        api_key = fetch_bzzoiro_api_key()

    headers = {"Authorization": f"Token {api_key}"}
    all_matches = []

    for year in years:
        print(f"Fetching {year} matches from BSD Tennis...")
        page = 1
        year_matches = 0

        while True:
            params = {
                "year": year,
                "page": page,
                "page_size": BATCH_SIZE,
                "ordering": "-date",  # most recent first
            }
            if surface:
                params["surface"] = surface

            try:
                resp = requests.get(
                    f"{BZZOIRO_BASE}/matches/",
                    headers=headers,
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"  Warning: API request failed ({e})")
                break

            data = resp.json()

            # Handle pagination (BSD Tennis returns {'count': N, 'results': [...]})
            if isinstance(data, dict):
                results = data.get("results", [])
                total = data.get("count", 0)
            else:
                results = data if isinstance(data, list) else []
                total = len(results)

            if not results:
                break

            for match in results:
                try:
                    # Extract nested player/tournament info
                    winner = match.get("winner") or {}
                    loser = match.get("loser") or {}
                    tournament = match.get("tournament") or {}
                    winner_stats = match.get("winner_stats") or {}
                    loser_stats = match.get("loser_stats") or {}

                    # Ensure we have winner and loser
                    if not (winner.get("name") and loser.get("name")):
                        continue

                    all_matches.append(
                        {
                            "tourney_date": match.get("date"),
                            "winner_id": winner.get("id") or 0,
                            "winner_name": winner.get("name"),
                            "loser_id": loser.get("id") or 0,
                            "loser_name": loser.get("name"),
                            "surface": match.get("surface"),
                            "tourney_level": tournament.get("level"),
                            "tourney_name": tournament.get("name"),
                            "score": match.get("score"),
                            "winner_aces": winner_stats.get("aces"),
                            "winner_df": winner_stats.get("double_faults"),
                            "winner_1st_svpt": winner_stats.get("first_serve_pct"),
                            "loser_aces": loser_stats.get("aces"),
                            "loser_df": loser_stats.get("double_faults"),
                            "loser_1st_svpt": loser_stats.get("first_serve_pct"),
                        }
                    )
                except (KeyError, TypeError, AttributeError):
                    continue

            year_matches += len(results)

            # Check if there are more pages
            if len(results) < BATCH_SIZE or page * BATCH_SIZE >= total:
                break

            page += 1

        print(f"  Fetched {year_matches} matches from {year}")

    if not all_matches:
        print("Warning: No matches returned from API. Check API key and network.")
        return pd.DataFrame()

    df = pd.DataFrame(all_matches)

    # Parse dates carefully
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")

    # Drop rows with invalid dates
    n_before = len(df)
    df = df[df["tourney_date"].notna()].copy()
    n_after = len(df)

    if n_before > n_after:
        print(f"  Warning: Dropped {n_before - n_after} matches with invalid dates")

    return df


def merge_with_jeffsackmann(
    jeffsackmann_df: pd.DataFrame,
    bzzoiro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine JeffSackmann (2000-2024) and bzzoiro (2025-2026) datasets.

    Parameters
    ----------
    jeffsackmann_df : DataFrame from fetch_matches() with data through 2024
    bzzoiro_df      : DataFrame from fetch_bzzoiro_matches() with 2025-2026 data

    Returns
    -------
    Combined DataFrame with both sources, deduplicated by (date, winner, loser)
    """
    # Standardise date columns to datetime
    jsa = jeffsackmann_df.copy()
    bsd = bzzoiro_df.copy()

    # Ensure both have the key columns
    for df in [jsa, bsd]:
        if "tourney_date" in df.columns:
            df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")

    # Keep only rows with valid dates
    jsa = jsa[jsa["tourney_date"].notna()].copy()
    bsd = bsd[bsd["tourney_date"].notna()].copy()

    if len(bsd) == 0:
        print("Warning: No valid dates in BSD Tennis data, using JeffSackmann only")
        return jsa

    # Concatenate
    combined = pd.concat([jsa, bsd], ignore_index=True, sort=False)

    # Deduplicate by match signature
    if "winner_name" in combined.columns and "loser_name" in combined.columns:
        combined = combined.drop_duplicates(
            subset=["tourney_date", "winner_name", "loser_name"],
            keep="first",  # prefer JeffSackmann (more reliable)
        )

    return combined.sort_values("tourney_date", na_position="last").reset_index(drop=True)


if __name__ == "__main__":
    # Quick test: fetch 2025 clay matches
    try:
        recent = fetch_bzzoiro_matches(years=[2025], surface="Clay")
        print(f"\n{len(recent)} 2025 clay matches loaded")
        print(recent[["tourney_date", "winner_name", "loser_name", "surface"]].head())
    except Exception as e:
        print(f"Error: {e}")
