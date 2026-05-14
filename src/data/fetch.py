"""Download ATP match and player data from JeffSackmann's GitHub repository."""

import os
from io import StringIO

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"


def fetch_matches(
    start_year: int = 2000,
    end_year: int = 2025,
    save_dir: str = "data/raw",
) -> pd.DataFrame:
    """Download ATP match CSVs, caching each year locally."""
    os.makedirs(save_dir, exist_ok=True)
    dfs = []

    for year in tqdm(range(start_year, end_year + 1), desc="Downloading matches"):
        local = os.path.join(save_dir, f"atp_matches_{year}.csv")
        if os.path.exists(local):
            df = pd.read_csv(local, low_memory=False)
        else:
            url = f"{BASE_URL}/atp_matches_{year}.csv"
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                print(f"  Warning: no data for {year} (HTTP {resp.status_code})")
                continue
            df = pd.read_csv(StringIO(resp.text), low_memory=False)
            df.to_csv(local, index=False)

        df["year"] = year
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No match data fetched — check internet connection.")

    matches = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(matches):,} matches ({start_year}–{end_year})")
    return matches


def fetch_players(save_dir: str = "data/raw") -> pd.DataFrame:
    """Download ATP player master table."""
    os.makedirs(save_dir, exist_ok=True)
    local = os.path.join(save_dir, "atp_players.csv")

    if os.path.exists(local):
        return pd.read_csv(local, low_memory=False)

    url = f"{BASE_URL}/atp_players.csv"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), low_memory=False)
    df.to_csv(local, index=False)
    print(f"Loaded {len(df):,} players")
    return df
