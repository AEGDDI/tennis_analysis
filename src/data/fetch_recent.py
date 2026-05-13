# src/data/fetch_recent.py
"""
Supplement JeffSackmann's repo with recent 2026 clay matches from tennis-data.co.uk.

JeffSackmann is updated with a lag — for Roland Garros predictions we need
clay matches from the full build-up season (Monte-Carlo, Barcelona, Madrid, Rome).
tennis-data.co.uk publishes results within days of each match.

Name format difference:
  JeffSackmann : "Jannik Sinner"   (first last, with player_id)
  tennis-data  : "Sinner J."       (last initial., no ID)

The resolver maps tennis-data names to JeffSackmann player_ids so both datasets
share the same identifier space.
"""

import os
import re
from io import BytesIO

import difflib
import pandas as pd
import requests

TENNISDATA_BASE = "http://www.tennis-data.co.uk"

SURFACE_MAP = {
    "Hard": "Hard",
    "Clay": "Clay",
    "Grass": "Grass",
    "Carpet": "Carpet",
    "Hard (I)": "Hard",
    "Indoor Hard": "Hard",
}

LEVEL_MAP = {
    "Grand Slam": "G",
    "Masters": "M",
    "Masters 1000": "M",
    "Masters Cup": "F",
    "ATP500": "A",
    "ATP250": "A",
    "International": "A",
    "International Gold": "A",
    "ATP Finals": "F",
    "Olympics": "O",
}

ROUND_MAP = {
    "1st Round": "R64",
    "2nd Round": "R32",
    "3rd Round": "R16",
    "4th Round": "R16",
    "Quarterfinals": "QF",
    "Semifinals": "SF",
    "The Final": "F",
    "Round Robin": "RR",
}


# ── 1. Download ──────────────────────────────────────────────────────────────

def fetch_tennisdata_year(year: int, save_dir: str = "data/raw") -> pd.DataFrame:
    """
    Download the full-year ATP Excel file from tennis-data.co.uk and cache it.

    URL pattern: http://www.tennis-data.co.uk/{YEAR}/{YEAR}.xlsx
    Returns raw DataFrame with tennis-data column names intact.
    Raises RuntimeError if the file is unavailable.
    """
    os.makedirs(save_dir, exist_ok=True)
    local = os.path.join(save_dir, f"tennisdata_{year}.xlsx")

    if os.path.exists(local):
        print(f"  Loading cached tennis-data {year} from {local}")
        return pd.read_excel(local)

    url = f"{TENNISDATA_BASE}/{year}/{year}.xlsx"
    print(f"  Downloading tennis-data {year} from {url} ...")
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"tennis-data.co.uk returned HTTP {resp.status_code} for {year}. "
            f"The file may not exist yet for this year."
        )

    with open(local, "wb") as f:
        f.write(resp.content)
    print(f"  Saved to {local}")
    return pd.read_excel(BytesIO(resp.content))


# ── 2. Name resolution ───────────────────────────────────────────────────────

def _parse_tennisdata_name(name: str):
    """
    Parse tennis-data abbreviated names into (last_name, first_initial).

    Examples:
      "Sinner J."        → ("sinner", "j")
      "Cerundolo J.M."   → ("cerundolo", "j")
      "Ugo Carabelli C." → ("ugo carabelli", "c")
      "Berrettini M."    → ("berrettini", "m")
    """
    name = name.strip()
    # Split off the trailing initials block, e.g. "J." or "J.M."
    parts = name.rsplit(" ", 1)
    if len(parts) == 2 and re.match(r"^[A-Z](\.[A-Z])*\.$", parts[1]):
        last = parts[0].strip().lower()
        initial = parts[1][0].lower()
        return last, initial
    # Fallback: treat the whole name as last name
    return name.lower(), ""


def build_name_index(players_df: pd.DataFrame):
    """
    Build two lookup dicts from atp_players.csv:

      primary  : (last_name_lower, first_initial_lower) → player_id
      secondary: last_name_lower → player_id  (None if the last name is ambiguous)

    Used by resolve_player_id() to map tennis-data names to JeffSackmann IDs.
    """
    primary = {}
    secondary = {}

    for _, row in players_df.iterrows():
        pid = int(row["player_id"])
        first = str(row.get("name_first", "")).strip()
        last = str(row.get("name_last", "")).strip()
        if not last:
            continue

        last_l = last.lower()
        init_l = first[0].lower() if first else ""

        primary[(last_l, init_l)] = pid

        if last_l in secondary:
            secondary[last_l] = None   # ambiguous — more than one player with this last name
        else:
            secondary[last_l] = pid

    return primary, secondary


def resolve_player_id(
    name: str,
    primary: dict,
    secondary: dict,
    all_last_names: list,
) -> int:
    """
    Map a tennis-data player name string to a JeffSackmann player_id.

    Resolution order:
      1. (last_name, first_initial) exact match  → most reliable
      2. last_name-only match if unambiguous      → works for rare surnames
      3. fuzzy match on last name (cutoff 0.75)  → handles typos / accents
      4. returns -1 if all strategies fail

    Parameters
    ----------
    name         : raw string from tennis-data, e.g. "Sinner J."
    primary      : (last, initial) → pid  from build_name_index()
    secondary    : last → pid (or None if ambiguous)  from build_name_index()
    all_last_names : list of all last_name keys for fuzzy matching
    """
    last, init = _parse_tennisdata_name(name)

    # 1. exact (last, initial) match
    if (last, init) in primary:
        return primary[(last, init)]

    # 2. unambiguous last-name match
    if secondary.get(last) is not None:
        return secondary[last]

    # 3. fuzzy last-name match
    close = difflib.get_close_matches(last, all_last_names, n=1, cutoff=0.75)
    if close:
        fuzzy_last = close[0]
        candidate = primary.get((fuzzy_last, init)) or secondary.get(fuzzy_last)
        if candidate:
            return candidate

    return -1  # unresolvable


# ── 3. Column mapping ────────────────────────────────────────────────────────

def map_tennisdata_to_sackmann(
    raw_df: pd.DataFrame,
    players_df: pd.DataFrame,
    surface_filter: str = "Clay",
    complete_only: bool = True,
) -> pd.DataFrame:
    """
    Convert a raw tennis-data.co.uk DataFrame to JeffSackmann-compatible format.

    Steps:
      1. Filter to the requested surface and completed matches
      2. Parse abbreviated player names into (last, initial) tuples
      3. Resolve those tuples to JeffSackmann player_ids
      4. Rename and cast columns to match the JeffSackmann schema

    Parameters
    ----------
    raw_df        : output of fetch_tennisdata_year()
    players_df    : loaded atp_players.csv  (provides the ID lookup table)
    surface_filter: keep only this surface ("Clay", "Hard", etc.)
    complete_only : if True, drop retired / walkover matches

    Returns
    -------
    DataFrame with JeffSackmann-compatible column names and player IDs,
    ready to concatenate with the main historical dataset.
    """
    df = raw_df.copy()

    # Surface and completion filters
    df = df[df["Surface"].str.strip() == surface_filter].copy()
    if complete_only:
        df = df[df["Comment"] == "Completed"].copy()

    if df.empty:
        return pd.DataFrame()

    # Build name index
    primary, secondary = build_name_index(players_df)
    all_last_names = list({k[0] for k in primary})

    df["winner_id"] = df["Winner"].apply(
        lambda n: resolve_player_id(str(n), primary, secondary, all_last_names)
    )
    df["loser_id"] = df["Loser"].apply(
        lambda n: resolve_player_id(str(n), primary, secondary, all_last_names)
    )

    unresolved = (df["winner_id"] == -1) | (df["loser_id"] == -1)
    if unresolved.sum():
        print(f"  Warning: {unresolved.sum()} matches dropped (player name unresolvable)")
    df = df[~unresolved].copy()

    # Build output in JeffSackmann schema
    out = pd.DataFrame()
    out["tourney_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    out["tourney_name"] = df["Tournament"].str.strip()
    out["surface"] = df["Surface"].map(SURFACE_MAP).fillna("Unknown")
    out["tourney_level"] = df["Series"].map(LEVEL_MAP).fillna("A")
    out["round"] = df["Round"].map(ROUND_MAP).fillna("R64")
    out["winner_id"] = df["winner_id"].astype(int)
    out["loser_id"] = df["loser_id"].astype(int)
    out["winner_name"] = df["Winner"].str.strip()
    out["loser_name"] = df["Loser"].str.strip()
    out["winner_rank"] = pd.to_numeric(df["WRank"], errors="coerce").fillna(9999).astype(int)
    out["loser_rank"] = pd.to_numeric(df["LRank"], errors="coerce").fillna(9999).astype(int)
    out["winner_rank_points"] = pd.to_numeric(df.get("WPts", 0), errors="coerce").fillna(0.0)
    out["loser_rank_points"] = pd.to_numeric(df.get("LPts", 0), errors="coerce").fillna(0.0)
    out["winner_age"] = float("nan")
    out["loser_age"] = float("nan")
    out["score"] = ""
    out["is_complete"] = True
    out["year"] = out["tourney_date"].dt.year

    return out.dropna(subset=["tourney_date"]).reset_index(drop=True)


# ── 4. Main entry point ──────────────────────────────────────────────────────

def fetch_supplement_clay(
    players_df: pd.DataFrame,
    year: int = 2026,
    existing_matches: pd.DataFrame = None,
    save_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Fetch clay matches from tennis-data.co.uk for a given year and remove
    any matches already present in existing_matches (deduplication).

    This is the single function called by 1_fetch_data.py to top-up the
    JeffSackmann dataset with the most recent clay results.

    Parameters
    ----------
    players_df      : atp_players.csv DataFrame for name resolution
    year            : year to fetch from tennis-data (default 2026)
    existing_matches: JeffSackmann matches already loaded; used to deduplicate.
                      If None, no deduplication is applied.
    save_dir        : where to cache the downloaded Excel file

    Returns
    -------
    DataFrame of new clay matches in JeffSackmann format, ready to pd.concat().
    """
    print(f"\nFetching {year} clay supplement from tennis-data.co.uk...")
    raw = fetch_tennisdata_year(year, save_dir=save_dir)
    new = map_tennisdata_to_sackmann(raw, players_df, surface_filter="Clay")

    if new.empty:
        print("  No new clay matches found.")
        return new

    # Deduplicate against existing dataset
    if existing_matches is not None and not existing_matches.empty:
        existing_clay = existing_matches[existing_matches["surface"] == "Clay"]
        existing_pairs = set(
            zip(existing_clay["winner_id"], existing_clay["loser_id"])
        )
        before = len(new)
        new = new[
            ~new.apply(lambda r: (r["winner_id"], r["loser_id"]) in existing_pairs, axis=1)
        ].copy()
        print(f"  Deduplicated: {before} → {len(new)} new clay matches "
              f"(removed {before - len(new)} already in JeffSackmann)")
    else:
        print(f"  {len(new)} clay matches (no deduplication)")

    return new.reset_index(drop=True)
