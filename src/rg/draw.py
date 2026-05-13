"""
Roland Garros draw generator — replicates Grand Slam seeding rules.

  - Seed 1  -> position   1  (top of top half)
  - Seed 2  -> position 128  (bottom of bottom half)
  - Seeds 3-4   -> randomly placed at half boundaries
  - Seeds 5-8   -> one per quarter
  - Seeds 9-16  -> one per eighth section
  - Seeds 17-32 -> one per 16th section
  - Unseeded    -> fill remaining 96 slots randomly
"""

import random
from typing import List, Optional

import pandas as pd

DRAW_SIZE = 128

_SEED_1_POS = 0
_SEED_2_POS = 127

_SLOTS_3_4   = [63, 64]
_SLOTS_5_8   = [31, 32, 95, 96]
_SLOTS_9_16  = [15, 16, 47, 48, 79, 80, 111, 112]
_SLOTS_17_32 = [7, 8, 23, 24, 39, 40, 55, 56, 71, 72, 87, 88, 103, 104, 119, 120]


def create_rg_draw(
    player_ids: List[int],
    rng_seed: Optional[int] = 42,
) -> List[int]:
    """
    Place 128 players into a Roland Garros bracket.

    player_ids must be ordered by seeding (index 0 = seed 1, ..., index 31 = seed 32,
    index 32-127 = unseeded in ranked order).
    """
    if len(player_ids) != DRAW_SIZE:
        raise ValueError(f"Need exactly {DRAW_SIZE} players, got {len(player_ids)}")

    rng = random.Random(rng_seed)
    draw: List[Optional[int]] = [None] * DRAW_SIZE

    seeds = player_ids[:32]
    unseeded = list(player_ids[32:])

    draw[_SEED_1_POS] = seeds[0]
    draw[_SEED_2_POS] = seeds[1]

    s34_slots = rng.sample(_SLOTS_3_4, 2)
    draw[s34_slots[0]] = seeds[2]
    draw[s34_slots[1]] = seeds[3]

    s58_slots = rng.sample(_SLOTS_5_8, 4)
    for i, slot in enumerate(s58_slots):
        draw[slot] = seeds[4 + i]

    s916_slots = rng.sample(_SLOTS_9_16, 8)
    for i, slot in enumerate(s916_slots):
        draw[slot] = seeds[8 + i]

    s1732_slots = rng.sample(_SLOTS_17_32, 16)
    for i, slot in enumerate(s1732_slots):
        draw[slot] = seeds[16 + i]

    rng.shuffle(unseeded)
    empty_slots = [i for i, x in enumerate(draw) if x is None]
    for slot, pid in zip(empty_slots, unseeded):
        draw[slot] = pid

    return draw  # type: ignore[return-value]


def draw_to_dataframe(draw: List[int], ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Annotate a draw list with player names, seeds, and clay strength."""
    id_to_info = ratings_df.set_index("player_id")[
        ["name", "clay_elo", "clay_strength", "rank"]
    ].to_dict("index")

    rows = []
    for pos, pid in enumerate(draw):
        info = id_to_info.get(pid, {"name": str(pid), "clay_elo": 1500, "clay_strength": 0, "rank": 999})
        if pos == _SEED_1_POS:
            seed_num = 1
        elif pos == _SEED_2_POS:
            seed_num = 2
        elif pos in _SLOTS_3_4:
            seed_num = "3/4"
        elif pos in _SLOTS_5_8:
            seed_num = "5-8"
        elif pos in _SLOTS_9_16:
            seed_num = "9-16"
        elif pos in _SLOTS_17_32:
            seed_num = "17-32"
        else:
            seed_num = None

        rows.append({
            "position": pos + 1,
            "player_id": pid,
            "name": info["name"],
            "seed": seed_num,
            "clay_elo": info["clay_elo"],
            "clay_strength": round(info["clay_strength"], 3),
            "rank": info["rank"],
        })

    return pd.DataFrame(rows)
