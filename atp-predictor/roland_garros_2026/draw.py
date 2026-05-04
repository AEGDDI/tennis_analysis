"""
Roland Garros draw generator — replicates Grand Slam seeding rules.

Grand Slam draw structure (128 players, 7 rounds):
  • Seed 1  → fixed at position   1  (top of top half)
  • Seed 2  → fixed at position 128  (bottom of bottom half)
  • Seeds 3-4   → randomly placed in positions 64 & 65
                  (bottom of top half / top of bottom half)
  • Seeds 5-8   → one per quarter: randomly in positions 32, 33, 96, 97
  • Seeds 9-16  → one per eighth section: fixed candidate slots
  • Seeds 17-32 → one per 16th section: fixed candidate slots
  • Unseeded    → randomly fill remaining 96 slots

The bracket is read left-to-right:
  draw[0] vs draw[1]  →  draw[2] vs draw[3]  → … → Final

Round labels:
  R1 (128→64), R2 (64→32), R3 (32→16), R4 (16→8=QF), SF, F
"""

import random
from typing import List, Optional

DRAW_SIZE = 128

# One fixed slot per seed group section.  These are the "protected" positions
# at which exactly one seeded player lands, ensuring they can only meet in
# the prescribed earliest round.
#
# Format: each inner list contains the candidate slots (0-indexed) for that
# group.  One slot per section is chosen randomly.
_SEED_1_POS = 0
_SEED_2_POS = 127

_SLOTS_3_4 = [63, 64]            # bottom of top half / top of bottom half

_SLOTS_5_8 = [31, 32, 95, 96]   # top/bottom of each quarter

_SLOTS_9_16 = [                  # top/bottom of each eighth
    15, 16, 47, 48,
    79, 80, 111, 112,
]

_SLOTS_17_32 = [                 # top/bottom of each 16th section
    7,   8,  23, 24,
    39, 40,  55, 56,
    71, 72,  87, 88,
    103, 104, 119, 120,
]


def create_rg_draw(
    player_ids: List[int],
    rng_seed: Optional[int] = 42,
) -> List[int]:
    """
    Place 128 players into a Roland Garros bracket.

    Parameters
    ----------
    player_ids : list of exactly 128 player IDs, already ordered by seeding:
                   index 0  → seed 1
                   index 1  → seed 2
                   …
                   index 31 → seed 32
                   index 32-127 → unseeded players (ranked order)
    rng_seed   : integer seed for the random draw lottery; None for truly random

    Returns
    -------
    draw : list of 128 player IDs in bracket order
           match pairs: (draw[0],draw[1]), (draw[2],draw[3]), …
    """
    if len(player_ids) != DRAW_SIZE:
        raise ValueError(f"Need exactly {DRAW_SIZE} players, got {len(player_ids)}")

    rng = random.Random(rng_seed)
    draw: List[Optional[int]] = [None] * DRAW_SIZE

    seeds = player_ids[:32]
    unseeded = list(player_ids[32:])

    # Fixed seeds
    draw[_SEED_1_POS] = seeds[0]
    draw[_SEED_2_POS] = seeds[1]

    # Seeds 3-4: random lottery into the two half-boundary slots
    s34_slots = rng.sample(_SLOTS_3_4, 2)
    draw[s34_slots[0]] = seeds[2]
    draw[s34_slots[1]] = seeds[3]

    # Seeds 5-8: one per quarter
    s58_slots = rng.sample(_SLOTS_5_8, 4)
    for i, slot in enumerate(s58_slots):
        draw[slot] = seeds[4 + i]

    # Seeds 9-16: one per eighth
    s916_slots = rng.sample(_SLOTS_9_16, 8)
    for i, slot in enumerate(s916_slots):
        draw[slot] = seeds[8 + i]

    # Seeds 17-32: one per 16th section
    s1732_slots = rng.sample(_SLOTS_17_32, 16)
    for i, slot in enumerate(s1732_slots):
        draw[slot] = seeds[16 + i]

    # Unseeded players fill remaining slots randomly
    rng.shuffle(unseeded)
    empty_slots = [i for i, x in enumerate(draw) if x is None]
    for slot, pid in zip(empty_slots, unseeded):
        draw[slot] = pid

    return draw  # type: ignore[return-value]


def draw_to_dataframe(draw: List[int], ratings_df) -> "pd.DataFrame":
    """
    Annotate a draw list with player names, seeds, and clay strength.

    Parameters
    ----------
    draw        : output of create_rg_draw()
    ratings_df  : DataFrame from clay_rating.compute_clay_ratings()
                  must contain columns: player_id, name, clay_strength, clay_elo, rank

    Returns
    -------
    DataFrame with columns: position, player_id, name, seed, clay_elo, clay_strength
    """
    import pandas as pd

    id_to_info = ratings_df.set_index("player_id")[
        ["name", "clay_elo", "clay_strength", "rank"]
    ].to_dict("index")

    rows = []
    for pos, pid in enumerate(draw):
        info = id_to_info.get(pid, {"name": str(pid), "clay_elo": 1500, "clay_strength": 0, "rank": 999})
        # Seed is inferred from position in the original sorted list
        seed_num = None
        if pos in [_SEED_1_POS]:
            seed_num = 1
        elif pos in [_SEED_2_POS]:
            seed_num = 2
        elif pos in _SLOTS_3_4:
            seed_num = "3/4"
        elif pos in _SLOTS_5_8:
            seed_num = "5-8"
        elif pos in _SLOTS_9_16:
            seed_num = "9-16"
        elif pos in _SLOTS_17_32:
            seed_num = "17-32"

        rows.append(
            {
                "position": pos + 1,
                "player_id": pid,
                "name": info["name"],
                "seed": seed_num,
                "clay_elo": info["clay_elo"],
                "clay_strength": round(info["clay_strength"], 3),
                "rank": info["rank"],
            }
        )

    import pandas as pd
    return pd.DataFrame(rows)
