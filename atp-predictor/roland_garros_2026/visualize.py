"""
Visualisations for the Roland Garros 2026 analysis.

Three chart types:
  1. plot_round_heatmap     — players × rounds probability matrix
  2. plot_champion_odds     — horizontal bar chart of title probabilities
  3. plot_clay_profile      — clay Elo vs clay win rate scatter (bubble = title prob)
  4. plot_draw_sensitivity  — box-plot of title probs across random draw realisations
  5. plot_matchup_matrix    — H2H probability heatmap for top contenders

All functions return a (fig, ax) tuple so callers can further customise or save.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Roland Garros colour palette
_CLAY   = "#C8421C"   # Suzanne Lenglen clay red
_CLAY_L = "#F0A882"   # light variant
_DARK   = "#1C1C1C"
_WHITE  = "#FAFAFA"

ROUNDS       = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "W"]
ROUND_LABELS = ["Draw\nEntry", "R2", "R3", "R4", "QF", "SF", "Final", "Champion"]


# ---------------------------------------------------------------------------
# 1. Round probability heatmap
# ---------------------------------------------------------------------------

def plot_round_heatmap(
    results: pd.DataFrame,
    top_n: int = 32,
    title: str = "Roland Garros 2026 — Round Probability Matrix",
    figsize: Tuple[int, int] = (15, 11),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Heatmap where each cell shows P(player reaches that round).

    Inspired by the Premier League season-end position probability matrix.
    Top *top_n* players (by champion probability) are shown.
    """
    top = results.head(top_n).copy()

    # Show R64 → W (skip R128 which is always 100 %)
    cols   = ROUNDS[1:]
    labels = ROUND_LABELS[1:]

    matrix = top[cols].values.astype(float)
    names  = top["name"].tolist()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_WHITE)
    ax.set_facecolor(_WHITE)

    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    im   = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Axes
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold", color=_DARK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9, color=_DARK)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)

    # Cell annotations
    for i in range(len(names)):
        for j in range(len(cols)):
            val = matrix[i, j]
            if val < 0.005:
                continue
            text_color = _WHITE if val > 0.45 else _DARK
            ax.text(
                j, i,
                f"{val:.0%}",
                ha="center", va="center",
                fontsize=7.5, color=text_color, fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("Probability", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20, color=_DARK)

    # Thin grid
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 2. Champion odds bar chart
# ---------------------------------------------------------------------------

def plot_champion_odds(
    results: pd.DataFrame,
    top_n: int = 16,
    title: str = "Roland Garros 2026 — Championship Probability",
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart of title probabilities for the top *top_n* players."""
    top = results.head(top_n).copy()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_WHITE)
    ax.set_facecolor(_WHITE)

    y_pos = range(len(top))
    bars  = ax.barh(y_pos, top["W"] * 100, color=_CLAY, edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["name"], fontsize=10, color=_DARK)
    ax.invert_yaxis()
    ax.set_xlabel("Championship Probability (%)", fontsize=10, color=_DARK)
    ax.set_title(title, fontsize=13, fontweight="bold", color=_DARK, pad=12)
    ax.tick_params(colors=_DARK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", zorder=0)

    # Baseline: equal probability for all 128 players
    baseline = 100 / 128
    ax.axvline(baseline, color="grey", linestyle="--", linewidth=0.9, alpha=0.7, label=f"Random baseline ({baseline:.1f}%)")
    ax.legend(fontsize=9)

    # Bar labels
    for bar, row in zip(bars, top.itertuples()):
        ax.text(
            bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
            f"{row.W:.1%}",
            va="center", fontsize=8.5, color=_DARK,
        )

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 3. Clay court profile scatter
# ---------------------------------------------------------------------------

def plot_clay_profile(
    results: pd.DataFrame,
    ratings_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Clay Court Profile — Roland Garros 2026 Contenders",
    figsize: Tuple[int, int] = (11, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot: Clay Elo (x) vs. Clay Win Rate (y).
    Bubble size ∝ championship probability.

    The two axes capture the two main dimensions of clay court ability:
      • x — career clay Elo: long-run track record on clay
      • y — rolling clay win rate: form over the last 12 months
    """
    top_ids = results.head(top_n)["player_id"].tolist()
    data = ratings_df[ratings_df["player_id"].isin(top_ids)].merge(
        results[["player_id", "W"]], on="player_id"
    )

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_WHITE)
    ax.set_facecolor(_WHITE)

    scatter = ax.scatter(
        data["clay_elo"],
        data["clay_winrate"],
        s=data["W"] * 8000 + 80,
        c=data["W"],
        cmap="YlOrRd",
        alpha=0.80,
        edgecolors=_DARK,
        linewidths=0.6,
        zorder=3,
    )

    for _, row in data.iterrows():
        ax.annotate(
            row["name"].split()[-1],
            (row["clay_elo"], row["clay_winrate"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8.5,
            color=_DARK,
        )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Championship probability", fontsize=9)

    ax.set_xlabel("Clay Court Elo Rating", fontsize=11, color=_DARK)
    ax.set_ylabel("Clay Win Rate — last 12 months", fontsize=11, color=_DARK)
    ax.set_title(title, fontsize=13, fontweight="bold", color=_DARK, pad=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#EEEEEE", zorder=0)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Draw sensitivity box-plot
# ---------------------------------------------------------------------------

def plot_draw_sensitivity(
    sensitivity_df: pd.DataFrame,
    top_n: int = 10,
    title: str = "Draw Luck — How Much Does the Bracket Lottery Matter?",
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Horizontal bar chart with error bars showing mean ± 1 std of championship
    probability across many random draw realisations.

    A wide error bar = high draw sensitivity = player's fate strongly depends
    on the bracket lottery.
    """
    top = sensitivity_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_WHITE)
    ax.set_facecolor(_WHITE)

    y = range(len(top))
    ax.barh(y, top["win_prob_mean"] * 100, xerr=top["win_prob_std"] * 100,
            color=_CLAY, alpha=0.85, edgecolor="white",
            error_kw={"ecolor": _DARK, "capsize": 4, "linewidth": 1.2}, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(top["name"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Championship Probability (%)\nmean ± 1 std across random draws", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", zorder=0)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 5. Matchup probability matrix
# ---------------------------------------------------------------------------

def plot_matchup_matrix(
    matchup_df: pd.DataFrame,
    title: str = "Head-to-Head Win Probabilities (Clay)",
    figsize: Tuple[int, int] = (9, 7),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Annotated heatmap of pairwise match probabilities for top contenders.
    Cell [i, j] = P(row player beats column player) on clay.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_WHITE)

    mask = np.eye(len(matchup_df), dtype=bool)  # hide diagonal
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        matchup_df,
        ax=ax,
        cmap=cmap,
        center=0.5,
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        mask=mask,
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.6, "label": "P(row beats column)"},
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=_DARK)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.tight_layout()
    return fig, ax
