"""Roland Garros 2026 — Championship Predictor  (Streamlit app)

Run with:
    streamlit run app.py
"""

import pickle

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Roland Garros 2026 · Predictor",
    page_icon="🎾",
    layout="wide",
)

# ── RG brand colours ──────────────────────────────────────────────────────────
_GREEN  = "#0a3d1f"
_GREEN2 = "#1a6b3c"
_RED    = "#c8102e"
_LIGHT  = "#f0f7f2"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* App background */
  .stApp {{ background: #fafaf8; }}

  /* Top header bar */
  [data-testid="stHeader"] {{ background: {_GREEN}; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background: {_GREEN};
  }}
  [data-testid="stSidebar"] * {{ color: rgba(255,255,255,0.9) !important; }}
  [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.2); }}

  /* Metric cards */
  [data-testid="metric-container"] {{
    background: {_LIGHT};
    border-left: 4px solid {_GREEN};
    border-radius: 8px;
    padding: 16px 20px;
  }}

  /* Tab bar */
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 2px solid #ddd; }}
  .stTabs [data-baseweb="tab"] {{
    background: #f5f5f2;
    border-radius: 6px 6px 0 0;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 0.9rem;
  }}
  .stTabs [aria-selected="true"] {{
    background: {_GREEN} !important;
    color: white !important;
  }}

  /* Section headers */
  h1 {{ color: {_GREEN}; }}
  h2, h3 {{ color: {_GREEN}; border-left: 4px solid {_RED}; padding-left: 12px; }}

  /* DataFrames */
  [data-testid="stDataFrame"] {{ border: 1px solid #e0e0d8; border-radius: 8px; }}

  /* Divider */
  hr {{ border-color: #e0e0d8; }}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "W"]
ROUND_LABELS = {
    "R128": "Entered",
    "R64":  "Round 1",
    "R32":  "Round 2",
    "R16":  "Round 3",
    "QF":   "Quarter-Final",
    "SF":   "Semi-Final",
    "F":    "Final",
    "W":    "Champion",
}
DISPLAY_ROUNDS = ROUNDS[1:]   # skip R128 (always 100%)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading simulation results…")
def load_data():
    with open("data/rg2026_results.pkl", "rb") as f:
        d = pickle.load(f)
    results: pd.DataFrame = d["results"].copy()
    ratings: pd.DataFrame = d["ratings"].copy()
    draw: list = d["draw"]

    results = results.reset_index(drop=True)
    results.insert(0, "clay_seed", results.index + 1)
    return results, ratings, draw

try:
    results, ratings, draw = load_data()
except FileNotFoundError:
    st.error(
        "**`data/rg2026_results.pkl` not found.**  "
        "Run the pipeline first:\n\n```\npython run_pipeline.py\n```"
    )
    st.stop()

# ── Helper indices ────────────────────────────────────────────────────────────
id_to_row   = results.set_index("player_id")          # player_id → result row
draw_pos_of = {pid: i for i, pid in enumerate(draw)}  # player_id → 0-based draw position


def potential_opponents(p_idx: int, round_num: int) -> list[int]:
    """Player IDs that could be faced in round_num (1 = R1 … 7 = Final)."""
    return [
        draw[i] for i in range(128)
        if (i >> round_num)       == (p_idx >> round_num)
        and (i >> (round_num - 1)) != (p_idx >> (round_num - 1))
    ]


def pct(v) -> str:
    return f"{float(v):.1%}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎾 RG 2026")
    st.markdown("**Roland Garros Predictor**")
    st.markdown("---")
    st.markdown(f"**Players in draw:** {len(draw)}")
    st.markdown(f"**Simulations:** 20,000")
    st.markdown("---")
    st.markdown("**Data sources**")
    st.markdown("JeffSackmann / tennis_atp")
    st.markdown("tennis-data.co.uk")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("XGBoost · Val acc 66%")
    st.markdown("Train: 2000–2022")
    st.markdown("Val: 2023–2025")

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="background:linear-gradient(135deg,{_GREEN} 0%,{_GREEN2} 60%,{_RED} 100%);
                color:#fff;padding:36px 32px 28px;border-radius:12px;margin-bottom:24px;">
      <div style="font-size:0.75rem;letter-spacing:0.12em;text-transform:uppercase;
                  opacity:0.7;margin-bottom:10px;">Data Science · Tennis Analytics</div>
      <h1 style="color:#fff;border:none;padding:0;margin:0 0 10px;font-size:2rem;">
        Roland Garros 2026 — Championship Predictor
      </h1>
      <p style="opacity:0.85;margin:0;font-size:1rem;">
        20,000 Monte Carlo simulations · XGBoost trained on 25 years of ATP data
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_odds, tab_draw, tab_compare, tab_ratings = st.tabs([
    "🏆  Championship Odds",
    "🎾  Draw Explorer",
    "⚔️  Compare Players",
    "📊  Clay Ratings",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Championship Odds
# ─────────────────────────────────────────────────────────────────────────────
with tab_odds:
    st.markdown("## Championship Odds")
    st.caption(
        "Probability of winning the title, reaching each round, based on 20,000 "
        "simulated brackets. Alcaraz excluded (injury withdrawal)."
    )

    # Top-3 metric cards
    top3 = results.head(3)
    cols = st.columns(3)
    for col, (_, row) in zip(cols, top3.iterrows()):
        col.metric(
            label=f"#{int(row['clay_seed'])}  {row['name']}",
            value=pct(row["W"]),
            delta=f"ATP Rank {int(row['rank'])}",
        )

    st.markdown("### Top 15 — Title Probability")

    top15 = results.head(15).copy()
    top15["win_pct"] = (top15["W"] * 100).round(1)

    fig_bar = px.bar(
        top15,
        x="win_pct",
        y="name",
        orientation="h",
        color="win_pct",
        color_continuous_scale=[[0, "#a8d5b5"], [0.5, _GREEN2], [1, _GREEN]],
        labels={"win_pct": "Title probability (%)", "name": ""},
        text=top15["win_pct"].apply(lambda x: f"{x:.1f}%"),
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=440,
        margin=dict(l=0, r=80, t=10, b=30),
        font=dict(size=13),
    )
    st.plotly_chart(fig_bar, width='stretch')

    # Full table
    st.markdown("### Full Probability Table")
    n_show = st.slider("Players to show", 16, 128, 32, step=16, key="odds_slider")

    disp = results.head(n_show)[["clay_seed", "name", "rank"] + DISPLAY_ROUNDS].copy()
    for r in DISPLAY_ROUNDS:
        disp[r] = disp[r].apply(pct)

    st.dataframe(
        disp.rename(columns={
            "clay_seed": "Clay Seed",
            "name": "Player",
            "rank": "ATP Rank",
            **{r: ROUND_LABELS[r] for r in DISPLAY_ROUNDS},
        }),
        width='stretch',
        hide_index=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Draw Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab_draw:
    st.markdown("## Draw Explorer")
    st.caption("Select a player to see their draw position, survival probabilities, and projected path.")

    player_list = results["name"].tolist()
    sel_name = st.selectbox("Select player", player_list, index=0, key="draw_sel")

    sel_row = results[results["name"] == sel_name].iloc[0]
    sel_id  = int(sel_row["player_id"])
    p_idx   = draw_pos_of[sel_id]

    # Summary metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Clay Seed",    f"#{int(sel_row['clay_seed'])}")
    mc2.metric("ATP Rank",     f"#{int(sel_row['rank'])}")
    mc3.metric("Draw Position", p_idx + 1)
    mc4.metric("Title Odds",   pct(sel_row["W"]))

    # Survival probability chart
    st.markdown(f"### {sel_name} — Survival Probabilities")

    surv_vals   = [float(sel_row[r]) for r in DISPLAY_ROUNDS]
    surv_labels = [ROUND_LABELS[r] for r in DISPLAY_ROUNDS]

    colors = []
    for v in surv_vals:
        if v >= 0.50:
            colors.append(_GREEN)
        elif v >= 0.15:
            colors.append(_GREEN2)
        elif v >= 0.05:
            colors.append("#e08000")
        else:
            colors.append(_RED)

    fig_surv = go.Figure(go.Bar(
        x=surv_labels,
        y=[v * 100 for v in surv_vals],
        marker_color=colors,
        text=[f"{v:.1%}" for v in surv_vals],
        textposition="outside",
    ))
    fig_surv.update_layout(
        yaxis=dict(title="Probability (%)", range=[0, 118]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=360,
        margin=dict(t=20, b=20, l=20, r=20),
        font=dict(size=13),
    )
    st.plotly_chart(fig_surv, width='stretch')

    # Projected path
    st.markdown("### Projected Path")
    st.caption(
        "For each round: the most dangerous projected opponent "
        "(highest title probability) from that section of the draw."
    )

    round_defs = [
        (1, "R64",  "Round 1 opponent"),
        (2, "R32",  "Round 2 — likely opponent"),
        (3, "R16",  "Round 3 — projected opponent"),
        (4, "QF",   "Quarter-Final — projected opponent"),
        (5, "SF",   "Semi-Final — projected opponent"),
        (6, "F",    "Final — projected opponent"),
    ]

    for rnd_num, rnd_col, rnd_title in round_defs:
        opp_ids  = potential_opponents(p_idx, rnd_num)
        opp_data = id_to_row.loc[[i for i in opp_ids if i in id_to_row.index]]
        if opp_data.empty:
            continue

        top_opp     = opp_data.sort_values("W", ascending=False).iloc[0]
        sel_survives = float(sel_row[rnd_col])

        badge_color = _GREEN if sel_survives >= 0.4 else (_GREEN2 if sel_survives >= 0.15 else _RED)

        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:16px;padding:12px 16px;
                        background:#fff;border:1px solid #e0e0d8;border-radius:8px;margin:6px 0;">
              <div style="background:{badge_color};color:#fff;border-radius:6px;
                          padding:4px 12px;font-weight:700;font-size:0.8rem;white-space:nowrap;">
                {ROUND_LABELS[rnd_col]}
              </div>
              <div style="flex:1">
                <strong>{rnd_title}:</strong> {top_opp['name']}
                &nbsp;(title odds: {pct(top_opp['W'])})
              </div>
              <div style="font-weight:700;color:{badge_color};font-size:1.05rem;white-space:nowrap;">
                {sel_name}: {pct(sel_survives)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Compare Players
# ─────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown("## Compare Players")
    st.caption("Side-by-side round-by-round probability comparison for any two players in the draw.")

    col_a, col_b = st.columns(2)
    with col_a:
        p1_name = st.selectbox("Player 1", player_list, index=0, key="cmp_p1")
    with col_b:
        default_p2 = min(1, len(player_list) - 1)
        p2_name = st.selectbox("Player 2", player_list, index=default_p2, key="cmp_p2")

    r1 = results[results["name"] == p1_name].iloc[0]
    r2 = results[results["name"] == p2_name].iloc[0]

    # Headline metrics
    mc1, sep, mc2 = st.columns([2, 1, 2])
    mc1.metric(p1_name, pct(r1["W"]), f"Clay Seed #{int(r1['clay_seed'])} · ATP #{int(r1['rank'])}")
    sep.markdown(
        "<div style='text-align:center;font-size:2rem;padding-top:18px;color:#aaa;'>vs</div>",
        unsafe_allow_html=True,
    )
    mc2.metric(p2_name, pct(r2["W"]), f"Clay Seed #{int(r2['clay_seed'])} · ATP #{int(r2['rank'])}")

    # Grouped bar chart
    cmp_rounds = DISPLAY_ROUNDS
    cmp_labels = [ROUND_LABELS[r] for r in cmp_rounds]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name=p1_name,
        x=cmp_labels,
        y=[float(r1[r]) * 100 for r in cmp_rounds],
        marker_color=_GREEN,
        text=[pct(r1[r]) for r in cmp_rounds],
        textposition="outside",
    ))
    fig_cmp.add_trace(go.Bar(
        name=p2_name,
        x=cmp_labels,
        y=[float(r2[r]) * 100 for r in cmp_rounds],
        marker_color=_RED,
        text=[pct(r2[r]) for r in cmp_rounds],
        textposition="outside",
    ))
    fig_cmp.update_layout(
        barmode="group",
        yaxis=dict(title="Probability (%)", range=[0, 118]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=13),
    )
    st.plotly_chart(fig_cmp, width='stretch')

    # Stat comparison table
    st.markdown("### Player Statistics")
    stat_data = {
        "Stat": ["Clay Elo", "Overall Elo", "Clay Win Rate (12m)", "Form (last 20)", "ATP Rank"],
        p1_name: [
            f"{float(r1['clay_elo']):.0f}",
            f"{float(r1['elo']):.0f}",
            pct(r1["clay_winrate"]),
            pct(r1["form"]),
            str(int(r1["rank"])),
        ],
        p2_name: [
            f"{float(r2['clay_elo']):.0f}",
            f"{float(r2['elo']):.0f}",
            pct(r2["clay_winrate"]),
            pct(r2["form"]),
            str(int(r2["rank"])),
        ],
    }
    st.dataframe(pd.DataFrame(stat_data), hide_index=True, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Clay Ratings
# ─────────────────────────────────────────────────────────────────────────────
with tab_ratings:
    st.markdown("## Clay Strength Ratings")
    st.caption(
        "Composite clay score = **45% Clay Elo** + **30% Clay Win Rate (12m)** + **25% Form (last 20 matches)** — "
        "all components z-score normalised before combining. Used to seed the draw."
    )

    n_clay = st.slider("Players to display", 16, 128, 32, step=16, key="clay_slider")
    rat_show = ratings.head(n_clay).copy()

    # Horizontal bar chart
    fig_clay = px.bar(
        rat_show,
        x="clay_strength",
        y="name",
        orientation="h",
        color="clay_strength",
        color_continuous_scale=[[0, "#c8e6c9"], [0.5, _GREEN2], [1, _GREEN]],
        labels={"clay_strength": "Clay Strength Score", "name": ""},
        text=rat_show["clay_strength"].round(2),
    )
    fig_clay.update_traces(textposition="outside")
    fig_clay.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(380, n_clay * 22),
        margin=dict(l=0, r=70, t=10, b=30),
        font=dict(size=12),
    )
    st.plotly_chart(fig_clay, width='stretch')

    # Component breakdown table
    st.markdown("### Component Breakdown")
    st.caption("Clay Elo and the underlying win-rate/form components behind each player's composite score.")

    rat_disp = rat_show[["name", "rank", "clay_elo", "clay_winrate", "clay_n", "form", "clay_strength"]].copy()
    rat_disp["clay_elo"]     = rat_disp["clay_elo"].round(1)
    rat_disp["clay_winrate"] = rat_disp["clay_winrate"].apply(pct)
    rat_disp["form"]         = rat_disp["form"].apply(pct)
    rat_disp["clay_strength"]= rat_disp["clay_strength"].round(3)

    st.dataframe(
        rat_disp.rename(columns={
            "name":          "Player",
            "rank":          "ATP Rank",
            "clay_elo":      "Clay Elo",
            "clay_winrate":  "Clay Win Rate (12m)",
            "clay_n":        "Clay Matches",
            "form":          "Form (last 20)",
            "clay_strength": "Clay Strength",
        }),
        hide_index=True,
        width='stretch',
    )
