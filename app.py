"""Roland Garros 2026 — Championship Predictor  (Streamlit app)

Run with:
    & <python> -m streamlit run app.py
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

# ── Brand colours (never change) ──────────────────────────────────────────────
_GREEN  = "#0a3d1f"
_GREEN2 = "#1a6b3c"
_RED    = "#c8102e"

# ── Chart colour constants (white bg, dark text — readable in both modes) ─────
_CHART_BG   = "#ffffff"
_CHART_FONT = "#333333"
_CHART_AXIS = "rgba(0,0,0,0.35)"
_CHART_GRID = "rgba(0,0,0,0.07)"

# ── Minimal CSS — only brand elements; let Streamlit handle everything else ───
st.markdown(f"""
<style>
  [data-testid="stHeader"]  {{ background: {_GREEN}; }}
  [data-testid="stSidebar"] {{ background: {_GREEN}; }}
  [data-testid="stSidebar"] * {{ color: rgba(255,255,255,0.92) !important; }}
  [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.2); }}
  .stTabs [aria-selected="true"] {{
    background: {_GREEN} !important;
    color: white !important;
  }}
</style>
""", unsafe_allow_html=True)

# ── Shared chart layout ───────────────────────────────────────────────────────
def _layout(**kw):
    base = dict(
        plot_bgcolor=_CHART_BG,
        paper_bgcolor=_CHART_BG,
        font=dict(color=_CHART_FONT, size=13),
        xaxis=dict(color=_CHART_AXIS, gridcolor=_CHART_GRID,
                   linecolor=_CHART_AXIS, tickfont=dict(color=_CHART_FONT)),
        yaxis=dict(color=_CHART_AXIS, gridcolor=_CHART_GRID,
                   linecolor=_CHART_AXIS, tickfont=dict(color=_CHART_FONT)),
        margin=dict(l=0, r=90, t=20, b=30),
    )
    base.update(kw)
    return base

# ── Constants ─────────────────────────────────────────────────────────────────
ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "W"]
ROUND_LABELS = {
    "R128": "Entered", "R64": "Round 1", "R32": "Round 2",
    "R16":  "Round 3", "QF":  "QF",      "SF":  "Semi-Final",
    "F":    "Final",   "W":   "Champion",
}
DISPLAY_ROUNDS = ROUNDS[1:]

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading simulation results…")
def load_data():
    with open("data/rg2026_results.pkl", "rb") as f:
        d = pickle.load(f)
    results = d["results"].copy().reset_index(drop=True)
    results.insert(0, "clay_seed", results.index + 1)
    return results, d["ratings"].copy(), d["draw"]

try:
    results, ratings, draw = load_data()
except FileNotFoundError:
    st.error(
        "**`data/rg2026_results.pkl` not found.**  "
        "Run the pipeline first:\n\n```\npython run_pipeline.py\n```"
    )
    st.stop()

id_to_row   = results.set_index("player_id")
draw_pos_of = {pid: i for i, pid in enumerate(draw)}


def potential_opponents(p_idx: int, rnd: int) -> list[int]:
    return [
        draw[i] for i in range(128)
        if (i >> rnd) == (p_idx >> rnd)
        and (i >> (rnd - 1)) != (p_idx >> (rnd - 1))
    ]


def pct(v) -> str:
    return f"{float(v):.1%}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎾 RG 2026")
    st.markdown("**Roland Garros Predictor**")
    st.markdown("---")
    st.markdown(f"**Players in draw:** {len(draw)}")
    st.markdown("**Simulations:** 20,000")
    st.markdown("---")
    st.markdown("**Data sources**")
    st.markdown("JeffSackmann / tennis_atp")
    st.markdown("tennis-data.co.uk")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("XGBoost · Val acc 66%")
    st.markdown("Train: 2000–2022  ·  Val: 2023–2025")

# ── Page header (always white text on green/red gradient — fine in both modes) ─
st.markdown(f"""
<div style="background:linear-gradient(135deg,{_GREEN} 0%,{_GREEN2} 60%,{_RED} 100%);
            color:#fff;padding:36px 32px 28px;border-radius:12px;margin-bottom:24px;">
  <div style="font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;
              opacity:0.7;margin-bottom:10px;">Data Science · Tennis Analytics</div>
  <div style="font-size:2rem;font-weight:700;margin-bottom:10px;">
    Roland Garros 2026 — Championship Predictor
  </div>
  <div style="opacity:0.85;font-size:1rem;">
    20,000 Monte Carlo simulations · XGBoost trained on 25 years of ATP data
  </div>
</div>
""", unsafe_allow_html=True)

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
    st.markdown("### Championship Odds")
    st.caption(
        "Probability of winning the title and reaching each round, based on 20,000 "
        "simulated brackets. Alcaraz excluded (injury withdrawal)."
    )

    cols = st.columns(3)
    for col, (_, row) in zip(cols, results.head(3).iterrows()):
        col.metric(
            label=f"#{int(row['clay_seed'])}  {row['name']}",
            value=pct(row["W"]),
            delta=f"ATP Rank {int(row['rank'])}",
        )

    st.markdown("#### Top 15 — Title Probability")
    top15 = results.head(15).copy()
    top15["win_pct"] = (top15["W"] * 100).round(1)

    fig_bar = px.bar(
        top15, x="win_pct", y="name", orientation="h",
        color="win_pct",
        color_continuous_scale=[[0, "#6abf8c"], [1, _GREEN]],
        labels={"win_pct": "Title probability (%)", "name": ""},
        text=top15["win_pct"].apply(lambda x: f"{x:.1f}%"),
    )
    fig_bar.update_traces(
        textposition="outside",
        textfont=dict(color=_CHART_FONT, size=12),
    )
    fig_bar.update_layout(**_layout(
        yaxis=dict(categoryorder="total ascending",
                   tickfont=dict(color=_CHART_FONT, size=12),
                   gridcolor=_CHART_GRID, linecolor=_CHART_AXIS),
        xaxis=dict(title="Title probability (%)",
                   tickfont=dict(color=_CHART_FONT),
                   gridcolor=_CHART_GRID, linecolor=_CHART_AXIS),
        coloraxis_showscale=False,
        height=440,
    ))
    st.plotly_chart(fig_bar, width="stretch")

    st.markdown("#### Full Probability Table")
    n_show = st.slider("Players to show", 16, 128, 32, step=16, key="odds_slider")
    disp = results.head(n_show)[["clay_seed", "name", "rank"] + DISPLAY_ROUNDS].copy()
    for r in DISPLAY_ROUNDS:
        disp[r] = disp[r].apply(pct)
    st.dataframe(
        disp.rename(columns={
            "clay_seed": "Clay Seed", "name": "Player", "rank": "ATP Rank",
            **{r: ROUND_LABELS[r] for r in DISPLAY_ROUNDS},
        }),
        width="stretch", hide_index=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Draw Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab_draw:
    st.markdown("### Draw Explorer")
    st.caption("Select a player to see their draw position, survival probabilities, and projected path.")

    player_list = results["name"].tolist()
    sel_name = st.selectbox("Select player", player_list, index=0, key="draw_sel")
    sel_row  = results[results["name"] == sel_name].iloc[0]
    sel_id   = int(sel_row["player_id"])
    p_idx    = draw_pos_of[sel_id]

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Clay Seed",     f"#{int(sel_row['clay_seed'])}")
    mc2.metric("ATP Rank",      f"#{int(sel_row['rank'])}")
    mc3.metric("Draw Position", p_idx + 1)
    mc4.metric("Title Odds",    pct(sel_row["W"]))

    st.markdown(f"#### {sel_name} — Survival Probabilities")
    surv_vals   = [float(sel_row[r]) for r in DISPLAY_ROUNDS]
    surv_labels = [ROUND_LABELS[r]   for r in DISPLAY_ROUNDS]
    bar_colors  = [
        _GREEN if v >= 0.50 else _GREEN2 if v >= 0.15 else "#e08000" if v >= 0.05 else _RED
        for v in surv_vals
    ]

    fig_surv = go.Figure(go.Bar(
        x=surv_labels, y=[v * 100 for v in surv_vals],
        marker_color=bar_colors,
        text=[f"{v:.1%}" for v in surv_vals],
        textposition="outside",
        textfont=dict(color=_CHART_FONT, size=12),
    ))
    fig_surv.update_layout(**_layout(
        yaxis=dict(title="Probability (%)", range=[0, 118],
                   tickfont=dict(color=_CHART_FONT),
                   gridcolor=_CHART_GRID, linecolor=_CHART_AXIS),
        xaxis=dict(tickfont=dict(color=_CHART_FONT, size=12),
                   gridcolor=_CHART_GRID, linecolor=_CHART_AXIS),
        height=360, margin=dict(t=20, b=20, l=20, r=20),
    ))
    st.plotly_chart(fig_surv, width="stretch")

    st.markdown("#### Projected Path")
    st.caption(
        "For each round: the most dangerous projected opponent "
        "(highest title probability) from that section of the draw."
    )
    for rnd_num, rnd_col, rnd_title in [
        (1, "R64", "Round 1 opponent"),
        (2, "R32", "Round 2 — likely opponent"),
        (3, "R16", "Round 3 — projected opponent"),
        (4, "QF",  "Quarter-Final — projected opponent"),
        (5, "SF",  "Semi-Final — projected opponent"),
        (6, "F",   "Final — projected opponent"),
    ]:
        opp_ids  = potential_opponents(p_idx, rnd_num)
        opp_data = id_to_row.loc[[i for i in opp_ids if i in id_to_row.index]]
        if opp_data.empty:
            continue
        top_opp      = opp_data.sort_values("W", ascending=False).iloc[0]
        sel_survives = float(sel_row[rnd_col])
        badge        = _GREEN if sel_survives >= 0.4 else (_GREEN2 if sel_survives >= 0.15 else _RED)

        c1, c2, c3 = st.columns([1, 4, 2])
        c1.markdown(
            f"<div style='background:{badge};color:#fff;border-radius:6px;"
            f"padding:6px 10px;font-weight:700;font-size:0.8rem;text-align:center;'>"
            f"{ROUND_LABELS[rnd_col]}</div>",
            unsafe_allow_html=True,
        )
        c2.markdown(f"**{rnd_title}:** {top_opp['name']}  "
                    f"*(title odds: {pct(top_opp['W'])})*")
        c3.markdown(
            f"<div style='font-weight:700;color:{badge};font-size:1.05rem;"
            f"text-align:right;padding-top:4px;'>{sel_name}: {pct(sel_survives)}</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Compare Players
# ─────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown("### Compare Players")
    st.caption("Side-by-side round-by-round probability comparison for any two players.")

    col_a, col_b = st.columns(2)
    with col_a:
        p1_name = st.selectbox("Player 1", player_list, index=0, key="cmp_p1")
    with col_b:
        p2_name = st.selectbox("Player 2", player_list,
                               index=min(1, len(player_list) - 1), key="cmp_p2")

    r1 = results[results["name"] == p1_name].iloc[0]
    r2 = results[results["name"] == p2_name].iloc[0]

    mc1, _, mc2 = st.columns([2, 1, 2])
    mc1.metric(p1_name, pct(r1["W"]),
               f"Clay Seed #{int(r1['clay_seed'])} · ATP #{int(r1['rank'])}")
    _.markdown("<div style='text-align:center;font-size:2rem;padding-top:18px;opacity:0.4;'>vs</div>",
               unsafe_allow_html=True)
    mc2.metric(p2_name, pct(r2["W"]),
               f"Clay Seed #{int(r2['clay_seed'])} · ATP #{int(r2['rank'])}")

    cmp_labels = [ROUND_LABELS[r] for r in DISPLAY_ROUNDS]
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name=p1_name, x=cmp_labels,
        y=[float(r1[r]) * 100 for r in DISPLAY_ROUNDS],
        marker_color=_GREEN,
        text=[pct(r1[r]) for r in DISPLAY_ROUNDS],
        textposition="outside",
        textfont=dict(color=_CHART_FONT, size=11),
    ))
    fig_cmp.add_trace(go.Bar(
        name=p2_name, x=cmp_labels,
        y=[float(r2[r]) * 100 for r in DISPLAY_ROUNDS],
        marker_color=_RED,
        text=[pct(r2[r]) for r in DISPLAY_ROUNDS],
        textposition="outside",
        textfont=dict(color=_CHART_FONT, size=11),
    ))
    fig_cmp.update_layout(**_layout(
        barmode="group",
        yaxis=dict(title="Probability (%)", range=[0, 118],
                   tickfont=dict(color=_CHART_FONT),
                   gridcolor=_CHART_GRID, linecolor=_CHART_AXIS),
        xaxis=dict(tickfont=dict(color=_CHART_FONT, size=12),
                   gridcolor=_CHART_GRID, linecolor=_CHART_AXIS),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    font=dict(color=_CHART_FONT)),
        height=420, margin=dict(t=20, b=20, l=20, r=20),
    ))
    st.plotly_chart(fig_cmp, width="stretch")

    st.markdown("#### Player Statistics")
    st.dataframe(
        pd.DataFrame({
            "Stat":  ["Clay Elo", "Overall Elo", "Clay Win Rate (12m)",
                      "Form (last 20)", "ATP Rank"],
            p1_name: [f"{float(r1['clay_elo']):.0f}", f"{float(r1['elo']):.0f}",
                      pct(r1["clay_winrate"]), pct(r1["form"]), str(int(r1["rank"]))],
            p2_name: [f"{float(r2['clay_elo']):.0f}", f"{float(r2['elo']):.0f}",
                      pct(r2["clay_winrate"]), pct(r2["form"]), str(int(r2["rank"]))],
        }),
        hide_index=True, width="stretch",
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Clay Ratings
# ─────────────────────────────────────────────────────────────────────────────
with tab_ratings:
    st.markdown("### Clay Strength Ratings")
    st.caption(
        "Composite score = **45% Clay Elo** + **30% Clay Win Rate (12m)** "
        "+ **25% Form (last 20 matches)** — z-score normalised. Used to seed the draw."
    )

    n_clay   = st.slider("Players to display", 16, 128, 32, step=16, key="clay_slider")
    rat_show = ratings.head(n_clay).copy()

    fig_clay = px.bar(
        rat_show, x="clay_strength", y="name", orientation="h",
        color="clay_strength",
        color_continuous_scale=[[0, "#6abf8c"], [0.5, _GREEN2], [1, _GREEN]],
        labels={"clay_strength": "Clay Strength Score", "name": ""},
        text=rat_show["clay_strength"].round(2),
    )
    fig_clay.update_traces(
        textposition="outside",
        textfont=dict(color=_CHART_FONT, size=11),
    )
    fig_clay.update_layout(**_layout(
        yaxis=dict(categoryorder="total ascending",
                   tickfont=dict(color=_CHART_FONT, size=11),
                   gridcolor=_CHART_GRID, linecolor=_CHART_AXIS),
        xaxis=dict(title="Clay Strength Score",
                   tickfont=dict(color=_CHART_FONT),
                   gridcolor=_CHART_GRID, linecolor=_CHART_AXIS),
        coloraxis_showscale=False,
        height=max(380, n_clay * 22),
        margin=dict(l=0, r=70, t=10, b=30),
    ))
    st.plotly_chart(fig_clay, width="stretch")

    st.markdown("#### Component Breakdown")
    rat_disp = rat_show[["name", "rank", "clay_elo", "clay_winrate",
                          "clay_n", "form", "clay_strength"]].copy()
    rat_disp["clay_elo"]      = rat_disp["clay_elo"].round(1)
    rat_disp["clay_winrate"]  = rat_disp["clay_winrate"].apply(pct)
    rat_disp["form"]          = rat_disp["form"].apply(pct)
    rat_disp["clay_strength"] = rat_disp["clay_strength"].round(3)
    st.dataframe(
        rat_disp.rename(columns={
            "name": "Player", "rank": "ATP Rank", "clay_elo": "Clay Elo",
            "clay_winrate": "Clay Win Rate (12m)", "clay_n": "Clay Matches",
            "form": "Form (last 20)", "clay_strength": "Clay Strength",
        }),
        hide_index=True, width="stretch",
    )
