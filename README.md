# 🎾 Roland Garros 2026 — Championship Predictor

A Monte Carlo simulation system that estimates Roland Garros 2026 championship odds using 25 years of ATP data, XGBoost, and live player-state tracking.

## What it does

1. Downloads ~74,000 ATP matches (2000–2026) from JeffSackmann and tennis-data.co.uk
2. Engineers player features: Elo ratings (overall + clay), head-to-head records, surface win rates, recent form
3. Trains an XGBoost binary classifier on 2000–2022 data (val accuracy ~66% on 2023–2025)
4. Runs 20,000 Monte Carlo bracket simulations to estimate round-by-round survival probabilities
5. Surfaces everything in an interactive Streamlit dashboard

---

## Quickstart

```bash
pip install -r requirements.txt
python run_pipeline.py        # runs all 4 steps (~15–20 min first time)
streamlit run app.py
```

---

## Pipeline

Four sequential steps, each a standalone script:

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `1_fetch_data.py` | Downloads ATP match data (Sackmann + tennis-data.co.uk) |
| 2 | `2_build_features.py` | Engineers Elo, H2H, surface stats, form |
| 3 | `3_train_model.py` | Trains XGBoost + builds live player states |
| 4 | `4_simulate_rg2026.py` | Runs 20,000-bracket Monte Carlo simulation |

```bash
python run_pipeline.py --from 3   # restart from step 3
python run_pipeline.py --only 4   # run step 4 only
```

---

## Streamlit Dashboard

Four tabs:

- **Championship Odds** — title probability bar chart + full probability table for all 128 players
- **Draw Explorer** — select any player to see their projected path and round-by-round survival odds
- **Compare Players** — side-by-side probability comparison across all rounds
- **Clay Ratings** — composite clay strength score used to seed the draw

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tennisanalysis-rmehhgyrsp47ozy7ddzmx8.streamlit.app/)

---

## Model

XGBoost binary classifier trained on symmetrised match data (each match appears twice, once per perspective).

**Features:** Elo difference (overall + clay), H2H win rate, clay surface win rate (12-month window), recent form (last 20 matches), ATP ranking difference.

| Split | Period | Rows | Accuracy | Log-loss |
|-------|--------|------|----------|----------|
| Train | 2000–2022 | ~120k | — | — |
| Val | 2023–2025 | ~30k | ~66% | ~0.63 |

---

## Draw construction

Players ranked by a composite **clay strength score**:

> 45% Clay Elo (z) + 30% Clay Win Rate 12m (z) + 25% Form last 20 (z)

Retired players and those inactive for >6 months are excluded automatically. The draw is seeded using this score (not ATP ranking), placing the strongest clay players at opposite ends of the bracket.

---

## Project structure

```
tennis_analysis/
├── src/
│   ├── data/        fetch.py  fetch_recent.py  process.py
│   ├── features/    elo.py  h2h.py  surface.py  form.py  builder.py
│   ├── models/      train.py  predict.py
│   └── rg/          clay_rating.py  draw.py  simulator.py
├── 1_fetch_data.py
├── 2_build_features.py
├── 3_train_model.py
├── 4_simulate_rg2026.py
├── run_pipeline.py
├── app.py
└── requirements.txt
```

---

## Data sources

- [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) — ATP match results 2000–2025 (open licence)
- [tennis-data.co.uk](http://www.tennis-data.co.uk/) — 2026 clay season supplement

---

## Requirements

Python 3.10+ · pandas · numpy · xgboost · scikit-learn · streamlit · plotly · joblib
