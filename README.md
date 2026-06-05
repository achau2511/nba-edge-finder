# NBA Finals Edge Finder

A machine learning pipeline that predicts NBA player stats and compares model-implied probabilities against Kalshi prediction market prices to identify mispriced contracts.

Built during the 2026 NBA Finals (Knicks vs Spurs).

---

## What It Does

1. Pulls player game logs from `nba_api` (6 seasons of regular season + playoffs)
2. Engineers look-ahead-safe rolling features per player
3. Trains an XGBoost regression model on playoff data only — tuned for playoff game contexts
4. Fetches live Kalshi player prop market prices
5. Converts point estimates to probabilities using empirical residual distributions
6. Flags markets where model probability exceeds Kalshi's implied price
7. Displays everything in a live Streamlit dashboard

---

## Live Dashboard

Run locally:
```bash
streamlit run app.py
```

The dashboard shows:
- **Best Bets** — markets with edge > 0.15 and model probability > 0.75
- **All Markets** — full predictions table across points, rebounds, and assists with color-coded edge, model probability, and prediction vs line
- **Refresh Predictions** button — re-fetches game logs, retrains the model, and pulls fresh Kalshi prices after each Finals game

Kalshi market prices update automatically every 10 minutes.

---

## Model

**Algorithm:** XGBoost regressor (separate model per stat — points, rebounds, assists)

**Training data:** 2024-25 full playoffs + 2025-26 playoffs (rounds 1 through Conference Finals)

**Recent games weighted more heavily:**
- Finals games: 5x weight
- Conference Finals games: 2x weight
- Earlier playoff games: 1x weight

**Features per player-game:**
- Rolling mean and std of target stat at 5, 10, and 20 game windows
- Rolling mean of minutes at 5, 10, and 20 game windows
- 3-game trend (slope of recent stat)
- Days of rest, back-to-back flag
- Home/away indicator
- Game number within season
- Opponent defensive rating, points in paint allowed, fastbreak points allowed

**Probability conversion:** Empirical residual distribution per player. For each player, historical residuals (actual − predicted) are stored. `P(stat > line)` is computed as the fraction of historical residuals that would need to exceed `(line − prediction)` — no normal distribution assumption.

**Walk-forward validation results:**

| Stat | MAE | RMSE |
|------|-----|------|
| Points | 4.11 | 5.75 |
| Rebounds | 1.72 | 2.38 |
| Assists | 1.15 | 1.66 |

Brier scores (probability calibration): Points 0.155, Rebounds 0.183, Assists 0.156 vs 0.25 naive baseline.

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database and fetch data
cd src/ingestion
python db.py
python nba_fetcher.py
python kalshi_fetcher.py

# Engineer features
cd ../features
python engineer.py

# Train playoff model
cd ../..
python playoff_model.py

# Run predictions
python finals_predictions.py

# Launch dashboard
streamlit run app.py
```

---

## Project Structure

```
nba-edge-finder/
├── app.py                    Streamlit dashboard
├── finals_predictions.py     Live predictions vs Kalshi markets
├── playoff_model.py          Playoff-specific XGBoost trainer
├── src/
│   ├── ingestion/
│   │   ├── db.py             SQLite schema + connection
│   │   ├── nba_fetcher.py    NBA game logs + team defensive ratings
│   │   └── kalshi_fetcher.py Historical Kalshi prop markets
│   ├── features/
│   │   ├── engineer.py       Look-ahead-safe feature engineering
│   │   └── team_mapping.py   Team name → abbreviation mapping
│   └── models/
│       ├── stat_predictor.py Regular season model (reference)
│       └── prob_converter.py Probability conversion utilities
├── data/
│   ├── nba_edge.db           SQLite database (not tracked)
│   ├── models/               Trained XGBoost models (not tracked)
│   └── processed/            Feature CSVs and predictions (not tracked)
├── requirements.txt
└── WRITEUP.md                Detailed technical writeup
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| NBA data | `nba_api` |
| Market data | Kalshi REST API (public, no auth) |
| Storage | SQLite |
| Feature engineering | pandas, numpy |
| Modeling | XGBoost, scikit-learn |
| Probability conversion | scipy, empirical residuals |
| Dashboard | Streamlit, Plotly |

---

## Game 1 Results (June 4, 2026)

Tracked 3 single bets from model best bets:

| Pick | Line | Actual | Result |
|------|------|--------|--------|
| KAT over 3.5 assists | 3.5 | 4 | ✅ |
| Castle over 5.5 rebounds | 5.5 | 8 | ✅ |
| Bridges over 9.5 points | 9.5 | 9 | ❌ |

**2-1 on Game 1.**

---

## Notes

Kalshi only has settled NBA prop data from April 2026 onward (the 2026 playoffs). Regular season Kalshi markets from November 2025–April 2026 are not accessible via the public API. The model was designed for regular season edge-finding but has been adapted for live Finals prediction tracking while the regular season data accumulates for the 2026-27 season.
