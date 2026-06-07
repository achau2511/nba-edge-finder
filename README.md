# NBA Finals Edge Finder

A machine learning pipeline that predicts NBA player stats and compares model-implied probabilities against Kalshi and Polymarket prediction market prices to identify mispriced contracts.

Built during the 2026 NBA Finals (Knicks vs Spurs).

---

## What It Does

1. Pulls player game logs from `nba_api` (6 seasons of regular season + playoffs)
2. Engineers look-ahead-safe rolling features per player
3. Trains XGBoost regression models on playoff data — tuned for playoff game contexts
4. Fetches live Kalshi and Polymarket player prop prices automatically
5. Converts point estimates to probabilities using empirical residual distributions
6. Flags markets where model probability exceeds the market's implied price
7. Displays everything in a live Streamlit dashboard

---

## Live Dashboard

```bash
streamlit run app.py
```

The dashboard has three tabs:

- **Best Bets** — Kalshi and Polymarket overs with edge > 0.12 and model probability > 0.75, ranked by edge × model probability
- **Kalshi Markets** — full predictions table across points, rebounds, assists, and threes with color-coded edge and model probability
- **Polymarket** — same table using auto-fetched Polymarket prices

**Sidebar buttons:**
- **⚡ Refresh Prices Only** — re-fetches live Kalshi and Polymarket prices instantly. Use this before tip-off.
- **🔄 Refresh Predictions** — runs the full post-game pipeline: fetches new box scores, re-engineers features, retrains the model. Use this after each game.

Both markets auto-refresh every 10 minutes in the background.

---

## Model

**Algorithm:** XGBoost regressor (separate model per stat — points, rebounds, assists, threes)

**Training data:** 2024-25 full playoffs + 2025-26 playoffs (rounds 1 through Conference Finals)

**Sample weights:**
- Finals games: 5x
- Conference Finals: 2x
- Earlier playoff rounds: 1x

**Features per player-game:**
- Rolling mean and std of target stat at 5, 10, and 20 game windows
- Rolling mean of minutes at 5, 10, and 20 game windows
- 3-game trend (slope of recent stat)
- Days of rest, back-to-back flag
- Home/away indicator
- Game number within season
- Opponent defensive rating, points in paint allowed, fastbreak points allowed

**Probability conversion:** Empirical residual distribution per player. For each player, historical residuals (actual − predicted) are stored. `P(stat > line)` is computed as the fraction of historical residuals that would need to exceed `(line − prediction)` — no normal distribution assumption.

**Walk-forward validation:**

| Stat | MAE | RMSE |
|------|-----|------|
| Points | 4.11 | 5.75 |
| Rebounds | 1.72 | 2.38 |
| Assists | 1.15 | 1.66 |

Brier scores: Points 0.155, Rebounds 0.183, Assists 0.156 vs 0.25 naive baseline.

---

## Market Data

**Kalshi** — public REST API, no authentication required. Fetches open markets per series ticker (`KXNBAPTS`, `KXNBAREB`, `KXNBAAST`, `KXNBA3PT`). Prices are mid of yes bid/ask.

**Polymarket** — fetched via `gateway.polymarket.us/v1/search`. Returns 191 live player prop markets per game with correct over prices. Illiquid markets (spread < 1.05) are filtered out automatically. No authentication required.

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database and fetch data
cd src/ingestion
python db.py
python nba_fetcher.py

# Engineer features
cd ../features
python engineer.py

# Train playoff model
cd ../..
python playoff_model.py

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
├── polymarket_fetcher.py     Auto-fetches Polymarket player prop prices
├── polymarket_data.py        Auto-generated Polymarket market data (not tracked)
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
│   └── processed/            Feature CSVs (not tracked)
├── requirements.txt
└── WRITEUP.md
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| NBA data | `nba_api` |
| Kalshi data | Kalshi REST API (public, no auth) |
| Polymarket data | `gateway.polymarket.us` REST API (public, no auth) |
| Storage | SQLite |
| Feature engineering | pandas, numpy |
| Modeling | XGBoost, scikit-learn |
| Probability conversion | scipy, empirical residuals |
| Dashboard | Streamlit, Plotly |

---

## Live Track Record — 2026 NBA Finals

The model generates forward-looking predictions before each game and tracks real outcomes.

### Game 1 — June 4, 2026 (Spurs win)

**Kalshi:**

| Pick | Line | Result |
|------|------|--------|
| KAT over 3.5 assists | 3.5 | ✅ |
| Castle over 5.5 rebounds | 5.5 | ✅ |
| Bridges over 9.5 points | 9.5 | ❌ |

**2-1 on Game 1 Kalshi picks.**

---

### Game 2 — June 6, 2026 (Spurs win)

**Kalshi:**

| Pick | Line | Result |
|------|------|--------|
| KAT over 3.5 assists | 3.5 | ✅ |
| Brunson over 19.5 points | 19.5 | ✅ |
| Castle over 3.5 rebounds | 3.5 | ✅ |

**3-0 on Game 2 Kalshi picks.**

**Polymarket:**

| Pick | Line | Result |
|------|------|--------|
| KAT over 3.5 assists | 3.5 | ✅ |
| Wembanyama over 1.5 threes | 1.5 | ✅ |
| Wembanyama under 12.5 rebounds | 12.5 | ✅ |
| Brunson under 6.5 assists | 6.5 | ✅ |

**4-0 on Game 2 Polymarket picks.**

---

### Series Record

| Game | Kalshi | Polymarket |
|------|--------|------------|
| Game 1 | 2-1 | — |
| Game 2 | 3-0 | 4-0 |
| **Total** | **5-1** | **4-0** |

---

## Notes

Kalshi only has settled NBA prop data from April 2026 onward (the 2026 playoffs). Regular season Kalshi markets are not accessible via the public API. The model was designed for regular season edge-finding but has been adapted for live Finals prediction tracking.

Deployment on cloud platforms (Streamlit Community Cloud, Railway, etc.) is not possible because `stats.nba.com` blocks requests from cloud server IPs. The app runs locally only.