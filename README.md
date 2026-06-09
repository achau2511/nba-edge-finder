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
6. Identifies edges on both overs and unders across both markets
7. Displays everything in a live Streamlit dashboard

---

## Live Dashboard

```bash
streamlit run app.py
```

The dashboard has four tabs:

- **Best Bets** — Kalshi best overs, Polymarket best overs, and Polymarket best unders, all filtered by edge > 0.12 and model probability > 0.75
- **Kalshi Markets** — full predictions table across points, rebounds, assists, and threes with color-coded edge and model probability
- **Poly Overs** — Polymarket over markets with edge and model probability
- **Poly Unders** — Polymarket under markets sorted by under edge, with under model probability vs market under price

**Sidebar buttons:**
- **⚡ Refresh Prices Only** — re-fetches live Kalshi and Polymarket prices instantly. Use before tip-off.
- **🔄 Refresh Predictions** — runs the full post-game pipeline: fetches new box scores, re-engineers features, retrains the model. Use after each game.

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

## Edge Calculation

**Over edge:** `model_prob − market_over_price`

**Under edge:** `(1 − model_prob) − market_under_price`

Best bets require edge > 0.12 and model probability > 0.75. Under best bets use the same thresholds applied to the under side.

---

## Market Data

**Kalshi** — public REST API, no authentication required. Fetches open markets per series ticker (`KXNBAPTS`, `KXNBAREB`, `KXNBAAST`, `KXNBA3PT`). Prices are mid of yes bid/ask.

**Polymarket** — fetched via `gateway.polymarket.us/v1/search`. Returns ~178 player prop markets per game covering points, rebounds, assists, and threes. Both over and under prices are stored. No authentication required.

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
├── push_to_supabase.py       Pushes predictions and prices to Supabase
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
| Storage | SQLite + Supabase |
| Feature engineering | pandas, numpy |
| Modeling | XGBoost, scikit-learn |
| Probability conversion | scipy, empirical residuals |
| Dashboard | Streamlit |

---

## Live Track Record — 2026 NBA Finals

All picks were selected from the dashboard's Best Bets list before tip-off.

### Game 1 — June 4, 2026 · Knicks 105, Spurs 95 (Knicks lead 1-0)

**Kalshi:**

| Pick | Line | Actual | Result |
|------|------|--------|--------|
| KAT over 3.5 assists | 3.5 | 4 ast | ✅ |
| Castle over 5.5 rebounds | 5.5 | 8 reb | ✅ |
| Bridges over 9.5 points | 9.5 | 9 pts | ❌ |

**2-1 on Game 1 Kalshi picks.**

---

### Game 2 — June 6, 2026 · Knicks 105, Spurs 104 (Knicks lead 2-0)

**Kalshi:**

| Pick | Line | Actual | Result |
|------|------|--------|--------|
| KAT over 3.5 assists | 3.5 | 4 ast | ✅ |
| Brunson over 19.5 points | 19.5 | 20 pts | ✅ |
| Castle over 3.5 rebounds | 3.5 | 5 reb | ✅ |

**3-0 on Game 2 Kalshi picks.**

**Polymarket:**

| Pick | Line | Actual | Result |
|------|------|--------|--------|
| KAT over 3.5 assists | 3.5 | 4 ast | ✅ |
| Wembanyama over 1.5 threes | 1.5 | 2 threes | ✅ |
| Wembanyama under 12.5 rebounds | 12.5 | 9 reb | ✅ |
| Brunson under 6.5 assists | 6.5 | 6 ast | ✅ |

**4-0 on Game 2 Polymarket picks.**

---

### Game 3 — June 9, 2026 · Spurs 115, Knicks 111 (Knicks lead 2-1)

**Kalshi:**

| Pick | Line | Actual | Result |
|------|------|--------|--------|
| Champagnie over 1.5 threes | 1.5 | 3 threes | ✅ |
| Vassell over 9.5 points | 9.5 | 11 pts | ✅ |
| Castle over 14.5 points | 14.5 | 23 pts | ✅ |
| Castle over 3.5 rebounds | 3.5 | 5 reb | ✅ |

**4-0 on Game 3 Kalshi picks.**

**Polymarket:**

| Pick | Line | Actual | Result |
|------|------|--------|--------|
| Wembanyama under 12.5 rebounds | 12.5 | 8 reb | ✅ |
| Castle under 6.5 assists | 6.5 | 5 ast | ✅ |
| Castle over 14.5 points | 14.5 | 23 pts | ✅ |
| Vassell over 1.5 assists | 1.5 | 1 ast | ❌ |
| Harper under 3.5 assists | 3.5 | 4 ast | ❌ |

**3-2 on Game 3 Polymarket picks.**

---

### Series Record

| Game | Kalshi | Polymarket |
|------|--------|------------|
| Game 1 | 2-1 | — |
| Game 2 | 3-0 | 4-0 |
| Game 3 | 4-0 | 3-2 |
| **Total** | **9-1** | **7-2** |

---

## Notes

Kalshi only has settled NBA prop data from April 2026 onward (the 2026 playoffs). Regular season Kalshi markets are not accessible via the public API. The model was designed for regular season edge-finding but has been adapted for live Finals prediction tracking.

Deployment on cloud platforms (Streamlit Community Cloud, Railway, etc.) is not possible because `stats.nba.com` blocks requests from cloud server IPs. The app runs locally only.