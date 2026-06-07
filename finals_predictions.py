"""
finals_predictions.py — Trains playoff model on all available data,
then pulls active Kalshi Finals markets and generates predictions.

Run this before each Finals game to see model predictions vs market prices.
Results are logged to data/processed/finals_predictions.csv for tracking.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "ingestion"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "models"))

from db import get_connection
from prob_converter import build_residual_map, predict_prob

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "data", "models")

FEATURE_COLS = {
    "points": [
        "pts_mean_5g", "pts_mean_10g", "pts_mean_20g",
        "pts_std_5g",  "pts_std_10g",  "pts_std_20g",
        "min_mean_5g", "min_mean_10g", "min_mean_20g",
        "pts_trend_3g",
        "days_rest", "is_b2b", "is_home", "game_num_in_season",
        "def_rating", "opp_pts_paint", "opp_pts_fb",
    ],
    "rebounds": [
        "reb_mean_5g", "reb_mean_10g", "reb_mean_20g",
        "reb_std_5g",  "reb_std_10g",  "reb_std_20g",
        "min_mean_5g", "min_mean_10g", "min_mean_20g",
        "reb_trend_3g",
        "days_rest", "is_b2b", "is_home", "game_num_in_season",
        "def_rating", "opp_pts_paint", "opp_pts_fb",
    ],
    "assists": [
        "ast_mean_5g", "ast_mean_10g", "ast_mean_20g",
        "ast_std_5g",  "ast_std_10g",  "ast_std_20g",
        "min_mean_5g", "min_mean_10g", "min_mean_20g",
        "ast_trend_3g",
        "days_rest", "is_b2b", "is_home", "game_num_in_season",
        "def_rating", "opp_pts_paint", "opp_pts_fb",
    ],
    "threes": [
        "fg3m_mean_5g", "fg3m_mean_10g", "fg3m_mean_20g",
        "fg3m_std_5g",  "fg3m_std_10g",  "fg3m_std_20g",
        "min_mean_5g",  "min_mean_10g",  "min_mean_20g",
        "fg3m_trend_3g",
        "days_rest", "is_b2b", "is_home", "game_num_in_season",
        "def_rating", "opp_pts_paint", "opp_pts_fb",
    ],
}

KALSHI_SERIES = {
    "KXNBAPTS": "points",
    "KXNBAREB": "rebounds",
    "KXNBAAST": "assists",
    "KXNBA3PT": "threes",
}

import requests
import time

def fetch_active_markets(series_ticker: str) -> list:
    """Fetch active (unsettled) Kalshi markets for a series."""
    markets = []
    cursor = None

    while True:
        params = {"limit": 200, "series_ticker": series_ticker, "status": "open"}
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(
                "https://external-api.kalshi.com/trade-api/v2/markets",
                params=params
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Error fetching {series_ticker}: {e}")
            break

        batch = data.get("markets", [])
        markets.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
        time.sleep(0.3)

    return markets


def parse_active_market(market: dict, stat_type: str) -> dict | None:
    """Parse an active market into a usable format."""
    title = market.get("title", "")
    if ":" not in title:
        return None

    player_name = title.split(":")[0].strip()
    line = market.get("floor_strike")
    if line is None:
        return None

    occurrence = market.get("occurrence_datetime", "")
    from datetime import datetime, timezone, timedelta
    if occurrence:
        utc_dt = datetime.fromisoformat(occurrence.replace("Z", "+00:00"))
        et_dt = utc_dt - timedelta(hours=4)  # UTC to ET
        game_date = et_dt.strftime("%Y-%m-%d")
    else:
        game_date = None
    if not game_date:
        return None

    # Current mid-market price as implied probability
    yes_ask = market.get("yes_ask_dollars")
    yes_bid = market.get("yes_bid_dollars")
    if yes_ask is None or yes_bid is None:
        return None

    implied_prob = (float(yes_ask) + float(yes_bid)) / 2.0
    if implied_prob == 0:
        implied_prob = float(yes_ask)

    return {
        "ticker":       market.get("ticker"),
        "player_name":  player_name,
        "stat_type":    stat_type,
        "line":         float(line),
        "game_date":    game_date,
        "kalshi_prob":  implied_prob,
        "volume":       float(market.get("volume_fp", 0)),
    }


def train_playoff_model(stat: str) -> tuple:
    """Train XGBoost on all available playoff data."""
    features_df = pd.read_csv(
        os.path.join(DATA_DIR, f"features_{stat}.csv"),
        parse_dates=["game_date"]
    )
    features_df["game_id"] = features_df["game_id"].astype(str).str.zfill(10)
    playoff_df = features_df[features_df["game_id"].str.startswith("0042")].copy()

    feature_cols = FEATURE_COLS[stat]
    train_df = playoff_df.dropna(subset=feature_cols + ["target"])

    print(f"  Training on {len(train_df):,} playoff rows...")

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(train_df[feature_cols], train_df["target"], verbose=False)

    # Build residual map
    train_df = train_df.copy()
    train_df["prediction"] = np.clip(model.predict(train_df[feature_cols]), 0, None)
    train_df["residual"] = train_df["target"] - train_df["prediction"]
    residual_map = build_residual_map(train_df, stat)

    return model, residual_map, train_df


def get_latest_features(player_name: str, game_date: str,
                         stat: str, features_df: pd.DataFrame) -> pd.Series | None:
    """
    Get the most recent feature row for a player before a given game date.
    Used to predict upcoming games.
    """
    player_norm = player_name.lower().strip()
    features_df["player_name_norm"] = features_df["player_name"].str.lower().str.strip()

    player_rows = features_df[
        (features_df["player_name_norm"] == player_norm) &
        (features_df["game_date"] < pd.to_datetime(game_date))
    ].sort_values("game_date")

    if player_rows.empty:
        return None

    return player_rows.iloc[-1]


def run():
    print("=" * 60)
    print("NBA Finals Predictions")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    all_predictions = []

    for series_ticker, stat_type in KALSHI_SERIES.items():
        print(f"\nFetching active {stat_type} markets...")
        active_markets = fetch_active_markets(series_ticker)
        print(f"  Found {len(active_markets)} active markets")

        if not active_markets:
            continue

        parsed = []
        for m in active_markets:
            p = parse_active_market(m, stat_type)
            if p:
                parsed.append(p)

        if not parsed:
            print("  No parseable markets found.")
            continue

        markets_df = pd.DataFrame(parsed)
        print(f"  Parsed {len(markets_df)} markets")
        print(f"  Game dates: {markets_df['game_date'].unique()}")

        # Train model
        print(f"\nTraining playoff model for {stat_type}...")
        features_df = pd.read_csv(
            os.path.join(DATA_DIR, f"features_{stat_type}.csv"),
            parse_dates=["game_date"]
        )
        model, residual_map, train_df = train_playoff_model(stat_type)
        feature_cols = FEATURE_COLS[stat_type]

        # Generate predictions for each market
        predictions = []
        for _, market in markets_df.iterrows():
            latest = get_latest_features(
                market["player_name"], market["game_date"],
                stat_type, features_df
            )

            if latest is None:
                continue

            if latest[feature_cols].isna().any():
                continue

            X = latest[feature_cols].values.reshape(1, -1)
            pred = float(np.clip(model.predict(X)[0], 0, None))

            player_id = int(latest["player_id"]) if "player_id" in latest else -1
            prob = predict_prob(pred, market["line"], player_id, stat_type, residual_map)
            edge = prob - market["kalshi_prob"]

            predictions.append({
                "stat":             stat_type,
                "player":           market["player_name"],
                "game_date":        market["game_date"],
                "line":             market["line"],
                "model_prediction": round(pred, 2),
                "model_prob":       round(prob, 3),
                "kalshi_prob":      round(market["kalshi_prob"], 3),
                "edge":             round(edge, 3),
                "ticker":           market["ticker"],
            })

        if not predictions:
            print("  No predictions generated — players may not have recent feature data.")
            continue

        pred_df = pd.DataFrame(predictions).sort_values("edge", ascending=False)
        all_predictions.append(pred_df)

        print(f"\n  {stat_type.upper()} Predictions:")
        print(f"  {'Player':<25} {'Line':>6} {'Pred':>6} {'Model%':>8} {'Kalshi%':>8} {'Edge':>7}")
        print(f"  {'-'*65}")
        for _, row in pred_df.iterrows():
            marker = " ◄" if row["edge"] > 0.07 else ""
            print(f"  {row['player']:<25} {row['line']:>6.1f} {row['model_prediction']:>6.1f} "
                  f"{row['model_prob']:>8.3f} {row['kalshi_prob']:>8.3f} {row['edge']:>7.3f}{marker}")

    if all_predictions:
        combined = pd.concat(all_predictions, ignore_index=True)
        out_path = os.path.join(DATA_DIR, "finals_predictions.csv")
        combined.to_csv(out_path, index=False)
        print(f"\n\nAll predictions saved → {out_path}")

        positive_edge = combined[combined["edge"] > 0.07].sort_values("edge", ascending=False)
        if not positive_edge.empty:
            print(f"\n{'='*60}")
            print(f"POSITIVE EDGE MARKETS (edge > 0.07):")
            print(f"{'='*60}")
            print(positive_edge[["stat", "player", "game_date", "line",
                                "model_prediction", "model_prob",
                                "kalshi_prob", "edge"]].to_string(index=False))

            best_bets = positive_edge[
                (positive_edge["edge"] > 0.12) &
                (positive_edge["model_prob"] > 0.75)
            ].sort_values("edge", ascending=False)

            print(f"\n{'='*60}")
            print(f"⚑  BEST BETS")
            print(f"{'='*60}")
            if best_bets.empty:
                print("  No markets meet both criteria tonight.")
            else:
                for _, row in best_bets.iterrows():
                    print(f"  {row['player']} {row['stat']} over {row['line']} "
                        f"| model {row['model_prob']:.0%} | edge +{row['edge']:.3f}")
        else:
            print("\nNo markets with edge > 0.07 found.")


if __name__ == "__main__":
    run()
