"""
playoff_model.py — Trains XGBoost models on playoff data only.

Train: 2024-25 full playoffs + 2025-26 rounds 1-2 (Apr 18 - May 12)
Test:  2025-26 Conference Finals (May 13 - May 30)

Usage:
    python playoff_model.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "ingestion"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "models"))

from db import get_connection
from prob_converter import build_residual_map, predict_prob, _GLOBAL_RESIDUALS

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "data", "models")

CONF_FINALS_START = "2026-05-13"
KALSHI_FEE = 0.05
EDGE_THRESHOLD = 0.07

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
}


def load_playoff_features(stat: str) -> pd.DataFrame:
    """Load features and filter to playoff games only using game_id prefix."""
    conn = get_connection()
    # Get playoff game_ids
    playoff_ids = pd.read_sql(
    "SELECT game_id FROM player_gamelogs WHERE game_id LIKE '0042%'", conn
    )["game_id"].astype(str).tolist()

    # Also handle integer version (leading zero stripped)
    playoff_ids_int = [int(g) for g in playoff_ids]
    conn.close()

    df = pd.read_csv(os.path.join(DATA_DIR, f"features_{stat}.csv"),
                     parse_dates=["game_date"])
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df = df[df["game_id"].isin(playoff_ids)].copy() 
    return df


def run():
    for stat, label in [("points", "Points"), ("rebounds", "Rebounds"), ("assists", "Assists")]:
        print(f"\n{'='*55}")
        print(f"Stat: {label.upper()} — Playoff Model")
        print(f"{'='*55}")

        df = load_playoff_features(stat)
        features = FEATURE_COLS[stat]

        print(f"  Total playoff rows: {len(df):,}")

        # Split
        train_df = df[
            (df["season"] == "2024-25") |
            ((df["season"] == "2025-26") & (df["game_date"] < CONF_FINALS_START))
        ].copy()

        test_df = df[
            (df["season"] == "2025-26") & (df["game_date"] >= CONF_FINALS_START)
        ].copy()

        train_df = train_df.dropna(subset=features + ["target"])
        test_df  = test_df.dropna(subset=features + ["target"])

        print(f"  Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

        if len(train_df) < 100 or len(test_df) < 10:
            print("  Insufficient data, skipping.")
            continue

        X_train = train_df[features]
        y_train = train_df["target"]
        X_test  = test_df[features]
        y_test  = test_df["target"]

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
        # Weight recent Finals games more heavily
        weights = np.ones(len(train_df))
        finals_mask = train_df["game_date"] >= "2026-06-01"
        conf_finals_mask = (train_df["game_date"] >= "2026-05-13") & (train_df["game_date"] < "2026-06-01")

        weights[finals_mask.values] = 5.0       # Finals games weighted 5x
        weights[conf_finals_mask.values] = 2.0  # Conference Finals weighted 2x

        model.fit(X_train, y_train, sample_weight=weights, verbose=False)

        preds = np.clip(model.predict(X_test), 0, None)
        mae  = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")

        # Save model
        model_path = os.path.join(MODEL_DIR, f"xgb_playoff_{stat}.joblib")
        joblib.dump(model, model_path)

        # Build residual map from training data
        train_df = train_df.copy()
        train_df["prediction"] = np.clip(model.predict(X_train), 0, None)
        train_df["residual"]   = y_train.values - train_df["prediction"].values
        residual_map = build_residual_map(train_df, stat)

        # Score test set against Kalshi markets
        conn = get_connection()
        kalshi_df = pd.read_sql(f"""
            SELECT ticker, player_name, stat_type, line, game_date, close_price, volume
            FROM kalshi_markets WHERE stat_type = '{stat}'
        """, conn)
        conn.close()
        kalshi_df["game_date"] = pd.to_datetime(kalshi_df["game_date"])
        kalshi_df = kalshi_df[kalshi_df["game_date"] >= CONF_FINALS_START]

        test_df["player_name_norm"] = test_df["player_name"].str.lower().str.strip()
        kalshi_df["player_name_norm"] = kalshi_df["player_name"].str.lower().str.strip()

        merged = kalshi_df.merge(
            test_df, on=["player_name_norm", "game_date"],
            how="inner", suffixes=("_kalshi", "_features")
        )

        if merged.empty:
            print("  No Kalshi markets matched for Conference Finals.")
            continue

        merged = merged.dropna(subset=features)
        merged["model_prediction"] = np.clip(model.predict(merged[features]), 0, None)
        merged["model_prob"] = merged.apply(
            lambda row: predict_prob(
                row["model_prediction"], row["line"],
                int(row["player_id"]), stat, residual_map
            ), axis=1
        )
        merged["kalshi_prob"] = merged["close_price"]
        merged["edge"] = merged["model_prob"] - merged["kalshi_prob"]

        print(f"\n  Matched {len(merged):,} Kalshi markets")
        print(f"  Mean edge: {merged['edge'].mean():.4f}")
        print(f"  Edge > {EDGE_THRESHOLD}: {(merged['edge'] > EDGE_THRESHOLD).sum()}")

        bets = merged[merged["edge"] > EDGE_THRESHOLD].copy()
        if bets.empty:
            print("  No bets above threshold.")
            continue

        bets["won"] = bets["target"] > bets["line"]
        bets["pnl"] = bets.apply(
            lambda r: (1 - r["kalshi_prob"]) * (1 - KALSHI_FEE) if r["won"] else -r["kalshi_prob"],
            axis=1
        )

        print(f"\n  Backtest ({len(bets)} bets):")
        print(f"    Win rate:  {bets['won'].mean():.3f}")
        print(f"    ROI:       {bets['pnl'].mean():.3f}")
        print(f"    Total P&L: ${bets['pnl'].sum():.2f}")
        print()
        print(bets[["player_name_norm", "game_date", "line",
                     "model_prediction", "model_prob", "kalshi_prob",
                     "edge", "won", "target"]].to_string())


if __name__ == "__main__":
    run()
