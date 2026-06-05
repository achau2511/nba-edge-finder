"""
prob_converter.py — Converts model point estimate predictions to probabilities.

Method: empirical quantile distribution per player.
For each player, we build their historical residual distribution (actual - predicted).
To compute P(stat > line), we shift the empirical residual CDF by the model's
point estimate and read off the probability directly.

This makes no distributional assumptions — no normal, no symmetry.
It captures the right-skew of NBA stats naturally.

Usage:
    python prob_converter.py --stat points --batch  # score all Kalshi markets
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from db import get_connection

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")

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

MIN_RESIDUAL_GAMES = 20
_GLOBAL_RESIDUALS: dict = {}


def load_residuals(stat: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"predictions_{stat}.csv")
    return pd.read_csv(path, parse_dates=["game_date"])


def build_residual_map(residuals_df: pd.DataFrame, stat: str) -> dict:
    global _GLOBAL_RESIDUALS
    all_residuals = residuals_df["residual"].dropna().values
    _GLOBAL_RESIDUALS[stat] = all_residuals

    residual_map = {}
    for player_id, group in residuals_df.groupby("player_id"):
        residuals = group["residual"].dropna().values
        if len(residuals) >= MIN_RESIDUAL_GAMES:
            residual_map[int(player_id)] = residuals
    return residual_map


def predict_prob(prediction, line, player_id, stat, residual_map):
    """
    P(stat > line) using empirical residual distribution.
    P(actual > line) = P(residual > line - prediction) = 1 - ECDF(line - prediction)
    """
    residuals = residual_map.get(player_id, _GLOBAL_RESIDUALS.get(stat, np.zeros(100)))
    threshold = line - prediction
    prob = np.mean(residuals > threshold)
    return float(np.clip(prob, 0.01, 0.99))


def evaluate_calibration(stat, residuals_df, residual_map):
    rows = residuals_df.dropna(subset=["prediction", "target"]).copy()
    median_line = rows["target"].median()

    probs, actuals = [], []
    for _, row in rows.iterrows():
        prob = predict_prob(row["prediction"], median_line,
                            int(row["player_id"]), stat, residual_map)
        probs.append(prob)
        actuals.append(int(row["target"] > median_line))

    probs   = np.array(probs)
    actuals = np.array(actuals)
    brier   = brier_score_loss(actuals, probs)
    print(f"  Brier score (line={median_line:.1f}): {brier:.4f}  "
          f"(baseline naive=0.25, perfect=0.0)")

    prob_df = pd.DataFrame({"prob": probs, "actual": actuals})
    prob_df["bucket"] = pd.cut(prob_df["prob"],
                                bins=[0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    print(f"\n  Calibration (predicted prob vs actual win rate):")
    print(f"    {'Bucket':<15} {'N':>6} {'Predicted':>10} {'Actual':>10}")
    for bucket, group in prob_df.groupby("bucket", observed=True):
        if len(group) == 0:
            continue
        print(f"    {str(bucket):<15} {len(group):>6} "
              f"{group['prob'].mean():>10.3f} "
              f"{group['actual'].mean():>10.3f}")
    return brier


def score_kalshi_markets(stat, residual_map):
    print(f"\nScoring Kalshi markets for {stat}...")

    features_df = pd.read_csv(
        os.path.join(DATA_DIR, f"features_{stat}.csv"),
        parse_dates=["game_date"],
    )
    model = joblib.load(os.path.join(MODEL_DIR, f"xgb_{stat}.joblib"))
    feature_cols = FEATURE_COLS[stat]

    conn = get_connection()
    kalshi_df = pd.read_sql(f"""
        SELECT ticker, player_name, stat_type, line, game_date, close_price, volume
        FROM kalshi_markets WHERE stat_type = '{stat}'
    """, conn)
    conn.close()
    kalshi_df["game_date"] = pd.to_datetime(kalshi_df["game_date"])

    features_df["player_name_norm"] = features_df["player_name"].str.lower().str.strip()
    kalshi_df["player_name_norm"]   = kalshi_df["player_name"].str.lower().str.strip()

    merged = kalshi_df.merge(features_df, on=["player_name_norm", "game_date"],
                              how="inner", suffixes=("_kalshi", "_features"))

    if merged.empty:
        print(f"  No matches found for {stat}.")
        return pd.DataFrame()

    print(f"  Matched {len(merged):,} Kalshi markets to feature rows.")
    merged = merged.dropna(subset=feature_cols)
    print(f"  {len(merged):,} rows after dropping NaN features.")

    merged["model_prediction"] = np.clip(model.predict(merged[feature_cols]), 0, None)
    merged["model_prob"] = merged.apply(
        lambda row: predict_prob(row["model_prediction"], row["line"],
                                  int(row["player_id"]), stat, residual_map), axis=1)
    merged["kalshi_prob"] = merged["close_price"]
    merged["edge"] = merged["model_prob"] - merged["kalshi_prob"]

    out_cols = ["ticker", "player_name_norm", "game_date", "line",
                "model_prediction", "model_prob", "kalshi_prob", "edge", "volume", "target"]
    result = merged[[c for c in out_cols if c in merged.columns]].copy()
    out_path = os.path.join(DATA_DIR, f"scored_{stat}.csv")
    result.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return result


def run(stat, batch=False):
    print(f"\nLoading residuals for {stat}...")
    residuals_df = load_residuals(stat)
    residual_map = build_residual_map(residuals_df, stat)
    coverage = len(residual_map) / residuals_df["player_id"].nunique()
    print(f"  Player residual coverage: {coverage:.1%} of players have {MIN_RESIDUAL_GAMES}+ games")
    evaluate_calibration(stat, residuals_df, residual_map)
    if batch:
        score_kalshi_markets(stat, residual_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", required=True, choices=["points", "rebounds", "assists"])
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()
    run(args.stat, batch=args.batch)