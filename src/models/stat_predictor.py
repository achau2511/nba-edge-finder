"""
stat_predictor.py — Trains XGBoost regression models to predict player stats.

One model per stat type (points, rebounds, assists).
Uses walk-forward validation: train on seasons N through N+2, test on N+3.

Usage:
    python stat_predictor.py                  # train all three models
    python stat_predictor.py --stat points    # train one model
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")

STATS = ["points", "rebounds", "assists"]

# Walk-forward splits: (train_seasons, test_season)
SPLITS = [
    (["2020-21", "2021-22", "2022-23"], "2023-24"),
    (["2021-22", "2022-23", "2023-24"], "2024-25"),
]

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


def load_data(stat: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"features_{stat}.csv")
    df = pd.read_csv(path, parse_dates=["game_date"])
    return df


def train_and_evaluate(stat: str):
    print(f"\n{'='*50}")
    print(f"Stat: {stat.upper()}")
    print(f"{'='*50}")

    df = load_data(stat)
    features = FEATURE_COLS[stat]

    all_preds = []

    for train_seasons, test_season in SPLITS:
        print(f"\n  Train: {train_seasons} | Test: {test_season}")

        train_df = df[df["season"].isin(train_seasons)].copy()
        test_df  = df[df["season"] == test_season].copy()

        # Drop rows with NaN in any feature
        train_df = train_df.dropna(subset=features + ["target"])
        test_df  = test_df.dropna(subset=features + ["target"])

        X_train = train_df[features]
        y_train = train_df["target"]
        X_test  = test_df[features]
        y_test  = test_df["target"]

        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,   # prevent overfitting on small player samples
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None)  # stats can't be negative

        mae  = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")

        test_df = test_df.copy()
        test_df["prediction"] = preds
        test_df["residual"]   = y_test.values - preds
        all_preds.append(test_df)

    # ── Train final model on all seasons for deployment ──
    print(f"\n  Training final model on all seasons...")
    all_seasons = df["season"].unique().tolist()
    full_df = df.dropna(subset=features + ["target"])

    X_full = full_df[features]
    y_full = full_df["target"]

    final_model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    final_model.fit(X_full, y_full, verbose=False)

    # ── Save model ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"xgb_{stat}.joblib")
    joblib.dump(final_model, model_path)
    print(f"  Model saved → {model_path}")

    # ── Save feature importances ──
    importances = pd.Series(
        final_model.feature_importances_, index=features
    ).sort_values(ascending=False)
    print(f"\n  Top 10 features:")
    print(importances.head(10).to_string())

    # ── Save predictions for prob converter ──
    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_path = os.path.join(DATA_DIR, f"predictions_{stat}.csv")
    preds_df[["player_id", "player_name", "season", "game_date",
              "opponent_abbr", "target", "prediction", "residual"]].to_csv(
        preds_path, index=False
    )
    print(f"\n  Predictions saved → {preds_path}")

    return final_model


def run(stats: list[str]):
    os.makedirs(MODEL_DIR, exist_ok=True)
    for stat in stats:
        train_and_evaluate(stat)
    print("\nAll models trained.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", type=str, default=None,
                        choices=STATS,
                        help="Single stat to train. Omit to train all.")
    args = parser.parse_args()

    stats = [args.stat] if args.stat else STATS
    run(stats)
