"""
engineer.py — Builds look-ahead-safe features for each player game.

Design principles:
- All rolling features use only data strictly BEFORE the game being predicted.
  We sort by date and use shift(1) before rolling to guarantee this.
- Features are built per player, then concatenated.
- Output is saved to data/processed/features_{stat}.csv for each stat type.

Usage:
    python engineer.py
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from db import get_connection

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
WINDOWS = [5, 10, 20]  # rolling window sizes in games


# ── Helpers ──────────────────────────────────────────────────────────────────

def season_to_year(season: str) -> int:
    """'2023-24' → 2023"""
    return int(season.split("-")[0])


def days_rest(dates: pd.Series) -> pd.Series:
    """
    Compute days since last game for each row.
    Returns NaN for a player's first game.
    """
    return dates.diff().dt.days


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Look-ahead-safe rolling mean: shift(1) ensures the current game's value
    is never included in its own feature.
    """
    return series.shift(1).rolling(window, min_periods=max(1, window // 2)).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=max(2, window // 2)).std()


# ── Core feature builder ──────────────────────────────────────────────────────

def build_features_for_stat(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    """
    Given the full game log dataframe, build features predicting `stat`
    (one of: 'pts', 'reb', 'ast').

    Returns a dataframe with one row per player-game, with features and target.
    """
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # Join opponent defensive rating — match on opponent_abbr + season
    # team_abbr in def_ratings is actually team_name (full name) due to API limitation
    # We join on team_id via a workaround: use opponent_abbr from matchup
    # For now we join on season-level def_rating using a separate query
    conn = get_connection()
    def_ratings = pd.read_sql("""
        SELECT team_abbr, season, def_rating, opp_pts_paint, opp_pts_fb
        FROM team_def_ratings
    """, conn)
    conn.close()

    from team_mapping import TEAM_NAME_TO_ABBR
    def_ratings["team_abbr"] = def_ratings["team_abbr"].map(TEAM_NAME_TO_ABBR)
    def_ratings = def_ratings.dropna(subset=["team_abbr"])

    records = []

    for player_id, player_df in df.groupby("player_id"):
        p = player_df.sort_values("game_date").reset_index(drop=True)

        # ── Rolling stat features ──
        for w in WINDOWS:
            p[f"{stat}_mean_{w}g"] = rolling_mean(p[stat], w)
            p[f"{stat}_std_{w}g"]  = rolling_std(p[stat], w)

        # ── Rolling minutes (proxy for role stability) ──
        for w in WINDOWS:
            p[f"min_mean_{w}g"] = rolling_mean(p["min"], w)

        # ── Rest and schedule ──
        p["days_rest"] = days_rest(p["game_date"])
        p["days_rest"] = p["days_rest"].clip(0, 14)  # cap extreme outliers

        # ── Back-to-back flag ──
        p["is_b2b"] = (p["days_rest"] == 1).astype(int)

        # ── Home/away already in data ──

        # ── Season progression (game number within season) ──
        p["game_num_in_season"] = p.groupby("season").cumcount() + 1

        # ── Recent form: last 3 games trend (slope of stat) ──
        def rolling_slope(series, window=3):
            def slope(x):
                if len(x) < 2:
                    return np.nan
                return np.polyfit(range(len(x)), x, 1)[0]
            return series.shift(1).rolling(window, min_periods=2).apply(slope, raw=True)

        p[f"{stat}_trend_3g"] = rolling_slope(p[stat], window=3)

        # ── Opponent defensive rating ──
        # Join season-level def_rating for the opponent
        # Note: team_abbr in def_ratings = full team name, opponent_abbr = abbreviation
        # We do a best-effort fuzzy match via season only for now;
        # a future improvement is to map abbr → full name
        p = p.merge(
            def_ratings[["team_abbr", "season", "def_rating", "opp_pts_paint", "opp_pts_fb"]],
            left_on=["opponent_abbr", "season"],
            right_on=["team_abbr", "season"],
            how="left",
        ).drop(columns=["team_abbr"])

        records.append(p)

    result = pd.concat(records, ignore_index=True)

    # ── Drop rows with no target or missing rolling features ──
    result = result.dropna(subset=[stat])

    # ── Label the target ──
    result = result.rename(columns={stat: "target"})

    # ── Keep only relevant columns ──
    feature_cols = (
        [f"{stat}_mean_{w}g" for w in WINDOWS] +
        [f"{stat}_std_{w}g"  for w in WINDOWS] +
        [f"min_mean_{w}g"    for w in WINDOWS] +
        [f"{stat}_trend_3g"] +
        ["days_rest", "is_b2b", "is_home", "game_num_in_season",
         "def_rating", "opp_pts_paint", "opp_pts_fb"]
    )

    meta_cols = ["player_id", "player_name", "season", "game_date",
                 "game_id", "opponent_abbr", "target"]

    keep = meta_cols + [c for c in feature_cols if c in result.columns]
    result = result[keep]

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading game logs...")
    conn = get_connection()
    df = pd.read_sql("""
        SELECT player_id, player_name, season, game_id, game_date,
               matchup, min, pts, reb, ast, is_home, opponent_abbr
        FROM player_gamelogs
        WHERE min IS NOT NULL AND min > 0
    """, conn)
    conn.close()
    print(f"  {len(df):,} rows loaded.")

    for stat, label in [("pts", "points"), ("reb", "rebounds"), ("ast", "assists")]:
        print(f"\nBuilding features for {label}...")
        features = build_features_for_stat(df, stat)
        out_path = os.path.join(OUTPUT_DIR, f"features_{label}.csv")
        features.to_csv(out_path, index=False)
        print(f"  {len(features):,} rows → {out_path}")
        print(f"  Columns: {[c for c in features.columns if c not in ['player_id','player_name','season','game_date','game_id','opponent_abbr','target']]}")

    print("\nDone.")


if __name__ == "__main__":
    run()
