"""
nba_fetcher.py — Pulls player game logs and team defensive ratings
from nba_api and stores them in SQLite.

Usage:
    python nba_fetcher.py                  # fetch all configured seasons
    python nba_fetcher.py --season 2024-25 # fetch a specific season
"""

import time
import argparse
import sqlite3
import pandas as pd
from tqdm import tqdm

from nba_api.stats.endpoints import playergamelogs, leaguedashteamstats
from nba_api.stats.static import players

from nba_api.stats.library.http import NBAStatsHTTP
NBAStatsHTTP.HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

from db import get_connection, init_db

# Seasons to pull by default — adjust range as needed
DEFAULT_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

# nba_api is rate-limited — sleep between requests to avoid 429s
REQUEST_DELAY = 0.8  # seconds


def fetch_player_gamelogs(season: str, conn: sqlite3.Connection, season_type: str = "Regular Season") -> int:
    """
    Fetch all player game logs for a given season and upsert into DB.
    Returns number of rows inserted/updated.
    """
    print(f"  Fetching player game logs for {season} ({season_type})...")

    try:
        logs = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable=season_type,
        )
        time.sleep(REQUEST_DELAY)

        df = logs.get_data_frames()[0]
    except Exception as e:
        print(f"  ERROR fetching game logs for {season}: {e}")
        return 0

    # Normalize columns
    df = df.rename(columns={
        "PLAYER_ID":    "player_id",
        "PLAYER_NAME":  "player_name",
        "GAME_ID":      "game_id",
        "GAME_DATE":    "game_date",
        "MATCHUP":      "matchup",
        "WL":           "wl",
        "MIN":          "min",
        "PTS":          "pts",
        "REB":          "reb",
        "AST":          "ast",
        "STL":          "stl",
        "BLK":          "blk",
        "FG_PCT":       "fg_pct",
        "FG3_PCT":      "fg3_pct",
        "FT_PCT":       "ft_pct",
        "PLUS_MINUS":   "plus_minus",
    })

    # Derive helper columns
    df["season"] = season
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    df["is_home"] = df["matchup"].apply(lambda m: 1 if "vs." in m else 0)
    df["opponent_abbr"] = df["matchup"].apply(
        lambda m: m.split("vs. ")[-1] if "vs." in m else m.split("@ ")[-1]
    )

    keep = [
        "player_id", "player_name", "season", "game_id", "game_date",
        "matchup", "wl", "min", "pts", "reb", "ast", "stl", "blk",
        "fg_pct", "fg3_pct", "ft_pct", "plus_minus", "is_home", "opponent_abbr",
    ]
    df = df[keep].dropna(subset=["pts", "reb", "ast"])

    cursor = conn.cursor()
    inserted = 0
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO player_gamelogs
                (player_id, player_name, season, game_id, game_date, matchup,
                 wl, min, pts, reb, ast, stl, blk, fg_pct, fg3_pct, ft_pct,
                 plus_minus, is_home, opponent_abbr)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                int(row.player_id), row.player_name, row.season,
                row.game_id, row.game_date, row.matchup,
                row.wl, row.get("min"), row.pts, row.reb, row.ast,
                row.get("stl"), row.get("blk"), row.get("fg_pct"),
                row.get("fg3_pct"), row.get("ft_pct"), row.get("plus_minus"),
                int(row.is_home), row.opponent_abbr,
            ))
            inserted += 1
        except Exception as e:
            print(f"  Row insert error: {e}")

    conn.commit()
    print(f"  {inserted} rows upserted for {season}.")
    return inserted


def fetch_team_def_ratings(season: str, conn: sqlite3.Connection) -> int:
    """
    Fetch team-level defensive ratings for a season and store them.
    """
    print(f"  Fetching team defensive ratings for {season}...")

    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Defense",
            per_mode_detailed="PerGame",
        )
        time.sleep(REQUEST_DELAY)
        df = stats.get_data_frames()[0]
    except Exception as e:
        print(f"  ERROR fetching team def ratings for {season}: {e}")
        return 0

    df = df.rename(columns={
        "TEAM_ID":    "team_id",
        "TEAM_NAME":  "team_abbr",
        "DEF_RATING": "def_rating",
        "OPP_PTS_PAINT": "opp_pts_paint",
        "OPP_PTS_FB": "opp_pts_fb",
    })
    df["season"] = season

    cursor = conn.cursor()
    inserted = 0
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO team_def_ratings
                (team_id, team_abbr, season, def_rating, opp_pts_paint, opp_pts_fb)
                VALUES (?,?,?,?,?,?)
            """, (
                int(row.team_id), row.team_abbr, row.season,
                row.get("def_rating"), row.get("opp_pts_paint"), row.get("opp_pts_fb"),
            ))
            inserted += 1
        except Exception as e:
            print(f"  Row insert error: {e}")

    conn.commit()
    print(f"  {inserted} teams upserted for {season}.")
    return inserted


def run(seasons: list[str]):
    init_db()
    conn = get_connection()

    for season in tqdm(seasons, desc="Seasons"):
        fetch_player_gamelogs(season, conn)
        fetch_player_gamelogs(season, conn, season_type="Playoffs")
        fetch_team_def_ratings(season, conn)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=str, default=None,
                        help="Single season to fetch, e.g. '2023-24'")
    args = parser.parse_args()

    seasons = [args.season] if args.season else DEFAULT_SEASONS
    run(seasons)
