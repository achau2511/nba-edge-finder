"""
db.py — SQLite connection and schema setup.
Run this once to initialize the database.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nba_edge.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Raw per-game player logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_gamelogs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id       INTEGER NOT NULL,
            player_name     TEXT NOT NULL,
            season          TEXT NOT NULL,       -- e.g. '2023-24'
            game_id         TEXT NOT NULL,
            game_date       TEXT NOT NULL,       -- YYYY-MM-DD
            matchup         TEXT NOT NULL,       -- e.g. 'BOS vs. MIA'
            wl              TEXT,                -- 'W' or 'L'
            min             REAL,
            pts             REAL,
            reb             REAL,
            ast             REAL,
            stl             REAL,
            blk             REAL,
            fg_pct          REAL,
            fg3_pct         REAL,
            ft_pct          REAL,
            plus_minus      REAL,
            is_home         INTEGER,             -- 1 = home, 0 = away
            opponent_abbr   TEXT NOT NULL,
            UNIQUE(player_id, game_id)
        )
    """)

    # Team defensive ratings per season (from LeagueDashTeamStats)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS team_def_ratings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id         INTEGER NOT NULL,
            team_abbr       TEXT NOT NULL,
            season          TEXT NOT NULL,
            def_rating      REAL,
            opp_pts_paint   REAL,
            opp_pts_fb      REAL,
            UNIQUE(team_id, season)
        )
    """)

    # Kalshi market data for player props
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kalshi_markets (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT NOT NULL,
            player_name     TEXT NOT NULL,
            stat_type       TEXT NOT NULL,       -- 'points', 'rebounds', 'assists'
            line            REAL NOT NULL,       -- e.g. 25.5
            game_date       TEXT NOT NULL,
            close_price     REAL,                -- 0-1, implied probability of OVER
            volume          INTEGER,
            UNIQUE(ticker, game_date)
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


if __name__ == "__main__":
    init_db()
