"""
app.py — NBA Finals Live Predictions Dashboard

Streamlit app that displays live model predictions vs Kalshi market prices.
Kalshi prices auto-refresh every 10 minutes.
Full pipeline (game logs + model retrain) runs on manual button click.

Run: streamlit run app.py
"""

import os
import sys
import time
import joblib
from networkx import display
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "ingestion"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "features"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "models"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Finals Edge Finder",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace !important;
    background-color: #0d0d10 !important;
    color: #e8e4d9 !important;
}

.stApp { background-color: #0d0d10 !important; }

section[data-testid="stSidebar"] { background-color: #0a0a0d !important; }

.main-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 48px;
    letter-spacing: 0.06em;
    color: #e8e4d9;
    line-height: 1;
    margin-bottom: 4px;
}

.subtitle {
    font-size: 11px;
    color: #444;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 24px;
}

.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #00d964;
    margin-right: 6px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.metric-box {
    background: #111116;
    border: 1px solid #1e1e26;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 8px;
}

.metric-label {
    font-size: 10px;
    color: #444;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 28px;
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 0.04em;
    color: #e8e4d9;
}

.best-bet-card {
    background: #0d130d;
    border: 1px solid #1a2e1a;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}

.best-bet-card:hover { border-color: #00d964; }

.bet-player { font-size: 15px; font-weight: 500; color: #e8e4d9; }
.bet-detail { font-size: 11px; color: #444; margin: 2px 0 12px; letter-spacing: 0.08em; }

.edge-big {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 32px;
    color: #00d964;
    letter-spacing: 0.04em;
}

.stat-pill {
    display: inline-block;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 999px;
    margin-right: 4px;
    letter-spacing: 0.06em;
}

.pill-nyk { background: #3d1a00; color: #f58426; }
.pill-sas { background: #1a1a1a; color: #c0c0c0; }

.section-header {
    font-size: 10px;
    color: #333;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin: 24px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a1a1a;
}

.refresh-time {
    font-size: 10px;
    color: #333;
    letter-spacing: 0.1em;
}

.stButton > button {
    background: #00d964 !important;
    color: #0d0d10 !important;
    border: none !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    padding: 10px 24px !important;
    border-radius: 4px !important;
    cursor: pointer !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: #00b854 !important;
}

.stSelectbox > div > div {
    background: #111116 !important;
    border-color: #1e1e26 !important;
    color: #e8e4d9 !important;
    font-family: 'DM Mono', monospace !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #1e1e26 !important;
    border-radius: 6px !important;
}

.stProgress > div > div > div {
    background: #00d964 !important;
}

hr { border-color: #1a1a1a !important; }

.warning-box {
    background: #1a1200;
    border: 1px solid #3d2d00;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 12px;
    color: #c8a000;
    margin-bottom: 16px;
}

.info-box {
    background: #001a0d;
    border: 1px solid #003d1a;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 12px;
    color: #00a844;
    margin-bottom: 16px;
}
            
.best-bet-card-poly {
    background: #0d1220;
    border: 1px solid #1a2a4a;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.best-bet-card-poly:hover { border-color: #4a90d9; }
            
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "data", "models")
DB_PATH   = os.path.join(os.path.dirname(__file__), "data", "nba_edge.db")

KALSHI_SERIES = {
    "KXNBAPTS": "points",
    "KXNBAREB": "rebounds",
    "KXNBAAST": "assists",
    "KXNBA3PT": "threes",
}

FEATURE_COLS = {
    "points":   ["pts_mean_5g","pts_mean_10g","pts_mean_20g","pts_std_5g","pts_std_10g","pts_std_20g","min_mean_5g","min_mean_10g","min_mean_20g","pts_trend_3g","days_rest","is_b2b","is_home","game_num_in_season","def_rating","opp_pts_paint","opp_pts_fb"],
    "rebounds": ["reb_mean_5g","reb_mean_10g","reb_mean_20g","reb_std_5g","reb_std_10g","reb_std_20g","min_mean_5g","min_mean_10g","min_mean_20g","reb_trend_3g","days_rest","is_b2b","is_home","game_num_in_season","def_rating","opp_pts_paint","opp_pts_fb"],
    "assists":  ["ast_mean_5g","ast_mean_10g","ast_mean_20g","ast_std_5g","ast_std_10g","ast_std_20g","min_mean_5g","min_mean_10g","min_mean_20g","ast_trend_3g","days_rest","is_b2b","is_home","game_num_in_season","def_rating","opp_pts_paint","opp_pts_fb"],
    "threes": [
        "fg3m_mean_5g", "fg3m_mean_10g", "fg3m_mean_20g",
        "fg3m_std_5g",  "fg3m_std_10g",  "fg3m_std_20g",
        "min_mean_5g",  "min_mean_10g",  "min_mean_20g",
        "fg3m_trend_3g",
        "days_rest", "is_b2b", "is_home", "game_num_in_season",
        "def_rating", "opp_pts_paint", "opp_pts_fb",
    ],
}

TEAMS = {
    "Jalen Brunson":"NYK","OG Anunoby":"NYK","Karl-Anthony Towns":"NYK",
    "Mikal Bridges":"NYK","Josh Hart":"NYK","Mitchell Robinson":"NYK",
    "Miles McBride":"NYK","Landry Shamet":"NYK","Julian Champagnie":"SAS",
    "Victor Wembanyama":"SAS","Stephon Castle":"SAS","De'Aaron Fox":"SAS",
    "Devin Vassell":"SAS","Keldon Johnson":"SAS","Dylan Harper":"SAS",
    "Jose Alvarado":"NYK","Zach Collins":"SAS",
}

STAT_EMOJI = {"points":"🏀","rebounds":"💪","assists":"🎯","threes":"🎳"}
KALSHI_FEE  = 0.05
EDGE_THRESH = 0.07
BEST_EDGE   = 0.12
BEST_PROB   = 0.75
KALSHI_REFRESH_SECS = 600  # 10 minutes


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def run_nba_fetch(progress_cb):
    """Fetch latest NBA game logs for 2025-26 season."""
    try:
        from nba_fetcher import fetch_player_gamelogs, fetch_team_def_ratings
        from db import get_connection, init_db
        import sqlite3

        progress_cb(0.1, "Initializing database...")
        init_db()
        conn = get_connection()

        progress_cb(0.2, "Fetching regular season logs...")
        fetch_player_gamelogs("2025-26", conn, season_type="Regular Season")

        progress_cb(0.4, "Fetching playoff logs...")
        fetch_player_gamelogs("2025-26", conn, season_type="Playoffs")

        progress_cb(0.5, "Fetching team defensive ratings...")
        fetch_team_def_ratings("2025-26", conn)

        conn.close()
        progress_cb(0.6, "Game logs updated.")
        return True
    except Exception as e:
        st.error(f"Error fetching NBA data: {e}")
        return False


def run_feature_engineering(progress_cb):
    """Re-run feature engineering."""
    try:
        from engineer import run as engineer_run
        progress_cb(0.65, "Engineering features...")
        engineer_run()
        progress_cb(0.75, "Features ready.")
        return True
    except Exception as e:
        st.error(f"Error engineering features: {e}")
        return False


def run_playoff_model(progress_cb):
    """Retrain playoff model."""
    try:
        from playoff_model import run as model_run
        progress_cb(0.8, "Retraining playoff model...")
        model_run()
        progress_cb(0.9, "Model trained.")
        return True
    except Exception as e:
        st.error(f"Error training model: {e}")
        return False


def run_polymarket_fetch(progress_cb):
    """Re-fetch Polymarket prices."""
    try:
        from polymarket_fetcher import fetch_polymarket_props, save_polymarket_data
        progress_cb(0.93, "Fetching Polymarket prices...")
        props = fetch_polymarket_props()
        if props:
            save_polymarket_data(props)
        progress_cb(0.97, "Polymarket prices updated.")
        return True
    except Exception as e:
        st.error(f"Error fetching Polymarket data: {e}")
        return False


# ── Polymarket auto-fetcher ───────────────────────────────────────────────────

@st.cache_data(ttl=KALSHI_REFRESH_SECS)
def fetch_polymarket_markets():
    """Fetch live Polymarket prices. Cached for 10 minutes."""
    try:
        from polymarket_fetcher import fetch_polymarket_props, save_polymarket_data
        props = fetch_polymarket_props()
        if props:
            save_polymarket_data(props)
            return pd.DataFrame(props)
    except Exception as e:
        st.warning(f"Could not auto-fetch Polymarket prices: {e}")
    # Fall back to saved file
    try:
        from polymarket_data import POLYMARKET_MARKETS
        return pd.DataFrame(POLYMARKET_MARKETS)
    except ImportError:
        return pd.DataFrame()


# ── Kalshi fetcher ────────────────────────────────────────────────────────────

@st.cache_data(ttl=KALSHI_REFRESH_SECS)
def fetch_kalshi_markets():
    """Fetch active Kalshi NBA Finals markets. Cached for 10 minutes."""
    all_markets = []
    for series_ticker, stat_type in KALSHI_SERIES.items():
        try:
            cursor = None
            while True:
                params = {"limit": 200, "series_ticker": series_ticker, "status": "open"}
                if cursor:
                    params["cursor"] = cursor
                resp = requests.get(
                    "https://external-api.kalshi.com/trade-api/v2/markets",
                    params=params, timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                batch = data.get("markets", [])
                for m in batch:
                    title = m.get("title", "")
                    if ":" not in title:
                        continue
                    player_name = title.split(":")[0].strip()
                    line = m.get("floor_strike")
                    occurrence = m.get("occurrence_datetime", "")
                    from datetime import datetime, timezone, timedelta
                    if occurrence:
                        utc_dt = datetime.fromisoformat(occurrence.replace("Z", "+00:00"))
                        et_dt = utc_dt - timedelta(hours=4)  # UTC to ET
                        game_date = et_dt.strftime("%Y-%m-%d")
                    else:
                        game_date = None
                    yes_ask = m.get("yes_ask_dollars")
                    yes_bid = m.get("yes_bid_dollars")
                    if not all([line, game_date, yes_ask, yes_bid]):
                        continue
                    implied_prob = (float(yes_ask) + float(yes_bid)) / 2.0
                    if implied_prob == 0:
                        implied_prob = float(yes_ask)
                    if implied_prob == 0:
                        continue
                    all_markets.append({
                        "ticker":      m.get("ticker"),
                        "player_name": player_name,
                        "stat_type":   stat_type,
                        "line":        float(line),
                        "game_date":   game_date,
                        "kalshi_prob": implied_prob,
                        "volume":      float(m.get("volume_fp", 0)),
                    })
                cursor = data.get("cursor")
                if not cursor or not batch:
                    break
                time.sleep(0.3)
        except Exception as e:
            st.warning(f"Could not fetch {series_ticker}: {e}")
    return pd.DataFrame(all_markets) if all_markets else pd.DataFrame()


# ── Prediction generator ──────────────────────────────────────────────────────

def generate_predictions(markets_df=None):
    """Run finals_predictions pipeline and return results as dataframe."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "finals_predictions",
        os.path.join(os.path.dirname(__file__), "finals_predictions.py")
    )
    fp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fp)
    
    all_preds = []
    for series_ticker, stat_type in KALSHI_SERIES.items():
        markets = fp.fetch_active_markets(series_ticker)
        parsed = [fp.parse_active_market(m, stat_type) for m in markets]
        parsed = [p for p in parsed if p]
        if not parsed:
            continue
        model, residual_map, _ = fp.train_playoff_model(stat_type)
        features_df = pd.read_csv(
            os.path.join(DATA_DIR, f"features_{stat_type}.csv"),
            parse_dates=["game_date"]
        )
        for market in parsed:
            latest = fp.get_latest_features(
                market["player_name"], market["game_date"],
                stat_type, features_df
            )
            if latest is None:
                continue
            feature_cols = fp.FEATURE_COLS[stat_type]
            if latest[feature_cols].isna().any():
                continue
            X = latest[feature_cols].values.reshape(1, -1)
            pred = float(np.clip(model.predict(X)[0], 0, None))
            player_id = int(latest["player_id"]) if "player_id" in latest else -1
            prob = fp.predict_prob(pred, market["line"], player_id, stat_type, residual_map)
            edge = prob - market["kalshi_prob"]
            all_preds.append({
                "player":      market["player_name"],
                "team":        TEAMS.get(market["player_name"], "—"),
                "stat":        stat_type,
                "game_date":   market["game_date"],
                "line":        market["line"],
                "prediction":  round(pred, 1),
                "model_prob":  round(prob, 3),
                "kalshi_prob": round(market["kalshi_prob"], 3),
                "edge":        round(edge, 3),
                "volume":      market["volume"],
            })
    return pd.DataFrame(all_preds) if all_preds else pd.DataFrame()


def generate_polymarket_predictions():
    """Generate model predictions for Polymarket markets."""
    markets_df = fetch_polymarket_markets()
    if markets_df.empty:
        return pd.DataFrame()

    # Use same game date as active Kalshi markets
    try:
        kalshi_resp = requests.get(
            "https://external-api.kalshi.com/trade-api/v2/markets",
            params={"series_ticker": "KXNBAPTS", "status": "open", "limit": 1},
            timeout=5
        )
        kalshi_data = kalshi_resp.json()
        if kalshi_data.get("markets"):
            occurrence = kalshi_data["markets"][0].get("occurrence_datetime", "")
            from datetime import datetime, timedelta, timezone
            if occurrence:
                game_date = occurrence[:10]
            else:
                game_date = datetime.now().strftime("%Y-%m-%d")
        else:
            game_date = datetime.now().strftime("%Y-%m-%d")
    except:
        game_date = datetime.now().strftime("%Y-%m-%d")

    markets_df["game_date"] = game_date

    markets_df.rename(columns={"over_price": "kalshi_prob"}, inplace=True)

    # Reuse same prediction logic
    all_preds = []
    for stat_type in ["points", "rebounds", "assists", "threes"]:
        import importlib.util
        spec = importlib.util.spec_from_file_location("finals_predictions",
            os.path.join(os.path.dirname(__file__), "finals_predictions.py"))
        fp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fp)
        model, residual_map, _ = fp.train_playoff_model(stat_type)


        import importlib.util
        spec = importlib.util.spec_from_file_location("finals_predictions", 
            os.path.join(os.path.dirname(__file__), "finals_predictions.py"))
        fp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fp)
        model, residual_map, _ = fp.train_playoff_model(stat_type)
        features_path = os.path.join(DATA_DIR, f"features_{stat_type}.csv")
        features_df = pd.read_csv(features_path, parse_dates=["game_date"])
        feature_cols = FEATURE_COLS[stat_type]

        stat_markets = markets_df[markets_df["stat"] == stat_type]
        for _, market in stat_markets.iterrows():
            player_norm = market["player"].lower().strip()
            features_df["player_norm"] = features_df["player_name"].str.lower().str.strip()
            player_rows = features_df[
                (features_df["player_norm"] == player_norm) &
                (features_df["game_date"] < pd.to_datetime(market["game_date"]))
            ].sort_values("game_date")
            if player_rows.empty:
                continue
            latest = player_rows.iloc[-1]
            if latest[feature_cols].isna().any():
                continue
            X = latest[feature_cols].values.reshape(1, -1)
            pred = float(np.clip(model.predict(X)[0], 0, None))

            player_id = int(latest.get("player_id", -1))
            prob = fp.predict_prob(pred, market["line"], player_id, stat_type, residual_map)
            edge = prob - market["kalshi_prob"]
            all_preds.append({
                "player":      market["player"],
                "team":        market["team"],
                "stat":        stat_type,
                "game_date":   market["game_date"],
                "line":        market["line"],
                "prediction":  round(pred, 1),
                "model_prob":  round(prob, 3),
                "kalshi_prob": round(market["kalshi_prob"], 3),
                "edge":        round(edge, 3),
                "volume":      market.get("volume", 0),
            })
    return pd.DataFrame(all_preds) if all_preds else pd.DataFrame()

# ── UI helpers ────────────────────────────────────────────────────────────────

def team_pill(team):
    cls = "pill-nyk" if team == "NYK" else "pill-sas"
    return f'<span class="stat-pill {cls}">{team}</span>'


def edge_color(edge):
    if edge > 0.15: return "#00d964"
    if edge > 0.07: return "#7ab82a"
    if edge < -0.15: return "#e05555"
    if edge < 0: return "#993333"
    return "#666"


def render_best_bets(df, source="kalshi"):
    best = df[(df["edge"] > BEST_EDGE) & (df["model_prob"] > BEST_PROB)].sort_values("edge", ascending=False)
    if best.empty:
        st.markdown('<div class="warning-box">No markets meet both criteria right now (edge > 0.15, model prob > 0.75).</div>', unsafe_allow_html=True)
        return

    cols = st.columns(4)
    for i, (_, row) in enumerate(best.iterrows()):
        with cols[i % 4]:
            bg = "#0d1220" if source == "poly" else "#0d130d"
            border = "#1a2a4a" if source == "poly" else "#1a2e1a"
            hover = "#4a90d9" if source == "poly" else "#00d964"
            market_label = "POLYMARKET %" if source == "poly" else "KALSHI %"

            st.markdown(f"""
                <div style="background:{bg};border:1px solid {border};border-radius:6px;padding:16px 20px;margin-bottom:10px">
                    <div class="bet-player">{row['player']}</div>
                    <div class="bet-detail">{team_pill(row['team'])} {STAT_EMOJI[row['stat']]} over {row['line']} {row['stat']}</div>
                    <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:12px">
                        <div><div class="edge-big">+{int(row['edge']*100)}</div><div style="font-size:10px;color:#333;letter-spacing:.1em">EDGE</div></div>
                        <div style="text-align:right"><div style="font-size:20px;font-family:'Bebas Neue',sans-serif;color:#e8e4d9">{int(row['model_prob']*100)}%</div><div style="font-size:10px;color:#333;letter-spacing:.1em">MODEL PROB</div></div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px">
                        <div style="background:#0a0a0d;border:1px solid {border};border-radius:4px;padding:8px">
                            <div style="font-size:9px;color:#333;letter-spacing:.1em;margin-bottom:2px">PRED</div>
                            <div style="font-size:13px;font-weight:500">{row['prediction']}</div>
                        </div>
                        <div style="background:#0a0a0d;border:1px solid {border};border-radius:4px;padding:8px">
                            <div style="font-size:9px;color:#333;letter-spacing:.1em;margin-bottom:2px">MODEL</div>
                            <div style="font-size:13px;font-weight:500">{int(row['model_prob']*100)}%</div>
                        </div>
                        <div style="background:#0a0a0d;border:1px solid {border};border-radius:4px;padding:8px">
                            <div style="font-size:9px;color:#333;letter-spacing:.1em;margin-bottom:2px">{market_label}</div>
                            <div style="font-size:13px;font-weight:500">{int(row['kalshi_prob']*100)}%</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

def render_best_unders(df, source="kalshi"):
    best = df[(df["edge"] < -0.12) & (df["model_prob"] < 0.25)].sort_values("edge", ascending=True)
    if best.empty:
        st.markdown('<div class="warning-box">No strong under picks right now.</div>', unsafe_allow_html=True)
        return

    cols = st.columns(4)
    for i, (_, row) in enumerate(best.iterrows()):
        bg = "#0d1220" if source == "poly" else "#1a0d0d"
        border = "#1a2a4a" if source == "poly" else "#2e1a1a"
        market_label = "POLYMARKET %" if source == "poly" else "KALSHI %"

        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:6px;padding:16px 20px;margin-bottom:10px">
                <div class="bet-player">{row['player']}</div>
                <div class="bet-detail">{team_pill(row['team'])} {STAT_EMOJI[row['stat']]} under {row['line']} {row['stat']}</div>
                <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:12px">
                    <div><div style="font-family:'Bebas Neue',sans-serif;font-size:32px;color:#e05555;letter-spacing:.04em">{int(row['edge']*100)}</div><div style="font-size:10px;color:#333;letter-spacing:.1em">EDGE</div></div>
                    <div style="text-align:right"><div style="font-size:20px;font-family:'Bebas Neue',sans-serif;color:#e8e4d9">{int(row['model_prob']*100)}%</div><div style="font-size:10px;color:#333;letter-spacing:.1em">MODEL PROB</div></div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px">
                    <div style="background:#0a0a0d;border:1px solid {border};border-radius:4px;padding:8px">
                        <div style="font-size:9px;color:#333;letter-spacing:.1em;margin-bottom:2px">PRED</div>
                        <div style="font-size:13px;font-weight:500">{row['prediction']}</div>
                    </div>
                    <div style="background:#0a0a0d;border:1px solid {border};border-radius:4px;padding:8px">
                        <div style="font-size:9px;color:#333;letter-spacing:.1em;margin-bottom:2px">MODEL</div>
                        <div style="font-size:13px;font-weight:500">{int(row['model_prob']*100)}%</div>
                    </div>
                    <div style="background:#0a0a0d;border:1px solid {border};border-radius:4px;padding:8px">
                        <div style="font-size:9px;color:#333;letter-spacing:.1em;margin-bottom:2px">{market_label}</div>
                        <div style="font-size:13px;font-weight:500">{int(row['kalshi_prob']*100)}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_markets_table(df, stat_filter, edge_only, show_stat_col=False, market_label="Kalshi %"):
    filtered = df.copy()
    if stat_filter != "All":
        filtered = filtered[filtered["stat"] == stat_filter.lower()]
    if edge_only:
        filtered = filtered[filtered["edge"] > EDGE_THRESH]
    filtered["score"] = filtered["model_prob"] * filtered["edge"]
    filtered["sort_key"] = filtered.apply(
        lambda r: r["score"] if r["edge"] > 0.05 else -1 + r["model_prob"] * 0.01,
        axis=1
    )
    filtered = filtered.sort_values("sort_key", ascending=False).drop(columns=["score", "sort_key"])

    if filtered.empty:
        st.markdown('<div class="warning-box">No markets match the current filters.</div>', unsafe_allow_html=True)
        return

    cols = ["player","team","line","prediction","model_prob","kalshi_prob","edge"]
    col_names = ["Player","Team","Line","Model Pred","Model %",market_label,"Edge"]
    if show_stat_col:
        cols = ["player","team","stat","line","prediction","model_prob","kalshi_prob","edge"]
        col_names = ["Player","Team","Stat","Line","Model Pred","Model %",market_label,"Edge"]
    display = filtered[cols].copy()
    display.columns = col_names
    display["Line"] = display["Line"].apply(lambda x: f"{float(x):.1f}")
    def style_table(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)

        # Team colors
        for i, team in enumerate(df["Team"]):
            if team == "NYK":
                styles.iloc[i, df.columns.get_loc("Player")] = "color: #f58426; font-weight: 500"
                styles.iloc[i, df.columns.get_loc("Team")] = "color: #f58426"
            elif team == "SAS":
                styles.iloc[i, df.columns.get_loc("Player")] = "color: #c0c0c0; font-weight: 500"
                styles.iloc[i, df.columns.get_loc("Team")] = "color: #c0c0c0"

        # Edge colors
        for i, edge in enumerate(df["Edge"]):
            val = float(str(edge).replace("+",""))
            if val > 15:
                styles.iloc[i, df.columns.get_loc("Edge")] = "color: #00d964; font-weight: 500"
            elif val > 7:
                styles.iloc[i, df.columns.get_loc("Edge")] = "color: #7ab82a; font-weight: 500"
            elif val < -15:
                styles.iloc[i, df.columns.get_loc("Edge")] = "color: #e05555; font-weight: 500"
            elif val < 0:
                styles.iloc[i, df.columns.get_loc("Edge")] = "color: #993333"

        # Model % color scale (low=red, mid=yellow, high=green)
        for i, val in enumerate(df["Model %"]):
            pct = int(str(val).replace("%",""))
            if pct >= 80:
                styles.iloc[i, df.columns.get_loc("Model %")] = "color: #00d964; font-weight: 500"
            elif pct >= 60:
                styles.iloc[i, df.columns.get_loc("Model %")] = "color: #c8f250"
            elif pct >= 40:
                styles.iloc[i, df.columns.get_loc("Model %")] = "color: #f0c040"
            else:
                styles.iloc[i, df.columns.get_loc("Model %")] = "color: #e05555"

        # Model Pred color scale relative to line
        for i, (pred, line) in enumerate(zip(display["Model Pred"], filtered["line"])):
            diff = float(pred) - float(line)
            if diff > 3:
                styles.iloc[i, df.columns.get_loc("Model Pred")] = "color: #00d964; font-weight: 500"
            elif diff > 0:
                styles.iloc[i, df.columns.get_loc("Model Pred")] = "color: #7ab82a"
            elif diff > -3:
                styles.iloc[i, df.columns.get_loc("Model Pred")] = "color: #f0c040"
            else:
                styles.iloc[i, df.columns.get_loc("Model Pred")] = "color: #e05555"

        return styles

    display["Model %"] = (filtered["model_prob"] * 100).round(0).astype(int).astype(str) + "%"
    display[market_label] = (filtered["kalshi_prob"] * 100).round(0).astype(int).astype(str) + "%"
    display["Edge"] = filtered["edge"].apply(lambda x: f"+{int(x*100)}" if x > 0 else str(int(x*100)))
    display["Model Pred"] = filtered["prediction"].astype(str)

    styled = display.style.apply(style_table, axis=None)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(600, 40 + len(display) * 35),
    )

# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    with st.spinner("Fetching markets and generating predictions..."):
        predictions_df = generate_predictions()
        poly_df = generate_polymarket_predictions()

    game_dates = predictions_df["game_date"].unique()
    next_game = sorted(game_dates)[0] if len(game_dates) > 0 else "Unknown"

    # Header
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.markdown('<div class="main-title">NBA Finals Edge Finder</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Model predictions vs Kalshi & Polymarket prices · NYK vs SAS</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:#444;letter-spacing:.1em;margin-top:-8px">NEXT GAME — {next_game}</div>', unsafe_allow_html=True)


    with col_status:
        st.markdown(f"""
        <div style="text-align:right;padding-top:12px">
            <span class="live-dot"></span>
            <span style="font-size:11px;color:#00d964;letter-spacing:.1em">LIVE</span>
            <div class="refresh-time">Prices refresh every 10 min</div>
            <div class="refresh-time">{datetime.now().strftime('%H:%M:%S')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar — refresh controls
    with st.sidebar:
        st.markdown('<div style="font-family:\'Bebas Neue\',sans-serif;font-size:24px;letter-spacing:.06em;margin-bottom:16px">Controls</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:11px;color:#555;line-height:1.6;margin-bottom:16px">
        Click below after a Finals game to fetch new game logs, retrain the model, and generate fresh predictions.
        Takes 2–3 minutes.
        </div>
        """, unsafe_allow_html=True)

        if st.button("⚡ Refresh Prices Only"):
            with st.spinner("Refreshing prices..."):
                st.cache_data.clear()
            st.success("Prices refreshed.")
            st.rerun()

        st.markdown("---")

        if st.button("🔄 Refresh Predictions"):
            progress = st.progress(0)
            status = st.empty()

            def update_progress(val, msg):
                progress.progress(val)
                status.markdown(f'<div style="font-size:11px;color:#555">{msg}</div>', unsafe_allow_html=True)

            success = True
            success &= run_nba_fetch(update_progress)
            if success:
                success &= run_feature_engineering(update_progress)
            if success:
                success &= run_playoff_model(update_progress)

            update_progress(1.0, "Done.")
            time.sleep(1)
            progress.empty()
            status.empty()

            if success:
                st.success("Predictions updated.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Update failed. Check logs.")

        st.markdown("---")
        st.markdown("""
        <div style="font-size:10px;color:#333;line-height:1.8">
        <b style="color:#555">Model:</b> XGBoost trained on 2024-25 + 2025-26 playoff games<br>
        <b style="color:#555">Features:</b> Rolling averages (5/10/20 games), rest, home/away, opponent defense<br>
        <b style="color:#555">Edge:</b> Model probability − Kalshi implied probability<br>
        <b style="color:#555">Best bets:</b> Edge > 0.15 · Model prob > 0.75
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div style="font-size:10px;color:#222">For informational purposes only. Not financial advice.</div>', unsafe_allow_html=True)

    
    if predictions_df.empty:
        st.markdown('<div class="warning-box">No active Kalshi NBA markets found. Game may not be open for betting yet.</div>', unsafe_allow_html=True)
        return

    

    if predictions_df.empty:
        st.markdown('<div class="warning-box">Could not generate predictions. Run the refresh pipeline first to train the model.</div>', unsafe_allow_html=True)
        return

    # Summary metrics
    total_markets = len(predictions_df)
    positive_edge = (predictions_df["edge"] > EDGE_THRESH).sum()
    best_bets_count = ((predictions_df["edge"] > BEST_EDGE) & (predictions_df["model_prob"] > BEST_PROB)).sum()
    mean_edge = predictions_df["edge"].mean()

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Kalshi Markets Analyzed</div><div class="metric-value">{total_markets}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Kalshi Positive Edge</div><div class="metric-value" style="color:#00d964">{positive_edge}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Kalshi Best Bets</div><div class="metric-value" style="color:#00d964">{best_bets_count}</div></div>', unsafe_allow_html=True)


    poly_total = len(poly_df) if not poly_df.empty else 0
    poly_pos_edge = (poly_df["edge"] > EDGE_THRESH).sum() if not poly_df.empty else 0
    poly_best = ((poly_df["edge"] > BEST_EDGE) & (poly_df["model_prob"] > BEST_PROB)).sum() if not poly_df.empty else 0

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Polymarket Markets Analyzed</div><div class="metric-value">{poly_total}</div></div>', unsafe_allow_html=True)
    with p2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Polymarket Positive Edge</div><div class="metric-value" style="color:#00d964">{poly_pos_edge}</div></div>', unsafe_allow_html=True)
    with p3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Polymarket Best Bets</div><div class="metric-value" style="color:#00d964">{poly_best}</div></div>', unsafe_allow_html=True)



    # Tabs
    tab1, tab2, tab3 = st.tabs(["⚑ Best Bets", "📊 Kalshi Markets", "🟣 Polymarket"])

    with tab1:
        st.markdown('<div style="font-size:13px;font-weight:500;color:#e8e4d9;margin-bottom:12px;letter-spacing:.08em">KALSHI BEST BETS</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Edge > 0.12 · Model prob > 0.75 · ranked by edge</div>', unsafe_allow_html=True)
        render_best_bets(predictions_df)

        st.markdown('<div style="font-size:13px;font-weight:500;color:#e8e4d9;margin:24px 0 12px;letter-spacing:.08em">POLYMARKET BEST BETS</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Edge > 0.12 · Model prob > 0.75 · ranked by edge</div>', unsafe_allow_html=True)
        if poly_df.empty:
            st.markdown('<div class="warning-box">No Polymarket data found.</div>', unsafe_allow_html=True)
        else:
            render_best_bets(poly_df, source="poly")

        st.markdown('<div style="font-size:13px;font-weight:500;color:#e8e4d9;margin:24px 0 12px;letter-spacing:.08em">POLYMARKET BEST UNDERS</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Edge < -0.12 · Model prob < 0.25 · ranked by edge</div>', unsafe_allow_html=True)
        render_best_unders(poly_df, source="poly")

    with tab2:
        edge_only = st.checkbox("Edge > 0.07 only", value=False)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div style="font-size:13px;font-weight:500;color:#e8e4d9;margin-bottom:8px">🏀 Points</div>', unsafe_allow_html=True)
            render_markets_table(predictions_df, "Points", edge_only)
        with col2:
            st.markdown('<div style="font-size:13px;font-weight:500;color:#e8e4d9;margin-bottom:8px">💪 Rebounds</div>', unsafe_allow_html=True)
            render_markets_table(predictions_df, "Rebounds", edge_only)
        with col3:
            st.markdown('<div style="font-size:13px;font-weight:500;color:#e8e4d9;margin-bottom:8px">🎯 Assists</div>', unsafe_allow_html=True)
            render_markets_table(predictions_df, "Assists", edge_only)
        with col4:
            st.markdown('<div style="font-size:13px;font-weight:500;color:#e8e4d9;margin-bottom:8px">🎳 Threes</div>', unsafe_allow_html=True)
            render_markets_table(predictions_df, "Threes", edge_only)

        st.markdown('<div style="font-size:13px;font-weight:500;color:#e8e4d9;margin:24px 0 8px">All Kalshi Markets</div>', unsafe_allow_html=True)
        render_markets_table(predictions_df, "All", edge_only, show_stat_col=True)

    with tab3:
        st.markdown('<div class="section-header">Polymarket player props — auto-fetched via API</div>', unsafe_allow_html=True)
        if poly_df.empty:
            st.markdown('<div class="warning-box">No Polymarket data. Add polymarket_data.py to project root.</div>', unsafe_allow_html=True)
        else:
            edge_only_poly = st.checkbox("Edge > 0.07 only ", value=False, key="poly_edge")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div style="font-size:13px;font-weight:500;color:#e8e4d9;margin-bottom:8px">🏀 Points</div>', unsafe_allow_html=True)
                render_markets_table(poly_df, "Points", edge_only_poly, market_label="Polymarket %")
            with col2:
                st.markdown('<div style="font-size:13px;font-weight:500;color:#e8e4d9;margin-bottom:8px">💪 Rebounds</div>', unsafe_allow_html=True)
                render_markets_table(poly_df, "Rebounds", edge_only_poly, market_label="Polymarket %")
            with col3:
                st.markdown('<div style="font-size:13px;font-weight:500;color:#e8e4d9;margin-bottom:8px">🎯 Assists</div>', unsafe_allow_html=True)
                render_markets_table(poly_df, "Assists", edge_only_poly, market_label="Polymarket %")
            with col4:
                st.markdown('<div style="font-size:13px;font-weight:500;color:#e8e4d9;margin-bottom:8px">🎳 Threes</div>', unsafe_allow_html=True)
                render_markets_table(poly_df, "Threes", edge_only_poly, market_label="Polymarket %")
            
            st.markdown('<div style="font-size:13px;font-weight:500;color:#e8e4d9;margin:24px 0 8px">All Polymarket</div>', unsafe_allow_html=True)
            render_markets_table(poly_df, "All", edge_only_poly, show_stat_col=True, market_label="Polymarket %")
    
    # Auto-refresh timestamp
    st.markdown(f"""
    <div style="text-align:center;margin-top:32px;font-size:10px;color:#222;letter-spacing:.1em">
    Last updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Kalshi prices cached 10 min
    </div>
    """, unsafe_allow_html=True)



if __name__ == "__main__" or True:
    main()