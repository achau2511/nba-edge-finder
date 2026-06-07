"""
push_to_supabase.py — Pushes predictions and market prices to Supabase.

Run after model retrain to update the web dashboard:
    python push_to_supabase.py

Requires in .env:
    SUPABASE_URL=https://rlzcwgptguitmzzhrgmt.supabase.co
    SUPABASE_SERVICE_KEY=your_service_role_key
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates",
}


def upsert(table, rows):
    """Upsert rows into a Supabase table."""
    if not rows:
        print(f"No rows to upsert into {table}.")
        return
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=HEADERS,
        json=rows,
        timeout=15
    )
    if resp.status_code in (200, 201):
        print(f"✅ Upserted {len(rows)} rows into {table}.")
    else:
        print(f"❌ Error upserting into {table}: {resp.status_code} {resp.text}")


def delete_table(table):
    """Delete all rows from a table before reinserting."""
    resp = requests.delete(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers={**HEADERS, "Prefer": "return=minimal"},
        params={"id": "gte.0"},
        timeout=15
    )
    if resp.status_code in (200, 204):
        print(f"🗑️  Cleared {table}.")
    else:
        print(f"⚠️  Could not clear {table}: {resp.status_code} {resp.text}")


def push_predictions():
    """Generate and push model predictions to Supabase for both Kalshi and Polymarket lines."""
    import importlib.util, sys, numpy as np
    sys.path.insert(0, os.path.dirname(__file__))

    spec = importlib.util.spec_from_file_location(
        "finals_predictions",
        os.path.join(os.path.dirname(__file__), "finals_predictions.py")
    )
    fp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fp)

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

    KALSHI_SERIES = {
        "KXNBAPTS": "points",
        "KXNBAREB": "rebounds",
        "KXNBAAST": "assists",
        "KXNBA3PT": "threes",
    }

    TEAMS = {
        "Jalen Brunson": "NYK", "OG Anunoby": "NYK", "Karl-Anthony Towns": "NYK",
        "Mikal Bridges": "NYK", "Josh Hart": "NYK", "Mitchell Robinson": "NYK",
        "Miles McBride": "NYK", "Landry Shamet": "NYK", "Julian Champagnie": "SAS",
        "Jose Alvarado": "NYK", "Victor Wembanyama": "SAS", "Stephon Castle": "SAS",
        "De'Aaron Fox": "SAS", "Devin Vassell": "SAS", "Keldon Johnson": "SAS",
        "Dylan Harper": "SAS", "Zach Collins": "SAS", "Luke Kornet": "SAS",
    }

    # Collect all markets to predict: Kalshi + Polymarket
    # Key: (player_name, stat_type, line, game_date)
    markets_to_predict: dict = {}

    # 1. Kalshi markets
    for series_ticker, stat_type in KALSHI_SERIES.items():
        markets = fp.fetch_active_markets(series_ticker)
        parsed = [fp.parse_active_market(m, stat_type) for m in markets]
        for p in parsed:
            if p:
                key = (p["player_name"], stat_type, p["line"], p["game_date"])
                markets_to_predict[key] = p

    # 2. Polymarket markets
    try:
        from polymarket_fetcher import fetch_polymarket_props
        poly_props = fetch_polymarket_props()

        # Get game date from Kalshi
        game_date = None
        for series_ticker in KALSHI_SERIES:
            try:
                resp = requests.get(
                    "https://external-api.kalshi.com/trade-api/v2/markets",
                    params={"series_ticker": series_ticker, "status": "open", "limit": 1},
                    timeout=5
                )
                markets_data = resp.json().get("markets", [])
                if markets_data:
                    from datetime import datetime, timedelta
                    occ = markets_data[0].get("occurrence_datetime", "")
                    if occ:
                        utc_dt = datetime.fromisoformat(occ.replace("Z", "+00:00"))
                        et_dt = utc_dt - timedelta(hours=4)
                        game_date = et_dt.strftime("%Y-%m-%d")
                        break
            except Exception:
                pass

        if not game_date:
            from datetime import datetime
            game_date = datetime.now().strftime("%Y-%m-%d")

        for prop in poly_props:
            stat_type = prop["stat"]
            key = (prop["player"], stat_type, prop["line"], game_date)
            if key not in markets_to_predict:
                markets_to_predict[key] = {
                    "player_name": prop["player"],
                    "stat_type": stat_type,
                    "line": prop["line"],
                    "game_date": game_date,
                }
    except Exception as e:
        print(f"Warning: Could not fetch Polymarket props: {e}")

    # Generate predictions for all unique markets
    all_preds = []
    stat_models: dict = {}

    for (player_name, stat_type, line, game_date), market in markets_to_predict.items():
        # Load model once per stat type
        if stat_type not in stat_models:
            try:
                model, residual_map, _ = fp.train_playoff_model(stat_type)
                features_df = pd.read_csv(
                    os.path.join(DATA_DIR, f"features_{stat_type}.csv"),
                    parse_dates=["game_date"]
                )
                stat_models[stat_type] = (model, residual_map, features_df)
            except Exception as e:
                print(f"Could not load model for {stat_type}: {e}")
                continue

        model, residual_map, features_df = stat_models[stat_type]
        feature_cols = fp.FEATURE_COLS[stat_type]

        latest = fp.get_latest_features(player_name, game_date, stat_type, features_df)
        if latest is None:
            continue
        if latest[feature_cols].isna().any():
            continue

        X = latest[feature_cols].values.reshape(1, -1)
        pred = float(np.clip(model.predict(X)[0], 0, None))
        player_id = int(latest["player_id"]) if "player_id" in latest else -1
        prob = fp.predict_prob(pred, line, player_id, stat_type, residual_map)

        all_preds.append({
            "player":     player_name,
            "team":       TEAMS.get(player_name, "—"),
            "stat":       stat_type,
            "line":       float(line),
            "prediction": round(float(pred), 1),
            "model_prob": round(float(prob), 4),
            "game_date":  game_date,
        })

    if all_preds:
        delete_table("predictions")
        upsert("predictions", all_preds)
        print(f"   ({len(all_preds)} total predictions across Kalshi + Polymarket lines)")
    else:
        print("No predictions generated.")


def push_kalshi_prices():
    """Fetch and push Kalshi prices to Supabase."""
    import time

    KALSHI_SERIES = {
        "KXNBAPTS": "points",
        "KXNBAREB": "rebounds",
        "KXNBAAST": "assists",
        "KXNBA3PT": "threes",
    }

    rows = []
    for series_ticker, stat_type in KALSHI_SERIES.items():
        try:
            resp = requests.get(
                "https://external-api.kalshi.com/trade-api/v2/markets",
                params={"limit": 200, "series_ticker": series_ticker, "status": "open"},
                timeout=10
            )
            resp.raise_for_status()
            markets = resp.json().get("markets", [])
            for m in markets:
                title = m.get("title", "")
                if ":" not in title:
                    continue
                player = title.split(":")[0].strip()
                line = m.get("floor_strike")
                yes_ask = m.get("yes_ask_dollars")
                yes_bid = m.get("yes_bid_dollars")
                if not all([line, yes_ask, yes_bid]):
                    continue
                price = (float(yes_ask) + float(yes_bid)) / 2.0
                if price == 0:
                    continue
                rows.append({
                    "player": player,
                    "stat": stat_type,
                    "line": float(line),
                    "market": "kalshi",
                    "price": round(price, 4),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
            time.sleep(0.3)
        except Exception as e:
            print(f"Error fetching Kalshi {series_ticker}: {e}")

    if rows:
        delete_table("market_prices")
        upsert("market_prices", rows)
    else:
        print("No Kalshi prices fetched.")


def push_polymarket_prices():
    """Fetch and push Polymarket prices to Supabase."""
    try:
        from polymarket_fetcher import fetch_polymarket_props
        props = fetch_polymarket_props()
        if not props:
            print("No Polymarket props fetched.")
            return
        rows = [{
            "player":      p["player"],
            "stat":        p["stat"],
            "line":        float(p["line"]),
            "market":      "polymarket",
            "price":       round(float(p["over_price"]), 4),
            "under_price": round(float(p["under_price"]), 4),
            "updated_at":  datetime.now(timezone.utc).isoformat(),
        } for p in props]
        # Delete existing Polymarket rows then reinsert
        requests.delete(
            f"{SUPABASE_URL}/rest/v1/market_prices",
            headers={**HEADERS, "Prefer": "return=minimal"},
            params={"market": "eq.polymarket"},
            timeout=15
        )
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/market_prices",
            headers=HEADERS,
            json=rows,
            timeout=15
        )
        if resp.status_code in (200, 201):
            print(f"✅ Upserted {len(rows)} Polymarket rows into market_prices.")
        else:
            print(f"❌ Error: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Error fetching Polymarket prices: {e}")


if __name__ == "__main__":
    print("Pushing predictions to Supabase...")
    push_predictions()
    print("\nPushing Kalshi prices...")
    push_kalshi_prices()
    print("\nPushing Polymarket prices...")
    push_polymarket_prices()
    print("\nDone.")