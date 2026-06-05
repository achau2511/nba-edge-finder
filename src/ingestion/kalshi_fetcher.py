"""
kalshi_fetcher.py — Pulls historical NBA player prop markets from Kalshi's
public API (no auth required) and stores pre-game implied probabilities in SQLite.

Kalshi docs: https://docs.kalshi.com/api-reference/market/get-markets

Usage:
    python kalshi_fetcher.py
"""

import time
import requests

from db import get_connection, init_db

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

# NBA player prop series tickers
NBA_SERIES = {
    "KXNBAPTS": "points",
    "KXNBAREB": "rebounds",
    "KXNBAAST": "assists",
}

REQUEST_DELAY = 0.3


def fetch_markets_for_series(series_ticker: str, limit: int = 200) -> list[dict]:
    """
    Fetch all finalized markets under a given series ticker.
    Paginates through all results automatically.
    """
    markets = []
    cursor = None

    while True:
        params = {
            "limit": limit,
            "series_ticker": series_ticker,
            "status": "settled",
        }
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(f"{BASE_URL}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            print(f"  HTTP error for {series_ticker}: {e}")
            break
        except Exception as e:
            print(f"  Error fetching {series_ticker}: {e}")
            break

        batch = data.get("markets", [])
        markets.extend(batch)
        print(f"  Fetched {len(markets)} markets so far...", end="\r")

        cursor = data.get("cursor")
        if not cursor or not batch:
            break

        time.sleep(REQUEST_DELAY)

    print()
    return markets


def parse_market(market: dict, stat_type: str) -> dict | None:
    """
    Extract relevant fields from a Kalshi market object.

    Key design decisions:
    - line: use floor_strike (e.g. 9.5 for "10+") — cleaner than parsing title
    - implied_prob: midpoint of previous_yes_bid and previous_yes_ask
      These are the pre-game prices, before in-game movement.
      last_price_dollars reflects post-game settlement — NOT what we want.
    - result: "yes" if player exceeded the line, "no" otherwise
    - game_date: from occurrence_datetime (scheduled tip-off time)
    """
    result = market.get("result")
    if result not in ("yes", "no"):
        return None

    # Pre-game implied probability — midpoint of bid/ask before game started
    prev_bid = market.get("previous_yes_bid_dollars")
    prev_ask = market.get("previous_yes_ask_dollars")
    if prev_bid is None or prev_ask is None:
        return None

    prev_bid_f = float(prev_bid)
    prev_ask_f = float(prev_ask)

    # Skip markets with no pre-game trading activity
    if prev_bid_f == 0.0 and prev_ask_f == 0.0:
        return None

    implied_prob = (prev_bid_f + prev_ask_f) / 2.0

    # Line from floor_strike — always X.5 format (e.g. 9.5 = "10+" market)
    line = market.get("floor_strike")
    if line is None:
        return None

    # Player name — everything before the colon in title
    title = market.get("title", "")
    if ":" not in title:
        return None
    player_name = title.split(":")[0].strip()

    # Game date from scheduled tip-off
    occurrence = market.get("occurrence_datetime", "")
    game_date = occurrence[:10] if occurrence else None
    if not game_date:
        return None

    return {
        "ticker":       market.get("ticker"),
        "player_name":  player_name,
        "stat_type":    stat_type,
        "line":         float(line),
        "game_date":    game_date,
        "implied_prob": implied_prob,
        "result":       result,
        "volume":       float(market.get("volume_fp", 0)),
    }


def store_markets(markets: list[dict], stat_type: str, conn):
    cursor = conn.cursor()
    inserted = 0
    skipped = 0

    for m in markets:
        parsed = parse_market(m, stat_type)
        if not parsed:
            skipped += 1
            continue
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO kalshi_markets
                (ticker, player_name, stat_type, line, game_date, close_price, volume)
                VALUES (?,?,?,?,?,?,?)
            """, (
                parsed["ticker"],
                parsed["player_name"],
                parsed["stat_type"],
                parsed["line"],
                parsed["game_date"],
                parsed["implied_prob"],
                parsed["volume"],
            ))
            inserted += 1
        except Exception as e:
            print(f"  Insert error: {e}")

    conn.commit()
    print(f"  {inserted} markets stored, {skipped} skipped.")


def run():
    init_db()
    conn = get_connection()

    for series_ticker, stat_type in NBA_SERIES.items():
        print(f"\nFetching {series_ticker} ({stat_type})...")
        markets = fetch_markets_for_series(series_ticker)
        print(f"  Total fetched: {len(markets)}")
        store_markets(markets, stat_type, conn)

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    run()