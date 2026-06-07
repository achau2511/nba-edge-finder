"""
polymarket_fetcher.py — Fetches live Polymarket US player prop markets.

Uses the gateway.polymarket.us search endpoint which returns full market data
including player props (points, rebounds, assists, threes, blocks).

Run: python polymarket_fetcher.py
"""

import requests
import re
from datetime import datetime

SEARCH_URL = "https://gateway.polymarket.us/v1/search"
SEARCH_QUERY = "Wembanyama points"  # Returns the current NBA Finals game event

PROP_TYPES = {
    "basketball_player_points":  "points",
    "basketball_player_rebounds": "rebounds",
    "basketball_player_assists":  "assists",
    "basketball_player_threes":   "threes",
}

TEAMS = {
    "Jalen Brunson":        "NYK",
    "OG Anunoby":           "NYK",
    "Karl-Anthony Towns":   "NYK",
    "Mikal Bridges":        "NYK",
    "Josh Hart":            "NYK",
    "Mitchell Robinson":    "NYK",
    "Miles McBride":        "NYK",
    "Landry Shamet":        "NYK",
    "Julian Champagnie":    "SAS",
    "Jose Alvarado":        "NYK",
    "Victor Wembanyama":    "SAS",
    "Stephon Castle":       "SAS",
    "De'Aaron Fox":         "SAS",
    "Devin Vassell":        "SAS",
    "Keldon Johnson":       "SAS",
    "Dylan Harper":         "SAS",
    "Zach Collins":         "SAS",
    "Luke Kornet":          "SAS",
}

# Regex: "Will {Player} record at least {N} {stat} in ..."
PATTERN = re.compile(
    r"Will (.+?) record at least (\d+(?:\.\d+)?) (.+?) in",
    re.IGNORECASE
)

STAT_KEYWORDS = {
    "points":          "points",
    "rebounds":        "rebounds",
    "assists":         "assists",
    "three pointers":  "threes",
    "3-point":         "threes",
    "threes":          "threes",
    "blocks":          "blocks",
}


def parse_question(question):
    """Parse 'Will X record at least N stat in ...' -> (player, line, stat)"""
    m = PATTERN.search(question)
    if not m:
        return None
    player = m.group(1).strip()
    line = float(m.group(2))
    stat_raw = m.group(3).strip().lower()

    stat = None
    for kw, mapped in STAT_KEYWORDS.items():
        if kw in stat_raw:
            stat = mapped
            break
    if not stat:
        return None

    return player, line, stat


def get_prices(market):
    """Extract Over and Under prices from marketSides.
    On Polymarket US: long=False/description='No' is the OVER side shown on app,
    long=True/description='Yes' is the UNDER side.
    Returns (over_price, under_price)."""
    over_price = None
    under_price = None
    for side in market.get("marketSides", []):
        price_str = side.get("price", "")
        try:
            price = round(float(price_str), 4)
        except (ValueError, TypeError):
            continue
        if side.get("description") == "No":
            over_price = price
        else:
            under_price = price
    return over_price, under_price


def fetch_polymarket_props():
    """Fetch all player prop markets from the current NBA Finals game."""
    print(f"Fetching Polymarket props via search...")
    try:
        resp = requests.get(SEARCH_URL, params={"query": SEARCH_QUERY, "limit": 5}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Error fetching: {e}")
        return []

    events = data.get("events", [])
    if not events:
        print("No events returned.")
        return []

    # Use the first active non-closed event
    target_event = None
    for e in events:
        if e.get("active") and not e.get("closed"):
            target_event = e
            break

    if not target_event:
        target_event = events[0]

    print(f"Using event: {target_event.get('title')} (ID: {target_event.get('id')})")
    markets = target_event.get("markets", [])
    print(f"Total markets: {len(markets)}")

    props = []
    seen = set()

    for m in markets:
        smt = m.get("sportsMarketType", "")
        if smt not in PROP_TYPES:
            continue

        question = m.get("question", "")
        parsed = parse_question(question)
        if not parsed:
            continue

        player, line, stat = parsed

        # Only include players we track
        if player not in TEAMS:
            continue

        # Skip blocks
        if stat == "blocks":
            continue

        key = (player, stat, line)
        if key in seen:
            continue
        seen.add(key)

        over_price, under_price = get_prices(m)
        if over_price is None or under_price is None:
            continue

        props.append({
            "player":      player,
            "team":        TEAMS[player],
            "stat":        stat,
            "line":        line - 0.5,
            "over_price":  over_price,
            "under_price": 1-under_price,
        })

    print(f"Found {len(props)} unique player prop markets.")
    return props


def save_polymarket_data(props, path="polymarket_data.py"):
    """Save fetched props to polymarket_data.py."""
    lines = [
        '"""',
        f"polymarket_data.py — Auto-generated Polymarket US player prop data.",
        f"Fetched: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}",
        '"""',
        "",
        "POLYMARKET_MARKETS = [",
    ]

    current_stat = None
    for p in sorted(props, key=lambda x: (x["stat"], x["player"], x["line"])):
        if p["stat"] != current_stat:
            current_stat = p["stat"]
            lines.append(f"    # ── {current_stat.upper()} {'─'*40}")
        lines.append(
            f'    {{"player": "{p["player"]}", "team": "{p["team"]}", '
            f'"stat": "{p["stat"]}", "line": {p["line"]}, '
            f'"over_price": {p["over_price"]}}},'
        )

    lines.append("]")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved to {path}")


if __name__ == "__main__":
    props = fetch_polymarket_props()
    if props:
        import pandas as pd
        df = pd.DataFrame(props)
        print("\nSample:")
        print(df[["player", "stat", "line", "over_price", "under_price"]].head(15).to_string(index=False))
        save_polymarket_data(props)
    else:
        print("No markets found. Event ID may have changed — check polymarket_fetcher.py.")