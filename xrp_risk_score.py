"""
XRP Risk Score — daily composite 0-1 score from Binance public OHLCV data.

Direct port of the BTC risk score methodology (same repo pattern) to XRPUSDT.
See the README caveat before trusting this blindly: the log-regression
component is a much better-established model for BTC than for XRP — read
that section before you rely on this for sizing decisions.

Components (all normalized 0-1 via expanding historical percentile rank,
so the score self-calibrates over time without hardcoded thresholds):

  1. Log-regression band position   (35%) — price vs. long-term log-log growth curve
  2. 200-day MA multiple            (25%) — price stretch vs. long-term trend
  3. RSI-14 (daily)                 (20%) — short-term overbought/oversold
  4. Volatility-adjusted momentum   (20%) — 30d return / 30d realized vol,
                                     3-day EMA smoothed pre-rank to dampen
                                     30d rolling-window edge effects

0 = cheap / accumulate harder.  1 = expensive / reduce or take profit.

Usage:
    python xrp_risk_score.py            # fetch full history, recompute, write data/xrp_risk_history.json
    python xrp_risk_score.py --update   # fetch only recent candles and append (fast daily run)
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
BINANCE_KLINES_URL_FALLBACK = "https://api.binance.com/api/v3/klines"
SYMBOL = "XRPUSDT"
INTERVAL = "1d"
DATA_DIR = Path(__file__).parent / "data"
HISTORY_FILE = DATA_DIR / "xrp_risk_history.json"
# Raw close-price cache: EVERY fetched date/close, including the pre-warmup
# rows that never get a composite_score (dropped from HISTORY_FILE by
# build_output). --update merges from THIS file, not from HISTORY_FILE, so
# the regression fit and expanding percentile ranks always see the full
# price series and match a full recompute exactly. Reconstructing prices
# from the scored output alone silently drops the earliest ~200-260 days,
# which skews the global log-regression fit (those rows anchor the low end
# of the log-days range) and can shift the composite score enough to flip
# a DCA zone near a boundary.
PRICES_FILE = DATA_DIR / "xrp_prices_raw.json"

WEIGHTS = {
    "log_regression": 0.35,
    "ma200_multiple": 0.25,
    "rsi14": 0.20,
    "vol_adj_momentum": 0.20,
}

# Earliest date we ASK Binance for — not a claim about when XRPUSDT actually
# started trading. Binance just returns whatever candles exist from here
# forward, so this only needs to be "early enough," not exact. The script
# prints the actual first returned date on every run; put that date in the
# README once you've run a full backfill (see README caveat on why this
# matters for the log-regression fit).
FETCH_START_DATE = pd.Timestamp("2017-08-01")

# XRP Ledger genesis ledger, 2012-06-02 (first ledger closed by rippled).
# This is the XRP equivalent of the BTC genesis-block date used in the
# original script's days_since_genesis calc — see README for why this
# component is on shakier theoretical ground for XRP than for BTC.
XRP_GENESIS_DATE = pd.Timestamp("2012-06-02")


def _get_with_fallback(params):
    """Try the geo-block-resistant mirror first, fall back to the main API domain."""
    try:
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        print(f"  Primary endpoint failed (status={status}), retrying via fallback domain...")
        resp = requests.get(BINANCE_KLINES_URL_FALLBACK, params=params, timeout=30)
        resp.raise_for_status()
        return resp


def fetch_klines(start_time_ms=None, limit=1000):
    """Fetch daily klines from Binance, paginating until caught up to now."""
    all_rows = []
    cursor = start_time_ms
    while True:
        params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": limit}
        if cursor is not None:
            params["startTime"] = cursor
        resp = _get_with_fallback(params)
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < limit:
            break
        # next page starts right after the last candle's open time
        cursor = rows[-1][0] + 1
        time.sleep(0.2)  # be polite to the public endpoint
    return all_rows


def klines_to_df(rows):
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    df["close"] = df["close"].astype(float)
    df = df[["date", "close"]].drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def percentile_rank_expanding(series: pd.Series, min_periods=60) -> pd.Series:
    """
    For each point, rank it against all prior history (inclusive), scaled 0-1.
    This is what makes each component self-calibrating: no hardcoded bounds,
    the definition of 'cheap' vs 'expensive' adapts as more history accumulates.
    """
    def rank_last(window):
        if len(window) < min_periods:
            return np.nan
        return (window <= window[-1]).sum() / len(window)

    return series.expanding(min_periods=min_periods).apply(rank_last, raw=True)


def compute_components(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_price"] = np.log(df["close"])
    df["days_since_genesis"] = (df["date"] - XRP_GENESIS_DATE).dt.days
    df["log_days"] = np.log(df["days_since_genesis"])

    # --- 1. Log-regression band position ---
    # Fit log(price) ~ a * log(days) + b using all available history (refit each run).
    # CAVEAT (see README): this power-law-vs-age model is a reasonably well
    # validated pattern for BTC, whose growth is driven by a fixed, disinflationary
    # supply schedule and a long, comparatively continuous adoption curve. XRP's
    # supply/float is controlled by Ripple's escrow release schedule (not fixed
    # issuance), and its price history includes discontinuous, event-driven jumps
    # (SEC lawsuit filing/resolution, exchange relistings, bank-partnership news)
    # that a smooth log-log curve doesn't model well. Treat this component's
    # output with more skepticism for XRP than you would for BTC.
    coeffs = np.polyfit(df["log_days"], df["log_price"], 1)
    df["log_price_fit"] = np.polyval(coeffs, df["log_days"])
    df["regression_residual"] = df["log_price"] - df["log_price_fit"]
    df["log_regression"] = percentile_rank_expanding(df["regression_residual"])

    # --- 2. 200-day MA multiple ---
    df["ma200"] = df["close"].rolling(200, min_periods=200).mean()
    df["ma200_ratio"] = df["close"] / df["ma200"]
    df["ma200_multiple"] = percentile_rank_expanding(df["ma200_ratio"])

    # --- 3. RSI-14 ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df["rsi14"] = (rsi / 100).clip(0, 1)

    # --- 4. Volatility-adjusted momentum ---
    df["ret"] = df["close"].pct_change()
    df["roc_30d"] = df["close"].pct_change(30)
    df["vol_30d"] = df["ret"].rolling(30, min_periods=30).std()
    df["vol_adj_mom_raw"] = df["roc_30d"] / df["vol_30d"].replace(0, np.nan)
    # 3-day EMA on the raw ratio before ranking — same fix as the BTC script,
    # carried over unvalidated for XRP (the BTC repo's correlation numbers in
    # its comment were computed on BTC data specifically; re-check against
    # XRP history if you want to confirm the same smoothing/lag tradeoff
    # holds here before leaning on it).
    df["vol_adj_mom_smoothed"] = df["vol_adj_mom_raw"].ewm(span=3, min_periods=1, adjust=False).mean()
    df["vol_adj_momentum"] = percentile_rank_expanding(df["vol_adj_mom_smoothed"])

    return df


def compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["composite_score"] = (
        df["log_regression"] * WEIGHTS["log_regression"]
        + df["ma200_multiple"] * WEIGHTS["ma200_multiple"]
        + df["rsi14"] * WEIGHTS["rsi14"]
        + df["vol_adj_momentum"] * WEIGHTS["vol_adj_momentum"]
    )
    return df


# Base weekly DCA size. Zone sizes below are BASE_WEEKLY_USD * multiplier.
# Same $10 base and same zone boundaries as the BTC script, carried over
# unchanged. XRP is materially more volatile than BTC, so these boundaries
# were NOT re-tuned against XRP's own history — see README before trusting
# the zone labels as-is.
BASE_WEEKLY_USD = 10

# Actual DCA rule table (XRP/USDT, base $10/week).
# Sell tiers only fire in practice once holdings >= $500 per asset — that's a
# portfolio-level gate this script can't see (it only knows price/score), so
# sell zones are always computed here and the $500 gate is applied by you
# (or by the dashboard, which does know your holdings) before acting on them.
ZONES = [
    # (upper_bound_exclusive, zone, tier, multiplier, action)
    (0.10, "Extreme Buy",   "buy",   3.0, "Max accumulate"),
    (0.20, "Strong Buy",    "buy",   1.5, "Accumulate"),
    (0.25, "Buy",           "buy",   1.0, "Normal DCA"),
    (0.35, "Reduced Buy",   "buy",   0.5, "Slow down"),
    (0.60, "Stop — Hold",   "hold",  0.0, "Accumulation done"),
    (0.70, "Sell Tier 1",   "sell1", None, "Exit 5% of holdings"),
    (0.80, "Sell Tier 2",   "sell2", None, "Exit 10% of holdings"),
    (1.01, "Sell Tier 3 / Exit", "sell3", None, "Exit 20% or full position"),
]


def zone_for_score(score):
    if pd.isna(score):
        return {"zone": "Insufficient history", "tier": "none", "multiplier": None,
                "size_usd": None, "action": "—"}
    for upper, zone, tier, mult, action in ZONES:
        if score < upper:
            size = round(BASE_WEEKLY_USD * mult, 2) if mult is not None else None
            return {"zone": zone, "tier": tier, "multiplier": mult, "size_usd": size, "action": action}
    # score == 1.0 edge case, falls into last zone above via < 1.01
    upper, zone, tier, mult, action = ZONES[-1]
    return {"zone": zone, "tier": tier, "multiplier": mult, "size_usd": None, "action": action}


def build_output(df: pd.DataFrame) -> list:
    out = []
    for _, row in df.iterrows():
        if pd.isna(row["composite_score"]):
            continue
        z = zone_for_score(row["composite_score"])
        out.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "close": round(row["close"], 4),
            "composite_score": round(row["composite_score"], 4),
            "zone": z["zone"],
            "tier": z["tier"],
            "multiplier": z["multiplier"],
            "size_usd": z["size_usd"],
            "action": z["action"],
            "components": {
                "log_regression": round(row["log_regression"], 4) if not pd.isna(row["log_regression"]) else None,
                "ma200_multiple": round(row["ma200_multiple"], 4) if not pd.isna(row["ma200_multiple"]) else None,
                "rsi14": round(row["rsi14"], 4) if not pd.isna(row["rsi14"]) else None,
                "vol_adj_momentum": round(row["vol_adj_momentum"], 4) if not pd.isna(row["vol_adj_momentum"]) else None,
            },
        })
    return out


def load_existing_closes() -> pd.DataFrame:
    """
    Load the FULL raw close-price cache (not the scored HISTORY_FILE, which is
    missing the pre-warmup rows — see PRICES_FILE comment above for why that
    distinction matters).
    """
    if not PRICES_FILE.exists():
        return pd.DataFrame(columns=["date", "close"])
    existing = json.loads(PRICES_FILE.read_text())
    if not existing:
        return pd.DataFrame(columns=["date", "close"])
    df = pd.DataFrame({
        "date": pd.to_datetime([r["date"] for r in existing]),
        "close": [r["close"] for r in existing],
    })
    return df


def save_prices_raw(df: pd.DataFrame) -> None:
    """Persist the full date/close series (including pre-warmup rows) for future merges."""
    rows = [{"date": d.strftime("%Y-%m-%d"), "close": round(c, 4)}
            for d, c in zip(df["date"], df["close"])]
    PRICES_FILE.write_text(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                         help="Only fetch recent candles (last 400 days) instead of the full "
                              "history from Binance. Indicators are still recomputed over the "
                              "FULL closing-price series (existing history + freshly fetched "
                              "tail merged) so results match a full recompute exactly — this "
                              "flag only speeds up the network fetch, not the math.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing_df = load_existing_closes() if args.update else pd.DataFrame(columns=["date", "close"])

    if args.update and not existing_df.empty:
        start_ms = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=400)).timestamp() * 1000)
    else:
        # No prior history to merge into (or a full run was requested) -> fetch everything.
        start_ms = int(FETCH_START_DATE.timestamp() * 1000)

    print(f"Fetching XRPUSDT daily klines from Binance (start={pd.to_datetime(start_ms, unit='ms')})...")
    rows = fetch_klines(start_time_ms=start_ms)
    fetched_df = klines_to_df(rows)
    print(f"Fetched {len(fetched_df)} daily candles, {fetched_df['date'].min().date()} to {fetched_df['date'].max().date()}")

    # Merge fetched candles on top of existing history so every component (log-regression fit,
    # expanding percentile ranks, 200d MA, etc.) is computed over the FULL price series, not
    # just the freshly fetched window. Fetched rows win on overlapping dates (fresher close).
    if not existing_df.empty:
        df = (
            pd.concat([existing_df, fetched_df], ignore_index=True)
            .drop_duplicates(subset="date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        print(f"Merged with existing history: {len(df)} total daily candles, "
              f"{df['date'].min().date()} to {df['date'].max().date()}")
    else:
        df = fetched_df

    df = compute_components(df)
    df = compute_composite(df)
    output = build_output(df)

    save_prices_raw(df[["date", "close"]])
    HISTORY_FILE.write_text(json.dumps(output, indent=2))
    print(f"Wrote {HISTORY_FILE}, {len(output)} rows ({PRICES_FILE.name} raw cache also updated)")

    if output:
        latest = output[-1]
        size = f"${latest['size_usd']}" if latest['size_usd'] is not None else "—"
        print(f"\nLatest ({latest['date']}): score={latest['composite_score']} "
              f"[{latest['zone']}] size={size}/wk — {latest['action']}")


if __name__ == "__main__":
    main()
