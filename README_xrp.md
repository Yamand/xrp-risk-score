# XRP Risk Score

A daily composite 0–1 risk score for XRP, built entirely from Binance public
OHLCV data (no API key, no paid on-chain data). Static site + GitHub Actions.

This is a direct port of [btc-risk-score](https://github.com/Yamand/btc-risk-score) —
same four components, same weights, same zone table, same repo layout. Read
**"Porting caveats"** below before you trust it the way you trust the BTC
version; the methodology transfers mechanically, but part of it is on
weaker theoretical footing for XRP.

**Live idea:** `0` = cheap, accumulate harder. `1` = expensive, reduce buys /
start distributing once holdings clear your $500 sell-tier threshold.

## How the score is built

Four components, each normalized to 0–1 via **expanding historical percentile
rank** (today's raw value ranked against every prior day back to the start of
data) — this means there are no hardcoded "cheap" / "expensive" thresholds to
maintain; the scale self-calibrates as more history accumulates.

| Component | Weight | What it captures |
|---|---|---|
| Log-regression band position | 35% | Price vs. long-term log-log growth curve, refit each run |
| 200-day MA multiple | 25% | Price stretch vs. long-term trend (price ÷ 200d MA) |
| RSI-14 (daily) | 20% | Short-term overbought/oversold |
| Volatility-adjusted momentum | 20% | 30d return ÷ 30d realized volatility |

Composite = weighted sum of the four, then smoothed with a **3-day EMA**
before being used for zone lookup. The raw (unsmoothed) score is still saved
in the output as `composite_score_raw`-equivalent field, `composite_score`,
same as the BTC repo.

## Porting caveats — read before trusting this like the BTC score

**1. The log-regression component (35% of the score) is on weaker ground for XRP.**
The BTC version fits `log(price)` against `log(days since genesis block)` —
a power-law model that's reasonably well documented for BTC, because BTC has
a fixed, disinflationary issuance schedule and a long, comparatively smooth
adoption curve. This script uses the **same math** but against
`log(days since the XRP Ledger genesis ledger, 2012-06-02)`. XRP's price
history doesn't share BTC's assumptions:
- XRP's circulating supply is governed by Ripple's monthly escrow releases,
  not a fixed protocol schedule.
- XRP's price history includes sharp, event-driven discontinuities (the SEC
  suit filed Dec 2020, the 2023 partial summary judgment, exchange
  relistings/delistings around those events) that a smooth log-log curve
  doesn't model — a regression fit across those regimes can produce a "fair
  value" band that doesn't mean the same thing it means for BTC.

  This doesn't make the component useless (it's still a legitimate measure
  of "how stretched is price relative to its own historical trend line"),
  but the "long-term growth curve" story behind it is a BTC-specific
  justification that doesn't fully carry over. Treat this component's signal
  with more skepticism than you would on the BTC dashboard, especially near
  known event dates.

**2. Zone boundaries (0.10 / 0.20 / 0.25 / 0.35 / 0.60 / 0.70 / 0.80) are carried
over unchanged, not re-tuned on XRP's own history.** XRP is meaningfully more
volatile than BTC, so these thresholds may fire more often / less
discriminatingly for XRP than they do for BTC. If the zone labels feel like
they're flipping too fast once you have a year+ of data, that's the first
place to look — either widen the bands or lengthen `min_periods` in
`percentile_rank_expanding`.

**3. Binance's XRPUSDT history is shorter than BTCUSDT's**, so the
log-regression fit and expanding percentile ranks will be even less stable
in year one than the BTC version's own (already-flagged) early-history
caveat. Run a full backfill and check the printed start date — the ranks
computed against a shallow history window are the least meaningful ones.

**4. Prices are formatted to 4 decimal places, not 2** — the BTC script
rounds to cents, which would collapse most of XRP's price history to
`$0.50`–`$3.00`-ish resolution. `xrp_risk_score.py` and the dashboard both
round to 4dp instead.

None of this means the score is wrong — it's the same well-reasoned
methodology, applied to an asset it wasn't originally validated against.
Use it as one more input, same as the BTC version, not as a standalone
signal.

## Repo structure

```
xrp_risk_score.py                    # fetch + compute + write data/
data/xrp_risk_history.json           # generated — one scored row per day
data/xrp_prices_raw.json             # generated — full raw close-price cache (see BTC repo's README for why this exists)
index.html                           # static site, reads data/ directly, Chart.js
.github/workflows/daily-update.yml   # cron job, runs xrp_risk_score.py --update daily
```

## Setup

1. Push this repo to GitHub, enable **GitHub Pages** (Settings → Pages →
   Deploy from branch → `main` / root).
2. Run a full backfill once, locally, so both `data/xrp_risk_history.json`
   and `data/xrp_prices_raw.json` exist before the site goes live — see
   "Local run" below.
3. Commit **both** files in `data/` — the workflow commits both on every run
   too, since `xrp_prices_raw.json` has to persist across runs for
   `--update` to work correctly.
4. If you're also running the BTC repo's Telegram alert, use a **separate
   bot token** (or clearly distinguish the messages) so the daily BTC and
   XRP alerts don't get confused with each other in the same chat.
5. The daily workflow (`daily-update.yml`) runs automatically at 00:15 UTC,
   fetches the last 400 days, merges with the cached full history,
   recomputes, and commits both updated JSON files. GitHub Pages redeploys
   automatically on push.

### Local run

```bash
pip install pandas numpy requests
python xrp_risk_score.py            # full history backfill (first run)
python xrp_risk_score.py --update   # fast daily run — same math, fewer requests
python -m http.server 8000          # then open localhost:8000
```

## Notes

- No API key required — Binance's `/api/v3/klines` endpoint is public.
- The score is descriptive, not a signal to auto-trade on.
- Not financial advice.
