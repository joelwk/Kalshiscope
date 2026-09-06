# Kalshiscope

Autonomous prediction-market trading bot for Kalshi that uses xAI Grok for research, confidence estimation, and trade execution decisions.

## What It Does

- Pulls active markets from Kalshi Trade API v2.
- Filters markets by liquidity, close window, category policy, ticker patterns, and event ladder shape.
- Uses Grok to analyze outcomes, confidence, and evidence quality with profile-aware sourcing.
- Applies layered gating before execution (confidence, edge, score, flip guard, and risk caps).
- Supports optional Bayesian updates, LMSR checks, and Kelly sizing.
- Submits Kalshi limit orders in live mode or simulates decisions in dry run.

## Prerequisites

- Python `>=3.10`
- [Poetry](https://python-poetry.org/docs/#installation) (recommended)
- Kalshi API credentials:
  - API key ID (`KALSHI_API_KEY_ID`)
  - RSA private key file path (`KALSHI_PRIVATE_KEY_PATH`)

## 5-Minute Quick Start

1. Copy env template:

```bash
cp .env.example .env
```

2. Edit `.env` and set:

- `XAI_API_KEY`
- `KALSHI_API_KEY_ID`
- `KALSHI_PRIVATE_KEY_PATH`

3. Install dependencies:

```bash
poetry install
```

4. Run bot:

```bash
poetry run predi
```

You can also run:

```bash
poetry run kalshi
```

To run a fixed number of cycles and stop automatically:

```bash
poetry run kalshi --cycles 30
```

## pip Fallback

```bash
pip install -r requirements.txt
python main.py
```

## Dry Run vs Live Trading

`DRY_RUN=true` is the safety-first mode and prevents real order placement.

- `DRY_RUN=true`: analyze and log candidate trades only.
- `DRY_RUN=false`: place live Kalshi orders when all trade gates pass.

`GUARANTEED_ORDERS_N` defaults to `0`. When set to a positive integer, the bot
locks this cycle's highest positive-EV analyzed names (chosen-side edge after
calibration × evidence quality × confidence), dives deeper on those slots, and
forces a Kelly-sized order from the researched side only when that side still
has clear positive chosen-side edge. Slots never share an event, and no market
family may hold more than three of them: demanding a distinct family per slot
handed four of every five slots to families with no fills on record, because
the one family that does fill was capped at a single slot. Absence-only, zero,
or negative-EV slots are replaced with another analyzed +EV name the same
cycle, or deferred to the next cycle — never forced to fill the quota.
Unlabeled `edge_mechanism` or proxy evidence quality below the ordinary trade
floor is not a hard skip when chosen-side edge still clears the floor.
`GUARANTEED_MIN_EDGE` (default `0.12`) is the hard chosen-side floor for
direct, computed-odds, named-mechanism, or weather sides; unlabeled
non-weather proxy must also clear `GUARANTEED_PROXY_MIN_EDGE` (default `0.15`).
`GUARANTEED_FAMILY_MIN_EDGE` (default `crypto:0.06`) replaces both of those
floors for the families it names. The default is calibrated on 857 resolved
trades: crypto returned +7% below a 0.12 edge and −13% above it, because a
large claimed edge on a continuously repriced ladder is overconfidence rather
than mispricing. Weather is the family `0.12` actually fits (+3% above it,
−16% below), so it stays on the default. Evidence strength is gated separately,
so an override only moves the edge magnitude a family has to clear.
Two guards keep the hunt from spending deep research where it cannot pay off.
A first pass sitting more than 20 points below the floor skips the deep call
outright, since deep research has historically moved confidence that far only
5% of the time. And a Kalshi series that misses the floor
`GUARANTEED_SERIES_MISS_LIMIT` times in a row (default `3`) without ever
filling stops being locked at all; the tally persists across runs, and any fill
resets it. This is what stops continuously repriced ladders such as crypto
strikes and index levels — the most liquid names on the exchange, and the ones
least likely to be mispriced — from consuming every slot.
`GUARANTEED_ORDER_MAX_RESEARCH_GAP_REPLACEMENTS` bounds churn across markets,
not the run itself: once it is spent, a slot whose market is still tradeable
holds its lock and re-prices on later cycles, and only a market that can never
fill is abandoned. Ordinary gate-cleared submissions are
suppressed in this mode so the run cannot exceed the target. Forced (and
normal) stakes scale with live portfolio value:
`clip(kelly_bet_pct × MAX_BET_PCT_OF_BANKROLL × portfolio, MIN_BET_PCT_OF_BANKROLL × portfolio, position/drawdown caps)`.
Dry runs persist up to that many attempted-order receipts when enough +EV
markets exist; bounded live runs fail explicitly if Kalshi does not accept all
target submissions and exit early once the target is complete. Live guarantee
plans exclude families the exchange has rejected for this account (Sports, and
when present Elections and Entertainment). If that restriction is first
discovered during forced submission, the rejected slot is retired and replaced
with an executable family using a new idempotency key. Set
`GUARANTEED_ORDERS_N=5` for a five-cycle `poetry run kalshi --cycles 5` run.

Start in dry run and switch to live only after reviewing behavior in logs.

## Environment Variables

Required:

- `XAI_API_KEY`
- `KALSHI_API_KEY_ID`
- `KALSHI_PRIVATE_KEY_PATH`

Common optional variables:

- `KALSHI_API_BASE_URL` (defaults to Kalshi v2 endpoint)
- `KALSHI_SERVER_SIDE_FILTERS_ENABLED`
- `POLL_INTERVAL_SEC`
- `MIN_LIQUIDITY_USDC`
- `MARKET_MIN_CLOSE_DAYS`, `MARKET_MAX_CLOSE_DAYS`

See `.env.example` for the full set of runtime controls.

## Strategy and Risk Controls

- `MIN_EDGE`, `LOW_PRICE_MIN_EDGE`, `FALLBACK_EDGE_MIN_EDGE` for edge thresholds.
- `SCORE_GATE_MODE` (`off`, `shadow`, `active`) for decision scoring rollout.
- `BAYESIAN_ENABLED`, `LMSR_ENABLED`, `KELLY_SIZING_ENABLED` for optional advanced layers. Keep `KELLY_SIZING_ENABLED=true` for normal orders; guaranteed slots always Kelly-size against the cycle's bankroll-derived max bet.
- `KELLY_MIN_BET_POLICY` controls handling when Kelly sizing is below minimum bet.
- `MIN_BET_PCT_OF_BANKROLL`, `MAX_BET_PCT_OF_BANKROLL` scale dollar bets with portfolio value (cash + positions).
- `MAX_POSITION_PCT_OF_BANKROLL`, `MAX_POSITION_PER_MARKET_USDC` cap exposure.
- `OPPOSITE_OUTCOME_STRATEGY` and flip-guard settings reduce churn from side flips.
- `MARKET_TICKER_BLOCKLIST_PREFIXES`, ladder collapse controls, and extreme-price filters reduce noisy candidates.
- Category-specific research profiles tune source domains and X handles for sports, crypto, politics, and generic markets.

## State and Logging

- State persistence: `STATE_DB_PATH` (SQLite) remains the complete audit store.
- `STATE_JSON_EXPORT_PATH` is an atomic, bounded schema-version-2 snapshot. It contains current-cycle markets, open positions, active orders, unresolved outcomes, recent settlements, calibration/research state, sync checkpoints, the current cycle receipt, and at most `STATE_JSON_RECENT_DECISIONS_LIMIT` decisions from that cycle. It intentionally excludes full receipt and trade history.
- `STATE_JSON_EXPORT_INTERVAL_CYCLES` controls snapshot frequency (default `1`). A cycle receipt is committed before its snapshot is replaced.
- Export complete history on demand without changing SQLite: `poetry run python scripts/export_state_audit.py --table decision_receipts --since 2026-07-01T00:00:00+00:00 --format ndjson --output decisions.ndjson`. Repeat `--table` to select multiple tables; omit it for every audit table.
- Exchange order/position reconciliation runs before execution. Live submissions are suppressed when either snapshot is incomplete or when an exchange resting order is not represented locally.
- `pending_orders` is retained as the compatibility-backed order lifecycle table; terminal rows are intentionally preserved. For direct SQLite inspection use `SELECT * FROM active_pending_orders` for actionable orders and `SELECT * FROM order_lifecycle_history` for the complete audit history.
- Resolution tracking runs on a configurable cycle interval.
- Logs are written under `LOG_DIR` (default `logs/`), including standard and error-focused outputs.

## Monitoring and Analytics

### PnL Report

View total realized PnL with category and monthly breakdowns:

```bash
poetry run python scripts/pnl_report.py
```

Sync latest settlements from Kalshi before reporting:

```bash
poetry run python scripts/pnl_report.py --sync
```

Run in offline mode (database only, no API calls):

```bash
poetry run python scripts/pnl_report.py --no-api
```

### Account Balance and Positions

Quick diagnostic of current balance and open positions:

```bash
poetry run python check_balance.py
```

### Calibration Analytics

Detailed calibration metrics, win rates, and decision quality diagnostics:

```bash
poetry run python analytics.py
```

### Tuning Recommendations

Generate threshold tuning recommendations from recent logs:

```bash
poetry run python scripts/daily_tuning_recommendations.py
```

## Troubleshooting

### Missing required environment variables

If startup fails with `Missing required environment variables`, verify:

- `XAI_API_KEY` is set.
- `KALSHI_API_KEY_ID` is set.
- `KALSHI_PRIVATE_KEY_PATH` points to an existing readable private key file.

### Kalshi authentication failures

- Confirm the API key ID matches the private key pair in your Kalshi account.
- Ensure the private key is in the expected PEM/plaintext format used by your account.
- Verify your system clock is accurate; signed request timestamps must be valid.

### No trades executing

- Confirm `DRY_RUN=false` for live placement.
- Check gating thresholds (`MIN_CONFIDENCE`, `MIN_EDGE`, score gate mode).
- Review liquidity, close-window, and category filters that may exclude candidates.

### Dependency issues

- Poetry path: run `poetry install`.
- pip path: run `pip install -r requirements.txt`.
- If imports fail, verify the active Python environment matches the install location.

## Security Notes

- Never commit `.env` or Kalshi private key files.
- Treat API credentials and private keys as compromised if leaked.
- Rotate keys immediately after accidental exposure.

## Run Tests

```bash
poetry run pytest -q -s
```
