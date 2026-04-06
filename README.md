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

## pip Fallback

```bash
pip install -r requirements.txt
python main.py
```

## Dry Run vs Live Trading

`DRY_RUN=true` is the safety-first mode and prevents real order placement.

- `DRY_RUN=true`: analyze and log candidate trades only.
- `DRY_RUN=false`: place live Kalshi orders when all trade gates pass.

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
- `BAYESIAN_ENABLED`, `LMSR_ENABLED`, `KELLY_SIZING_ENABLED` for optional advanced layers.
- `KELLY_MIN_BET_POLICY` controls handling when Kelly sizing is below minimum bet.
- `MAX_POSITION_PCT_OF_BANKROLL`, `MAX_POSITION_PER_MARKET_USDC` cap exposure.
- `OPPOSITE_OUTCOME_STRATEGY` and flip-guard settings reduce churn from side flips.
- `MARKET_TICKER_BLOCKLIST_PREFIXES`, ladder collapse controls, and extreme-price filters reduce noisy candidates.
- Category-specific research profiles tune source domains and X handles for sports, crypto, politics, and generic markets.

## State and Logging

- State persistence: `STATE_DB_PATH` (SQLite) and optional JSON export (`STATE_JSON_EXPORT_PATH`).
- Resolution tracking runs on a configurable cycle interval.
- Logs are written under `LOG_DIR` (default `logs/`), including standard and error-focused outputs.

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
