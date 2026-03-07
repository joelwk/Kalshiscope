# Prediscope

Autonomous prediction-market trading bot using xAI Grok for research, edge scoring, and PredictBase for market execution.

## What It Does

- Pulls active markets from PredictBase.
- Filters by liquidity/category/close window.
- Uses Grok to analyze outcomes, confidence, and supporting evidence.
- Applies edge, evidence-quality, and position-risk gates before trading.
- Can layer Bayesian updating, LMSR-based price checks, and Kelly sizing behind feature flags.
- Optionally submits orders and on-chain approvals.

## Prerequisites

- Python `>=3.10`
- [Poetry](https://python-poetry.org/docs/#installation) (recommended)

## 5-Minute Quick Start (Recommended)

1. Copy env template:

```bash
cp .env.example .env
```

2. Edit `.env` and fill only:

- `XAI_API_KEY`
- `WALLET_PRIVATE_KEY`

3. Install dependencies:

```bash
poetry install
```

4. Run bot:

```bash
poetry run predi
```

## pip Fallback

If you prefer `pip`:

```bash
pip install -r requirements.txt
python main.py
```

## Safe Mode vs Live Mode

Default `.env.example` is safety-first:

- `DRY_RUN=true`
- `AUTO_APPROVE_USDC=false`
- `EXECUTE_ONCHAIN=false`

This means no live orders or approvals by default.

### Enable Live Trading (Explicit Opt-In)

Only after validating behavior in dry run:

1. Set `DRY_RUN=false`
2. Set `EXECUTE_ONCHAIN=true` if you want on-chain execution
3. Optionally set `AUTO_APPROVE_USDC=true` for automatic approvals

When `EXECUTE_ONCHAIN=true` or `AUTO_APPROVE_USDC=true`, `ALCHEMY_RPC_URL` is required.

When `AUTO_APPROVE_USDC=true`, `PREDICTBASE_CONTRACT_ADDRESS` is required.

## Environment Variables

Required by default:

- `XAI_API_KEY`
- `WALLET_PRIVATE_KEY`

Conditionally required:

- `ALCHEMY_RPC_URL` when `EXECUTE_ONCHAIN=true` or `AUTO_APPROVE_USDC=true`
- `PREDICTBASE_CONTRACT_ADDRESS` when `AUTO_APPROVE_USDC=true`

Everything else has defaults in `.env.example`, including conservative rollout settings for edge thresholds, Kelly sizing, and optional Bayesian/LMSR layers.

## Strategy Controls

- `MIN_EDGE` sets the minimum edge required before a market can pass trade gating.
- `KELLY_SIZING_ENABLED` switches sizing from edge scaling to fractional Kelly.
- `BAYESIAN_ENABLED` enables posterior updates from model likelihood ratios across cycles.
- `LMSR_ENABLED` enables an independent LMSR-based price verification layer.

The template keeps Bayesian, LMSR, and Kelly conservative by default so they can be enabled gradually.

## Troubleshooting

### Missing required environment variables

If startup fails with `Missing required environment variables`, check `.env` and mode flags:

- Safe mode needs only `XAI_API_KEY` and `WALLET_PRIVATE_KEY`.
- Live/on-chain mode requires `ALCHEMY_RPC_URL`.
- Auto-approve also requires `PREDICTBASE_CONTRACT_ADDRESS`.

### RPC / chain issues

- Ensure `CHAIN_ID=8453` for Base mainnet.
- Ensure your RPC endpoint is reachable and funded account has gas.

### Dependency issues

- Poetry path: run `poetry install`.
- pip path: run `pip install -r requirements.txt`.
- If imports fail, verify active Python environment matches install location.

## Security Notes

- Never commit `.env`.
- Treat private keys and API keys as compromised if leaked.
- Rotate credentials immediately after any accidental exposure.

## Run Tests

```bash
poetry run pytest -q -s
```
