from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

from dotenv import load_dotenv

# .env should be the single source of truth for local bot runs.
# Using override=True avoids stale exported shell vars shadowing updated .env values.
load_dotenv(override=True)

XAI_WEB_SEARCH_ALLOWED_DOMAINS_LIMIT = 5
XAI_X_SEARCH_ALLOWED_HANDLES_LIMIT = 10


@dataclass(frozen=True)
class Settings:
    # Risk controls - Conservative defaults for value betting
    MIN_BET_USDC: float = 1.0
    MAX_BET_USDC: float = 50.0
    MIN_CONFIDENCE: float = 0.62  # Raised to avoid low-confidence churn and improve calibration
    CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED: bool = True
    CONFIDENCE_GATE_MIN_EDGE: float = 0.08
    CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY: float = 0.70
    CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE: float = 0.58
    # Direct-evidence posterior floor: for direct + computed + high-evidence
    # decisions, floor the posterior used by the edge gate, Kelly sizing, and the
    # score gate at the model's own outcome estimate (implied + edge_external) so
    # confidence calibration cannot invert a genuine positive edge into a
    # negative market edge. Bounded by MAX_GLOBAL_CONFIDENCE_DIRECT.
    DIRECT_POSTERIOR_FLOOR_ENABLED: bool = True
    DIRECT_POSTERIOR_FLOOR_MIN_EVIDENCE_QUALITY: float = 0.80
    # Scope guard for numeric-strike price markets (commodity/index/crypto
    # -T<strike> tickers): a live quote confirms the CURRENT value, not the
    # settlement value, so the floor only applies when settlement is within
    # this many hours (or the decision passes definitive validation). Weather
    # keeps the floor: forecasts predict the settlement quantity itself.
    # June 2026 evidence: floored commodity strikes placed 3-4h out ran a 52%
    # realized win rate at ~0.57 entries. Set to 0 to disable the scope guard.
    DIRECT_POSTERIOR_FLOOR_MAX_HOURS_TO_CLOSE: float = 1.5
    MIN_EVIDENCE_QUALITY_FOR_TRADE: float = 0.55
    SPORTS_MIN_EVIDENCE_QUALITY: float = 0.55
    MIN_LIQUIDITY_USDC: float = 15.0
    POLL_INTERVAL_SEC: int = 300
    DRY_STREAK_SLEEP_ENABLED: bool = True

    # Edge gating / sizing
    MIN_EDGE: float = 0.12
    MIN_EDGE_HIGH_LIQUIDITY_THRESHOLD: float = 300.0
    MIN_EDGE_HIGH_LIQUIDITY_MULTIPLIER: float = 0.70
    MIN_EDGE_MEDIUM_LIQUIDITY_THRESHOLD: float = 100.0
    MIN_EDGE_MEDIUM_LIQUIDITY_MULTIPLIER: float = 0.85
    LOW_PRICE_THRESHOLD: float = 0.50
    VERY_LOW_PRICE_THRESHOLD: float = 0.25
    # Allow high-edge, direct, settlement-aligned trades to bypass the hard
    # VERY_LOW_PRICE_THRESHOLD entry floor. The min-edge ladder already prices
    # low-entry risk (VERY_LOW_PRICE_MIN_EDGE), so the hard floor otherwise
    # discards legitimate cheap longshots backed by direct settlement evidence
    # (e.g. a 0.21-priced market with a 0.49 direct edge).
    ENTRY_PRICE_FLOOR_EDGE_OVERRIDE_ENABLED: bool = True
    ENTRY_PRICE_FLOOR_OVERRIDE_MIN_EDGE: float = 0.20
    # eq floor set to MIN_EVIDENCE_QUALITY_FOR_TRADE level (0.60) rather than 0.80
    # so the override actually recovers the observed lost class (direct,
    # settlement-aligned, edge 0.49, eq 0.60). The direct + settlement_aligned +
    # strong-edge gates already make this a high bar; the market must still clear
    # the separate evidence-quality and edge gates downstream.
    ENTRY_PRICE_FLOOR_OVERRIDE_MIN_EVIDENCE_QUALITY: float = 0.60
    HIGH_PRICE_THRESHOLD: float = 0.65
    LOW_PRICE_MIN_EDGE: float = 0.18
    VERY_LOW_PRICE_MIN_EDGE: float = 0.28
    LOW_PRICE_MIN_EDGE_MULTIPLIER: float = 0.85
    COINFLIP_PRICE_LOWER: float = 0.48
    COINFLIP_PRICE_UPPER: float = 0.52
    EDGE_SCALING_RANGE: float = 0.15
    LOW_PRICE_BET_PENALTY: float = 0.50
    FALLBACK_EDGE_MIN_EDGE: float = 0.30
    FALLBACK_EDGE_MIN_EDGE_MULTIPLIER: float = 0.90
    WEATHER_MIN_EDGE: float = 0.14
    WEATHER_HIGH_EQ_EDGE_MULTIPLIER: float = 0.85
    WEATHER_FALLBACK_EDGE_MIN_EDGE: float = 0.34
    # Block weather entries when the chosen-outcome market price is an underdog
    # (< LOW_PRICE_THRESHOLD). Lifetime weather underdog WR ~32% / large negative PnL.
    WEATHER_BLOCK_UNDERDOG_ENTRIES: bool = True
    # Cap the edge preserved by the weather posterior floor so extreme raw claims
    # cannot be fully resurrected after calibration shrink.
    WEATHER_POSTERIOR_FLOOR_MAX_EDGE: float = 0.20
    # When raw−calibrated confidence gap is large, shrink weather Kelly sizing.
    WEATHER_CALIBRATION_GAP_FOR_KELLY_SHRINK: float = 0.20
    WEATHER_CALIBRATION_GAP_KELLY_MULTIPLIER: float = 0.50
    # Commodity futures (WTI/NATGAS/BRENT/etc.) historically underperform; raise
    # the edge bar without hard-blocking analysis eligibility.
    COMMODITY_MIN_EDGE: float = 0.22
    # Near-settlement high-EQ commodity decisions may multiply COMMODITY_MIN_EDGE
    # (0.22 * 0.95 ≈ 0.209) so knife-edge buffers clear without lowering the
    # long-horizon / low-EQ floor that protects the toxic edge≥0.20 strike class.
    COMMODITY_HIGH_EQ_EDGE_MULTIPLIER: float = 0.95
    COMMODITY_HIGH_EQ_MIN_EVIDENCE_QUALITY: float = 0.75
    REQUIRE_IMPLIED_PRICE: bool = True
    
    # Confidence caps to prevent overconfidence on high-variance events
    MAX_GLOBAL_CONFIDENCE: float = 0.82
    MAX_GLOBAL_CONFIDENCE_DIRECT: float = 0.89
    MAX_SPORTS_CONFIDENCE: float = 0.80
    MAX_ESPORTS_CONFIDENCE: float = 0.75
    MAX_WEATHER_CONFIDENCE: float = 0.65
    MAX_INDEX_CONFIDENCE: float = 0.70
    MAX_COMMODITY_CONFIDENCE: float = 0.78
    MAX_LIVESTOCK_CONFIDENCE: float = 0.65
    MAX_HEATING_OIL_CONFIDENCE: float = 0.70
    MAX_CORN_CONFIDENCE: float = 0.70
    MAX_CRYPTO_CONFIDENCE: float = 0.72
    # Hard sanity cap on model-vs-market edge to catch hallucinated
    # opportunities. Raised from 0.32 to 0.40 after cycle 1 review showed a
    # validated direct + settlement-aligned music chart trade with edge=0.56
    # being hard-blocked before its score could be evaluated. The score gate
    # threshold + stacked penalties continue to scale on edges above 0.32.
    MAX_REASONABLE_EDGE: float = 0.40
    NON_SPORTS_REQUIRES_DIRECT_EVIDENCE: bool = True
    NON_SPORTS_REQUIRES_PRIMARY_SOURCE_URL: bool = True
    # Families with a universal canonical settlement source are exempt from the
    # per-market primary_source_url requirement for direct evidence: sports
    # (ESPN/league sites) and weather (NWS/NOAA). Other non-sports markets still
    # require a settlement-grade primary_source_url to qualify as direct.
    PRIMARY_SOURCE_URL_EXEMPT_FAMILIES: tuple[str, ...] = ("sports", "weather")
    MAX_SPEECH_CONFIDENCE: float = 0.65

    # Filtering
    MARKET_CATEGORIES_ALLOWLIST: tuple[str, ...] = ()
    MARKET_CATEGORIES_BLOCKLIST: tuple[str, ...] = ()
    MARKET_FAMILY_BLOCKLIST: tuple[str, ...] = ()
    MARKET_TICKER_BLOCKLIST_PREFIXES: tuple[str, ...] = ()
    SKIP_WEATHER_BIN_MARKETS: bool = False
    CRYPTO_BIN_MARKET_BLOCKLIST_ENABLED: bool = False
    MIN_VOLUME_24H: float = 10.0
    MIN_OPEN_INTEREST: float = 25.0
    EXTREME_YES_PRICE_LOWER: float = 0.02
    EXTREME_YES_PRICE_UPPER: float = 0.98
    MIN_TRADEABLE_IMPLIED_PRICE: float = 0.12
    MAX_TRADEABLE_IMPLIED_PRICE: float = 0.95
    LADDER_COLLAPSE_THRESHOLD: int = 3
    MAX_BRACKETS_PER_EVENT: int = 3
    # Date range filtering: only consider markets closing within this window (days from now)
    # Set to 0 or None to disable the filter
    MARKET_MIN_CLOSE_DAYS: int | None = None  # Minimum days until close (skip markets closing too soon)
    MARKET_MAX_CLOSE_DAYS: int | None = None  # Maximum days until close (skip markets closing too far out)

    # xAI Grok
    XAI_API_KEY: str = ""
    GROK_MODEL: str = "grok-4-1-fast-reasoning"
    GROK_MODEL_DEEP: str = "grok-4.20-beta-0309-reasoning"
    SEARCH_LOOKBACK_HOURS: int = 24
    SEARCH_ALLOWED_DOMAINS: tuple[str, ...] = (
        "espn.com",
        "cbssports.com",
        "nba.com",
        "espncricinfo.com",
        "cricbuzz.com",
        "iplt20.com",
        "koreabaseball.com",
        "covers.com",
        "sportsbookreview.com",
        "theathletic.com",
        "rotowire.com",
        "actionnetwork.com",
        "atptour.com",
        "wtatennis.com",
        "tennisexplorer.com",
        "flashscore.com",
    )
    SEARCH_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "ESPN",
        "CBSSports",
        "NBA",
        "ESPNcricinfo",
        "cricbuzz",
        "IPL",
        "KBO_ENG",
        "SportsCenter",
        "ShamsCharania",
        "wojespn",
        "FDSportsbook",
        "DKSportsbook",
        "BetMGM",
        "coinbase",
        "krakenfx",
        "business",
        "Reuters",
        "ReutersBiz",
        "WSJ",
        "FT",
        "CNBC",
        "MarketWatch",
        "TheEconomist",
        "YahooFinance",
        "GoUncensored",
        "ZssBecker",
        "WallStreetMav",
        "CryptoHayes",
        "elonmusk",
        "TrustlessState",
        "WhaleInsider",
        "WallStreetApes",
        "WatcherGuru",
        "intocryptoverse",
    )
    MULTIMEDIA_CONFIDENCE_THRESHOLD: tuple[float, float] = (0.55, 0.75)
    SEARCH_PROFILE_MAX_DOMAINS: int = 5
    SEARCH_PROFILE_MAX_X_HANDLES: int = 10
    # Domains whose pages count as settlement-grade primary sources for direct
    # evidence on non-sports markets. URLs outside this allowlist (e.g. commodity
    # aggregators) are treated as proxy-tier and cannot satisfy direct evidence.
    SETTLEMENT_SOURCE_ALLOWLIST_DOMAINS: tuple[str, ...] = (
        "cmegroup.com",
        "theice.com",
        "eia.gov",
        "treasury.gov",
        "bls.gov",
        "bea.gov",
        "federalreserve.gov",
        "sec.gov",
        "nasdaq.com",
        "weather.gov",
        "noaa.gov",
        "wsj.com",
        "bloomberg.com",
        "reuters.com",
        "apnews.com",
        "coinbase.com",
        "binance.com",
        "kraken.com",
        "coindesk.com",
        "spotify.com",
        "billboard.com",
        "luminate.com",
        "netflix.com",
        "boxofficemojo.com",
    )
    EXTENDED_RESEARCH_SOURCE_OFFSET: int = 5
    EXTENDED_RESEARCH_X_HANDLE_OFFSET: int = 10
    # Dynamic search windows by market horizon
    SEARCH_LOOKBACK_SHORT_HOURS: int = 24
    SEARCH_LOOKBACK_MEDIUM_HOURS: int = 72
    SEARCH_LOOKBACK_LONG_HOURS: int = 168
    # Category-specific source profiles
    SPORTS_ALLOWED_DOMAINS: tuple[str, ...] = (
        "espn.com",
        "cbssports.com",
        "nba.com",
        "espncricinfo.com",
        "cricbuzz.com",
        "iplt20.com",
        "koreabaseball.com",
        "covers.com",
        "sportsbookreview.com",
        "theathletic.com",
        "rotowire.com",
        "actionnetwork.com",
        "atptour.com",
        "wtatennis.com",
        "tennisexplorer.com",
        "flashscore.com",
    )
    SPORTS_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "ESPN",
        "CBSSports",
        "NBA",
        "ESPNcricinfo",
        "cricbuzz",
        "IPL",
        "KBO_ENG",
        "SportsCenter",
        "ShamsCharania",
        "wojespn",
        "FDSportsbook",
        "DKSportsbook",
        "BetMGM",
        "ataborasso",
        "TennisChannel",
        "WTA",
        "atptour",
    )
    # Ordered so the first SEARCH_PROFILE_MAX_DOMAINS entries are settlement-grade
    # exchanges (in SETTLEMENT_SOURCE_ALLOWLIST_DOMAINS); analytics/news sites that
    # cannot satisfy a direct-evidence primary_source_url follow as fallback.
    CRYPTO_ALLOWED_DOMAINS: tuple[str, ...] = (
        "coinbase.com",
        "kraken.com",
        "binance.com",
        "coindesk.com",
        "cointelegraph.com",
        "theblock.co",
        "decrypt.co",
        "messari.io",
    )
    CRYPTO_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "coinbase",
        "krakenfx",
        "CoinDesk",
        "TheBlock__",
        "WatcherGuru",
        "intocryptoverse",
        "WhaleInsider",
    )
    POLITICS_ALLOWED_DOMAINS: tuple[str, ...] = (
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "politico.com",
        "economist.com",
        "ft.com",
    )
    POLITICS_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "Reuters",
        "ReutersBiz",
        "AP",
        "BBCWorld",
        "politico",
        "WSJ",
        "FT",
    )
    SPEECH_ALLOWED_DOMAINS: tuple[str, ...] = (
        "drudgereport.com",
        "realclearpolitics.com",
        "zerohedge.com",
        "c-span.org",
        "youtube.com",
        "whitehouse.gov",
        "pm.gc.ca",
        "parl.ca",
        "politico.com",
        "reuters.com",
        "apnews.com",
    )
    SPEECH_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "CSPAN",
        "WhiteHouse",
        "POTUS",
        "CanadianPM",
        "Reuters",
        "AP",
        "politico",
        "CBCNews",
        "BBCWorld",
        "WSJ",
    )
    MUSIC_ALLOWED_DOMAINS: tuple[str, ...] = (
        "billboard.com",
        "hitsdailydouble.com",
        "luminate.com",
        "spotifycharts.com",
        "chartmasters.org",
    )
    MUSIC_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "Billboard",
        "SpotifyCharts",
        "LuminateData",
        "HitsDailyDouble",
    )
    WEATHER_ALLOWED_DOMAINS: tuple[str, ...] = (
        "weather.gov",
        "forecast.weather.gov",
        "noaa.gov",
        "tropicaltidbits.com",
        "wunderground.com",
    )
    WEATHER_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "NWS",
        "NWSSPC",
        "NHC_Atlantic",
        "NWSChicago",
        "NWSNewYorkNY",
        "NWSLosAngeles",
        "NWSHouston",
        "NWSMiami",
        "weatherchannel",
    )
    GENERIC_ALLOWED_DOMAINS: tuple[str, ...] = (
        "reuters.com",
        "apnews.com",
        "wsj.com",
        "nasdaq.com",
        "ft.com",
        "economist.com",
    )
    GENERIC_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "Reuters",
        "ReutersBiz",
        "WSJ",
        "FT",
        "CNBC",
        "MarketWatch",
        "YahooFinance",
    )
    ENTERTAINMENT_ALLOWED_DOMAINS: tuple[str, ...] = (
        "netflix.com",
        "top10.netflix.com",
        "flixpatrol.com",
        "boxofficemojo.com",
        "the-numbers.com",
        "variety.com",
        "hollywoodreporter.com",
        "deadline.com",
    )
    ENTERTAINMENT_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "Netflix",
        "NetflixTudum",
        "flixpatrol",
        "BoxOfficeMojo",
        "Variety",
        "THR",
        "DEADLINE",
    )
    # Commodity/index markets classify as the "generic" family but settle on
    # exchange data, so they need a dedicated search profile whose first
    # SEARCH_PROFILE_MAX_DOMAINS entries are reachable settlement-grade pages
    # (CME/ICE/EIA + Tier-1 wires). Without this they inherit GENERIC_ALLOWED_DOMAINS
    # (news wires) and can never cite the exchange settlement URL the commodities
    # prompt requires, so direct evidence is impossible and edges are suppressed.
    COMMODITY_ALLOWED_DOMAINS: tuple[str, ...] = (
        "cmegroup.com",
        "theice.com",
        "eia.gov",
        "wsj.com",
        "bloomberg.com",
        "reuters.com",
        "apnews.com",
    )
    COMMODITY_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "CMEGroup",
        "EIAgov",
        "Reuters",
        "ReutersBiz",
        "business",
        "WSJmarkets",
    )

    # Kalshi
    KALSHI_API_BASE_URL: str = "https://api.elections.kalshi.com/trade-api/v2"
    KALSHI_API_KEY_ID: str = ""
    KALSHI_PRIVATE_KEY_PATH: str = "kalshi-scope.txt"
    KALSHI_SERVER_SIDE_FILTERS_ENABLED: bool = True
    # Hard cap on paginated market fetch requests per cycle (0 = unlimited).
    # Cycle 1 review settled at 30 pages, cycle 2 follow-up found those pages
    # were dominated by KXMVE combo markets and added KALSHI_MVE_FILTER=exclude
    # to drop them server-side. Estimating ~10% of the unbounded 494K-market
    # catalog is non-MVE (~50K markets) means 30 pages * 1000 covers ~60%
    # of the non-MVE window — 50 pages closes the gap. Adaptive top-up
    # (KALSHI_FETCH_TOPUP_ENABLED) and the eligible_floor warning will tell
    # us if we still need to push higher (e.g. 80) under particular catalog
    # conditions.
    KALSHI_MAX_FETCH_PAGES: int = 50
    # Server-side multivariate-event filter for /markets. Cycle 2 follow-up
    # found multivariate-event combo markets dominate the catalog ordering,
    # so excluding them server-side is the cheapest way to keep the page cap
    # focused on individual markets. Allowed values: "exclude" (drop MVE),
    # "only" (keep only MVE), "" / unset (Kalshi default = include both).
    KALSHI_MVE_FILTER: str = "exclude"
    # When eligible_markets falls below this floor on a cycle that hit the
    # page cap, log a structured WARNING so operators can detect "we are
    # running out of catalog before we run out of cap" before the symptom
    # becomes a sustained cycle_yield_alert ERROR.
    KALSHI_ELIGIBLE_FLOOR: int = 100
    # Reserved for the future adaptive top-up path: when True, a cycle that
    # falls below KALSHI_ELIGIBLE_FLOOR with the page cap hit will
    # automatically issue a one-shot follow-up fetch with a doubled cap.
    # Default False — observe the warnings first, then enable.
    KALSHI_FETCH_TOPUP_ENABLED: bool = False

    # Execution
    DRY_RUN: bool = True
    POSITION_SYNC_ENABLED: bool = True
    POSITION_SYNC_INTERVAL_CYCLES: int = 3
    PRE_ORDER_MARKET_REFRESH: bool = True
    MAX_MARKET_DATA_AGE_SECONDS: int = 120
    ORDERBOOK_PRECHECK_ENABLED: bool = True
    ORDERBOOK_PRECHECK_MIN_CONFIDENCE: float = 0.75
    ORDERBOOK_MIN_RESTING_VOLUME: int = 3
    ORDER_PRICE_IMPROVEMENT_CENTS: int = 1
    ORDER_DEFAULT_TIF: str = "gtc"
    ORDER_SUBMISSION_MIN_PRICE: float = 0.03
    ORDER_SUBMISSION_MAX_PRICE: float = 0.97
    ORDER_FALLBACK_TO_MARKET: bool = True
    ORDER_FALLBACK_MIN_CONFIDENCE: float = 0.85
    ORDER_FALLBACK_MIN_LIQUIDITY_USDC: float = 200.0
    CALIBRATION_MODE_ENABLED: bool = True
    CALIBRATION_MIN_SAMPLES: int = 20

    # Probe trade
    PROBE_TRADE_ENABLED: bool = False
    PROBE_TRADE_MAX_USDC: float = 1.0

    # State management
    STATE_DB_PATH: str = "data/market_state.db"
    STATE_JSON_EXPORT_PATH: str = "data/market_state.json"
    EXPORT_STATE_JSON: bool = True

    # Definitive side override
    MAX_DEFINITIVE_OVERRIDES_PER_CYCLE: int = 2

    # Re-analysis controls
    MAX_REANALYSES_PER_MARKET_PER_DAY: int = 2
    REANALYSIS_COOLDOWN_HOURS: int = 6
    URGENT_REANALYSIS_DAYS_BEFORE_CLOSE: int = 1
    URGENT_REANALYSIS_COOLDOWN_HOURS: int = 1
    # Families for which the borderline-trade-confidence refinement trigger
    # should be skipped. Refinement currently spends ~1.5-2 minutes on a
    # second Grok pass for any should_trade=True with confidence in
    # [0.60, 0.78]; for fast-moving markets (sports player props, F5, RFI)
    # the price moves enough during refinement to drop edge below MIN_EDGE,
    # killing the trade. Sports markets already have well-defined statistical
    # priors so the deep pass adds little new information beyond fresher
    # market price. Other refinement triggers (low_evidence_quality,
    # missing_implied_probability, high_conf_small_edge, legacy_borderline_urgent)
    # still fire for these families. Empty tuple disables this behavior.
    REFINEMENT_SKIP_BORDERLINE_FAMILIES: tuple[str, ...] = ()
    PARALLEL_ANALYSIS_ENABLED: bool = True
    ANALYSIS_MAX_WORKERS: int = 2
    MAX_MARKETS_PER_CYCLE: int = 20
    MAX_WEATHER_CANDIDATES_PER_CYCLE: int = 1
    MAX_CRYPTO_CANDIDATES_PER_CYCLE: int = 1
    # Music/speech candidates per cycle were raised from 1 to 2 after the
    # cycle 1 review: the only high-conviction trade in cycle 1 was a music
    # market (KXPUREALBUMS-KEH26APR30-39K) and pre-analysis routinely surfaces
    # a dozen+ music candidates per cycle. Cycle 1 family availability was
    # weather=53, music=14, crypto=10, generic=5 — bumping the music/speech
    # caps shifts selection toward families where direct evidence is most
    # likely. Total per-cycle throughput is still bounded by
    # MAX_MARKETS_PER_CYCLE.
    MAX_SPEECH_CANDIDATES_PER_CYCLE: int = 2
    MAX_MUSIC_CANDIDATES_PER_CYCLE: int = 2
    # Cycle 4 review: sports props consumed every analysis slot even when
    # other families had eligible candidates, leaving direct-evidence
    # opportunities (weather, music, generic) un-analyzed. A positive
    # value caps sports candidates per cycle to reserve room for other
    # families. 0 (default) preserves legacy behavior (no sports-specific cap).
    MAX_SPORTS_CANDIDATES_PER_CYCLE: int = 0
    # Generic is the catch-all family (speech/album/photo-count/macro/etc). A
    # 15-cycle review found it dominated analysis (~48% of slots) yet was
    # majority absence-only (no findable settlement data) and produced 0 fills.
    # A positive value caps generic candidates per cycle so freed slots flow to
    # direct-evidence families. 0 (default) preserves legacy behavior (no cap).
    MAX_GENERIC_CANDIDATES_PER_CYCLE: int = 0
    MAX_TRADES_PER_CYCLE: int = 4
    MAX_BETS_PER_EVENT: int = 2
    MAX_TRADES_PER_DAY: int = 6
    MAX_DAILY_DRAWDOWN_USDC: float = 30.0
    # When the daily drawdown cap is already exceeded, skip Grok analysis for
    # the remainder of the day and route candidates to research_queue with
    # tier=MONITOR_ONLY so we capture the conviction signal without spending
    # API tokens on trades that will be blocked downstream by the same cap.
    DAILY_DRAWDOWN_PREFLIGHT_ENABLED: bool = True
    XAI_CIRCUIT_BREAKER_MAX_FAILURES: int = 3
    XAI_QUOTA_BREAKER_ENABLED: bool = True
    XAI_QUOTA_PAUSE_MINUTES: int = 30
    XAI_CLIENT_TIMEOUT_SECONDS: int = 120
    GROK_STREAM_TIMEOUT_SECONDS: int = 75
    # Deep refinement runs richer prompts and tends to take longer than the
    # initial pass; keep its per-attempt deadline at the pre-fix value of 90s
    # so legitimate refinements (e.g. KXINXU at ~75s) don't timeout.
    GROK_STREAM_TIMEOUT_SECONDS_DEEP: int = 90
    GROK_DEEP_ANALYSIS_MAX_ATTEMPTS: int = 2
    GROK_ANALYSIS_MAX_BUDGET_SECONDS: int = 420
    GROK_SELF_CONSISTENCY_ENABLED: bool = True
    GROK_SELF_CONSISTENCY_LIQUIDITY_THRESHOLD: float = 400.0
    GROK_SELF_CONSISTENCY_EDGE_THRESHOLD: float = 0.15
    # When > 0, the second self-consistency pass only runs for the top-N
    # candidates by pre-analysis score each cycle (the markets most likely to
    # trade), sharply cutting Grok API cost. 0 keeps self-consistency eligible
    # for every analyzed candidate.
    GROK_SELF_CONSISTENCY_TOP_CANDIDATES: int = 0
    GROK_SELF_CONSISTENCY_PRIMARY_TEMPERATURE: float = 0.3
    GROK_SELF_CONSISTENCY_SECONDARY_TEMPERATURE: float = 0.7
    EDGE_REPAIR_ENABLED: bool = True
    EDGE_BAND_CALIBRATION_ENABLED: bool = True
    CONVICTION_REPAIR_ENABLED: bool = True
    # Aligned with the execution edge baseline (MIN_EDGE-tier) so repair
    # eligibility matches the standard actually used to execute; 0.20 parked
    # hundreds of repairable decisions as edge_below_repair_min.
    CONVICTION_REPAIR_MIN_EDGE: float = 0.12
    CONVICTION_REPAIR_MIN_EVIDENCE_QUALITY: float = 0.90
    CONVICTION_REPAIR_SCORE_GAP_MAX: float = 0.08
    CONVICTION_REPAIR_CONFIDENCE_SCORE_FLOOR: float = 0.0
    DAILY_EXPECTANCY_ENABLED: bool = True
    DAILY_EXPECTANCY_PRIMARY_TARGETS: int = 2
    DAILY_EXPECTANCY_SATELLITE_MAX_BET_PCT: float = 0.25

    # Resolution tracking
    RESOLUTION_SYNC_INTERVAL_CYCLES: int = 3

    # Position limits
    MAX_POSITION_PER_MARKET_USDC: float = 200.0
    MAX_POSITION_PCT_OF_BANKROLL: float = 0.15
    MIN_CONFIDENCE_INCREASE_FOR_ADD: float = 0.10
    MIN_PRICE_MOVE_FOR_READD: float = 0.05
    HIGH_CONFIDENCE_POSITION_OVERRIDE: float = 0.85  # Allow adding to position if conf >= this
    OPPOSITE_OUTCOME_STRATEGY: str = "block"  # block|hedge

    # Score gate (phase A/B can run in shadow mode)
    SCORE_GATE_MODE: str = "active"  # off|shadow|active
    SCORE_GATE_THRESHOLD: float = 0.52
    SCORE_GATE_THRESHOLD_WEATHER_DIRECT: float = 0.30
    SCORE_GATE_THRESHOLD_DIRECT_HIGH_QUALITY: float = 0.30
    SCORE_GATE_PROFITABLE_FAMILY_CONVERGENT_ENABLED: bool = True
    SCORE_GATE_THRESHOLD_PROFITABLE_FAMILY_CONVERGENT: float = 0.08
    SCORE_GATE_PROFITABLE_FAMILY_CONVERGENT_MIN_SAMPLES: int = 30
    # Weights for the Kelly / LMSR-inefficiency / Bayesian-posterior model-edge
    # signals inside compute_final_score. These are the strategy signals the bot
    # is meant to follow; they are now applied at both ranking and the execution
    # score gate so genuine edge can clear the gate. See score_engine defaults.
    SCORE_KELLY_COMPONENT_WEIGHT: float = 0.30
    SCORE_INEFFICIENCY_COMPONENT_WEIGHT: float = 0.18
    SCORE_BAYESIAN_COMPONENT_WEIGHT: float = 0.10
    SCORE_LOW_INFO_PENALTY_THRESHOLD: float = 0.60
    SCORE_LOW_INFO_PENALTY_BASE: float = 0.08
    SCORE_REPEATED_ANALYSIS_PENALTY_BASE: float = 0.025
    SCORE_REPEATED_ANALYSIS_PENALTY_START_COUNT: int = 1
    SCORE_CONFIDENCE_CALIBRATION_FLOOR: float = 0.55
    SCORE_CONFIDENCE_CALIBRATION_PENALTY_SCALE: float = 0.10
    SCORE_FALLBACK_EDGE_PENALTY_BASE: float = 0.12
    SCORE_OVERCONFIDENCE_PENALTY_BASE: float = 0.05
    SCORE_COMPUTED_EDGE_BONUS: float = 0.03
    SCORE_SOURCE_CONFIRMED_EDGE_BONUS: float = 0.06
    SCORE_PROXY_EVIDENCE_PENALTY_BASE: float = 0.11
    SCORE_GENERIC_BIN_PENALTY_BASE: float = 0.015
    SCORE_AMBIGUOUS_RESOLUTION_PENALTY_BASE: float = 0.08
    SCORE_HALLUCINATED_EDGE_PENALTY_BASE: float = 0.08
    SCORE_VOLUME_AMPLIFIER_ENABLED: bool = True
    SCORE_EXTREME_MARKET_EDGE_PENALTY_BASE: float = 0.08
    SCORE_LATE_STAGE_OVERCONFIDENCE_PENALTY_BASE: float = 0.12
    SCORE_EXTREME_CONFIDENCE_THRESHOLD: float = 0.90
    SCORE_EXTREME_CONFIDENCE_PENALTY_BASE: float = 0.08
    MENTION_MARKET_SCORE_PENALTY: float = 0.10
    WEATHER_SCORE_PENALTY: float = 0.12
    WEATHER_MIN_EVIDENCE_QUALITY: float = 0.60
    DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_WEATHER: float = 0.72
    DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_SPORTS: float = 0.65
    DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_DEFAULT: float = 0.75
    DIRECT_SOURCE_WHITELIST: tuple[str, ...] = (
        "weather.gov",
        "noaa.gov",
        "wsj.com",
        "bloomberg.com",
        "reuters.com",
        "coindesk.com",
        "kalshi.com",
        "drudgereport.com",
        "zerohedge.com",
        "apnews.com",
        "espn.com",
        "espncricinfo.com",
        "cricbuzz.com",
        "iplt20.com",
        "koreabaseball.com",
        "mlb.com",
        "nfl.com",
        "nba.com",
        "nhl.com",
        "netflix.com",
        "top10.netflix.com",
        "flixpatrol.com",
        "boxofficemojo.com",
        "the-numbers.com",
    )
    PRE_ANALYSIS_OPPORTUNITY_ENABLED: bool = True
    PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE: float = 0.28
    PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND: float = 0.20
    PRE_ANALYSIS_OPPORTUNITY_RESEARCH_ENABLED: bool = True
    # Adaptive widening of the soft-research band when sustained zero-execution
    # cycles indicate calibration drift. Only widens the routing band that
    # captures markets for learning; the deep-analysis MIN_SCORE itself is
    # NEVER moved by this knob, so execution gating stays unchanged. The
    # widening is linear: per cycle beyond 2 * CYCLE_YIELD_ALERT_ESCALATE_AFTER
    # consecutive zero-execution cycles, the band grows by 0.02 up to BAND_MAX.
    PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_ADAPTIVE_ENABLED: bool = True
    PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_MAX: float = 0.30
    PRE_ANALYSIS_MUST_ANALYZE_THRESHOLD: float = 0.50
    PRE_ANALYSIS_REDUCED_MAX_CANDIDATES: int = 8
    PRE_ANALYSIS_NON_ACTIONABLE_STREAK_PENALTY: float = 0.25
    PRE_ANALYSIS_NON_ACTIONABLE_STREAK_CAP: int = 8
    PRE_ANALYSIS_ANALYSIS_COUNT_PENALTY: float = 0.15
    PRE_ANALYSIS_ANALYSIS_COUNT_START: int = 1
    PRE_ANALYSIS_FAMILY_PENALTY_SPEECH: float = 0.10
    PRE_ANALYSIS_FAMILY_PENALTY_MUSIC: float = 0.08
    PRE_ANALYSIS_FAMILY_PENALTY_SPORTS: float = 0.0
    PRE_ANALYSIS_FAMILY_PENALTY_WEATHER_BIN: float = 0.05
    PRE_ANALYSIS_FAMILY_PENALTY_GENERIC_BIN: float = 0.10
    PRE_ANALYSIS_FAMILY_PENALTY_CRYPTO_BIN: float = 0.06
    PRE_ANALYSIS_ZERO_TRADE_RATE_PENALTY: float = 0.04
    PRE_ANALYSIS_FALLBACK_FAMILY_RATE_THRESHOLD: float = 0.80
    PRE_ANALYSIS_FALLBACK_FAMILY_PENALTY: float = 0.12
    PRE_ANALYSIS_FALLBACK_FAMILY_MIN_SAMPLES: int = 20
    PRE_ANALYSIS_HISTORICAL_FAMILY_MIN_SAMPLES: int = 10
    PRE_ANALYSIS_HISTORICAL_FAMILY_WIN_RATE_THRESHOLD: float = 0.45
    PRE_ANALYSIS_HISTORICAL_FAMILY_PENALTY: float = 0.12
    PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_MIN_SAMPLES: int = 20
    PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_THRESHOLD: float = -5.0
    PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_PENALTY: float = 0.10
    PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_THRESHOLD: float = -15.0
    PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_PENALTY: float = 0.15
    PRE_ANALYSIS_ADAPTIVE_BOOST: float = 0.03
    # Cap on the combined "stacked historical-family penalty" set in
    # _pre_analysis_opportunity_score. The fallback-family, historical-family
    # win-rate, historical-family PnL, zero-trade-rate, negative-prefix and
    # historical-gate score-penalty terms all draw from overlapping historical
    # data sources; without a cap a single bad family/prefix can collapse a
    # market's pre-analysis score by ~0.65pp and force soft-research routing
    # even for liquid, well-priced near-event markets. Setting this to <=0
    # disables the cap (legacy behavior); otherwise the excess is credited
    # back to the score and recorded under
    # ``pre_score_stacked_historical_excess_credited`` for telemetry.
    PRE_ANALYSIS_STACKED_HISTORICAL_PENALTY_CAP: float = 0.25
    PRE_ANALYSIS_CRYPTO_NEGATIVE_PNL_BLOCK_ENABLED: bool = True
    PRE_ANALYSIS_CRYPTO_NEGATIVE_PNL_THRESHOLD: float = 0.0
    PRE_ANALYSIS_CRYPTO_FALLBACK_RATE_BLOCK_THRESHOLD: float = 0.55
    PRE_ANALYSIS_CRYPTO_MIN_SAMPLES: int = 20
    MAX_LIFETIME_ANALYSES_PER_MARKET: int = 8
    PRE_ANALYSIS_HARD_REJECTION_ENABLED: bool = True
    PRE_ANALYSIS_HARD_REJECTION_MIN_STREAK: int = 3
    PRE_ANALYSIS_HARD_REJECTION_MIN_ANALYSES: int = 5
    HISTORICAL_TICKER_PREFIX_GATE_ENABLED: bool = True
    HISTORICAL_TICKER_PREFIX_LEN: int = 12
    HISTORICAL_TICKER_PREFIX_LOOKBACK_DAYS: int = 30
    HISTORICAL_TICKER_PREFIX_MIN_SAMPLES: int = 3
    HISTORICAL_TICKER_PREFIX_PNL_CUTOFF: float = -2.0
    HISTORICAL_TICKER_PREFIX_WIN_RATE_CUTOFF: float = 0.40
    HISTORICAL_FAMILY_GATE_ENABLED: bool = True
    HISTORICAL_FAMILY_LOOKBACK_DAYS: int = 30
    HISTORICAL_FAMILY_MIN_SAMPLES: int = 12
    HISTORICAL_FAMILY_PNL_CUTOFF: float = -12.0
    HISTORICAL_FAMILY_WIN_RATE_CUTOFF: float = 0.40
    # Per-trade Bayesian-shrunk PnL cutoff for family hard-deny (distinct from
    # HISTORICAL_TICKER_PREFIX_SHRUNK_PNL_CUTOFF so family/prefix bars can diverge).
    HISTORICAL_FAMILY_SHRUNK_PNL_CUTOFF: float = -0.50
    HISTORICAL_FAMILY_SIGNAL_ENABLED: bool = True
    HISTORICAL_FAMILY_SCORE_SCALE: float = 0.06
    HISTORICAL_FAMILY_SIZE_SCALE_MAX: float = 0.25
    # Downward size authority for families with a negative historical signal.
    # Kept >= HISTORICAL_FAMILY_SIZE_SCALE_MAX so persistent losers can be
    # shrunk more aggressively than winners are inflated (oversizing a losing
    # family is the dominant drawdown risk). Defaults to the symmetric value;
    # raise it in .env to harden the de-risk.
    HISTORICAL_FAMILY_SIZE_SCALE_MAX_NEGATIVE: float = 0.25
    HISTORICAL_SHORT_PREFIX_LEN: int = 5
    HISTORICAL_SHORT_PREFIX_MIN_SAMPLES: int = 3
    HISTORICAL_SHORT_PREFIX_PNL_CUTOFF: float = -5.0
    HISTORICAL_SHORT_PREFIX_SCORE_PENALTY: float = 0.10
    HISTORICAL_TICKER_PREFIX_HARD_BLOCK_MIN_SAMPLES: int = 20
    HISTORICAL_TICKER_PREFIX_SHRINKAGE_ENABLED: bool = True
    HISTORICAL_TICKER_PREFIX_PRIOR_WIN_RATE: float = 0.50
    HISTORICAL_TICKER_PREFIX_PRIOR_STRENGTH: float = 10.0
    HISTORICAL_TICKER_PREFIX_SHRUNK_PNL_CUTOFF: float = -0.50
    HISTORICAL_TICKER_PREFIX_SOFT_DEMOTE_SCORE_PENALTY: float = 0.08
    STRONG_EVIDENCE_CONFIDENCE_FLOOR: float = 0.55
    STRONG_EVIDENCE_MIN_EVIDENCE_QUALITY: float = 0.85
    STRONG_EVIDENCE_PROXY_MIN_EVIDENCE_QUALITY: float = 0.95
    STRONG_EVIDENCE_PROXY_MIN_EDGE: float = 0.20
    # Proxy markets whose market edge clears this bar bypass the preview/proxy
    # validation blocks in grok_client. The downstream edge gate and per-family
    # size multiplier still apply, so historically weak families are sized down
    # rather than hard-blocked at validation. Set to 1.0 to disable.
    PROXY_HIGH_EDGE_PARTICIPATION_MIN_EDGE: float = 0.15
    # Generic family historically underperforms on proxy evidence; require a
    # higher market edge before the proxy high-edge participation override fires.
    GENERIC_PROXY_HIGH_EDGE_MIN: float = 0.18
    GROK_PROXY_CONFIDENCE_CAP: float = 0.78
    GROK_LOW_INFO_CONFIDENCE_CAP: float = 0.70
    GROK_FALLBACK_MIN_EVIDENCE_QUALITY: float = 0.45
    GROK_ABSTAIN_EVIDENCE_THRESHOLD: float = 0.35
    CONFIDENCE_SHRINKAGE_FLOOR: float = 0.55
    CONFIDENCE_SHRINKAGE_FACTOR: float = 0.32
    CONFIDENCE_SHRINKAGE_FACTOR_HIGH: float = 0.28
    CALIBRATION_DIRECT_SHRINKAGE_FACTOR_BOOST: float = 2.0
    CALIBRATION_ONLINE_UPDATE_ENABLED: bool = True
    CALIBRATION_ONLINE_ALPHA: float = 0.15
    CALIBRATION_ONLINE_MAX_SAMPLES_PER_BUCKET: int = 500
    HISTORICAL_CONFIDENCE_SHRINK_ENABLED: bool = True
    HISTORICAL_CONFIDENCE_SHRINK_MIN_SAMPLES: int = 15
    HISTORICAL_CONFIDENCE_SHRINK_LOOKBACK_DAYS: int = 30
    # Cap on how far the historical-bucket shrink may pull confidence down in a
    # single pass. Without it the bucket can deflate confidence so much (a
    # 10-cycle review saw 0.57 -> 0.46 across 130/136 markets) that nothing
    # clears MIN_CONFIDENCE / the score gate, which prevents the winning trades
    # that would recalibrate the bucket -- a self-reinforcing no-trade spiral.
    HISTORICAL_CONFIDENCE_SHRINK_MAX_DELTA: float = 0.05
    # Only apply the historical shrink when stage-one confidence is above this
    # band; below it there is no overconfidence to correct and shrinking only
    # destroys tradeable edge. 0.0 disables the band (cap still applies).
    HISTORICAL_CONFIDENCE_SHRINK_MIN_CONFIDENCE: float = 0.0
    RESEARCH_QUEUE_ENABLED: bool = True
    RESEARCH_QUEUE_PERSIST_TO_DB: bool = True
    RESEARCH_QUEUE_REUSE_LOOKBACK_HOURS: int = 6
    RESEARCH_QUEUE_PRIORITY_ENABLED: bool = True
    # Bound on the per-cycle research-queue capture log used for cycle-receipt
    # telemetry and the "Research queue captured N blocked opportunities" log
    # line. The DB persists EVERY entry regardless of this cap; only the
    # in-memory cycle log is bounded. Operators can raise this when triaging
    # large queue cycles or lower to reduce log payload size.
    RESEARCH_QUEUE_CYCLE_LOG_MAXLEN: int = 200
    # Periodically promote stale research-queued markets back to deep analysis so
    # the queue is not a write-only black hole. Conservative defaults: at most one
    # forced probe per cycle, only after the entry has aged at least an hour.
    RESEARCH_QUEUE_DRAIN_ENABLED: bool = True
    RESEARCH_QUEUE_DRAIN_PER_CYCLE: int = 1
    RESEARCH_QUEUE_DRAIN_MIN_AGE_HOURS: float = 1.0
    RESEARCH_QUEUE_DRAIN_MAX_AGE_HOURS: float = 12.0
    RESEARCH_QUEUE_DRAIN_MIN_PRIORITY: float = 0.40
    RESEARCH_QUEUE_DRAIN_FORCE_EXTENDED_RESEARCH: bool = True
    RESEARCH_QUEUE_DRAIN_RETRY_COOLDOWN_MINUTES: float = 45.0
    RESEARCH_QUEUE_ZERO_YIELD_PROMOTIONS: int = 1
    RESEARCH_QUEUE_ZERO_YIELD_PROMOTIONS_MAX: int = 1
    RESEARCH_QUEUE_SCORE_PROMOTION_GAP: float = 0.05
    RESEARCH_QUEUE_LOW_YIELD_PLACEHOLDER_MIN_ATTEMPTS: int = 4
    RESEARCH_QUEUE_LOW_YIELD_PLACEHOLDER_MIN_TIMES_SEEN: int = 8
    EXTENDED_RESEARCH_AFTER_STREAK: int = 2
    EXTENDED_RESEARCH_COOLDOWN_CYCLES: int = 3
    # Near-miss / research_queued after extended research uses a shorter cooldown
    # so soft candidates re-enter the normal analysis pool sooner than hard skips.
    EXTENDED_RESEARCH_QUEUE_COOLDOWN_CYCLES: int = 2
    DEFINITIVE_OUTCOME_EVIDENCE_QUALITY_FLOOR: float = 0.80
    # Edge cap for definitive-outcome and high-quality direct settled markets.
    # Raised from 0.40 to 0.50 alongside the MAX_REASONABLE_EDGE bump so the
    # "validated" path retains a meaningful headroom over the generic cap.
    DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX: float = 0.50
    # Minimum evidence_quality for the high-quality settlement-aligned exemption
    # used by the edge gate and hallucinated_edge penalty suppression. The
    # exemption recognizes direct + settlement_aligned + whitelisted-source
    # evidence without requiring definitive_outcome_detected=True; the strict
    # eq floor compensates by demanding near-perfect evidence quality.
    HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ: float = 0.95
    PARTICIPATION_TIER_AUDIT_ENABLED: bool = True
    PARTICIPATION_TIER_GATING_ENABLED: bool = True
    # Escalate the per-cycle "zero execution candidates with research_queue >50"
    # warning to ERROR after this many consecutive cycles so predictbot_errors.log
    # captures sustained selection failure. 0 disables escalation.
    CYCLE_YIELD_ALERT_ESCALATE_AFTER: int = 2
    NEGATIVE_BEST_SCORE_DEEP_ANALYSIS_FLOOR: float = 0.05
    CRYPTO_PREFLIGHT_ENABLED: bool = False
    CRYPTO_THRESHOLD_BUFFER_AUTO_NO_TRADE_PCT: float = 0.50
    GROK_STREAM_TIMEOUT_SECONDS_CRYPTO: int = 120
    # Weather profile per-attempt timeout. Falls back to
    # GROK_STREAM_TIMEOUT_SECONDS when set to 0. Heavy NWS observation
    # prompts routinely stream 90-100s; 120s avoids clipping the p95 tail.
    GROK_STREAM_TIMEOUT_SECONDS_WEATHER: int = 120
    TIMEOUT_RETRY_AS_MONITOR_ONLY_ENABLED: bool = True

    # Bayesian + LMSR + Kelly experimental layers
    BAYESIAN_ENABLED: bool = False
    BAYESIAN_SKIP_STALE_UPDATES: bool = True
    BAYESIAN_PRIOR_DEFAULT: float = 0.50
    # Raised 1 -> 3: a single neutral update leaves the posterior ~= the 0.50
    # prior, and that uninformative posterior was overwriting the model's
    # calibrated confidence (collapsing it to 0.50) on fresh threshold markets.
    BAYESIAN_MIN_UPDATES_FOR_TRADE: int = 3
    # Do not apply the posterior when it is within this distance of the prior
    # (uninformative). Keeps the model's calibrated confidence instead of
    # reverting it to the prior. 0 disables the guard (legacy behavior).
    BAYESIAN_MIN_POSTERIOR_DIVERGENCE: float = 0.05
    BAYESIAN_MAX_POSTERIOR: float = 0.90
    BAYESIAN_MAX_CONFIDENCE_BOOST: float = 0.15
    LMSR_ENABLED: bool = False
    LMSR_LIQUIDITY_PARAM_B: float = 100000.0
    LMSR_MIN_INEFFICIENCY: float = 0.05
    KELLY_SIZING_ENABLED: bool = False
    KELLY_DYNAMIC_ENABLED: bool = True
    KELLY_FRACTION_DEFAULT: float = 0.45
    KELLY_FRACTION_SHORT_HORIZON_HOURS: int = 1
    KELLY_FRACTION_SHORT_HORIZON: float = 0.10
    KELLY_FRACTION_WEATHER: float = 0.50
    KELLY_MIN_BET_POLICY: str = "skip"  # skip|floor|fallback_edge_scaling
    # When policy=skip and dynamic Kelly qualifies, floor bets within this
    # fraction of MIN_BET instead of skipping (e.g. 0.60 → $1.20 of $2.00).
    KELLY_MIN_BET_NEAR_MISS_RATIO: float = 0.60
    KELLY_MIN_BANKROLL_USDC: float = 30.0

    # Side-flip guardrails
    FLIP_GUARD_ENABLED: bool = True
    FLIP_GUARD_MIN_ABS_CONFIDENCE: float = 0.65
    FLIP_GUARD_MIN_CONF_GAIN: float = 0.08
    FLIP_GUARD_MIN_EDGE_GAIN: float = 0.03
    FLIP_GUARD_MIN_EVIDENCE_QUALITY: float = 0.60
    # Direct-evidence flip bypass: fresh direct, settlement-aligned evidence with
    # a strong edge legitimately overrides a stale anchor even when the new
    # confidence is lower than the anchor's (negative conf_delta). Without this,
    # a deliberate refinement flip on ground-truth evidence is wrongly blocked.
    FLIP_GUARD_DIRECT_EVIDENCE_OVERRIDE_ENABLED: bool = True
    FLIP_GUARD_DIRECT_MIN_EDGE: float = 0.15
    FLIP_GUARD_DIRECT_MIN_LIKELIHOOD_RATIO: float = 5.0
    FLIP_CIRCUIT_BREAKER_ENABLED: bool = True
    FLIP_CIRCUIT_BREAKER_MAX_FLIPS: int = 3
    EVIDENCE_QUALITY_HIGH_CONFIDENCE_OVERRIDE: bool = False
    EVIDENCE_QUALITY_CONVERGENT_FLOOR_ENABLED: bool = True
    EVIDENCE_QUALITY_CONVERGENT_FLOOR_VALUE: float = 0.60
    PROXY_PENALTY_CONVERGENT_REDUCTION_ENABLED: bool = True
    HISTORICAL_FAMILY_HIGH_CONF_LOSS_RELAX_THRESHOLD: float = 0.05
    HISTORICAL_FAMILY_BOOST_EVIDENCE_MIN: float = 0.44
    HISTORICAL_FAMILY_LOSS_DRAG_SCALE: float = 1.8
    HISTORICAL_FAMILY_LOSS_DRAG_SAMPLE_MIN: int = 30
    PRE_ANALYSIS_HISTORICAL_FAMILY_PROFIT_BONUS: float = 0.10
    BORDERLINE_CRITIQUE_REFINEMENT_ENABLED: bool = True
    BORDERLINE_CRITIQUE_REFINEMENT_SCORE_BAND: float = 0.10
    CODE_EXECUTION_FOR_DEEP_ANALYSIS_ENABLED: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE_LEVEL: str = "DEBUG"
    LOG_DIR: str = "logs"
    ENABLE_FILE_LOGGING: bool = True
    ENABLE_JSON_LOGGING: bool = True
    ENABLE_COLORED_LOGGING: bool = True
    API_COST_INPUT_PER_1K_TOKENS_USD: float = 0.0
    API_COST_OUTPUT_PER_1K_TOKENS_USD: float = 0.0


BASE_REQUIRED_ENV_VARS = (
    "XAI_API_KEY",
    "KALSHI_API_KEY_ID",
    "KALSHI_PRIVATE_KEY_PATH",
)


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    items = [item.strip() for item in value.split(",")]
    return tuple(item for item in items if item)


def _read_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw


def _read_env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    return _split_csv(raw)


def _read_env_float_pair(
    name: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        left, right = [part.strip() for part in raw.split(",", maxsplit=1)]
        return (float(left), float(right))
    except (ValueError, TypeError):
        return default


def _read_env_int_optional(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if not raw or raw.strip().lower() in {"", "none", "null"}:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_settings() -> Settings:
    legacy_model_aliases: dict[str, str] = {}

    requested_model_initial = _read_env_str("GROK_MODEL", Settings.GROK_MODEL).strip()
    normalized_model_initial = legacy_model_aliases.get(
        requested_model_initial,
        requested_model_initial,
    )
    requested_model_deep = _read_env_str(
        "GROK_MODEL_DEEP",
        Settings.GROK_MODEL_DEEP,
    ).strip()
    normalized_model_deep = legacy_model_aliases.get(
        requested_model_deep,
        requested_model_deep,
    )

    settings = Settings(
        MIN_BET_USDC=_read_env_float("MIN_BET_USDC", Settings.MIN_BET_USDC),
        MAX_BET_USDC=_read_env_float("MAX_BET_USDC", Settings.MAX_BET_USDC),
        MIN_CONFIDENCE=_read_env_float("MIN_CONFIDENCE", Settings.MIN_CONFIDENCE),
        CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED=_read_env_bool(
            "CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED",
            Settings.CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED,
        ),
        CONFIDENCE_GATE_MIN_EDGE=_read_env_float(
            "CONFIDENCE_GATE_MIN_EDGE", Settings.CONFIDENCE_GATE_MIN_EDGE
        ),
        CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY=_read_env_float(
            "CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY",
            Settings.CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY,
        ),
        CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE=_read_env_float(
            "CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE",
            Settings.CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE,
        ),
        DIRECT_POSTERIOR_FLOOR_ENABLED=_read_env_bool(
            "DIRECT_POSTERIOR_FLOOR_ENABLED",
            Settings.DIRECT_POSTERIOR_FLOOR_ENABLED,
        ),
        DIRECT_POSTERIOR_FLOOR_MIN_EVIDENCE_QUALITY=_read_env_float(
            "DIRECT_POSTERIOR_FLOOR_MIN_EVIDENCE_QUALITY",
            Settings.DIRECT_POSTERIOR_FLOOR_MIN_EVIDENCE_QUALITY,
        ),
        DIRECT_POSTERIOR_FLOOR_MAX_HOURS_TO_CLOSE=_read_env_float(
            "DIRECT_POSTERIOR_FLOOR_MAX_HOURS_TO_CLOSE",
            Settings.DIRECT_POSTERIOR_FLOOR_MAX_HOURS_TO_CLOSE,
        ),
        MIN_EVIDENCE_QUALITY_FOR_TRADE=_read_env_float(
            "MIN_EVIDENCE_QUALITY_FOR_TRADE",
            Settings.MIN_EVIDENCE_QUALITY_FOR_TRADE,
        ),
        SPORTS_MIN_EVIDENCE_QUALITY=_read_env_float(
            "SPORTS_MIN_EVIDENCE_QUALITY",
            Settings.SPORTS_MIN_EVIDENCE_QUALITY,
        ),
        MIN_EDGE=_read_env_float("MIN_EDGE", Settings.MIN_EDGE),
        MIN_EDGE_HIGH_LIQUIDITY_THRESHOLD=_read_env_float(
            "MIN_EDGE_HIGH_LIQUIDITY_THRESHOLD",
            Settings.MIN_EDGE_HIGH_LIQUIDITY_THRESHOLD,
        ),
        MIN_EDGE_HIGH_LIQUIDITY_MULTIPLIER=_read_env_float(
            "MIN_EDGE_HIGH_LIQUIDITY_MULTIPLIER",
            Settings.MIN_EDGE_HIGH_LIQUIDITY_MULTIPLIER,
        ),
        MIN_EDGE_MEDIUM_LIQUIDITY_THRESHOLD=_read_env_float(
            "MIN_EDGE_MEDIUM_LIQUIDITY_THRESHOLD",
            Settings.MIN_EDGE_MEDIUM_LIQUIDITY_THRESHOLD,
        ),
        MIN_EDGE_MEDIUM_LIQUIDITY_MULTIPLIER=_read_env_float(
            "MIN_EDGE_MEDIUM_LIQUIDITY_MULTIPLIER",
            Settings.MIN_EDGE_MEDIUM_LIQUIDITY_MULTIPLIER,
        ),
        LOW_PRICE_THRESHOLD=_read_env_float(
            "LOW_PRICE_THRESHOLD", Settings.LOW_PRICE_THRESHOLD
        ),
        VERY_LOW_PRICE_THRESHOLD=_read_env_float(
            "VERY_LOW_PRICE_THRESHOLD", Settings.VERY_LOW_PRICE_THRESHOLD
        ),
        ENTRY_PRICE_FLOOR_EDGE_OVERRIDE_ENABLED=_read_env_bool(
            "ENTRY_PRICE_FLOOR_EDGE_OVERRIDE_ENABLED",
            Settings.ENTRY_PRICE_FLOOR_EDGE_OVERRIDE_ENABLED,
        ),
        ENTRY_PRICE_FLOOR_OVERRIDE_MIN_EDGE=_read_env_float(
            "ENTRY_PRICE_FLOOR_OVERRIDE_MIN_EDGE",
            Settings.ENTRY_PRICE_FLOOR_OVERRIDE_MIN_EDGE,
        ),
        ENTRY_PRICE_FLOOR_OVERRIDE_MIN_EVIDENCE_QUALITY=_read_env_float(
            "ENTRY_PRICE_FLOOR_OVERRIDE_MIN_EVIDENCE_QUALITY",
            Settings.ENTRY_PRICE_FLOOR_OVERRIDE_MIN_EVIDENCE_QUALITY,
        ),
        HIGH_PRICE_THRESHOLD=_read_env_float(
            "HIGH_PRICE_THRESHOLD", Settings.HIGH_PRICE_THRESHOLD
        ),
        LOW_PRICE_MIN_EDGE=_read_env_float(
            "LOW_PRICE_MIN_EDGE", Settings.LOW_PRICE_MIN_EDGE
        ),
        VERY_LOW_PRICE_MIN_EDGE=_read_env_float(
            "VERY_LOW_PRICE_MIN_EDGE", Settings.VERY_LOW_PRICE_MIN_EDGE
        ),
        LOW_PRICE_MIN_EDGE_MULTIPLIER=_read_env_float(
            "LOW_PRICE_MIN_EDGE_MULTIPLIER",
            Settings.LOW_PRICE_MIN_EDGE_MULTIPLIER,
        ),
        COINFLIP_PRICE_LOWER=_read_env_float(
            "COINFLIP_PRICE_LOWER", Settings.COINFLIP_PRICE_LOWER
        ),
        COINFLIP_PRICE_UPPER=_read_env_float(
            "COINFLIP_PRICE_UPPER", Settings.COINFLIP_PRICE_UPPER
        ),
        EDGE_SCALING_RANGE=_read_env_float(
            "EDGE_SCALING_RANGE", Settings.EDGE_SCALING_RANGE
        ),
        LOW_PRICE_BET_PENALTY=_read_env_float(
            "LOW_PRICE_BET_PENALTY", Settings.LOW_PRICE_BET_PENALTY
        ),
        FALLBACK_EDGE_MIN_EDGE=_read_env_float(
            "FALLBACK_EDGE_MIN_EDGE", Settings.FALLBACK_EDGE_MIN_EDGE
        ),
        FALLBACK_EDGE_MIN_EDGE_MULTIPLIER=_read_env_float(
            "FALLBACK_EDGE_MIN_EDGE_MULTIPLIER",
            Settings.FALLBACK_EDGE_MIN_EDGE_MULTIPLIER,
        ),
        WEATHER_MIN_EDGE=_read_env_float(
            "WEATHER_MIN_EDGE", Settings.WEATHER_MIN_EDGE
        ),
        WEATHER_HIGH_EQ_EDGE_MULTIPLIER=_read_env_float(
            "WEATHER_HIGH_EQ_EDGE_MULTIPLIER",
            Settings.WEATHER_HIGH_EQ_EDGE_MULTIPLIER,
        ),
        WEATHER_FALLBACK_EDGE_MIN_EDGE=_read_env_float(
            "WEATHER_FALLBACK_EDGE_MIN_EDGE",
            Settings.WEATHER_FALLBACK_EDGE_MIN_EDGE,
        ),
        WEATHER_BLOCK_UNDERDOG_ENTRIES=_read_env_bool(
            "WEATHER_BLOCK_UNDERDOG_ENTRIES",
            Settings.WEATHER_BLOCK_UNDERDOG_ENTRIES,
        ),
        WEATHER_POSTERIOR_FLOOR_MAX_EDGE=_read_env_float(
            "WEATHER_POSTERIOR_FLOOR_MAX_EDGE",
            Settings.WEATHER_POSTERIOR_FLOOR_MAX_EDGE,
        ),
        WEATHER_CALIBRATION_GAP_FOR_KELLY_SHRINK=_read_env_float(
            "WEATHER_CALIBRATION_GAP_FOR_KELLY_SHRINK",
            Settings.WEATHER_CALIBRATION_GAP_FOR_KELLY_SHRINK,
        ),
        WEATHER_CALIBRATION_GAP_KELLY_MULTIPLIER=_read_env_float(
            "WEATHER_CALIBRATION_GAP_KELLY_MULTIPLIER",
            Settings.WEATHER_CALIBRATION_GAP_KELLY_MULTIPLIER,
        ),
        COMMODITY_MIN_EDGE=_read_env_float(
            "COMMODITY_MIN_EDGE", Settings.COMMODITY_MIN_EDGE
        ),
        COMMODITY_HIGH_EQ_EDGE_MULTIPLIER=_read_env_float(
            "COMMODITY_HIGH_EQ_EDGE_MULTIPLIER",
            Settings.COMMODITY_HIGH_EQ_EDGE_MULTIPLIER,
        ),
        COMMODITY_HIGH_EQ_MIN_EVIDENCE_QUALITY=_read_env_float(
            "COMMODITY_HIGH_EQ_MIN_EVIDENCE_QUALITY",
            Settings.COMMODITY_HIGH_EQ_MIN_EVIDENCE_QUALITY,
        ),
        REQUIRE_IMPLIED_PRICE=_read_env_bool(
            "REQUIRE_IMPLIED_PRICE", Settings.REQUIRE_IMPLIED_PRICE
        ),
        MAX_GLOBAL_CONFIDENCE=_read_env_float(
            "MAX_GLOBAL_CONFIDENCE", Settings.MAX_GLOBAL_CONFIDENCE
        ),
        MAX_GLOBAL_CONFIDENCE_DIRECT=_read_env_float(
            "MAX_GLOBAL_CONFIDENCE_DIRECT", Settings.MAX_GLOBAL_CONFIDENCE_DIRECT
        ),
        MAX_SPORTS_CONFIDENCE=_read_env_float(
            "MAX_SPORTS_CONFIDENCE", Settings.MAX_SPORTS_CONFIDENCE
        ),
        MAX_ESPORTS_CONFIDENCE=_read_env_float(
            "MAX_ESPORTS_CONFIDENCE", Settings.MAX_ESPORTS_CONFIDENCE
        ),
        MAX_WEATHER_CONFIDENCE=_read_env_float(
            "MAX_WEATHER_CONFIDENCE", Settings.MAX_WEATHER_CONFIDENCE
        ),
        MAX_INDEX_CONFIDENCE=_read_env_float(
            "MAX_INDEX_CONFIDENCE", Settings.MAX_INDEX_CONFIDENCE
        ),
        MAX_COMMODITY_CONFIDENCE=_read_env_float(
            "MAX_COMMODITY_CONFIDENCE", Settings.MAX_COMMODITY_CONFIDENCE
        ),
        MAX_LIVESTOCK_CONFIDENCE=_read_env_float(
            "MAX_LIVESTOCK_CONFIDENCE", Settings.MAX_LIVESTOCK_CONFIDENCE
        ),
        MAX_HEATING_OIL_CONFIDENCE=_read_env_float(
            "MAX_HEATING_OIL_CONFIDENCE", Settings.MAX_HEATING_OIL_CONFIDENCE
        ),
        MAX_CORN_CONFIDENCE=_read_env_float(
            "MAX_CORN_CONFIDENCE", Settings.MAX_CORN_CONFIDENCE
        ),
        MAX_CRYPTO_CONFIDENCE=_read_env_float(
            "MAX_CRYPTO_CONFIDENCE", Settings.MAX_CRYPTO_CONFIDENCE
        ),
        MAX_SPEECH_CONFIDENCE=_read_env_float(
            "MAX_SPEECH_CONFIDENCE", Settings.MAX_SPEECH_CONFIDENCE
        ),
        MAX_REASONABLE_EDGE=_read_env_float(
            "MAX_REASONABLE_EDGE", Settings.MAX_REASONABLE_EDGE
        ),
        NON_SPORTS_REQUIRES_DIRECT_EVIDENCE=_read_env_bool(
            "NON_SPORTS_REQUIRES_DIRECT_EVIDENCE",
            Settings.NON_SPORTS_REQUIRES_DIRECT_EVIDENCE,
        ),
        NON_SPORTS_REQUIRES_PRIMARY_SOURCE_URL=_read_env_bool(
            "NON_SPORTS_REQUIRES_PRIMARY_SOURCE_URL",
            Settings.NON_SPORTS_REQUIRES_PRIMARY_SOURCE_URL,
        ),
        PRIMARY_SOURCE_URL_EXEMPT_FAMILIES=_read_env_csv(
            "PRIMARY_SOURCE_URL_EXEMPT_FAMILIES",
            Settings.PRIMARY_SOURCE_URL_EXEMPT_FAMILIES,
        ),
        MIN_LIQUIDITY_USDC=_read_env_float(
            "MIN_LIQUIDITY_USDC", Settings.MIN_LIQUIDITY_USDC
        ),
        POLL_INTERVAL_SEC=_read_env_int(
            "POLL_INTERVAL_SEC", Settings.POLL_INTERVAL_SEC
        ),
        DRY_STREAK_SLEEP_ENABLED=_read_env_bool(
            "DRY_STREAK_SLEEP_ENABLED",
            Settings.DRY_STREAK_SLEEP_ENABLED,
        ),
        MARKET_CATEGORIES_ALLOWLIST=_split_csv(
            os.getenv("MARKET_CATEGORIES_ALLOWLIST")
        ),
        MARKET_CATEGORIES_BLOCKLIST=_split_csv(
            os.getenv("MARKET_CATEGORIES_BLOCKLIST")
        ),
        MARKET_FAMILY_BLOCKLIST=_read_env_csv(
            "MARKET_FAMILY_BLOCKLIST",
            Settings.MARKET_FAMILY_BLOCKLIST,
        ),
        MARKET_TICKER_BLOCKLIST_PREFIXES=_read_env_csv(
            "MARKET_TICKER_BLOCKLIST_PREFIXES",
            Settings.MARKET_TICKER_BLOCKLIST_PREFIXES,
        ),
        SKIP_WEATHER_BIN_MARKETS=_read_env_bool(
            "SKIP_WEATHER_BIN_MARKETS", Settings.SKIP_WEATHER_BIN_MARKETS
        ),
        CRYPTO_BIN_MARKET_BLOCKLIST_ENABLED=_read_env_bool(
            "CRYPTO_BIN_MARKET_BLOCKLIST_ENABLED",
            Settings.CRYPTO_BIN_MARKET_BLOCKLIST_ENABLED,
        ),
        MIN_VOLUME_24H=_read_env_float("MIN_VOLUME_24H", Settings.MIN_VOLUME_24H),
        MIN_OPEN_INTEREST=_read_env_float(
            "MIN_OPEN_INTEREST", Settings.MIN_OPEN_INTEREST
        ),
        EXTREME_YES_PRICE_LOWER=_read_env_float(
            "EXTREME_YES_PRICE_LOWER",
            Settings.EXTREME_YES_PRICE_LOWER,
        ),
        EXTREME_YES_PRICE_UPPER=_read_env_float(
            "EXTREME_YES_PRICE_UPPER",
            Settings.EXTREME_YES_PRICE_UPPER,
        ),
        MIN_TRADEABLE_IMPLIED_PRICE=_read_env_float(
            "MIN_TRADEABLE_IMPLIED_PRICE",
            Settings.MIN_TRADEABLE_IMPLIED_PRICE,
        ),
        MAX_TRADEABLE_IMPLIED_PRICE=_read_env_float(
            "MAX_TRADEABLE_IMPLIED_PRICE",
            Settings.MAX_TRADEABLE_IMPLIED_PRICE,
        ),
        LADDER_COLLAPSE_THRESHOLD=_read_env_int(
            "LADDER_COLLAPSE_THRESHOLD",
            Settings.LADDER_COLLAPSE_THRESHOLD,
        ),
        MAX_BRACKETS_PER_EVENT=_read_env_int(
            "MAX_BRACKETS_PER_EVENT",
            Settings.MAX_BRACKETS_PER_EVENT,
        ),
        MARKET_MIN_CLOSE_DAYS=_read_env_int_optional(
            "MARKET_MIN_CLOSE_DAYS", Settings.MARKET_MIN_CLOSE_DAYS
        ),
        MARKET_MAX_CLOSE_DAYS=_read_env_int_optional(
            "MARKET_MAX_CLOSE_DAYS", Settings.MARKET_MAX_CLOSE_DAYS
        ),
        XAI_API_KEY=_read_env_str("XAI_API_KEY", Settings.XAI_API_KEY),
        GROK_MODEL=normalized_model_initial,
        GROK_MODEL_DEEP=normalized_model_deep,
        SEARCH_LOOKBACK_HOURS=_read_env_int(
            "SEARCH_LOOKBACK_HOURS", Settings.SEARCH_LOOKBACK_HOURS
        ),
        SEARCH_ALLOWED_DOMAINS=_read_env_csv(
            "SEARCH_ALLOWED_DOMAINS", Settings.SEARCH_ALLOWED_DOMAINS
        ),
        SEARCH_ALLOWED_X_HANDLES=_read_env_csv(
            "SEARCH_ALLOWED_X_HANDLES", Settings.SEARCH_ALLOWED_X_HANDLES
        ),
        SETTLEMENT_SOURCE_ALLOWLIST_DOMAINS=_read_env_csv(
            "SETTLEMENT_SOURCE_ALLOWLIST_DOMAINS",
            Settings.SETTLEMENT_SOURCE_ALLOWLIST_DOMAINS,
        ),
        MULTIMEDIA_CONFIDENCE_THRESHOLD=_read_env_float_pair(
            "MULTIMEDIA_CONFIDENCE_THRESHOLD",
            Settings.MULTIMEDIA_CONFIDENCE_THRESHOLD,
        ),
        SEARCH_PROFILE_MAX_DOMAINS=min(
            XAI_WEB_SEARCH_ALLOWED_DOMAINS_LIMIT,
            max(
                1,
                _read_env_int(
                    "SEARCH_PROFILE_MAX_DOMAINS",
                    Settings.SEARCH_PROFILE_MAX_DOMAINS,
                ),
            ),
        ),
        SEARCH_PROFILE_MAX_X_HANDLES=min(
            XAI_X_SEARCH_ALLOWED_HANDLES_LIMIT,
            max(
                1,
                _read_env_int(
                    "SEARCH_PROFILE_MAX_X_HANDLES",
                    Settings.SEARCH_PROFILE_MAX_X_HANDLES,
                ),
            ),
        ),
        EXTENDED_RESEARCH_SOURCE_OFFSET=_read_env_int(
            "EXTENDED_RESEARCH_SOURCE_OFFSET",
            Settings.EXTENDED_RESEARCH_SOURCE_OFFSET,
        ),
        EXTENDED_RESEARCH_X_HANDLE_OFFSET=_read_env_int(
            "EXTENDED_RESEARCH_X_HANDLE_OFFSET",
            Settings.EXTENDED_RESEARCH_X_HANDLE_OFFSET,
        ),
        SEARCH_LOOKBACK_SHORT_HOURS=_read_env_int(
            "SEARCH_LOOKBACK_SHORT_HOURS",
            Settings.SEARCH_LOOKBACK_SHORT_HOURS,
        ),
        SEARCH_LOOKBACK_MEDIUM_HOURS=_read_env_int(
            "SEARCH_LOOKBACK_MEDIUM_HOURS",
            Settings.SEARCH_LOOKBACK_MEDIUM_HOURS,
        ),
        SEARCH_LOOKBACK_LONG_HOURS=_read_env_int(
            "SEARCH_LOOKBACK_LONG_HOURS",
            Settings.SEARCH_LOOKBACK_LONG_HOURS,
        ),
        SPORTS_ALLOWED_DOMAINS=_read_env_csv(
            "SPORTS_ALLOWED_DOMAINS", Settings.SPORTS_ALLOWED_DOMAINS
        ),
        SPORTS_ALLOWED_X_HANDLES=_read_env_csv(
            "SPORTS_ALLOWED_X_HANDLES", Settings.SPORTS_ALLOWED_X_HANDLES
        ),
        CRYPTO_ALLOWED_DOMAINS=_read_env_csv(
            "CRYPTO_ALLOWED_DOMAINS", Settings.CRYPTO_ALLOWED_DOMAINS
        ),
        CRYPTO_ALLOWED_X_HANDLES=_read_env_csv(
            "CRYPTO_ALLOWED_X_HANDLES", Settings.CRYPTO_ALLOWED_X_HANDLES
        ),
        POLITICS_ALLOWED_DOMAINS=_read_env_csv(
            "POLITICS_ALLOWED_DOMAINS", Settings.POLITICS_ALLOWED_DOMAINS
        ),
        POLITICS_ALLOWED_X_HANDLES=_read_env_csv(
            "POLITICS_ALLOWED_X_HANDLES", Settings.POLITICS_ALLOWED_X_HANDLES
        ),
        SPEECH_ALLOWED_DOMAINS=_read_env_csv(
            "SPEECH_ALLOWED_DOMAINS", Settings.SPEECH_ALLOWED_DOMAINS
        ),
        SPEECH_ALLOWED_X_HANDLES=_read_env_csv(
            "SPEECH_ALLOWED_X_HANDLES", Settings.SPEECH_ALLOWED_X_HANDLES
        ),
        MUSIC_ALLOWED_DOMAINS=_read_env_csv(
            "MUSIC_ALLOWED_DOMAINS", Settings.MUSIC_ALLOWED_DOMAINS
        ),
        MUSIC_ALLOWED_X_HANDLES=_read_env_csv(
            "MUSIC_ALLOWED_X_HANDLES", Settings.MUSIC_ALLOWED_X_HANDLES
        ),
        WEATHER_ALLOWED_DOMAINS=_read_env_csv(
            "WEATHER_ALLOWED_DOMAINS", Settings.WEATHER_ALLOWED_DOMAINS
        ),
        WEATHER_ALLOWED_X_HANDLES=_read_env_csv(
            "WEATHER_ALLOWED_X_HANDLES", Settings.WEATHER_ALLOWED_X_HANDLES
        ),
        GENERIC_ALLOWED_DOMAINS=_read_env_csv(
            "GENERIC_ALLOWED_DOMAINS", Settings.GENERIC_ALLOWED_DOMAINS
        ),
        GENERIC_ALLOWED_X_HANDLES=_read_env_csv(
            "GENERIC_ALLOWED_X_HANDLES", Settings.GENERIC_ALLOWED_X_HANDLES
        ),
        ENTERTAINMENT_ALLOWED_DOMAINS=_read_env_csv(
            "ENTERTAINMENT_ALLOWED_DOMAINS",
            Settings.ENTERTAINMENT_ALLOWED_DOMAINS,
        ),
        ENTERTAINMENT_ALLOWED_X_HANDLES=_read_env_csv(
            "ENTERTAINMENT_ALLOWED_X_HANDLES",
            Settings.ENTERTAINMENT_ALLOWED_X_HANDLES,
        ),
        COMMODITY_ALLOWED_DOMAINS=_read_env_csv(
            "COMMODITY_ALLOWED_DOMAINS", Settings.COMMODITY_ALLOWED_DOMAINS
        ),
        COMMODITY_ALLOWED_X_HANDLES=_read_env_csv(
            "COMMODITY_ALLOWED_X_HANDLES", Settings.COMMODITY_ALLOWED_X_HANDLES
        ),
        KALSHI_API_BASE_URL=_read_env_str(
            "KALSHI_API_BASE_URL", Settings.KALSHI_API_BASE_URL
        ),
        KALSHI_API_KEY_ID=_read_env_str(
            "KALSHI_API_KEY_ID", Settings.KALSHI_API_KEY_ID
        ),
        KALSHI_PRIVATE_KEY_PATH=_read_env_str(
            "KALSHI_PRIVATE_KEY_PATH", Settings.KALSHI_PRIVATE_KEY_PATH
        ),
        KALSHI_SERVER_SIDE_FILTERS_ENABLED=_read_env_bool(
            "KALSHI_SERVER_SIDE_FILTERS_ENABLED",
            Settings.KALSHI_SERVER_SIDE_FILTERS_ENABLED,
        ),
        KALSHI_MAX_FETCH_PAGES=_read_env_int(
            "KALSHI_MAX_FETCH_PAGES", Settings.KALSHI_MAX_FETCH_PAGES
        ),
        KALSHI_MVE_FILTER=_read_env_str(
            "KALSHI_MVE_FILTER", Settings.KALSHI_MVE_FILTER
        ),
        KALSHI_ELIGIBLE_FLOOR=_read_env_int(
            "KALSHI_ELIGIBLE_FLOOR", Settings.KALSHI_ELIGIBLE_FLOOR
        ),
        KALSHI_FETCH_TOPUP_ENABLED=_read_env_bool(
            "KALSHI_FETCH_TOPUP_ENABLED",
            Settings.KALSHI_FETCH_TOPUP_ENABLED,
        ),
        DRY_RUN=_read_env_bool("DRY_RUN", Settings.DRY_RUN),
        POSITION_SYNC_ENABLED=_read_env_bool(
            "POSITION_SYNC_ENABLED", Settings.POSITION_SYNC_ENABLED
        ),
        POSITION_SYNC_INTERVAL_CYCLES=_read_env_int(
            "POSITION_SYNC_INTERVAL_CYCLES",
            Settings.POSITION_SYNC_INTERVAL_CYCLES,
        ),
        PRE_ORDER_MARKET_REFRESH=_read_env_bool(
            "PRE_ORDER_MARKET_REFRESH", Settings.PRE_ORDER_MARKET_REFRESH
        ),
        MAX_MARKET_DATA_AGE_SECONDS=_read_env_int(
            "MAX_MARKET_DATA_AGE_SECONDS",
            Settings.MAX_MARKET_DATA_AGE_SECONDS,
        ),
        ORDERBOOK_PRECHECK_ENABLED=_read_env_bool(
            "ORDERBOOK_PRECHECK_ENABLED", Settings.ORDERBOOK_PRECHECK_ENABLED
        ),
        ORDERBOOK_PRECHECK_MIN_CONFIDENCE=_read_env_float(
            "ORDERBOOK_PRECHECK_MIN_CONFIDENCE",
            Settings.ORDERBOOK_PRECHECK_MIN_CONFIDENCE,
        ),
        ORDERBOOK_MIN_RESTING_VOLUME=_read_env_int(
            "ORDERBOOK_MIN_RESTING_VOLUME",
            Settings.ORDERBOOK_MIN_RESTING_VOLUME,
        ),
        ORDER_PRICE_IMPROVEMENT_CENTS=_read_env_int(
            "ORDER_PRICE_IMPROVEMENT_CENTS",
            Settings.ORDER_PRICE_IMPROVEMENT_CENTS,
        ),
        ORDER_DEFAULT_TIF=_read_env_str(
            "ORDER_DEFAULT_TIF",
            Settings.ORDER_DEFAULT_TIF,
        ),
        ORDER_SUBMISSION_MIN_PRICE=_read_env_float(
            "ORDER_SUBMISSION_MIN_PRICE",
            Settings.ORDER_SUBMISSION_MIN_PRICE,
        ),
        ORDER_SUBMISSION_MAX_PRICE=_read_env_float(
            "ORDER_SUBMISSION_MAX_PRICE",
            Settings.ORDER_SUBMISSION_MAX_PRICE,
        ),
        ORDER_FALLBACK_TO_MARKET=_read_env_bool(
            "ORDER_FALLBACK_TO_MARKET",
            Settings.ORDER_FALLBACK_TO_MARKET,
        ),
        ORDER_FALLBACK_MIN_CONFIDENCE=_read_env_float(
            "ORDER_FALLBACK_MIN_CONFIDENCE",
            Settings.ORDER_FALLBACK_MIN_CONFIDENCE,
        ),
        ORDER_FALLBACK_MIN_LIQUIDITY_USDC=_read_env_float(
            "ORDER_FALLBACK_MIN_LIQUIDITY_USDC",
            Settings.ORDER_FALLBACK_MIN_LIQUIDITY_USDC,
        ),
        CALIBRATION_MODE_ENABLED=_read_env_bool(
            "CALIBRATION_MODE_ENABLED", Settings.CALIBRATION_MODE_ENABLED
        ),
        CALIBRATION_MIN_SAMPLES=_read_env_int(
            "CALIBRATION_MIN_SAMPLES", Settings.CALIBRATION_MIN_SAMPLES
        ),
        PROBE_TRADE_ENABLED=_read_env_bool(
            "PROBE_TRADE_ENABLED", Settings.PROBE_TRADE_ENABLED
        ),
        PROBE_TRADE_MAX_USDC=_read_env_float(
            "PROBE_TRADE_MAX_USDC", Settings.PROBE_TRADE_MAX_USDC
        ),
        MAX_DEFINITIVE_OVERRIDES_PER_CYCLE=_read_env_int(
            "MAX_DEFINITIVE_OVERRIDES_PER_CYCLE",
            Settings.MAX_DEFINITIVE_OVERRIDES_PER_CYCLE,
        ),
        MAX_REANALYSES_PER_MARKET_PER_DAY=_read_env_int(
            "MAX_REANALYSES_PER_MARKET_PER_DAY",
            Settings.MAX_REANALYSES_PER_MARKET_PER_DAY,
        ),
        STATE_DB_PATH=_read_env_str(
            "STATE_DB_PATH", Settings.STATE_DB_PATH
        ),
        STATE_JSON_EXPORT_PATH=_read_env_str(
            "STATE_JSON_EXPORT_PATH", Settings.STATE_JSON_EXPORT_PATH
        ),
        EXPORT_STATE_JSON=_read_env_bool(
            "EXPORT_STATE_JSON", Settings.EXPORT_STATE_JSON
        ),
        REANALYSIS_COOLDOWN_HOURS=_read_env_int(
            "REANALYSIS_COOLDOWN_HOURS",
            Settings.REANALYSIS_COOLDOWN_HOURS,
        ),
        URGENT_REANALYSIS_DAYS_BEFORE_CLOSE=_read_env_int(
            "URGENT_REANALYSIS_DAYS_BEFORE_CLOSE",
            Settings.URGENT_REANALYSIS_DAYS_BEFORE_CLOSE,
        ),
        URGENT_REANALYSIS_COOLDOWN_HOURS=_read_env_int(
            "URGENT_REANALYSIS_COOLDOWN_HOURS",
            Settings.URGENT_REANALYSIS_COOLDOWN_HOURS,
        ),
        REFINEMENT_SKIP_BORDERLINE_FAMILIES=_read_env_csv(
            "REFINEMENT_SKIP_BORDERLINE_FAMILIES",
            Settings.REFINEMENT_SKIP_BORDERLINE_FAMILIES,
        ),
        PARALLEL_ANALYSIS_ENABLED=_read_env_bool(
            "PARALLEL_ANALYSIS_ENABLED", Settings.PARALLEL_ANALYSIS_ENABLED
        ),
        ANALYSIS_MAX_WORKERS=_read_env_int(
            "ANALYSIS_MAX_WORKERS", Settings.ANALYSIS_MAX_WORKERS
        ),
        MAX_MARKETS_PER_CYCLE=_read_env_int(
            "MAX_MARKETS_PER_CYCLE", Settings.MAX_MARKETS_PER_CYCLE
        ),
        MAX_WEATHER_CANDIDATES_PER_CYCLE=_read_env_int(
            "MAX_WEATHER_CANDIDATES_PER_CYCLE",
            Settings.MAX_WEATHER_CANDIDATES_PER_CYCLE,
        ),
        MAX_CRYPTO_CANDIDATES_PER_CYCLE=_read_env_int(
            "MAX_CRYPTO_CANDIDATES_PER_CYCLE",
            Settings.MAX_CRYPTO_CANDIDATES_PER_CYCLE,
        ),
        MAX_SPEECH_CANDIDATES_PER_CYCLE=_read_env_int(
            "MAX_SPEECH_CANDIDATES_PER_CYCLE",
            Settings.MAX_SPEECH_CANDIDATES_PER_CYCLE,
        ),
        MAX_MUSIC_CANDIDATES_PER_CYCLE=_read_env_int(
            "MAX_MUSIC_CANDIDATES_PER_CYCLE",
            Settings.MAX_MUSIC_CANDIDATES_PER_CYCLE,
        ),
        MAX_SPORTS_CANDIDATES_PER_CYCLE=_read_env_int(
            "MAX_SPORTS_CANDIDATES_PER_CYCLE",
            Settings.MAX_SPORTS_CANDIDATES_PER_CYCLE,
        ),
        MAX_GENERIC_CANDIDATES_PER_CYCLE=_read_env_int(
            "MAX_GENERIC_CANDIDATES_PER_CYCLE",
            Settings.MAX_GENERIC_CANDIDATES_PER_CYCLE,
        ),
        MAX_TRADES_PER_CYCLE=_read_env_int(
            "MAX_TRADES_PER_CYCLE",
            Settings.MAX_TRADES_PER_CYCLE,
        ),
        MAX_BETS_PER_EVENT=_read_env_int(
            "MAX_BETS_PER_EVENT",
            Settings.MAX_BETS_PER_EVENT,
        ),
        MAX_TRADES_PER_DAY=_read_env_int(
            "MAX_TRADES_PER_DAY",
            Settings.MAX_TRADES_PER_DAY,
        ),
        MAX_DAILY_DRAWDOWN_USDC=_read_env_float(
            "MAX_DAILY_DRAWDOWN_USDC",
            Settings.MAX_DAILY_DRAWDOWN_USDC,
        ),
        DAILY_DRAWDOWN_PREFLIGHT_ENABLED=_read_env_bool(
            "DAILY_DRAWDOWN_PREFLIGHT_ENABLED",
            Settings.DAILY_DRAWDOWN_PREFLIGHT_ENABLED,
        ),
        XAI_CIRCUIT_BREAKER_MAX_FAILURES=_read_env_int(
            "XAI_CIRCUIT_BREAKER_MAX_FAILURES",
            Settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES,
        ),
        XAI_QUOTA_BREAKER_ENABLED=_read_env_bool(
            "XAI_QUOTA_BREAKER_ENABLED",
            Settings.XAI_QUOTA_BREAKER_ENABLED,
        ),
        XAI_QUOTA_PAUSE_MINUTES=_read_env_int(
            "XAI_QUOTA_PAUSE_MINUTES",
            Settings.XAI_QUOTA_PAUSE_MINUTES,
        ),
        XAI_CLIENT_TIMEOUT_SECONDS=_read_env_int(
            "XAI_CLIENT_TIMEOUT_SECONDS",
            Settings.XAI_CLIENT_TIMEOUT_SECONDS,
        ),
        GROK_STREAM_TIMEOUT_SECONDS=_read_env_int(
            "GROK_STREAM_TIMEOUT_SECONDS",
            Settings.GROK_STREAM_TIMEOUT_SECONDS,
        ),
        GROK_STREAM_TIMEOUT_SECONDS_DEEP=_read_env_int(
            "GROK_STREAM_TIMEOUT_SECONDS_DEEP",
            Settings.GROK_STREAM_TIMEOUT_SECONDS_DEEP,
        ),
        GROK_DEEP_ANALYSIS_MAX_ATTEMPTS=_read_env_int(
            "GROK_DEEP_ANALYSIS_MAX_ATTEMPTS",
            Settings.GROK_DEEP_ANALYSIS_MAX_ATTEMPTS,
        ),
        GROK_ANALYSIS_MAX_BUDGET_SECONDS=_read_env_int(
            "GROK_ANALYSIS_MAX_BUDGET_SECONDS",
            Settings.GROK_ANALYSIS_MAX_BUDGET_SECONDS,
        ),
        GROK_SELF_CONSISTENCY_ENABLED=_read_env_bool(
            "GROK_SELF_CONSISTENCY_ENABLED",
            Settings.GROK_SELF_CONSISTENCY_ENABLED,
        ),
        GROK_SELF_CONSISTENCY_LIQUIDITY_THRESHOLD=_read_env_float(
            "GROK_SELF_CONSISTENCY_LIQUIDITY_THRESHOLD",
            Settings.GROK_SELF_CONSISTENCY_LIQUIDITY_THRESHOLD,
        ),
        GROK_SELF_CONSISTENCY_EDGE_THRESHOLD=_read_env_float(
            "GROK_SELF_CONSISTENCY_EDGE_THRESHOLD",
            Settings.GROK_SELF_CONSISTENCY_EDGE_THRESHOLD,
        ),
        GROK_SELF_CONSISTENCY_TOP_CANDIDATES=_read_env_int(
            "GROK_SELF_CONSISTENCY_TOP_CANDIDATES",
            Settings.GROK_SELF_CONSISTENCY_TOP_CANDIDATES,
        ),
        GROK_SELF_CONSISTENCY_PRIMARY_TEMPERATURE=_read_env_float(
            "GROK_SELF_CONSISTENCY_PRIMARY_TEMPERATURE",
            Settings.GROK_SELF_CONSISTENCY_PRIMARY_TEMPERATURE,
        ),
        GROK_SELF_CONSISTENCY_SECONDARY_TEMPERATURE=_read_env_float(
            "GROK_SELF_CONSISTENCY_SECONDARY_TEMPERATURE",
            Settings.GROK_SELF_CONSISTENCY_SECONDARY_TEMPERATURE,
        ),
        EDGE_REPAIR_ENABLED=_read_env_bool(
            "EDGE_REPAIR_ENABLED",
            Settings.EDGE_REPAIR_ENABLED,
        ),
        EDGE_BAND_CALIBRATION_ENABLED=_read_env_bool(
            "EDGE_BAND_CALIBRATION_ENABLED",
            Settings.EDGE_BAND_CALIBRATION_ENABLED,
        ),
        CONVICTION_REPAIR_ENABLED=_read_env_bool(
            "CONVICTION_REPAIR_ENABLED",
            Settings.CONVICTION_REPAIR_ENABLED,
        ),
        CONVICTION_REPAIR_MIN_EDGE=_read_env_float(
            "CONVICTION_REPAIR_MIN_EDGE",
            Settings.CONVICTION_REPAIR_MIN_EDGE,
        ),
        CONVICTION_REPAIR_MIN_EVIDENCE_QUALITY=_read_env_float(
            "CONVICTION_REPAIR_MIN_EVIDENCE_QUALITY",
            Settings.CONVICTION_REPAIR_MIN_EVIDENCE_QUALITY,
        ),
        CONVICTION_REPAIR_SCORE_GAP_MAX=_read_env_float(
            "CONVICTION_REPAIR_SCORE_GAP_MAX",
            Settings.CONVICTION_REPAIR_SCORE_GAP_MAX,
        ),
        CONVICTION_REPAIR_CONFIDENCE_SCORE_FLOOR=_read_env_float(
            "CONVICTION_REPAIR_CONFIDENCE_SCORE_FLOOR",
            Settings.CONVICTION_REPAIR_CONFIDENCE_SCORE_FLOOR,
        ),
        DAILY_EXPECTANCY_ENABLED=_read_env_bool(
            "DAILY_EXPECTANCY_ENABLED",
            Settings.DAILY_EXPECTANCY_ENABLED,
        ),
        DAILY_EXPECTANCY_PRIMARY_TARGETS=_read_env_int(
            "DAILY_EXPECTANCY_PRIMARY_TARGETS",
            Settings.DAILY_EXPECTANCY_PRIMARY_TARGETS,
        ),
        DAILY_EXPECTANCY_SATELLITE_MAX_BET_PCT=_read_env_float(
            "DAILY_EXPECTANCY_SATELLITE_MAX_BET_PCT",
            Settings.DAILY_EXPECTANCY_SATELLITE_MAX_BET_PCT,
        ),
        RESOLUTION_SYNC_INTERVAL_CYCLES=_read_env_int(
            "RESOLUTION_SYNC_INTERVAL_CYCLES",
            Settings.RESOLUTION_SYNC_INTERVAL_CYCLES,
        ),
        MAX_POSITION_PER_MARKET_USDC=_read_env_float(
            "MAX_POSITION_PER_MARKET_USDC",
            Settings.MAX_POSITION_PER_MARKET_USDC,
        ),
        MAX_POSITION_PCT_OF_BANKROLL=_read_env_float(
            "MAX_POSITION_PCT_OF_BANKROLL",
            Settings.MAX_POSITION_PCT_OF_BANKROLL,
        ),
        MIN_CONFIDENCE_INCREASE_FOR_ADD=_read_env_float(
            "MIN_CONFIDENCE_INCREASE_FOR_ADD",
            Settings.MIN_CONFIDENCE_INCREASE_FOR_ADD,
        ),
        MIN_PRICE_MOVE_FOR_READD=_read_env_float(
            "MIN_PRICE_MOVE_FOR_READD",
            Settings.MIN_PRICE_MOVE_FOR_READD,
        ),
        HIGH_CONFIDENCE_POSITION_OVERRIDE=_read_env_float(
            "HIGH_CONFIDENCE_POSITION_OVERRIDE",
            Settings.HIGH_CONFIDENCE_POSITION_OVERRIDE,
        ),
        OPPOSITE_OUTCOME_STRATEGY=_read_env_str(
            "OPPOSITE_OUTCOME_STRATEGY",
            Settings.OPPOSITE_OUTCOME_STRATEGY,
        ),
        SCORE_GATE_MODE=_read_env_str(
            "SCORE_GATE_MODE",
            Settings.SCORE_GATE_MODE,
        ),
        SCORE_GATE_THRESHOLD=_read_env_float(
            "SCORE_GATE_THRESHOLD",
            Settings.SCORE_GATE_THRESHOLD,
        ),
        SCORE_GATE_THRESHOLD_WEATHER_DIRECT=_read_env_float(
            "SCORE_GATE_THRESHOLD_WEATHER_DIRECT",
            Settings.SCORE_GATE_THRESHOLD_WEATHER_DIRECT,
        ),
        SCORE_GATE_THRESHOLD_DIRECT_HIGH_QUALITY=_read_env_float(
            "SCORE_GATE_THRESHOLD_DIRECT_HIGH_QUALITY",
            Settings.SCORE_GATE_THRESHOLD_DIRECT_HIGH_QUALITY,
        ),
        SCORE_GATE_PROFITABLE_FAMILY_CONVERGENT_ENABLED=_read_env_bool(
            "SCORE_GATE_PROFITABLE_FAMILY_CONVERGENT_ENABLED",
            Settings.SCORE_GATE_PROFITABLE_FAMILY_CONVERGENT_ENABLED,
        ),
        SCORE_GATE_THRESHOLD_PROFITABLE_FAMILY_CONVERGENT=_read_env_float(
            "SCORE_GATE_THRESHOLD_PROFITABLE_FAMILY_CONVERGENT",
            Settings.SCORE_GATE_THRESHOLD_PROFITABLE_FAMILY_CONVERGENT,
        ),
        SCORE_GATE_PROFITABLE_FAMILY_CONVERGENT_MIN_SAMPLES=_read_env_int(
            "SCORE_GATE_PROFITABLE_FAMILY_CONVERGENT_MIN_SAMPLES",
            Settings.SCORE_GATE_PROFITABLE_FAMILY_CONVERGENT_MIN_SAMPLES,
        ),
        SCORE_KELLY_COMPONENT_WEIGHT=_read_env_float(
            "SCORE_KELLY_COMPONENT_WEIGHT",
            Settings.SCORE_KELLY_COMPONENT_WEIGHT,
        ),
        SCORE_INEFFICIENCY_COMPONENT_WEIGHT=_read_env_float(
            "SCORE_INEFFICIENCY_COMPONENT_WEIGHT",
            Settings.SCORE_INEFFICIENCY_COMPONENT_WEIGHT,
        ),
        SCORE_BAYESIAN_COMPONENT_WEIGHT=_read_env_float(
            "SCORE_BAYESIAN_COMPONENT_WEIGHT",
            Settings.SCORE_BAYESIAN_COMPONENT_WEIGHT,
        ),
        SCORE_LOW_INFO_PENALTY_THRESHOLD=_read_env_float(
            "SCORE_LOW_INFO_PENALTY_THRESHOLD",
            Settings.SCORE_LOW_INFO_PENALTY_THRESHOLD,
        ),
        SCORE_LOW_INFO_PENALTY_BASE=_read_env_float(
            "SCORE_LOW_INFO_PENALTY_BASE",
            Settings.SCORE_LOW_INFO_PENALTY_BASE,
        ),
        SCORE_REPEATED_ANALYSIS_PENALTY_BASE=_read_env_float(
            "SCORE_REPEATED_ANALYSIS_PENALTY_BASE",
            Settings.SCORE_REPEATED_ANALYSIS_PENALTY_BASE,
        ),
        SCORE_REPEATED_ANALYSIS_PENALTY_START_COUNT=_read_env_int(
            "SCORE_REPEATED_ANALYSIS_PENALTY_START_COUNT",
            Settings.SCORE_REPEATED_ANALYSIS_PENALTY_START_COUNT,
        ),
        SCORE_CONFIDENCE_CALIBRATION_FLOOR=_read_env_float(
            "SCORE_CONFIDENCE_CALIBRATION_FLOOR",
            Settings.SCORE_CONFIDENCE_CALIBRATION_FLOOR,
        ),
        SCORE_CONFIDENCE_CALIBRATION_PENALTY_SCALE=_read_env_float(
            "SCORE_CONFIDENCE_CALIBRATION_PENALTY_SCALE",
            Settings.SCORE_CONFIDENCE_CALIBRATION_PENALTY_SCALE,
        ),
        SCORE_FALLBACK_EDGE_PENALTY_BASE=_read_env_float(
            "SCORE_FALLBACK_EDGE_PENALTY_BASE",
            Settings.SCORE_FALLBACK_EDGE_PENALTY_BASE,
        ),
        SCORE_OVERCONFIDENCE_PENALTY_BASE=_read_env_float(
            "SCORE_OVERCONFIDENCE_PENALTY_BASE",
            Settings.SCORE_OVERCONFIDENCE_PENALTY_BASE,
        ),
        SCORE_COMPUTED_EDGE_BONUS=_read_env_float(
            "SCORE_COMPUTED_EDGE_BONUS",
            Settings.SCORE_COMPUTED_EDGE_BONUS,
        ),
        SCORE_SOURCE_CONFIRMED_EDGE_BONUS=_read_env_float(
            "SCORE_SOURCE_CONFIRMED_EDGE_BONUS",
            Settings.SCORE_SOURCE_CONFIRMED_EDGE_BONUS,
        ),
        SCORE_PROXY_EVIDENCE_PENALTY_BASE=_read_env_float(
            "SCORE_PROXY_EVIDENCE_PENALTY_BASE",
            Settings.SCORE_PROXY_EVIDENCE_PENALTY_BASE,
        ),
        SCORE_GENERIC_BIN_PENALTY_BASE=_read_env_float(
            "SCORE_GENERIC_BIN_PENALTY_BASE",
            Settings.SCORE_GENERIC_BIN_PENALTY_BASE,
        ),
        SCORE_AMBIGUOUS_RESOLUTION_PENALTY_BASE=_read_env_float(
            "SCORE_AMBIGUOUS_RESOLUTION_PENALTY_BASE",
            Settings.SCORE_AMBIGUOUS_RESOLUTION_PENALTY_BASE,
        ),
        SCORE_HALLUCINATED_EDGE_PENALTY_BASE=_read_env_float(
            "SCORE_HALLUCINATED_EDGE_PENALTY_BASE",
            Settings.SCORE_HALLUCINATED_EDGE_PENALTY_BASE,
        ),
        SCORE_VOLUME_AMPLIFIER_ENABLED=_read_env_bool(
            "SCORE_VOLUME_AMPLIFIER_ENABLED",
            Settings.SCORE_VOLUME_AMPLIFIER_ENABLED,
        ),
        SCORE_EXTREME_MARKET_EDGE_PENALTY_BASE=_read_env_float(
            "SCORE_EXTREME_MARKET_EDGE_PENALTY_BASE",
            Settings.SCORE_EXTREME_MARKET_EDGE_PENALTY_BASE,
        ),
        SCORE_LATE_STAGE_OVERCONFIDENCE_PENALTY_BASE=_read_env_float(
            "SCORE_LATE_STAGE_OVERCONFIDENCE_PENALTY_BASE",
            Settings.SCORE_LATE_STAGE_OVERCONFIDENCE_PENALTY_BASE,
        ),
        SCORE_EXTREME_CONFIDENCE_THRESHOLD=_read_env_float(
            "SCORE_EXTREME_CONFIDENCE_THRESHOLD",
            Settings.SCORE_EXTREME_CONFIDENCE_THRESHOLD,
        ),
        SCORE_EXTREME_CONFIDENCE_PENALTY_BASE=_read_env_float(
            "SCORE_EXTREME_CONFIDENCE_PENALTY_BASE",
            Settings.SCORE_EXTREME_CONFIDENCE_PENALTY_BASE,
        ),
        MENTION_MARKET_SCORE_PENALTY=_read_env_float(
            "MENTION_MARKET_SCORE_PENALTY",
            Settings.MENTION_MARKET_SCORE_PENALTY,
        ),
        WEATHER_SCORE_PENALTY=_read_env_float(
            "WEATHER_SCORE_PENALTY",
            Settings.WEATHER_SCORE_PENALTY,
        ),
        WEATHER_MIN_EVIDENCE_QUALITY=_read_env_float(
            "WEATHER_MIN_EVIDENCE_QUALITY",
            Settings.WEATHER_MIN_EVIDENCE_QUALITY,
        ),
        DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_WEATHER=_read_env_float(
            "DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_WEATHER",
            Settings.DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_WEATHER,
        ),
        DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_SPORTS=_read_env_float(
            "DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_SPORTS",
            Settings.DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_SPORTS,
        ),
        DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_DEFAULT=_read_env_float(
            "DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_DEFAULT",
            Settings.DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_DEFAULT,
        ),
        DIRECT_SOURCE_WHITELIST=_read_env_csv(
            "DIRECT_SOURCE_WHITELIST",
            Settings.DIRECT_SOURCE_WHITELIST,
        ),
        PRE_ANALYSIS_OPPORTUNITY_ENABLED=_read_env_bool(
            "PRE_ANALYSIS_OPPORTUNITY_ENABLED",
            Settings.PRE_ANALYSIS_OPPORTUNITY_ENABLED,
        ),
        PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE=_read_env_float(
            "PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE",
            Settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE,
        ),
        PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND=_read_env_float(
            "PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND",
            Settings.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND,
        ),
        PRE_ANALYSIS_OPPORTUNITY_RESEARCH_ENABLED=_read_env_bool(
            "PRE_ANALYSIS_OPPORTUNITY_RESEARCH_ENABLED",
            Settings.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_ENABLED,
        ),
        PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_ADAPTIVE_ENABLED=_read_env_bool(
            "PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_ADAPTIVE_ENABLED",
            Settings.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_ADAPTIVE_ENABLED,
        ),
        PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_MAX=_read_env_float(
            "PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_MAX",
            Settings.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_MAX,
        ),
        PRE_ANALYSIS_MUST_ANALYZE_THRESHOLD=_read_env_float(
            "PRE_ANALYSIS_MUST_ANALYZE_THRESHOLD",
            Settings.PRE_ANALYSIS_MUST_ANALYZE_THRESHOLD,
        ),
        PRE_ANALYSIS_REDUCED_MAX_CANDIDATES=_read_env_int(
            "PRE_ANALYSIS_REDUCED_MAX_CANDIDATES",
            Settings.PRE_ANALYSIS_REDUCED_MAX_CANDIDATES,
        ),
        PRE_ANALYSIS_NON_ACTIONABLE_STREAK_PENALTY=_read_env_float(
            "PRE_ANALYSIS_NON_ACTIONABLE_STREAK_PENALTY",
            Settings.PRE_ANALYSIS_NON_ACTIONABLE_STREAK_PENALTY,
        ),
        PRE_ANALYSIS_NON_ACTIONABLE_STREAK_CAP=_read_env_int(
            "PRE_ANALYSIS_NON_ACTIONABLE_STREAK_CAP",
            Settings.PRE_ANALYSIS_NON_ACTIONABLE_STREAK_CAP,
        ),
        PRE_ANALYSIS_ANALYSIS_COUNT_PENALTY=_read_env_float(
            "PRE_ANALYSIS_ANALYSIS_COUNT_PENALTY",
            Settings.PRE_ANALYSIS_ANALYSIS_COUNT_PENALTY,
        ),
        PRE_ANALYSIS_ANALYSIS_COUNT_START=_read_env_int(
            "PRE_ANALYSIS_ANALYSIS_COUNT_START",
            Settings.PRE_ANALYSIS_ANALYSIS_COUNT_START,
        ),
        PRE_ANALYSIS_FAMILY_PENALTY_SPEECH=_read_env_float(
            "PRE_ANALYSIS_FAMILY_PENALTY_SPEECH",
            Settings.PRE_ANALYSIS_FAMILY_PENALTY_SPEECH,
        ),
        PRE_ANALYSIS_FAMILY_PENALTY_MUSIC=_read_env_float(
            "PRE_ANALYSIS_FAMILY_PENALTY_MUSIC",
            Settings.PRE_ANALYSIS_FAMILY_PENALTY_MUSIC,
        ),
        PRE_ANALYSIS_FAMILY_PENALTY_SPORTS=_read_env_float(
            "PRE_ANALYSIS_FAMILY_PENALTY_SPORTS",
            Settings.PRE_ANALYSIS_FAMILY_PENALTY_SPORTS,
        ),
        PRE_ANALYSIS_FAMILY_PENALTY_WEATHER_BIN=_read_env_float(
            "PRE_ANALYSIS_FAMILY_PENALTY_WEATHER_BIN",
            Settings.PRE_ANALYSIS_FAMILY_PENALTY_WEATHER_BIN,
        ),
        PRE_ANALYSIS_FAMILY_PENALTY_GENERIC_BIN=_read_env_float(
            "PRE_ANALYSIS_FAMILY_PENALTY_GENERIC_BIN",
            Settings.PRE_ANALYSIS_FAMILY_PENALTY_GENERIC_BIN,
        ),
        PRE_ANALYSIS_FAMILY_PENALTY_CRYPTO_BIN=_read_env_float(
            "PRE_ANALYSIS_FAMILY_PENALTY_CRYPTO_BIN",
            Settings.PRE_ANALYSIS_FAMILY_PENALTY_CRYPTO_BIN,
        ),
        PRE_ANALYSIS_ZERO_TRADE_RATE_PENALTY=_read_env_float(
            "PRE_ANALYSIS_ZERO_TRADE_RATE_PENALTY",
            Settings.PRE_ANALYSIS_ZERO_TRADE_RATE_PENALTY,
        ),
        PRE_ANALYSIS_FALLBACK_FAMILY_RATE_THRESHOLD=_read_env_float(
            "PRE_ANALYSIS_FALLBACK_FAMILY_RATE_THRESHOLD",
            Settings.PRE_ANALYSIS_FALLBACK_FAMILY_RATE_THRESHOLD,
        ),
        PRE_ANALYSIS_FALLBACK_FAMILY_PENALTY=_read_env_float(
            "PRE_ANALYSIS_FALLBACK_FAMILY_PENALTY",
            Settings.PRE_ANALYSIS_FALLBACK_FAMILY_PENALTY,
        ),
        PRE_ANALYSIS_FALLBACK_FAMILY_MIN_SAMPLES=_read_env_int(
            "PRE_ANALYSIS_FALLBACK_FAMILY_MIN_SAMPLES",
            Settings.PRE_ANALYSIS_FALLBACK_FAMILY_MIN_SAMPLES,
        ),
        PRE_ANALYSIS_HISTORICAL_FAMILY_MIN_SAMPLES=_read_env_int(
            "PRE_ANALYSIS_HISTORICAL_FAMILY_MIN_SAMPLES",
            Settings.PRE_ANALYSIS_HISTORICAL_FAMILY_MIN_SAMPLES,
        ),
        PRE_ANALYSIS_HISTORICAL_FAMILY_WIN_RATE_THRESHOLD=_read_env_float(
            "PRE_ANALYSIS_HISTORICAL_FAMILY_WIN_RATE_THRESHOLD",
            Settings.PRE_ANALYSIS_HISTORICAL_FAMILY_WIN_RATE_THRESHOLD,
        ),
        PRE_ANALYSIS_HISTORICAL_FAMILY_PENALTY=_read_env_float(
            "PRE_ANALYSIS_HISTORICAL_FAMILY_PENALTY",
            Settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PENALTY,
        ),
        PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_MIN_SAMPLES=_read_env_int(
            "PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_MIN_SAMPLES",
            Settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_MIN_SAMPLES,
        ),
        PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_THRESHOLD=_read_env_float(
            "PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_THRESHOLD",
            Settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_THRESHOLD,
        ),
        PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_PENALTY=_read_env_float(
            "PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_PENALTY",
            Settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_PENALTY,
        ),
        PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_THRESHOLD=_read_env_float(
            "PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_THRESHOLD",
            Settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_THRESHOLD,
        ),
        PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_PENALTY=_read_env_float(
            "PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_PENALTY",
            Settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_PENALTY,
        ),
        PRE_ANALYSIS_ADAPTIVE_BOOST=_read_env_float(
            "PRE_ANALYSIS_ADAPTIVE_BOOST",
            Settings.PRE_ANALYSIS_ADAPTIVE_BOOST,
        ),
        PRE_ANALYSIS_STACKED_HISTORICAL_PENALTY_CAP=_read_env_float(
            "PRE_ANALYSIS_STACKED_HISTORICAL_PENALTY_CAP",
            Settings.PRE_ANALYSIS_STACKED_HISTORICAL_PENALTY_CAP,
        ),
        PRE_ANALYSIS_CRYPTO_NEGATIVE_PNL_BLOCK_ENABLED=_read_env_bool(
            "PRE_ANALYSIS_CRYPTO_NEGATIVE_PNL_BLOCK_ENABLED",
            Settings.PRE_ANALYSIS_CRYPTO_NEGATIVE_PNL_BLOCK_ENABLED,
        ),
        PRE_ANALYSIS_CRYPTO_NEGATIVE_PNL_THRESHOLD=_read_env_float(
            "PRE_ANALYSIS_CRYPTO_NEGATIVE_PNL_THRESHOLD",
            Settings.PRE_ANALYSIS_CRYPTO_NEGATIVE_PNL_THRESHOLD,
        ),
        PRE_ANALYSIS_CRYPTO_FALLBACK_RATE_BLOCK_THRESHOLD=_read_env_float(
            "PRE_ANALYSIS_CRYPTO_FALLBACK_RATE_BLOCK_THRESHOLD",
            Settings.PRE_ANALYSIS_CRYPTO_FALLBACK_RATE_BLOCK_THRESHOLD,
        ),
        PRE_ANALYSIS_CRYPTO_MIN_SAMPLES=_read_env_int(
            "PRE_ANALYSIS_CRYPTO_MIN_SAMPLES",
            Settings.PRE_ANALYSIS_CRYPTO_MIN_SAMPLES,
        ),
        MAX_LIFETIME_ANALYSES_PER_MARKET=_read_env_int(
            "MAX_LIFETIME_ANALYSES_PER_MARKET",
            Settings.MAX_LIFETIME_ANALYSES_PER_MARKET,
        ),
        # The PRE_ANALYSIS_HARD_REJECTION_* settings are misnamed: they actually
        # control soft research-routing (final_action="research_queued"), not
        # terminal rejection. Accept the clearer PRE_ANALYSIS_PARTICIPATION_*
        # aliases first; fall back to the legacy names so existing .env files
        # keep working without any operator action.
        PRE_ANALYSIS_HARD_REJECTION_ENABLED=_read_env_bool(
            "PRE_ANALYSIS_PARTICIPATION_GATING_ENABLED",
            _read_env_bool(
                "PRE_ANALYSIS_HARD_REJECTION_ENABLED",
                Settings.PRE_ANALYSIS_HARD_REJECTION_ENABLED,
            ),
        ),
        PRE_ANALYSIS_HARD_REJECTION_MIN_STREAK=_read_env_int(
            "PRE_ANALYSIS_PARTICIPATION_MIN_STREAK",
            _read_env_int(
                "PRE_ANALYSIS_HARD_REJECTION_MIN_STREAK",
                Settings.PRE_ANALYSIS_HARD_REJECTION_MIN_STREAK,
            ),
        ),
        PRE_ANALYSIS_HARD_REJECTION_MIN_ANALYSES=_read_env_int(
            "PRE_ANALYSIS_PARTICIPATION_MIN_ANALYSES",
            _read_env_int(
                "PRE_ANALYSIS_HARD_REJECTION_MIN_ANALYSES",
                Settings.PRE_ANALYSIS_HARD_REJECTION_MIN_ANALYSES,
            ),
        ),
        HISTORICAL_TICKER_PREFIX_GATE_ENABLED=_read_env_bool(
            "HISTORICAL_TICKER_PREFIX_GATE_ENABLED",
            Settings.HISTORICAL_TICKER_PREFIX_GATE_ENABLED,
        ),
        HISTORICAL_TICKER_PREFIX_LEN=_read_env_int(
            "HISTORICAL_TICKER_PREFIX_LEN",
            Settings.HISTORICAL_TICKER_PREFIX_LEN,
        ),
        HISTORICAL_TICKER_PREFIX_LOOKBACK_DAYS=_read_env_int(
            "HISTORICAL_TICKER_PREFIX_LOOKBACK_DAYS",
            Settings.HISTORICAL_TICKER_PREFIX_LOOKBACK_DAYS,
        ),
        HISTORICAL_TICKER_PREFIX_MIN_SAMPLES=_read_env_int(
            "HISTORICAL_TICKER_PREFIX_MIN_SAMPLES",
            Settings.HISTORICAL_TICKER_PREFIX_MIN_SAMPLES,
        ),
        HISTORICAL_TICKER_PREFIX_PNL_CUTOFF=_read_env_float(
            "HISTORICAL_TICKER_PREFIX_PNL_CUTOFF",
            Settings.HISTORICAL_TICKER_PREFIX_PNL_CUTOFF,
        ),
        HISTORICAL_TICKER_PREFIX_WIN_RATE_CUTOFF=_read_env_float(
            "HISTORICAL_TICKER_PREFIX_WIN_RATE_CUTOFF",
            Settings.HISTORICAL_TICKER_PREFIX_WIN_RATE_CUTOFF,
        ),
        HISTORICAL_FAMILY_GATE_ENABLED=_read_env_bool(
            "HISTORICAL_FAMILY_GATE_ENABLED",
            Settings.HISTORICAL_FAMILY_GATE_ENABLED,
        ),
        HISTORICAL_FAMILY_LOOKBACK_DAYS=_read_env_int(
            "HISTORICAL_FAMILY_LOOKBACK_DAYS",
            Settings.HISTORICAL_FAMILY_LOOKBACK_DAYS,
        ),
        HISTORICAL_FAMILY_MIN_SAMPLES=_read_env_int(
            "HISTORICAL_FAMILY_MIN_SAMPLES",
            Settings.HISTORICAL_FAMILY_MIN_SAMPLES,
        ),
        HISTORICAL_FAMILY_PNL_CUTOFF=_read_env_float(
            "HISTORICAL_FAMILY_PNL_CUTOFF",
            Settings.HISTORICAL_FAMILY_PNL_CUTOFF,
        ),
        HISTORICAL_FAMILY_WIN_RATE_CUTOFF=_read_env_float(
            "HISTORICAL_FAMILY_WIN_RATE_CUTOFF",
            Settings.HISTORICAL_FAMILY_WIN_RATE_CUTOFF,
        ),
        HISTORICAL_FAMILY_SHRUNK_PNL_CUTOFF=_read_env_float(
            "HISTORICAL_FAMILY_SHRUNK_PNL_CUTOFF",
            Settings.HISTORICAL_FAMILY_SHRUNK_PNL_CUTOFF,
        ),
        HISTORICAL_FAMILY_SIGNAL_ENABLED=_read_env_bool(
            "HISTORICAL_FAMILY_SIGNAL_ENABLED",
            Settings.HISTORICAL_FAMILY_SIGNAL_ENABLED,
        ),
        HISTORICAL_FAMILY_SCORE_SCALE=_read_env_float(
            "HISTORICAL_FAMILY_SCORE_SCALE",
            Settings.HISTORICAL_FAMILY_SCORE_SCALE,
        ),
        HISTORICAL_FAMILY_SIZE_SCALE_MAX=_read_env_float(
            "HISTORICAL_FAMILY_SIZE_SCALE_MAX",
            Settings.HISTORICAL_FAMILY_SIZE_SCALE_MAX,
        ),
        HISTORICAL_FAMILY_SIZE_SCALE_MAX_NEGATIVE=_read_env_float(
            "HISTORICAL_FAMILY_SIZE_SCALE_MAX_NEGATIVE",
            Settings.HISTORICAL_FAMILY_SIZE_SCALE_MAX_NEGATIVE,
        ),
        HISTORICAL_SHORT_PREFIX_LEN=_read_env_int(
            "HISTORICAL_SHORT_PREFIX_LEN",
            Settings.HISTORICAL_SHORT_PREFIX_LEN,
        ),
        HISTORICAL_SHORT_PREFIX_MIN_SAMPLES=_read_env_int(
            "HISTORICAL_SHORT_PREFIX_MIN_SAMPLES",
            Settings.HISTORICAL_SHORT_PREFIX_MIN_SAMPLES,
        ),
        HISTORICAL_SHORT_PREFIX_PNL_CUTOFF=_read_env_float(
            "HISTORICAL_SHORT_PREFIX_PNL_CUTOFF",
            Settings.HISTORICAL_SHORT_PREFIX_PNL_CUTOFF,
        ),
        HISTORICAL_SHORT_PREFIX_SCORE_PENALTY=_read_env_float(
            "HISTORICAL_SHORT_PREFIX_SCORE_PENALTY",
            Settings.HISTORICAL_SHORT_PREFIX_SCORE_PENALTY,
        ),
        HISTORICAL_TICKER_PREFIX_HARD_BLOCK_MIN_SAMPLES=_read_env_int(
            "HISTORICAL_TICKER_PREFIX_HARD_BLOCK_MIN_SAMPLES",
            Settings.HISTORICAL_TICKER_PREFIX_HARD_BLOCK_MIN_SAMPLES,
        ),
        HISTORICAL_TICKER_PREFIX_SHRINKAGE_ENABLED=_read_env_bool(
            "HISTORICAL_TICKER_PREFIX_SHRINKAGE_ENABLED",
            Settings.HISTORICAL_TICKER_PREFIX_SHRINKAGE_ENABLED,
        ),
        HISTORICAL_TICKER_PREFIX_PRIOR_WIN_RATE=_read_env_float(
            "HISTORICAL_TICKER_PREFIX_PRIOR_WIN_RATE",
            Settings.HISTORICAL_TICKER_PREFIX_PRIOR_WIN_RATE,
        ),
        HISTORICAL_TICKER_PREFIX_PRIOR_STRENGTH=_read_env_float(
            "HISTORICAL_TICKER_PREFIX_PRIOR_STRENGTH",
            Settings.HISTORICAL_TICKER_PREFIX_PRIOR_STRENGTH,
        ),
        HISTORICAL_TICKER_PREFIX_SHRUNK_PNL_CUTOFF=_read_env_float(
            "HISTORICAL_TICKER_PREFIX_SHRUNK_PNL_CUTOFF",
            Settings.HISTORICAL_TICKER_PREFIX_SHRUNK_PNL_CUTOFF,
        ),
        HISTORICAL_TICKER_PREFIX_SOFT_DEMOTE_SCORE_PENALTY=_read_env_float(
            "HISTORICAL_TICKER_PREFIX_SOFT_DEMOTE_SCORE_PENALTY",
            Settings.HISTORICAL_TICKER_PREFIX_SOFT_DEMOTE_SCORE_PENALTY,
        ),
        STRONG_EVIDENCE_CONFIDENCE_FLOOR=_read_env_float(
            "STRONG_EVIDENCE_CONFIDENCE_FLOOR",
            Settings.STRONG_EVIDENCE_CONFIDENCE_FLOOR,
        ),
        STRONG_EVIDENCE_MIN_EVIDENCE_QUALITY=_read_env_float(
            "STRONG_EVIDENCE_MIN_EVIDENCE_QUALITY",
            Settings.STRONG_EVIDENCE_MIN_EVIDENCE_QUALITY,
        ),
        STRONG_EVIDENCE_PROXY_MIN_EVIDENCE_QUALITY=_read_env_float(
            "STRONG_EVIDENCE_PROXY_MIN_EVIDENCE_QUALITY",
            Settings.STRONG_EVIDENCE_PROXY_MIN_EVIDENCE_QUALITY,
        ),
        STRONG_EVIDENCE_PROXY_MIN_EDGE=_read_env_float(
            "STRONG_EVIDENCE_PROXY_MIN_EDGE",
            Settings.STRONG_EVIDENCE_PROXY_MIN_EDGE,
        ),
        PROXY_HIGH_EDGE_PARTICIPATION_MIN_EDGE=_read_env_float(
            "PROXY_HIGH_EDGE_PARTICIPATION_MIN_EDGE",
            Settings.PROXY_HIGH_EDGE_PARTICIPATION_MIN_EDGE,
        ),
        GENERIC_PROXY_HIGH_EDGE_MIN=_read_env_float(
            "GENERIC_PROXY_HIGH_EDGE_MIN",
            Settings.GENERIC_PROXY_HIGH_EDGE_MIN,
        ),
        GROK_PROXY_CONFIDENCE_CAP=_read_env_float(
            "GROK_PROXY_CONFIDENCE_CAP",
            Settings.GROK_PROXY_CONFIDENCE_CAP,
        ),
        GROK_LOW_INFO_CONFIDENCE_CAP=_read_env_float(
            "GROK_LOW_INFO_CONFIDENCE_CAP",
            Settings.GROK_LOW_INFO_CONFIDENCE_CAP,
        ),
        GROK_FALLBACK_MIN_EVIDENCE_QUALITY=_read_env_float(
            "GROK_FALLBACK_MIN_EVIDENCE_QUALITY",
            Settings.GROK_FALLBACK_MIN_EVIDENCE_QUALITY,
        ),
        GROK_ABSTAIN_EVIDENCE_THRESHOLD=_read_env_float(
            "GROK_ABSTAIN_EVIDENCE_THRESHOLD",
            Settings.GROK_ABSTAIN_EVIDENCE_THRESHOLD,
        ),
        CONFIDENCE_SHRINKAGE_FLOOR=_read_env_float(
            "CONFIDENCE_SHRINKAGE_FLOOR", Settings.CONFIDENCE_SHRINKAGE_FLOOR
        ),
        CONFIDENCE_SHRINKAGE_FACTOR=_read_env_float(
            "CONFIDENCE_SHRINKAGE_FACTOR", Settings.CONFIDENCE_SHRINKAGE_FACTOR
        ),
        CONFIDENCE_SHRINKAGE_FACTOR_HIGH=_read_env_float(
            "CONFIDENCE_SHRINKAGE_FACTOR_HIGH",
            Settings.CONFIDENCE_SHRINKAGE_FACTOR_HIGH,
        ),
        CALIBRATION_DIRECT_SHRINKAGE_FACTOR_BOOST=_read_env_float(
            "CALIBRATION_DIRECT_SHRINKAGE_FACTOR_BOOST",
            Settings.CALIBRATION_DIRECT_SHRINKAGE_FACTOR_BOOST,
        ),
        CALIBRATION_ONLINE_UPDATE_ENABLED=_read_env_bool(
            "CALIBRATION_ONLINE_UPDATE_ENABLED",
            Settings.CALIBRATION_ONLINE_UPDATE_ENABLED,
        ),
        CALIBRATION_ONLINE_ALPHA=_read_env_float(
            "CALIBRATION_ONLINE_ALPHA",
            Settings.CALIBRATION_ONLINE_ALPHA,
        ),
        CALIBRATION_ONLINE_MAX_SAMPLES_PER_BUCKET=_read_env_int(
            "CALIBRATION_ONLINE_MAX_SAMPLES_PER_BUCKET",
            Settings.CALIBRATION_ONLINE_MAX_SAMPLES_PER_BUCKET,
        ),
        HISTORICAL_CONFIDENCE_SHRINK_ENABLED=_read_env_bool(
            "HISTORICAL_CONFIDENCE_SHRINK_ENABLED",
            Settings.HISTORICAL_CONFIDENCE_SHRINK_ENABLED,
        ),
        HISTORICAL_CONFIDENCE_SHRINK_MIN_SAMPLES=_read_env_int(
            "HISTORICAL_CONFIDENCE_SHRINK_MIN_SAMPLES",
            Settings.HISTORICAL_CONFIDENCE_SHRINK_MIN_SAMPLES,
        ),
        HISTORICAL_CONFIDENCE_SHRINK_LOOKBACK_DAYS=_read_env_int(
            "HISTORICAL_CONFIDENCE_SHRINK_LOOKBACK_DAYS",
            Settings.HISTORICAL_CONFIDENCE_SHRINK_LOOKBACK_DAYS,
        ),
        HISTORICAL_CONFIDENCE_SHRINK_MAX_DELTA=_read_env_float(
            "HISTORICAL_CONFIDENCE_SHRINK_MAX_DELTA",
            Settings.HISTORICAL_CONFIDENCE_SHRINK_MAX_DELTA,
        ),
        HISTORICAL_CONFIDENCE_SHRINK_MIN_CONFIDENCE=_read_env_float(
            "HISTORICAL_CONFIDENCE_SHRINK_MIN_CONFIDENCE",
            Settings.HISTORICAL_CONFIDENCE_SHRINK_MIN_CONFIDENCE,
        ),
        RESEARCH_QUEUE_ENABLED=_read_env_bool(
            "RESEARCH_QUEUE_ENABLED",
            Settings.RESEARCH_QUEUE_ENABLED,
        ),
        RESEARCH_QUEUE_PERSIST_TO_DB=_read_env_bool(
            "RESEARCH_QUEUE_PERSIST_TO_DB",
            Settings.RESEARCH_QUEUE_PERSIST_TO_DB,
        ),
        RESEARCH_QUEUE_REUSE_LOOKBACK_HOURS=_read_env_int(
            "RESEARCH_QUEUE_REUSE_LOOKBACK_HOURS",
            Settings.RESEARCH_QUEUE_REUSE_LOOKBACK_HOURS,
        ),
        RESEARCH_QUEUE_PRIORITY_ENABLED=_read_env_bool(
            "RESEARCH_QUEUE_PRIORITY_ENABLED",
            Settings.RESEARCH_QUEUE_PRIORITY_ENABLED,
        ),
        RESEARCH_QUEUE_CYCLE_LOG_MAXLEN=_read_env_int(
            "RESEARCH_QUEUE_CYCLE_LOG_MAXLEN",
            Settings.RESEARCH_QUEUE_CYCLE_LOG_MAXLEN,
        ),
        RESEARCH_QUEUE_DRAIN_ENABLED=_read_env_bool(
            "RESEARCH_QUEUE_DRAIN_ENABLED",
            Settings.RESEARCH_QUEUE_DRAIN_ENABLED,
        ),
        RESEARCH_QUEUE_DRAIN_PER_CYCLE=_read_env_int(
            "RESEARCH_QUEUE_DRAIN_PER_CYCLE",
            Settings.RESEARCH_QUEUE_DRAIN_PER_CYCLE,
        ),
        RESEARCH_QUEUE_DRAIN_MIN_AGE_HOURS=_read_env_float(
            "RESEARCH_QUEUE_DRAIN_MIN_AGE_HOURS",
            Settings.RESEARCH_QUEUE_DRAIN_MIN_AGE_HOURS,
        ),
        RESEARCH_QUEUE_DRAIN_MAX_AGE_HOURS=_read_env_float(
            "RESEARCH_QUEUE_DRAIN_MAX_AGE_HOURS",
            Settings.RESEARCH_QUEUE_DRAIN_MAX_AGE_HOURS,
        ),
        RESEARCH_QUEUE_DRAIN_MIN_PRIORITY=_read_env_float(
            "RESEARCH_QUEUE_DRAIN_MIN_PRIORITY",
            Settings.RESEARCH_QUEUE_DRAIN_MIN_PRIORITY,
        ),
        RESEARCH_QUEUE_DRAIN_FORCE_EXTENDED_RESEARCH=_read_env_bool(
            "RESEARCH_QUEUE_DRAIN_FORCE_EXTENDED_RESEARCH",
            Settings.RESEARCH_QUEUE_DRAIN_FORCE_EXTENDED_RESEARCH,
        ),
        RESEARCH_QUEUE_DRAIN_RETRY_COOLDOWN_MINUTES=_read_env_float(
            "RESEARCH_QUEUE_DRAIN_RETRY_COOLDOWN_MINUTES",
            Settings.RESEARCH_QUEUE_DRAIN_RETRY_COOLDOWN_MINUTES,
        ),
        RESEARCH_QUEUE_ZERO_YIELD_PROMOTIONS=_read_env_int(
            "RESEARCH_QUEUE_ZERO_YIELD_PROMOTIONS",
            Settings.RESEARCH_QUEUE_ZERO_YIELD_PROMOTIONS,
        ),
        RESEARCH_QUEUE_ZERO_YIELD_PROMOTIONS_MAX=_read_env_int(
            "RESEARCH_QUEUE_ZERO_YIELD_PROMOTIONS_MAX",
            Settings.RESEARCH_QUEUE_ZERO_YIELD_PROMOTIONS_MAX,
        ),
        RESEARCH_QUEUE_SCORE_PROMOTION_GAP=_read_env_float(
            "RESEARCH_QUEUE_SCORE_PROMOTION_GAP",
            Settings.RESEARCH_QUEUE_SCORE_PROMOTION_GAP,
        ),
        RESEARCH_QUEUE_LOW_YIELD_PLACEHOLDER_MIN_ATTEMPTS=_read_env_int(
            "RESEARCH_QUEUE_LOW_YIELD_PLACEHOLDER_MIN_ATTEMPTS",
            Settings.RESEARCH_QUEUE_LOW_YIELD_PLACEHOLDER_MIN_ATTEMPTS,
        ),
        RESEARCH_QUEUE_LOW_YIELD_PLACEHOLDER_MIN_TIMES_SEEN=_read_env_int(
            "RESEARCH_QUEUE_LOW_YIELD_PLACEHOLDER_MIN_TIMES_SEEN",
            Settings.RESEARCH_QUEUE_LOW_YIELD_PLACEHOLDER_MIN_TIMES_SEEN,
        ),
        EXTENDED_RESEARCH_AFTER_STREAK=_read_env_int(
            "EXTENDED_RESEARCH_AFTER_STREAK",
            Settings.EXTENDED_RESEARCH_AFTER_STREAK,
        ),
        EXTENDED_RESEARCH_COOLDOWN_CYCLES=_read_env_int(
            "EXTENDED_RESEARCH_COOLDOWN_CYCLES",
            Settings.EXTENDED_RESEARCH_COOLDOWN_CYCLES,
        ),
        EXTENDED_RESEARCH_QUEUE_COOLDOWN_CYCLES=_read_env_int(
            "EXTENDED_RESEARCH_QUEUE_COOLDOWN_CYCLES",
            Settings.EXTENDED_RESEARCH_QUEUE_COOLDOWN_CYCLES,
        ),
        DEFINITIVE_OUTCOME_EVIDENCE_QUALITY_FLOOR=_read_env_float(
            "DEFINITIVE_OUTCOME_EVIDENCE_QUALITY_FLOOR",
            Settings.DEFINITIVE_OUTCOME_EVIDENCE_QUALITY_FLOOR,
        ),
        DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX=_read_env_float(
            "DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX",
            Settings.DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX,
        ),
        HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ=_read_env_float(
            "HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ",
            Settings.HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ,
        ),
        PARTICIPATION_TIER_AUDIT_ENABLED=_read_env_bool(
            "PARTICIPATION_TIER_AUDIT_ENABLED",
            Settings.PARTICIPATION_TIER_AUDIT_ENABLED,
        ),
        PARTICIPATION_TIER_GATING_ENABLED=_read_env_bool(
            "PARTICIPATION_TIER_GATING_ENABLED",
            Settings.PARTICIPATION_TIER_GATING_ENABLED,
        ),
        CYCLE_YIELD_ALERT_ESCALATE_AFTER=_read_env_int(
            "CYCLE_YIELD_ALERT_ESCALATE_AFTER",
            Settings.CYCLE_YIELD_ALERT_ESCALATE_AFTER,
        ),
        NEGATIVE_BEST_SCORE_DEEP_ANALYSIS_FLOOR=_read_env_float(
            "NEGATIVE_BEST_SCORE_DEEP_ANALYSIS_FLOOR",
            Settings.NEGATIVE_BEST_SCORE_DEEP_ANALYSIS_FLOOR,
        ),
        CRYPTO_PREFLIGHT_ENABLED=_read_env_bool(
            "CRYPTO_PREFLIGHT_ENABLED",
            Settings.CRYPTO_PREFLIGHT_ENABLED,
        ),
        CRYPTO_THRESHOLD_BUFFER_AUTO_NO_TRADE_PCT=_read_env_float(
            "CRYPTO_THRESHOLD_BUFFER_AUTO_NO_TRADE_PCT",
            Settings.CRYPTO_THRESHOLD_BUFFER_AUTO_NO_TRADE_PCT,
        ),
        GROK_STREAM_TIMEOUT_SECONDS_CRYPTO=_read_env_int(
            "GROK_STREAM_TIMEOUT_SECONDS_CRYPTO",
            Settings.GROK_STREAM_TIMEOUT_SECONDS_CRYPTO,
        ),
        GROK_STREAM_TIMEOUT_SECONDS_WEATHER=_read_env_int(
            "GROK_STREAM_TIMEOUT_SECONDS_WEATHER",
            Settings.GROK_STREAM_TIMEOUT_SECONDS_WEATHER,
        ),
        TIMEOUT_RETRY_AS_MONITOR_ONLY_ENABLED=_read_env_bool(
            "TIMEOUT_RETRY_AS_MONITOR_ONLY_ENABLED",
            Settings.TIMEOUT_RETRY_AS_MONITOR_ONLY_ENABLED,
        ),
        BAYESIAN_ENABLED=_read_env_bool(
            "BAYESIAN_ENABLED",
            Settings.BAYESIAN_ENABLED,
        ),
        BAYESIAN_SKIP_STALE_UPDATES=_read_env_bool(
            "BAYESIAN_SKIP_STALE_UPDATES",
            Settings.BAYESIAN_SKIP_STALE_UPDATES,
        ),
        BAYESIAN_PRIOR_DEFAULT=_read_env_float(
            "BAYESIAN_PRIOR_DEFAULT",
            Settings.BAYESIAN_PRIOR_DEFAULT,
        ),
        BAYESIAN_MIN_UPDATES_FOR_TRADE=_read_env_int(
            "BAYESIAN_MIN_UPDATES",
            _read_env_int(
                "BAYESIAN_MIN_UPDATES_FOR_TRADE",
                Settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
            ),
        ),
        BAYESIAN_MIN_POSTERIOR_DIVERGENCE=_read_env_float(
            "BAYESIAN_MIN_POSTERIOR_DIVERGENCE",
            Settings.BAYESIAN_MIN_POSTERIOR_DIVERGENCE,
        ),
        BAYESIAN_MAX_POSTERIOR=_read_env_float(
            "BAYESIAN_MAX_POSTERIOR",
            Settings.BAYESIAN_MAX_POSTERIOR,
        ),
        BAYESIAN_MAX_CONFIDENCE_BOOST=_read_env_float(
            "BAYESIAN_MAX_CONFIDENCE_BOOST",
            Settings.BAYESIAN_MAX_CONFIDENCE_BOOST,
        ),
        LMSR_ENABLED=_read_env_bool(
            "LMSR_ENABLED",
            Settings.LMSR_ENABLED,
        ),
        LMSR_LIQUIDITY_PARAM_B=_read_env_float(
            "LMSR_LIQUIDITY_PARAM_B",
            Settings.LMSR_LIQUIDITY_PARAM_B,
        ),
        LMSR_MIN_INEFFICIENCY=_read_env_float(
            "LMSR_MIN_INEFFICIENCY",
            Settings.LMSR_MIN_INEFFICIENCY,
        ),
        KELLY_SIZING_ENABLED=_read_env_bool(
            "KELLY_SIZING_ENABLED",
            Settings.KELLY_SIZING_ENABLED,
        ),
        KELLY_DYNAMIC_ENABLED=_read_env_bool(
            "KELLY_DYNAMIC_ENABLED",
            Settings.KELLY_DYNAMIC_ENABLED,
        ),
        KELLY_FRACTION_DEFAULT=_read_env_float(
            "KELLY_FRACTION_DEFAULT",
            Settings.KELLY_FRACTION_DEFAULT,
        ),
        KELLY_FRACTION_SHORT_HORIZON_HOURS=_read_env_int(
            "KELLY_FRACTION_SHORT_HORIZON_HOURS",
            Settings.KELLY_FRACTION_SHORT_HORIZON_HOURS,
        ),
        KELLY_FRACTION_SHORT_HORIZON=_read_env_float(
            "KELLY_FRACTION_SHORT_HORIZON",
            Settings.KELLY_FRACTION_SHORT_HORIZON,
        ),
        KELLY_FRACTION_WEATHER=_read_env_float(
            "KELLY_FRACTION_WEATHER",
            Settings.KELLY_FRACTION_WEATHER,
        ),
        KELLY_MIN_BET_POLICY=_read_env_str(
            "KELLY_MIN_BET_POLICY",
            Settings.KELLY_MIN_BET_POLICY,
        ),
        KELLY_MIN_BET_NEAR_MISS_RATIO=_read_env_float(
            "KELLY_MIN_BET_NEAR_MISS_RATIO",
            Settings.KELLY_MIN_BET_NEAR_MISS_RATIO,
        ),
        KELLY_MIN_BANKROLL_USDC=_read_env_float(
            "KELLY_MIN_BANKROLL_USDC",
            Settings.KELLY_MIN_BANKROLL_USDC,
        ),
        FLIP_GUARD_ENABLED=_read_env_bool(
            "FLIP_GUARD_ENABLED",
            Settings.FLIP_GUARD_ENABLED,
        ),
        FLIP_GUARD_MIN_ABS_CONFIDENCE=_read_env_float(
            "FLIP_GUARD_MIN_ABS_CONFIDENCE",
            Settings.FLIP_GUARD_MIN_ABS_CONFIDENCE,
        ),
        FLIP_GUARD_MIN_CONF_GAIN=_read_env_float(
            "FLIP_GUARD_MIN_CONF_GAIN",
            Settings.FLIP_GUARD_MIN_CONF_GAIN,
        ),
        FLIP_GUARD_MIN_EDGE_GAIN=_read_env_float(
            "FLIP_GUARD_MIN_EDGE_GAIN",
            Settings.FLIP_GUARD_MIN_EDGE_GAIN,
        ),
        FLIP_GUARD_MIN_EVIDENCE_QUALITY=_read_env_float(
            "FLIP_GUARD_MIN_EVIDENCE_QUALITY",
            Settings.FLIP_GUARD_MIN_EVIDENCE_QUALITY,
        ),
        FLIP_GUARD_DIRECT_EVIDENCE_OVERRIDE_ENABLED=_read_env_bool(
            "FLIP_GUARD_DIRECT_EVIDENCE_OVERRIDE_ENABLED",
            Settings.FLIP_GUARD_DIRECT_EVIDENCE_OVERRIDE_ENABLED,
        ),
        FLIP_GUARD_DIRECT_MIN_EDGE=_read_env_float(
            "FLIP_GUARD_DIRECT_MIN_EDGE",
            Settings.FLIP_GUARD_DIRECT_MIN_EDGE,
        ),
        FLIP_GUARD_DIRECT_MIN_LIKELIHOOD_RATIO=_read_env_float(
            "FLIP_GUARD_DIRECT_MIN_LIKELIHOOD_RATIO",
            Settings.FLIP_GUARD_DIRECT_MIN_LIKELIHOOD_RATIO,
        ),
        FLIP_CIRCUIT_BREAKER_ENABLED=_read_env_bool(
            "FLIP_CIRCUIT_BREAKER_ENABLED",
            Settings.FLIP_CIRCUIT_BREAKER_ENABLED,
        ),
        FLIP_CIRCUIT_BREAKER_MAX_FLIPS=_read_env_int(
            "FLIP_CIRCUIT_BREAKER_MAX_FLIPS",
            Settings.FLIP_CIRCUIT_BREAKER_MAX_FLIPS,
        ),
        EVIDENCE_QUALITY_HIGH_CONFIDENCE_OVERRIDE=_read_env_bool(
            "EVIDENCE_QUALITY_HIGH_CONFIDENCE_OVERRIDE",
            Settings.EVIDENCE_QUALITY_HIGH_CONFIDENCE_OVERRIDE,
        ),
        EVIDENCE_QUALITY_CONVERGENT_FLOOR_ENABLED=_read_env_bool(
            "EVIDENCE_QUALITY_CONVERGENT_FLOOR_ENABLED",
            Settings.EVIDENCE_QUALITY_CONVERGENT_FLOOR_ENABLED,
        ),
        EVIDENCE_QUALITY_CONVERGENT_FLOOR_VALUE=_read_env_float(
            "EVIDENCE_QUALITY_CONVERGENT_FLOOR_VALUE",
            Settings.EVIDENCE_QUALITY_CONVERGENT_FLOOR_VALUE,
        ),
        PROXY_PENALTY_CONVERGENT_REDUCTION_ENABLED=_read_env_bool(
            "PROXY_PENALTY_CONVERGENT_REDUCTION_ENABLED",
            Settings.PROXY_PENALTY_CONVERGENT_REDUCTION_ENABLED,
        ),
        HISTORICAL_FAMILY_HIGH_CONF_LOSS_RELAX_THRESHOLD=_read_env_float(
            "HISTORICAL_FAMILY_HIGH_CONF_LOSS_RELAX_THRESHOLD",
            Settings.HISTORICAL_FAMILY_HIGH_CONF_LOSS_RELAX_THRESHOLD,
        ),
        HISTORICAL_FAMILY_BOOST_EVIDENCE_MIN=_read_env_float(
            "HISTORICAL_FAMILY_BOOST_EVIDENCE_MIN",
            Settings.HISTORICAL_FAMILY_BOOST_EVIDENCE_MIN,
        ),
        HISTORICAL_FAMILY_LOSS_DRAG_SCALE=_read_env_float(
            "HISTORICAL_FAMILY_LOSS_DRAG_SCALE",
            Settings.HISTORICAL_FAMILY_LOSS_DRAG_SCALE,
        ),
        HISTORICAL_FAMILY_LOSS_DRAG_SAMPLE_MIN=_read_env_int(
            "HISTORICAL_FAMILY_LOSS_DRAG_SAMPLE_MIN",
            Settings.HISTORICAL_FAMILY_LOSS_DRAG_SAMPLE_MIN,
        ),
        PRE_ANALYSIS_HISTORICAL_FAMILY_PROFIT_BONUS=_read_env_float(
            "PRE_ANALYSIS_HISTORICAL_FAMILY_PROFIT_BONUS",
            Settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PROFIT_BONUS,
        ),
        BORDERLINE_CRITIQUE_REFINEMENT_ENABLED=_read_env_bool(
            "BORDERLINE_CRITIQUE_REFINEMENT_ENABLED",
            Settings.BORDERLINE_CRITIQUE_REFINEMENT_ENABLED,
        ),
        BORDERLINE_CRITIQUE_REFINEMENT_SCORE_BAND=_read_env_float(
            "BORDERLINE_CRITIQUE_REFINEMENT_SCORE_BAND",
            Settings.BORDERLINE_CRITIQUE_REFINEMENT_SCORE_BAND,
        ),
        CODE_EXECUTION_FOR_DEEP_ANALYSIS_ENABLED=_read_env_bool(
            "CODE_EXECUTION_FOR_DEEP_ANALYSIS_ENABLED",
            Settings.CODE_EXECUTION_FOR_DEEP_ANALYSIS_ENABLED,
        ),
        LOG_LEVEL=_read_env_str("LOG_LEVEL", Settings.LOG_LEVEL),
        LOG_FILE_LEVEL=_read_env_str("LOG_FILE_LEVEL", Settings.LOG_FILE_LEVEL),
        LOG_DIR=_read_env_str("LOG_DIR", Settings.LOG_DIR),
        ENABLE_FILE_LOGGING=_read_env_bool(
            "ENABLE_FILE_LOGGING", Settings.ENABLE_FILE_LOGGING
        ),
        ENABLE_JSON_LOGGING=_read_env_bool(
            "ENABLE_JSON_LOGGING", Settings.ENABLE_JSON_LOGGING
        ),
        ENABLE_COLORED_LOGGING=_read_env_bool(
            "ENABLE_COLORED_LOGGING", Settings.ENABLE_COLORED_LOGGING
        ),
        API_COST_INPUT_PER_1K_TOKENS_USD=_read_env_float(
            "API_COST_INPUT_PER_1K_TOKENS_USD",
            Settings.API_COST_INPUT_PER_1K_TOKENS_USD,
        ),
        API_COST_OUTPUT_PER_1K_TOKENS_USD=_read_env_float(
            "API_COST_OUTPUT_PER_1K_TOKENS_USD",
            Settings.API_COST_OUTPUT_PER_1K_TOKENS_USD,
        ),
    )
    strategy = settings.OPPOSITE_OUTCOME_STRATEGY.strip().lower()
    if strategy not in {"block", "hedge"}:
        strategy = Settings.OPPOSITE_OUTCOME_STRATEGY
    score_mode = settings.SCORE_GATE_MODE.strip().lower()
    if score_mode not in {"off", "shadow", "active"}:
        score_mode = Settings.SCORE_GATE_MODE
    kelly_min_bet_policy = settings.KELLY_MIN_BET_POLICY.strip().lower()
    if kelly_min_bet_policy not in {"skip", "floor", "fallback_edge_scaling"}:
        kelly_min_bet_policy = Settings.KELLY_MIN_BET_POLICY
    kelly_near_miss_ratio = max(
        0.0,
        min(1.0, float(settings.KELLY_MIN_BET_NEAR_MISS_RATIO)),
    )

    settings = Settings(
        **{
            **settings.__dict__,
            "OPPOSITE_OUTCOME_STRATEGY": strategy,
            "SCORE_GATE_MODE": score_mode,
            "KELLY_MIN_BET_POLICY": kelly_min_bet_policy,
            "KELLY_MIN_BET_NEAR_MISS_RATIO": kelly_near_miss_ratio,
        }
    )

    _validate_required(settings)
    return settings


def _required_env_vars(settings: Settings) -> tuple[str, ...]:
    return tuple(BASE_REQUIRED_ENV_VARS)


def _validate_required(
    settings: Settings, required: Iterable[str] | None = None
) -> None:
    required_vars = tuple(required) if required is not None else _required_env_vars(settings)
    missing = [name for name in required_vars if not getattr(settings, name)]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Missing required environment variables: {names}")


def build_search_config(settings: Settings) -> SearchConfig:
    """Build SearchConfig from settings to keep wiring centralized."""
    from datetime import datetime, timedelta, timezone

    search_now = datetime.now(timezone.utc)
    return SearchConfig(
        from_date=search_now - timedelta(hours=settings.SEARCH_LOOKBACK_HOURS),
        to_date=search_now,
        allowed_domains=list(settings.SEARCH_ALLOWED_DOMAINS),
        allowed_x_handles=list(settings.SEARCH_ALLOWED_X_HANDLES),
        source_domains_pool=list(settings.SEARCH_ALLOWED_DOMAINS),
        source_x_handles_pool=list(settings.SEARCH_ALLOWED_X_HANDLES),
        max_allowed_domains=settings.SEARCH_PROFILE_MAX_DOMAINS,
        max_allowed_x_handles=settings.SEARCH_PROFILE_MAX_X_HANDLES,
        multimedia_confidence_range=settings.MULTIMEDIA_CONFIDENCE_THRESHOLD,
    )


@dataclass
class SearchConfig:
    from_date: "datetime | None" = None
    to_date: "datetime | None" = None
    allowed_domains: list[str] = field(default_factory=list)
    allowed_x_handles: list[str] = field(default_factory=list)
    source_domains_pool: list[str] = field(default_factory=list)
    source_x_handles_pool: list[str] = field(default_factory=list)
    max_allowed_domains: int = 5
    max_allowed_x_handles: int = 10
    enable_multimedia: bool = False
    multimedia_confidence_range: tuple[float, float] = (0.55, 0.75)
    profile_name: str = "generic"
    lookback_hours: int | None = None
