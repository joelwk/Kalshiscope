import hashlib

from config import SearchConfig
from grok_client import GrokClient, _SYSTEM_PROMPT_ANALYZE, _SYSTEM_PROMPT_DEEP
from models import Market, MarketOutcome, TradeDecision


EXPECTED_SYSTEM_PROMPT_HASHES = {
    "analyze": "965a8140f6296556298c0d8cd9ce8cb55c37471fdee544166522153c58bb8887",
    "deep": "d0550226ab14dcc2ec52fff8733f51df5daa8c207e329b685a4b0e07c532dfc6",
}

EXPECTED_MARKET_PROMPT_HASHES = {
    "commodities:deep_false": "42d9e5aebb3c3d08f0e6c9e8733d20326d2b919f7002300a843b83fcde744126",
    "commodities:deep_true": "586d62780d355c337c66bd2b2640e40962c35071c4d2cefb93491472ba507c3a",
    "crypto:deep_false": "fd22ee2738f59d695bb106d30a2bdbddd10899b6ae0bfba2e2782bc8d2ae16ce",
    "crypto:deep_true": "1271f3c348401d8d48fc0e6d757aca966e540339145ced4ffecf92fb046d5bf3",
    "generic:deep_false": "f760221c069205bc55bb73d0740ceca7cecc49b82019d5abc6cbdee52458723a",
    "generic:deep_true": "2d90e4a052badf599dd8bf0cb10f157cecb9eee763ba4e3ae70126f918185d4b",
    "music:deep_false": "d449b6a05834eae11c55a9a6969446f576a380d8f4290e1bcadee3761651e8fc",
    "music:deep_true": "1d9b1debb4d0b92ab011c8f2274aec504bf25250d3be68eea14feaeca7b3c140",
    "politics:deep_false": "773c1d3bd5bf7f84e62a0d8be804cb1b7f528d111695d10f9ac487d6d0b8b5a6",
    "politics:deep_true": "3a5e1cdeb8e26c57bf22f3a72983ed407efbe17b9237468ea724d799c7c281cd",
    "speech:deep_false": "44004d077b07f695867b9a2608adab8c2ac36219ff4d5b793a1c7f5103361dc5",
    "speech:deep_true": "902c7cb40d7eb1316f7280654160bd344f9ebe662d5f9085db98b0ec4e0254c2",
    "speech_mention:deep_false": "5e6d3e65ce329d6f31915dafab60fba664add664b6bb525ca7494883ddec28cf",
    "speech_mention:deep_true": "c9d6660a2a5a9c6850cc53cd5f45834281f52a20fe140d9a7dc5b4b85f213e45",
    "sports:deep_false": "39a6bd21f86b13793f4911e1d507c2eed23b7e34a6185ab5e641528c415b9744",
    "sports:deep_true": "f5d783281897d9d2e1cb73570ff675bcf37ad40a89a484680158faad2cf2b817",
    "weather:deep_false": "609dfd16b1bacb249d309815aae40ea82c8b99f74f26286f3d77af03ea3e62c5",
    "weather:deep_true": "0eb16aeae1efccc65a0c8b0bec326f595b72185c65a4be2cfbb8ec9e63be10b9",
}

EXPECTED_SCHEMA_DESCRIPTION_HASHES = {
    "abstain": "139411c8d6a39816135c7602c019ef257a8bc4cbe6de148d6accbf5ac05de84b",
    "bet_size_pct": "641a0cd6d5347f4b1e64447cfe665638765238c6505b67697d4b2e2750d80dc4",
    "cached_tokens": "17c3feb94f70c94185ca5932f3717f8808124cb104f6a59ca433d72b6970a180",
    "completion_tokens": "787a3235dae3e9a5d93be988eef22495d944c55cda92b20ce5ac21669f97524a",
    "confidence": "3993782437221b5549e1c1d527d246a5a9899919641c5bceb63a90a89d4ea137",
    "definitive_outcome_detected": "d5b6ac13a07ebfbbb084577dbf0de381318ac0ce8020bcdec550804cd5ddc51e",
    "edge_external": "c85f0f881724bd6cb9c2626a4b3497e6540511c9dbbaef1b125b79c484ad16ad",
    "edge_source": "2901c6e3da6168e6b4205bf0660b22475968ecc6b946ebb1a66841b517cd2bce",
    "evidence_basis": "4b71f131ebb0cf364858c8ccad2d966cf3cd0330e6f6d28d09e64ee3b0c63bf5",
    "evidence_floor_suppressed_reason": "68e10d20cd67474640a8ee43d46ff0c38013d9eb0de86e1b4deb395cc0aa1b9f",
    "evidence_quality": "6658519042cb5d88eae008c8641d15e4c7bd16eeb7b9929bec12f82cfbc2045f",
    "evidence_quality_floor_applied": "6435a1f1d466bcbb12a753b06e319e98d3f3eef7812cf48a155eb5bb3f43427a",
    "implied_prob_external": "db0b9497d1f31ef2601d3130975e4b4ea115c49758eedb266f4986410ef69765",
    "likelihood_ratio": "837b46f6e4d7d577f488326851fa2b036d12f4ef27e0628fa9230c1cc2b05b65",
    "my_prob": "5abe352ec13d685677213b1120c1914090c2eb9115cd8450156de34b1c8b31d0",
    "outcome": "7642b21e60176506e653b20b3183cc5bc0c361804028e0a8a98a697e8b65f94c",
    "primary_source_url": "6101dd4040210985d00f82c41313d2d69a513361eca215aeab2883f35387af70",
    "prompt_tokens": "49db0a66d30b18b71326920cd8ceea07a4c204889a27ad00bc6a275ca0c97b36",
    "raw_bet_size_pct": "8f72e04cf77115c2893583367d0f8ae98d15f7725ccca7072b3c9fa3308a0cce",
    "raw_confidence": "e44485c16aeb9954bc3709398af748c4126ff9b1eff3c46fce31f81f28ec30d9",
    "raw_evidence_quality": "5a06f668a3cc075d9b7dad2ca7e01cdfbf0786df7cf12c0e7269ee05f2d28b4f",
    "raw_outcome": "907648115ca3f1fb58fa62d08cbce43ea30930d19ce410f3c1e39031aebd0c53",
    "raw_reasoning": "a26f9a108c792b0d03e1ba7626b7a8d5f68f9c1a48fd1875818ae23d4d1eecc3",
    "raw_should_trade": "4e06a691859874c81a7e47e8eb005df24c905457147959d13b924f507df9027e",
    "reasoning": "c07771931c2070d60b09b1b1e7701ceb33017c7da58cd8a8deaf48ad7fbc1bcf",
    "reasoning_tokens": "6b47f605e7844debc22ecbf3b199a6f581dd2a72a1ebdaaca8c06f282f7122b9",
    "should_trade": "997fe2b7115c9a5af68197e3bb41b5e2d0e653d5b8ba8533b0e28a1233488e53",
    "source_match_class": "8bca00a6b3b1f3d3edc79c9a23303957f8c3971dc39208283368c5b6c1f07efc",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_market_prompt_hashes() -> dict[str, str]:
    client = GrokClient(api_key="x")
    base_market = Market(
        id="MKT-BASE",
        question="Will event happen?",
        subtitle="Sub",
        resolution_criteria="Rules",
        outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
        liquidity_usdc=123.0,
    )
    mention_market = Market(
        id="KXGOVERNORMENTION-26APR09-OIL",
        question="Will the governor mention oil today?",
        outcomes=[MarketOutcome(name="YES", price=0.5), MarketOutcome(name="NO", price=0.5)],
        liquidity_usdc=100.0,
    )
    commodity_market = Market(
        id="GOLD-TEST",
        question="Will gold close above 4600?",
        category="business",
        outcomes=[MarketOutcome(name="YES", price=0.51), MarketOutcome(name="NO", price=0.49)],
        liquidity_usdc=200.0,
    )
    previous_summary = "None"

    prompt_hashes: dict[str, str] = {}
    for profile_name in ["sports", "politics", "crypto", "weather", "speech", "music"]:
        config = SearchConfig(profile_name=profile_name, lookback_hours=24)
        prompt_hashes[f"{profile_name}:deep_false"] = _sha256(
            client._build_market_prompt(base_market, config, previous_summary, False)
        )
        prompt_hashes[f"{profile_name}:deep_true"] = _sha256(
            client._build_market_prompt(base_market, config, previous_summary, True)
        )

    prompt_hashes["speech_mention:deep_false"] = _sha256(
        client._build_market_prompt(
            mention_market,
            SearchConfig(profile_name="speech", lookback_hours=24),
            previous_summary,
            False,
        )
    )
    prompt_hashes["speech_mention:deep_true"] = _sha256(
        client._build_market_prompt(
            mention_market,
            SearchConfig(profile_name="speech", lookback_hours=24),
            previous_summary,
            True,
        )
    )
    prompt_hashes["commodities:deep_false"] = _sha256(
        client._build_market_prompt(
            commodity_market,
            SearchConfig(profile_name="generic", lookback_hours=24),
            previous_summary,
            False,
        )
    )
    prompt_hashes["commodities:deep_true"] = _sha256(
        client._build_market_prompt(
            commodity_market,
            SearchConfig(profile_name="generic", lookback_hours=24),
            previous_summary,
            True,
        )
    )
    prompt_hashes["generic:deep_false"] = _sha256(
        client._build_market_prompt(
            base_market,
            SearchConfig(profile_name="generic", lookback_hours=24),
            previous_summary,
            False,
        )
    )
    prompt_hashes["generic:deep_true"] = _sha256(
        client._build_market_prompt(
            base_market,
            SearchConfig(profile_name="generic", lookback_hours=24),
            previous_summary,
            True,
        )
    )
    return dict(sorted(prompt_hashes.items()))


def test_system_prompt_hashes_are_stable() -> None:
    assert _sha256(_SYSTEM_PROMPT_ANALYZE) == EXPECTED_SYSTEM_PROMPT_HASHES["analyze"]
    assert _sha256(_SYSTEM_PROMPT_DEEP) == EXPECTED_SYSTEM_PROMPT_HASHES["deep"]


def test_market_prompt_hashes_are_stable() -> None:
    assert _build_market_prompt_hashes() == EXPECTED_MARKET_PROMPT_HASHES


def test_trade_decision_schema_descriptions_are_stable() -> None:
    properties = TradeDecision.model_json_schema().get("properties", {})
    current_hashes = {
        field_name: _sha256(metadata.get("description", ""))
        for field_name, metadata in properties.items()
        if "description" in metadata
    }
    assert dict(sorted(current_hashes.items())) == EXPECTED_SCHEMA_DESCRIPTION_HASHES


def test_trade_decision_schema_includes_primary_source_url() -> None:
    properties = TradeDecision.model_json_schema().get("properties", {})
    assert "primary_source_url" in properties


def test_system_prompt_contains_hallucination_and_direct_evidence_rules() -> None:
    assert "Confidence must not exceed evidence_quality + 0.05." in _SYSTEM_PROMPT_ANALYZE
    assert "edge versus market implied probability exceeds 32 percentage points" in _SYSTEM_PROMPT_ANALYZE
    assert "Treat live-price threshold evidence as direct when URL-cited" in _SYSTEM_PROMPT_ANALYZE
    assert "you MUST populate primary_source_url with the exact URL used" in _SYSTEM_PROMPT_ANALYZE
    assert "Fallback/no-external-odds trades must clear the configured fallback edge threshold" in _SYSTEM_PROMPT_ANALYZE
