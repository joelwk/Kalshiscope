import hashlib

from config import SearchConfig
from grok_client import GrokClient, _SYSTEM_PROMPT_ANALYZE, _SYSTEM_PROMPT_DEEP
from models import Market, MarketOutcome, TradeDecision


EXPECTED_SYSTEM_PROMPT_HASHES = {
    # Re-pinned Aug 12 2026: anti-hunch edge_mechanism requirement.
    "analyze": "897c6280f9f172449868b373c57ee415a68e936d83258ceebe508c3513ab4658",
    "deep": "bb94bf528ff9a0d05e5b369d01d3a23e5b6ed23dc63796f362f9e36ac902eaeb",
}

EXPECTED_MARKET_PROMPT_HASHES = {
    # Re-pinned Aug 12 2026: mechanism-or-abstain constraints and category hints.
    "commodities:deep_false": "e88c352293f537572e6fa93884405b59bce4ceb5115bbe84f58bde5ce1747765",
    "commodities:deep_true": "bbb6989d478adde7092ce30a8a04961ea4d15f1c9dd9c08c09e8fb6d4aad0a48",
    "crypto:deep_false": "d0dc68ab06acb33fb6e1c98ac92a094fac8af83314948f420115b6b2a3931db0",
    "crypto:deep_true": "ba0e7a492f5358ac4525ede5db3e28d626913bb2dd9d8f26cac8a2c7de46297a",
    "generic:deep_false": "5c0e3518b2f7493da63a8f14a1be11dda02c95baaa3461fcbd7ce6aa9db8eadb",
    "generic:deep_true": "febc8590bbe2b8fa0da65218d29d1653d7ed54a4a218895d1f9207404e908571",
    "music:deep_false": "bb8f75e9b8f502862441105fb061d3032c2c6ae6d62d04bcd0acbcb021172151",
    "music:deep_true": "f339f4dd6def06e4353750dd69d1f042e5520bbbe79e9a4438404ff93137c41d",
    "politics:deep_false": "f12753300d7f3943b15947e551c1479745fec45bdd1586dbb1cb71c27f11edaf",
    "politics:deep_true": "32073fdfa7e985523c91eae818e976aff413f70c2e4a8b498e84767cbd88f7eb",
    "speech:deep_false": "243ebd8c59080e2a1bdc4604b96680541704be5a31d91f8a54d4438241ac35b8",
    "speech:deep_true": "a8ef2663766b640b8d4574ff7d808fb35e72367b141b30c8a9823d174f8fb2f6",
    "speech_mention:deep_false": "8d513a8ba0905b8df6211e36485a714137a5b00002c364b62e460b82abdd33ec",
    "speech_mention:deep_true": "47b8bacff4ac722925d08861f0cb76e6e9206c0745334c4b455605b674bbb87b",
    "sports:deep_false": "47eb8cd85ca3e940a622ab2c5482ee20a48102a9ded6ccfb8ac7e6270df56e93",
    "sports:deep_true": "cdfdec0b0afc5536c0f87b49bcd436b517f648afb702f7e2fba14587d72dd97b",
    "weather:deep_false": "ea82703262b4d9c780c0b93f3d1ea303828699512a510dfa235a958a159b9570",
    "weather:deep_true": "96cb46693a1e62cebac0e8b1b0bc19d68e32604bd599953f9ba23969908348df",
}

EXPECTED_SCHEMA_DESCRIPTION_HASHES = {
    "abstain": "139411c8d6a39816135c7602c019ef257a8bc4cbe6de148d6accbf5ac05de84b",
    "base_rate_used": "37af25f50be73dbfece673e830b4583f4c3efeaca815fd0fffeeae0b1f68ac5a",
    "bet_size_pct": "641a0cd6d5347f4b1e64447cfe665638765238c6505b67697d4b2e2750d80dc4",
    "cached_tokens": "17c3feb94f70c94185ca5932f3717f8808124cb104f6a59ca433d72b6970a180",
    "code_execution_used": "01fa8db002c20f439e072fdc6edbb98dd81754b999e3cd1ae3e763ddb84956a0",
    "completion_tokens": "787a3235dae3e9a5d93be988eef22495d944c55cda92b20ce5ac21669f97524a",
    "confidence": "04b522eca80f69fed0bcb20dfee763f111f267ef6693886f4e9f91f80405271c",
    "definitive_outcome_detected": "d5b6ac13a07ebfbbb084577dbf0de381318ac0ce8020bcdec550804cd5ddc51e",
    "edge_external": "f282db8b1f7a551cd97b4d59c60a19c4b081c6dc016a6a302b347376897862b6",
    "edge_mechanism": "3997dfef2a05ad911f4010fca0afd6fbf392172366e8fa91efe28db711b483a1",
    "edge_source": "b9776d71fbfe07f42694983863a13414ab896d3e57b689559af48d367499daee",
    "evidence_basis": "f95084f2b5778968536af2848391348a95eac3b52869a7b065975a69dcbe5d0b",
    "evidence_floor_suppressed_reason": "68e10d20cd67474640a8ee43d46ff0c38013d9eb0de86e1b4deb395cc0aa1b9f",
    "evidence_quality": "476f825d5b7a15e377c7e8f53749beb0a39db6c6f965065a801ece51ff07b0b6",
    "evidence_quality_floor_applied": "6435a1f1d466bcbb12a753b06e319e98d3f3eef7812cf48a155eb5bb3f43427a",
    "implied_prob_external": "bcee52dd7b4d7128147948661abcf9b8c71f040a12bfb61123b5f147ce544cdd",
    "key_sources": "3aa0cc5c44f0f8a2ee451eeb25844c11c7d34426c2a2627651c814e5a4f38747",
    "likelihood_ratio": "665e8eb8c191397f65c92dc151a02aca85b7638f3c2c640751da9d89c69917d8",
    "my_prob": "fe7f5f44a678d62caa666e93054cf80e20cbf465c650ed67984deab8209f5a2f",
    "outcome": "67cf74e5d00e1ebece3635535fb09d9dc67b9a4a997a3246761c44ccd6198399",
    "primary_source_url": "bdd16205b16f6b72d3c9228c2f90468953d813dca5f11c657f2000539c9c1fea",
    "probability_yes": "b539166cc0a6aa8aa9d5f77845fd0c94631914dd2834648a0b5c27c26272dcf6",
    "prompt_tokens": "49db0a66d30b18b71326920cd8ceea07a4c204889a27ad00bc6a275ca0c97b36",
    "raw_bet_size_pct": "8f72e04cf77115c2893583367d0f8ae98d15f7725ccca7072b3c9fa3308a0cce",
    "raw_confidence": "e44485c16aeb9954bc3709398af748c4126ff9b1eff3c46fce31f81f28ec30d9",
    "raw_evidence_quality": "5a06f668a3cc075d9b7dad2ca7e01cdfbf0786df7cf12c0e7269ee05f2d28b4f",
    "raw_outcome": "907648115ca3f1fb58fa62d08cbce43ea30930d19ce410f3c1e39031aebd0c53",
    "raw_reasoning": "a26f9a108c792b0d03e1ba7626b7a8d5f68f9c1a48fd1875818ae23d4d1eecc3",
    "raw_should_trade": "4e06a691859874c81a7e47e8eb005df24c905457147959d13b924f507df9027e",
    "reasoning": "247e17a3b1e2943c08f8ca99cfe0422652a66c814b0b29443d3d57dd21639b7f",
    "reasoning_tokens": "6b47f605e7844debc22ecbf3b199a6f581dd2a72a1ebdaaca8c06f282f7122b9",
    "self_critique": "ab2d939c15dcfe7ab68bd6af225432bdedfd4596e126f76a74ac6afde5bb7bbe",
    "should_trade": "bf0c9e7de6bf47aa43bda0103abf9462462438e8ce3b89ae93e503f1011a108e",
    "source_match_class": "8bca00a6b3b1f3d3edc79c9a23303957f8c3971dc39208283368c5b6c1f07efc",
    "uncertainty_note": "0c8956f9b9c3e8e792696a9e4f2ab4c901795cd01e5d9b963b1ae33093f7539f",
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
    assert "probability_yes" in properties
    assert "self_critique" in properties
    assert "edge_mechanism" in properties


def test_system_prompt_contains_hallucination_and_direct_evidence_rules() -> None:
    assert "Confidence must not exceed evidence_quality + 0.05." in _SYSTEM_PROMPT_ANALYZE
    assert "edge versus market implied probability exceeds 40 percentage points" in _SYSTEM_PROMPT_ANALYZE
    assert "Self-consistency check" in _SYSTEM_PROMPT_ANALYZE
    assert "Treat live-price threshold evidence as direct when URL-cited" in _SYSTEM_PROMPT_ANALYZE
    assert "you MUST populate primary_source_url with the exact URL used" in _SYSTEM_PROMPT_ANALYZE
    assert "primary_source_url must be a real https:// link" in _SYSTEM_PROMPT_ANALYZE
    assert "Side consistency:" in _SYSTEM_PROMPT_ANALYZE
    assert "Fallback/no-external-odds trades must clear the configured fallback edge threshold" in _SYSTEM_PROMPT_ANALYZE
    assert "A hunch, \"form,\" vibe, or unexplained directional view is edge_mechanism=none" in _SYSTEM_PROMPT_ANALYZE


def test_system_prompt_contains_edge_honesty_and_field_hygiene() -> None:
    assert "Edge honesty" in _SYSTEM_PROMPT_ANALYZE
    assert "External odds hygiene" in _SYSTEM_PROMPT_ANALYZE
    assert "Never copy Kalshi yes_price into implied_prob_external" in _SYSTEM_PROMPT_ANALYZE
    assert "Absence vs quote" in _SYSTEM_PROMPT_ANALYZE
    assert "Aggregator discipline" in _SYSTEM_PROMPT_ANALYZE
    assert "Prefer null over a reflexive 1.0" in _SYSTEM_PROMPT_ANALYZE


def test_weather_hint_blocks_mapclick_optimism() -> None:
    from prompts.loader import load_prompt

    weather = load_prompt("user/category_hints/weather")
    assert "MapClick edge honesty" in weather
    assert "Same-day KXHIGHT" in weather
    assert "never invent my_prob far from the Kalshi price" in weather or "do not invent my_prob far from the Kalshi price" in weather


def test_crypto_hint_blocks_absence_with_quote_url() -> None:
    from prompts.loader import load_prompt

    crypto = load_prompt("user/category_hints/crypto")
    assert "never absence_only" in crypto
    assert "15-minute / ultra-short direction markets" in crypto
    assert "Do not set evidence_basis=direct merely because an exchange URL exists" in crypto


def test_deep_constraints_prefer_repair_before_abstain() -> None:
    from prompts.loader import load_lines

    lines = load_lines("user/fragments/constraints_deep")
    joined = "\n".join(lines)
    assert "Repair before abstain" in joined
    assert "paste the real https primary_source_url" in joined
    assert "Only abstain after repair attempts" in joined
    assert "Edge honesty on deep pass" in joined
