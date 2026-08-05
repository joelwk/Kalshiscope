import hashlib

from config import SearchConfig
from grok_client import GrokClient, _SYSTEM_PROMPT_ANALYZE, _SYSTEM_PROMPT_DEEP
from models import Market, MarketOutcome, TradeDecision


EXPECTED_SYSTEM_PROMPT_HASHES = {
    # Re-pinned Aug 4 2026: edge honesty, external-odds hygiene, aggregator/MapClick caps.
    "analyze": "03d857e58c10a71f64b8da7c6ac49b84ee30fb1d7f0fd0577687960be1044c58",
    "deep": "d04c2d3565b836fac6a8f42fb1bc2e992d1eac1aceba1e119857929910edfc9e",
}

EXPECTED_MARKET_PROMPT_HASHES = {
    # Re-pinned Aug 4 2026: category/constraint field-hygiene updates.
    "commodities:deep_false": "49ff9f19b3a207180d690fb26bfacbe9710de56992d33f07fc89eaae1582e5d4",
    "commodities:deep_true": "f2d19e7c8290ba76a36a3ad778c8acd94421d402804984ac6eba2be01e66cef8",
    "crypto:deep_false": "7e6c1956a717c9f845cc80f44f48bf8120a33c851a78e4917587c210f7bbfbca",
    "crypto:deep_true": "85d821b8e4c4f06f1d4adb6fc583913d67e18bcb09389072053df9c10dbefa58",
    "generic:deep_false": "6849bff3192109e4bdb5cc79e64b796db8bc0cf7333030a6807d9ae4e78e09ed",
    "generic:deep_true": "a98ff2d218e0beb6fb5cfd81eb08139a60feb117367e7fec3fd0f36d2302a18c",
    "music:deep_false": "7303d1797ea93804fe6eb871789970967ccbc36b8cafe6a6463321130efc340a",
    "music:deep_true": "d1e933d58c00c77d63e08e7217617fc8044f3f091467b24192c9303204699339",
    "politics:deep_false": "82721d5546da9a9a4443eb6f6afcbbcf739e75dcae3a25b8a8cb577c6c65d59f",
    "politics:deep_true": "c50baf88c2b94ab274e5c06fdb8d6e138435542b3291bba2e5d035e40e17a74d",
    "speech:deep_false": "a2b9ac7999318002b4b51b71c74b5c92416107cce77d6ac6b131322f14dd81d1",
    "speech:deep_true": "e7e0fc1dff8c7e6c3ba9dfbf857dffbd474738ff430a37a7375d8111eac55ebf",
    "speech_mention:deep_false": "d46ab7b93a8010202e09129168c99394bed3ec1914d23847a9c7a9b7418934b6",
    "speech_mention:deep_true": "a3bd7e3ffcdc91a03ea0016cd999be6f58c649add310be5dbeeba99416ee8552",
    "sports:deep_false": "66bf19af75717e70c6e9914848d155c2314e684220c024efe47a42faec910d91",
    "sports:deep_true": "b5664f285b1390718ad1cd7d5b2ebb6e434bd87ec23922f155d0b5103961d54c",
    "weather:deep_false": "cf722d0988eb77f560b3bac1a13cc39c4952584e0030519fe5bd37417b06a997",
    "weather:deep_true": "770f9b1214b7ef8f0051181cfaf9ffa707763f098c731697814a5cacf163c4d1",
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
    "should_trade": "215fad0b1f319687dc14c538d174d5c12dccdfb0dac72b7af305004f73f8c82c",
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


def test_system_prompt_contains_hallucination_and_direct_evidence_rules() -> None:
    assert "Confidence must not exceed evidence_quality + 0.05." in _SYSTEM_PROMPT_ANALYZE
    assert "edge versus market implied probability exceeds 40 percentage points" in _SYSTEM_PROMPT_ANALYZE
    assert "Self-consistency check" in _SYSTEM_PROMPT_ANALYZE
    assert "Treat live-price threshold evidence as direct when URL-cited" in _SYSTEM_PROMPT_ANALYZE
    assert "you MUST populate primary_source_url with the exact URL used" in _SYSTEM_PROMPT_ANALYZE
    assert "primary_source_url must be a real https:// link" in _SYSTEM_PROMPT_ANALYZE
    assert "Side consistency:" in _SYSTEM_PROMPT_ANALYZE
    assert "Fallback/no-external-odds trades must clear the configured fallback edge threshold" in _SYSTEM_PROMPT_ANALYZE


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
