import hashlib

from config import SearchConfig
from grok_client import GrokClient, _SYSTEM_PROMPT_ANALYZE, _SYSTEM_PROMPT_DEEP
from models import Market, MarketOutcome, TradeDecision


EXPECTED_SYSTEM_PROMPT_HASHES = {
    # Re-pinned Aug 19 2026: weather/numeric field repair and favorite-price cap.
    "analyze": "a00aa8f24f5569c54f41d3ac39e0b91e905c2f5cf20975c156521b1346e4238f",
    "deep": "447595d7487c060bd07539fc0327c185a955ce8c8dd4102ea12dcd4938c0bd14",
}

EXPECTED_MARKET_PROMPT_HASHES = {
    # Re-pinned Aug 19 2026: NWS PoP implied, fallback-not-none, family hint repairs.
    "commodities:deep_false": "b1baaaa661424945224d271f2fb44bed899c3fe00308545ef00b75a466c2ba46",
    "commodities:deep_true": "9313bbf962c43e38684d908823fdd4c8ee1b4b2d5b66a598a077cbcecfe04f14",
    "crypto:deep_false": "9fc9744dc0c6afa67d38903a49e27673bf0c2802018c6ea0bbcb9df9394035c2",
    "crypto:deep_true": "b2ab083c4fdf277dc264767d7aa946bd71ee0dafb118e0fdd762ea9b989df9e2",
    "generic:deep_false": "9aa0ba4496ade868d5252a038c09756ad637d6ff21c91366da112a99e84d5cd9",
    "generic:deep_true": "cb39d6611ba96c75550b0ac2daca43b083e6abab6dd1707ccc3262c6a8fe81ac",
    "music:deep_false": "fb11f93439a3cc643d0081f4bc46824bd4490d63eb2b9cb5613a9f0fa29c171d",
    "music:deep_true": "f816441dbee48166f01d02323c7a395ca96125effb3b48757f10d86689278ded",
    "politics:deep_false": "8c2f5fee9a0cffbdbecc492c4e0575e5c74630166f9283b13d089d9621f2c135",
    "politics:deep_true": "1e6b0ee975c552397c952fbb4ec9c6c2a0f1d7538023275e37071c390baf09f1",
    "speech:deep_false": "ffaccf37a2e1684934c1325571a155d8d2344d0864889be6dc8c8627b8a17fad",
    "speech:deep_true": "849f8072f798804baee46c7bd4c8113e7e06e07b3447a1bfea46eec07157ad98",
    "speech_mention:deep_false": "abe982c2939059f1a8ff2632a890d2dabd321ca84bdc0a83c472163b082827ae",
    "speech_mention:deep_true": "4be155b5ffa5557acfdb5ce9e7c422a1470028ab34d830ed26fd7f931cf8c8aa",
    "sports:deep_false": "9135576cb7f4ebafa9ea6ab74ca51cb2614089dc974377a18ad6db4810a1ca03",
    "sports:deep_true": "38b3d70329442ee4479b48c1273b959b85d9a93f365eb6cc15018e77e1612c6b",
    "weather:deep_false": "3b190401a6c7187a2d50ab6d3e68b25778852e6b40df37d24b722ad83520f01d",
    "weather:deep_true": "d4d3a1ccad3245a3c2d7c702526fcd96d7d6a4ffe99f018b6cf93c8fe50be1df",
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
    "edge_mechanism": "7e41188d8cee5efee2937ec4614d8e2cf301f1da7318db73014c2ba3dee98561",
    "edge_source": "71c6ba27fd276fa50f55c8d27ff8852a0e92e85be1adddccff783cc7d67e3b4b",
    "evidence_basis": "f95084f2b5778968536af2848391348a95eac3b52869a7b065975a69dcbe5d0b",
    "evidence_floor_suppressed_reason": "68e10d20cd67474640a8ee43d46ff0c38013d9eb0de86e1b4deb395cc0aa1b9f",
    "evidence_quality": "476f825d5b7a15e377c7e8f53749beb0a39db6c6f965065a801ece51ff07b0b6",
    "evidence_quality_floor_applied": "6435a1f1d466bcbb12a753b06e319e98d3f3eef7812cf48a155eb5bb3f43427a",
    "implied_prob_external": "14cb90cd70f2f2a0d60a5e505ff85fe32ab7577305891ea680c75eaf9372b8c4",
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
    assert "(10) Kalshi copy ban" in _SYSTEM_PROMPT_ANALYZE
    assert "(9) Weather/numeric field repair" in _SYSTEM_PROMPT_ANALYZE
    assert "implied_prob_external must not equal the Kalshi yes_price" in _SYSTEM_PROMPT_ANALYZE
    assert "If chosen-side Kalshi implied >= 0.55, do not emit raw confidence >= 0.70 unless settlement_already_known" in _SYSTEM_PROMPT_ANALYZE
    assert '"Not final yet" is remaining-session uncertainty, not edge_mechanism=none' in _SYSTEM_PROMPT_ANALYZE


def test_weather_hint_blocks_mapclick_optimism() -> None:
    from prompts.loader import load_prompt

    weather = load_prompt("user/category_hints/weather")
    assert "MapClick edge honesty" in weather
    assert "Same-day KXHIGHT" in weather
    assert "never invent my_prob far from the Kalshi price" in weather or "do not invent my_prob far from the Kalshi price" in weather
    assert "PoP is implied_prob_external" in weather
    assert "implied_prob_external" in weather
    assert "morning-dry" in weather
    assert "Never edge_source=none when an NWS URL is cited" in weather
    assert "Current page is not CLI" in weather
    assert "WFO hygiene" in weather


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
    assert "this pass is field repair" in joined
    assert "do not abstain solely because the prior pass stamped none" in joined


def test_schema_edge_source_forbids_none_when_nws_or_quote_cited() -> None:
    properties = TradeDecision.model_json_schema().get("properties", {})
    edge_source = properties["edge_source"]["description"]
    implied = properties["implied_prob_external"]["description"]
    mechanism = properties["edge_mechanism"]["description"]
    assert "NWS PoP and exchange/Tier-1 quotes are never none" in edge_source
    assert "NWS PoP for rain markets is a valid implied_prob_external" in implied
    assert "Do not use none when a live quote, CLI, or PoP URL is in the reasoning" in mechanism


def test_entertainment_and_politics_hints_block_adjacent_news() -> None:
    from prompts.loader import load_prompt

    entertainment = load_prompt("user/category_hints/entertainment")
    politics = load_prompt("user/category_hints/politics")
    generic = load_prompt("user/category_hints/generic")
    assert "settlement chart for the exact period" in entertainment
    assert "leave primary_source_url empty if the chart is missing" in entertainment
    assert "Exact RCP/poll bin markets" in politics
    assert "Trading Economics" in generic
    assert "must not be primary_source_url for KXINXU" in generic
    assert "edge_mechanism=observed_vs_strike even before the official close" in generic
