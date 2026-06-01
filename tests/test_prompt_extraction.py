import hashlib

from config import SearchConfig
from grok_client import GrokClient, _SYSTEM_PROMPT_ANALYZE, _SYSTEM_PROMPT_DEEP
from models import Market, MarketOutcome, TradeDecision


EXPECTED_SYSTEM_PROMPT_HASHES = {
    # Updated after reinforcing primary_source_url emission for settlement-aligned
    # evidence in system/analyze_market <source_requirements> and the
    # shared_output_rules self-check (post-update review: trades only fire on
    # direct evidence, but settlement-aligned markets were demoted to proxy for a
    # missing URL). shared_output_rules is shared, so the deep hash moves too.
    "analyze": "b83948bd7afc18b8fc1a734abd8312d94188cfcbb5686f234ad357a29ac7eb38",
    "deep": "f248d43a97597d0791b8dc66045285b4b218635c78814707075a2e4e38fc2928",
}

EXPECTED_MARKET_PROMPT_HASHES = {
    "commodities:deep_false": "38564eb54eb00390a2bb02dfa18c8ed60625a088f21d7abcc965630bb0c79c2c",
    "commodities:deep_true": "90eabefb8d2025d4089f347302b56f611632509f6b912ffcf086b1a6e82d2430",
    "crypto:deep_false": "e6733dda89e222611b6cf4d9d3aec418a302529085cba8c068355c82c391b277",
    "crypto:deep_true": "258a262e5ab29618859866fc982e5b3068307a4e8aa9183474535a993518cae7",
    "generic:deep_false": "6cab3e34150bab0aaaf4c9728669c6396115bee98f8585f333540d309ed30eb8",
    "generic:deep_true": "b66efa020191233efaf6c1daaebc17b32cb8db4efeffbb6d5908c0a3ea95a4da",
    "music:deep_false": "968f3ef54858161afcd2655f8dea85473b064c4b4d5f5ef7827bcc7978510fb9",
    "music:deep_true": "d290e7604e2ce963b03e52083a20fb76e3d5406fe8b422d256bdb6fb60a2ad2a",
    "politics:deep_false": "2da431f1b2f58ea7ec490d1b74246e70afdde90665eb9b6af000dc151fcf2e98",
    "politics:deep_true": "8528ae8d3b1103e97800d54225082205c954048c30c693272ea3374997abedf5",
    "speech:deep_false": "7450e4fe419e6e9a4e5e0dadd4ef5ae360d298e00dc5d9eb1b17bcb1500dfe88",
    "speech:deep_true": "bdd23e00194f0ef062c403bd7c2273a7c867f3b24b463158baf30ec2e2ba463a",
    "speech_mention:deep_false": "d9d6c65ce7d55646a15eb1531e0cc3c5b5a8c7a3e2509e05a73229da11edc5bd",
    "speech_mention:deep_true": "fc8ce1fca9f8854d8306d649b2857f4714014b66671ce598ce0986a5fc97de37",
    "sports:deep_false": "fe0dbbd28b8b597cad72470437da79ae7dc19f8f516e998feb7d85d2196c6234",
    "sports:deep_true": "124db8a8dfd3c057d7699b6d2b8477fa7388935f171010ccaff1490f00a93909",
    "weather:deep_false": "bca33a890b34d265e75899905e63dc40d364c8442791b3982070d6ce008a65d2",
    "weather:deep_true": "e6cda107427fcc000b420a4161ca3ebfa7f64aad1f25d9940349d6dee753cbd6",
}

EXPECTED_SCHEMA_DESCRIPTION_HASHES = {
    "abstain": "139411c8d6a39816135c7602c019ef257a8bc4cbe6de148d6accbf5ac05de84b",
    "base_rate_used": "37af25f50be73dbfece673e830b4583f4c3efeaca815fd0fffeeae0b1f68ac5a",
    "bet_size_pct": "641a0cd6d5347f4b1e64447cfe665638765238c6505b67697d4b2e2750d80dc4",
    "cached_tokens": "17c3feb94f70c94185ca5932f3717f8808124cb104f6a59ca433d72b6970a180",
    "code_execution_used": "01fa8db002c20f439e072fdc6edbb98dd81754b999e3cd1ae3e763ddb84956a0",
    "completion_tokens": "787a3235dae3e9a5d93be988eef22495d944c55cda92b20ce5ac21669f97524a",
    "confidence": "3993782437221b5549e1c1d527d246a5a9899919641c5bceb63a90a89d4ea137",
    "definitive_outcome_detected": "d5b6ac13a07ebfbbb084577dbf0de381318ac0ce8020bcdec550804cd5ddc51e",
    "edge_external": "c85f0f881724bd6cb9c2626a4b3497e6540511c9dbbaef1b125b79c484ad16ad",
    "edge_source": "2901c6e3da6168e6b4205bf0660b22475968ecc6b946ebb1a66841b517cd2bce",
    "evidence_basis": "ff492060d94fea1e70d1623d7bc0d3c4348302c9d19dda58e233aa703ec71682",
    "evidence_floor_suppressed_reason": "68e10d20cd67474640a8ee43d46ff0c38013d9eb0de86e1b4deb395cc0aa1b9f",
    "evidence_quality": "476f825d5b7a15e377c7e8f53749beb0a39db6c6f965065a801ece51ff07b0b6",
    "evidence_quality_floor_applied": "6435a1f1d466bcbb12a753b06e319e98d3f3eef7812cf48a155eb5bb3f43427a",
    "implied_prob_external": "db0b9497d1f31ef2601d3130975e4b4ea115c49758eedb266f4986410ef69765",
    "key_sources": "655b9b2751cca8b59fee6971dfcb2d682165d25855a5d41a9fe4684fa9ca6690",
    "likelihood_ratio": "d2f282fd76a34dec00a06dbfafac1f4760b46dbab9020d2f690e61611cb26a75",
    "my_prob": "5abe352ec13d685677213b1120c1914090c2eb9115cd8450156de34b1c8b31d0",
    "outcome": "7642b21e60176506e653b20b3183cc5bc0c361804028e0a8a98a697e8b65f94c",
    "primary_source_url": "6101dd4040210985d00f82c41313d2d69a513361eca215aeab2883f35387af70",
    "probability_yes": "c23e1e0ff03c759f6c24ec0a60249568a7f2d449f3e16c6a0efa03d4b8b72802",
    "prompt_tokens": "49db0a66d30b18b71326920cd8ceea07a4c204889a27ad00bc6a275ca0c97b36",
    "raw_bet_size_pct": "8f72e04cf77115c2893583367d0f8ae98d15f7725ccca7072b3c9fa3308a0cce",
    "raw_confidence": "e44485c16aeb9954bc3709398af748c4126ff9b1eff3c46fce31f81f28ec30d9",
    "raw_evidence_quality": "5a06f668a3cc075d9b7dad2ca7e01cdfbf0786df7cf12c0e7269ee05f2d28b4f",
    "raw_outcome": "907648115ca3f1fb58fa62d08cbce43ea30930d19ce410f3c1e39031aebd0c53",
    "raw_reasoning": "a26f9a108c792b0d03e1ba7626b7a8d5f68f9c1a48fd1875818ae23d4d1eecc3",
    "raw_should_trade": "4e06a691859874c81a7e47e8eb005df24c905457147959d13b924f507df9027e",
    "reasoning": "d354569b3ae0b524436f727bd1d0ed5e3f56e5973a4952eff6d05d341cfaed31",
    "reasoning_tokens": "6b47f605e7844debc22ecbf3b199a6f581dd2a72a1ebdaaca8c06f282f7122b9",
    "self_critique": "ab2d939c15dcfe7ab68bd6af225432bdedfd4596e126f76a74ac6afde5bb7bbe",
    "should_trade": "997fe2b7115c9a5af68197e3bb41b5e2d0e653d5b8ba8533b0e28a1233488e53",
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
    assert "Fallback/no-external-odds trades must clear the configured fallback edge threshold" in _SYSTEM_PROMPT_ANALYZE
