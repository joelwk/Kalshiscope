import hashlib

from config import SearchConfig
from grok_client import GrokClient, _SYSTEM_PROMPT_ANALYZE, _SYSTEM_PROMPT_DEEP
from models import Market, MarketOutcome, TradeDecision


EXPECTED_SYSTEM_PROMPT_HASHES = {
    "analyze": "59dba101d969a58f6512148fb289c76a74855df71f76c2441f532a2ebee0eff4",
    "deep": "d0550226ab14dcc2ec52fff8733f51df5daa8c207e329b685a4b0e07c532dfc6",
}

EXPECTED_MARKET_PROMPT_HASHES = {
    "commodities:deep_false": "043d8e1ae1e587e97c14ba6e1096787c2356c6f98ba8b7a8a26095672158db3e",
    "commodities:deep_true": "c7cf0607b4a603448b091b79f2e85b078e571c0e7917bc35d66d38dc4f2000a6",
    "crypto:deep_false": "3fa0ca87f71435ca1c8434c085065133909ef5ec7fb8690d02f72805994b57b7",
    "crypto:deep_true": "fd400865be30b4050922e20e382a4e9b8c1672be35963e6f36c96402219db782",
    "generic:deep_false": "e65daff654c96c570d601aec670aa84c1b7875eb3f949c359b924846d255271c",
    "generic:deep_true": "aff7076824e76bf3426a8b98b46ec8ad72134eb161d4ff765587cc4585d4139b",
    "music:deep_false": "c2a5d40116795fd2449215a4f011c29ad2e4711f41807f4b071b0f35c541be98",
    "music:deep_true": "0f19eb9553c4621a053c6e99f1ef52fcaca5592e0a489877ac006e55d1f7ea5a",
    "politics:deep_false": "17324635ace631dc33a98aa60ecfbc2d92ef379b485b176e81f483d883937830",
    "politics:deep_true": "421a4936b9a6e409bdb6bc05d211e5b08c68cc797a2fd32bbca5446f287bf736",
    "speech:deep_false": "f1228e100eb52a41dba37f0546c6ee27653f864d89e4721c719f40e83c202ac6",
    "speech:deep_true": "e6b281d33197c76711bbed8f2b024644b6e7610a33ff4f705393929a7b988649",
    "speech_mention:deep_false": "d9cb905bcba77b47c52f280ed293779b7d433ec2d6d305db0ec52bda45098cb6",
    "speech_mention:deep_true": "d8d261da18ad5a74bdb4d105379ae10b708ea49af1c02818ca07872a2cacc5ac",
    "sports:deep_false": "10acbb06d4b428298db0652257487b8797d18884cdb4cb820426ca4975e9c629",
    "sports:deep_true": "eff61a1498b14eda9cec8c4a9b4ef22f18c8ecfc2d8e9da5423296336791457e",
    "weather:deep_false": "7d90eda56d83089e7885a47fd2e0d3f20b5144c27e728285701893ea8b2f9923",
    "weather:deep_true": "239cb2f91bbf8c553a7f2d120ed0b4854eac7df39005332f7c6b571b4347c6a1",
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
    "evidence_basis": "ff492060d94fea1e70d1623d7bc0d3c4348302c9d19dda58e233aa703ec71682",
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
