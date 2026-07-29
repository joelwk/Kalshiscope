import hashlib

from config import SearchConfig
from grok_client import GrokClient, _SYSTEM_PROMPT_ANALYZE, _SYSTEM_PROMPT_DEEP
from models import Market, MarketOutcome, TradeDecision


EXPECTED_SYSTEM_PROMPT_HASHES = {
    # Re-pinned Jul 28 2026: https URL hygiene examples, side-consistency rule,
    # and crypto live-quote few-shot (execution-yield alignment).
    "analyze": "7db641bc7aee12b95d30aaee4de88c20249a8fe56658d73dcf637af06e49ad2a",
    "deep": "ea0c9f26bb4f9b1dcb492186239980483037c0ba60c05308aecde420bd2b6f1c",
}

EXPECTED_MARKET_PROMPT_HASHES = {
    # Re-pinned Jul 29 2026: deep constraints repair-before-abstain lines.
    "commodities:deep_false": "5ace0c9e5c8448d4ad6caad4a3e64d30b178078b63237bdc4a6b0aa1cb84650a",
    "commodities:deep_true": "619ef9ec2575c891090092db00fdce3bf1e225e6e746bd187bbefebbbf1e789b",
    "crypto:deep_false": "296581f0b94cf1a90e39fc0e6b981cf96283143a0f7fbb64aee5f8c5f895f553",
    "crypto:deep_true": "7de5dbdef1c2fd6c1cc7db844aec15a71313bdeba6a2c3dc3bd12a5fecc950f4",
    "generic:deep_false": "de34974965eea79e2a926963f5a7b444f12ab7f8621a5120d1879f26481c258d",
    "generic:deep_true": "b6e5d0d3186eb8b32babf3fe3a943d78974eeacf61c983dd37507b82e8736dfc",
    "music:deep_false": "a0d39c5d325a41abdd64dd72aa6662e7283fbc1c70b1272a10fe0c6b72a52b59",
    "music:deep_true": "f74dce78195e106bf56220b4c29de25871b10b19721048d07187b2109a38c5ce",
    "politics:deep_false": "877eb11281828ff2aade2c1253170f423539068a0c1f609bdb9ac03931cb2f47",
    "politics:deep_true": "cd980413d86def36d37cf4f173e0ec75af9f9a64dd8b493475be3033d4254713",
    "speech:deep_false": "a4dfa6c4e3d77f919a3ca7b9f454249f629e2b0e5df5cad8ad4e0783ba046473",
    "speech:deep_true": "f904c60a2c1248799fe5fdc559e528041302b5f7433919be03ea58cca233ccdd",
    "speech_mention:deep_false": "f15420e53452fbb09616a8bea172b6181d90e2f84584fbcbfa4c09b80860d9d2",
    "speech_mention:deep_true": "ea76dc238b78eaec219b4c681c6ffb025ab61e19f9b95b87404afad671774ac2",
    "sports:deep_false": "6d6783072a778b067bf364ed473c16226e0c4438fe891440ca67c8f56b5a42f9",
    "sports:deep_true": "17b6e5c35259d837df3217c14bf662d6dfa05e9162735a4a79d736e765626476",
    "weather:deep_false": "eeacbb3c933b623e93880a80f8bd27003800ec17393f78547b2357805b9e4db2",
    "weather:deep_true": "e9a577927c76d9bc0bc5262b441b09b20c9a03525cbd54e3093f57a62b20ebf0",
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
    "edge_external": "af45ae57027c853dd040990e6be9dfe0b045772e14690dfc378f91ff56380e2a",
    "edge_source": "2901c6e3da6168e6b4205bf0660b22475968ecc6b946ebb1a66841b517cd2bce",
    "evidence_basis": "fd09eed4586de60e1a994291d7b04a2de5603dc17a2b94dce732066888d389da",
    "evidence_floor_suppressed_reason": "68e10d20cd67474640a8ee43d46ff0c38013d9eb0de86e1b4deb395cc0aa1b9f",
    "evidence_quality": "476f825d5b7a15e377c7e8f53749beb0a39db6c6f965065a801ece51ff07b0b6",
    "evidence_quality_floor_applied": "6435a1f1d466bcbb12a753b06e319e98d3f3eef7812cf48a155eb5bb3f43427a",
    "implied_prob_external": "76bcee210033e309fd75807a13b8ec2cee01051ee4687b4d8dc70a433bf90848",
    "key_sources": "3aa0cc5c44f0f8a2ee451eeb25844c11c7d34426c2a2627651c814e5a4f38747",
    "likelihood_ratio": "d2f282fd76a34dec00a06dbfafac1f4760b46dbab9020d2f690e61611cb26a75",
    "my_prob": "752cdbc9aba6b7fdc735e8f3c81944467b6598eba3d139041430fd03afdec0c6",
    "outcome": "67cf74e5d00e1ebece3635535fb09d9dc67b9a4a997a3246761c44ccd6198399",
    "primary_source_url": "bdd16205b16f6b72d3c9228c2f90468953d813dca5f11c657f2000539c9c1fea",
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
    "should_trade": "a3af674bf96502c698bea6b9e0d42968e98a39b106920cac9e0751f92ee67bef",
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


def test_deep_constraints_prefer_repair_before_abstain() -> None:
    from prompts.loader import load_lines

    lines = load_lines("user/fragments/constraints_deep")
    joined = "\n".join(lines)
    assert "Repair before abstain" in joined
    assert "paste the real https primary_source_url" in joined
    assert "Only abstain after repair attempts" in joined
