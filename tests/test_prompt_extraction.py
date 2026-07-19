import hashlib

from config import SearchConfig
from grok_client import GrokClient, _SYSTEM_PROMPT_ANALYZE, _SYSTEM_PROMPT_DEEP
from models import Market, MarketOutcome, TradeDecision


EXPECTED_SYSTEM_PROMPT_HASHES = {
    # Updated with the source-repair constraints that distinguish true
    # absence-only gaps from exceptional, settlement-predictive proxy evidence.
    "analyze": "8d3f47805aef707fe86cc1eb409b3510ff746ade87487554e4627c5cbe3491ca",
    "deep": "62a1686290f1ec6907c9e976b8d964ea57a3b1c3dfce48ab06d7a5f7ffcc2fdc",
}

EXPECTED_MARKET_PROMPT_HASHES = {
    "commodities:deep_false": "e253db2cd7e7015cbe9c08e204b1ac99e9fdaa93f6b2fe80843d01d872204354",
    "commodities:deep_true": "578c225154f2805017a73b09417ec91daa52df59af68dd51bb68cf6ef34fd94a",
    # Updated after strengthening the crypto category hint to mandate citing a
    # settlement-grade exchange primary_source_url (coinbase/kraken/binance/coindesk)
    # with observed price + timestamp, with a proxy/EQ<=0.60 fallback otherwise.
    "crypto:deep_false": "a024c18c866151bb7210534cb9abf6c9d3dbf15edd2829e0df8a08554ffb7e28",
    "crypto:deep_true": "477cb80dabfa6ea65efd2d7ed9353f9d41b75ea928aa76df2110293c9037d5b2",
    # Updated after generic Kelly-sized primary_source_url guidance + deep repair action.
    "generic:deep_false": "b0669c2a1b046ca2e1ee4c2c83afeb7e06258e1cee74119a60e98bc231cc3a23",
    "generic:deep_true": "4abf11e2e7aeddbb2ca02dabeecf218fe7356b04bd65371c957a406ec7f34a91",
    "music:deep_false": "bdf644f27fa99a37e5a0fa3d45892b3d77abbc860645f9e7e4c7ddc8373da467",
    "music:deep_true": "60022b4465445c5fb0d52a6cc570c0e67ce03b906eb9187c1f0b998ab0bb3710",
    "politics:deep_false": "ef404dda88e14d32ee5f777c62afef8f8939d18bd35debb8e11346a1627a2d11",
    "politics:deep_true": "c0989422b26e0331b0aa3322cfec832add49aa955b93204d2f3a2fd2c4882564",
    "speech:deep_false": "05081d0e0ee47d01c1ec63a3b3611e5c170f13f36c400e0959167fe943f67cb2",
    "speech:deep_true": "8aea5f22631375e79c6aa83d97016636bd4bcbe659fc0d4c7e15f1ca6d3d4a53",
    "speech_mention:deep_false": "28a8d412a8066613a8148cd5fd220706c82c1d907cacdc163b1b935f2de76030",
    "speech_mention:deep_true": "957a1ee66306d3412149be3dcc2750c599098821e0d4cb90aef4de0b912ee27e",
    # Updated after enriching the sports category hint to mirror weather/commodity:
    # verify game status first, cite an official box-score/result primary_source_url
    # for settled props (direct evidence), ground pre-game edges in a cited sportsbook
    # line via implied_prob_external/edge_external (edge_source=computed not none), and
    # abstain at evidence_quality<=0.55 when neither odds nor result is citable.
    "sports:deep_false": "32e5449e854a7ec4273430d40af3b74157c643d422c1b7667675e6dc546aa035",
    "sports:deep_true": "f5fb2ef308074d2649c2d2917ade76247626fd44509561677d526e61fce36896",
    "weather:deep_false": "aad1b1901da03a6746022e9e3c9b802b82f2aa07dc244d3528fdf7a8dc8b5fc2",
    "weather:deep_true": "a770501aab67b7364e2b46c81919c137e729d381fa3092568001227759e10000",
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
    "key_sources": "655b9b2751cca8b59fee6971dfcb2d682165d25855a5d41a9fe4684fa9ca6690",
    "likelihood_ratio": "d2f282fd76a34dec00a06dbfafac1f4760b46dbab9020d2f690e61611cb26a75",
    "my_prob": "752cdbc9aba6b7fdc735e8f3c81944467b6598eba3d139041430fd03afdec0c6",
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
