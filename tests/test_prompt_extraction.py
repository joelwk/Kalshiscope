import hashlib

from config import SearchConfig
from grok_client import GrokClient, _SYSTEM_PROMPT_ANALYZE, _SYSTEM_PROMPT_DEEP
from models import Market, MarketOutcome, TradeDecision


EXPECTED_SYSTEM_PROMPT_HASHES = {
    "analyze": "d9a12956584e4c7e83de7037ecf6dca92ed620244e7ba4f58c3bb0f6d68f5789",
    "deep": "ea93885d617ad6ed5500a18c9cfd17fc97b1a22977348751c8bcf472e2487135",
}

EXPECTED_MARKET_PROMPT_HASHES = {
    "commodities:deep_false": "a8fd225992256dc29604638e6ebd5bf2cfc0b06419beed7087ee7b805ced0f85",
    "commodities:deep_true": "200b0832746f5e531964e6030fa9aa0308b466fc3569b31671673205abea93da",
    "crypto:deep_false": "1e5d175489a6be68d424a430896f26609c2e9cf068dfd36a4b3bd26781362784",
    "crypto:deep_true": "124ba6b02822bfd66f1065c8042f38d42a2f94ec6978d068440998eb66dbe112",
    "generic:deep_false": "2e57e0e6c57728e0ea014ab4ad99e5839fa6a81015ddab37157b406a12b9aa6a",
    "generic:deep_true": "c710a8cb6cc3dcb9c938e2972658daa7d1319acf9231f94f8970e5f23b75d683",
    "music:deep_false": "a772497f41f0fa13dc567bf5da1a026dd65875c7a5c9547a34c663df245caf6f",
    "music:deep_true": "593d477927c98c26746b3e52cade149f4dd3fb8754cc3fcc26c3dce6831d6b06",
    "politics:deep_false": "e2d10f171523cbff342a380dd2482e771ac8c647c7aa3f8ef4d15e199b7573ec",
    "politics:deep_true": "89c5c681aaa343e0312f5f0a60a96a85f13f9d37caaa93b6b47b3dd62b581729",
    "speech:deep_false": "419b9777cd29ec1754c4ba3c2bf18c25f6b1c3870f35738aac5e9ec42820ebef",
    "speech:deep_true": "6b9fbb3fcbb0d31c0c424eb7eb2f86e41ae9551e2b48db0492939d86a4c49689",
    "speech_mention:deep_false": "b799950fdd3b703450ee198c61e60b6e0b4a78c9405f270e5aa6d9250f0c409b",
    "speech_mention:deep_true": "fc89fdca91236d909b61e561caac2f6ccc951f3f225add969b0483048e658687",
    "sports:deep_false": "5f46ae12369f121b2489b62bb8f5509153d68b170bc5be32fb14839e7e177562",
    "sports:deep_true": "a298a42c3166c862a719e3b9c8b90539f689bf388bfcce9140c252d8669bbb99",
    "weather:deep_false": "fbee20ac2eb0394bb7a078cdfa897cbdcd75de96907b029cfa09c3135ccda170",
    "weather:deep_true": "fca3a2ae7cd64a1eb7b6c802f3714c98795f263ee03c1c72a4a49e87c3b7ec6d",
}

EXPECTED_SCHEMA_DESCRIPTION_HASHES = {
    "abstain": "139411c8d6a39816135c7602c019ef257a8bc4cbe6de148d6accbf5ac05de84b",
    "bet_size_pct": "641a0cd6d5347f4b1e64447cfe665638765238c6505b67697d4b2e2750d80dc4",
    "cached_tokens": "17c3feb94f70c94185ca5932f3717f8808124cb104f6a59ca433d72b6970a180",
    "completion_tokens": "787a3235dae3e9a5d93be988eef22495d944c55cda92b20ce5ac21669f97524a",
    "confidence": "2e0ae289279f3f159b702a396d633e4310b0a4b10107a6f620d601803cdf8815",
    "definitive_outcome_detected": "d5b6ac13a07ebfbbb084577dbf0de381318ac0ce8020bcdec550804cd5ddc51e",
    "edge_external": "c85f0f881724bd6cb9c2626a4b3497e6540511c9dbbaef1b125b79c484ad16ad",
    "edge_source": "e937b173f62fa5690733910a3bd0f27bd3c727cd8f9f6d43220a8d688cece695",
    "evidence_basis": "b8b5a7ae4d8ecc391e4e0a72cf60be664ae4fd825337504e1273d39a5f0e3f31",
    "evidence_quality": "16da4528a2879172de8b4370c697655df6fe0dfff3fd99bb339cb1585137bc2d",
    "evidence_quality_floor_applied": "6435a1f1d466bcbb12a753b06e319e98d3f3eef7812cf48a155eb5bb3f43427a",
    "implied_prob_external": "db0b9497d1f31ef2601d3130975e4b4ea115c49758eedb266f4986410ef69765",
    "likelihood_ratio": "837b46f6e4d7d577f488326851fa2b036d12f4ef27e0628fa9230c1cc2b05b65",
    "my_prob": "5abe352ec13d685677213b1120c1914090c2eb9115cd8450156de34b1c8b31d0",
    "outcome": "7642b21e60176506e653b20b3183cc5bc0c361804028e0a8a98a697e8b65f94c",
    "prompt_tokens": "49db0a66d30b18b71326920cd8ceea07a4c204889a27ad00bc6a275ca0c97b36",
    "raw_bet_size_pct": "8f72e04cf77115c2893583367d0f8ae98d15f7725ccca7072b3c9fa3308a0cce",
    "raw_confidence": "e44485c16aeb9954bc3709398af748c4126ff9b1eff3c46fce31f81f28ec30d9",
    "raw_evidence_quality": "5a06f668a3cc075d9b7dad2ca7e01cdfbf0786df7cf12c0e7269ee05f2d28b4f",
    "raw_outcome": "907648115ca3f1fb58fa62d08cbce43ea30930d19ce410f3c1e39031aebd0c53",
    "raw_reasoning": "a26f9a108c792b0d03e1ba7626b7a8d5f68f9c1a48fd1875818ae23d4d1eecc3",
    "raw_should_trade": "4e06a691859874c81a7e47e8eb005df24c905457147959d13b924f507df9027e",
    "reasoning": "61f5445ef509335cc97c1a4f03d2705e94482c04acd9a536558a32fb4762c6da",
    "reasoning_tokens": "6b47f605e7844debc22ecbf3b199a6f581dd2a72a1ebdaaca8c06f282f7122b9",
    "should_trade": "a032e9ccd47335be8454e456b10c5c1bae121e2a5c6d5cbecf94d6f900c1f519",
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
