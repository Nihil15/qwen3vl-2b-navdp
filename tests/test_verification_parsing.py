"""The verifier's reply parser.

Every case here was observed from Qwen3-VL-2B in a real run. The truncated-
fence case is not hypothetical: it silently discarded the best-matching table
(0.97 m from ground truth) while a 4.47 m distractor was accepted, because a
strict raw_decode threw and the candidate was recorded as "uncertain".
"""

from nav_pipeline.fact3r_live_memory import parse_verification


def test_plain_json():
    out = parse_verification(
        '{"decision":"yes","confidence":0.95,"predicted_object":"a table",'
        '"reason":"wooden surface with legs"}')
    assert out["decision"] == "yes"
    assert out["confidence"] == 0.95
    assert out["predicted_object"] == "a table"


def test_fenced_json():
    out = parse_verification(
        '```json\n{"decision":"no","confidence":0.9,"predicted_object":"lampshade",'
        '"reason":"cylindrical layered object"}\n```')
    assert out["decision"] == "no"
    assert out["predicted_object"] == "lampshade"


def test_truncated_fenced_json_recovers_the_verdict():
    """Cut off mid-`reason` by the token limit -- the verdict was already
    stated and must survive."""
    out = parse_verification(
        '```json\n{"decision":"yes","confidence":0.95,"predicted_object":"a table",'
        '"reason":"The provided images show a long, wooden table with drawers, which is')
    assert out["decision"] == "yes"
    assert out["confidence"] == 0.95
    assert out["predicted_object"] == "a table"
    assert "truncated" in out["reason"]


def test_truncated_before_any_verdict_is_uncertain():
    out = parse_verification("```json\n{")
    assert out["decision"] == "uncertain"
    assert out["confidence"] == 0.0


def test_prose_reply_is_uncertain_not_a_crash():
    out = parse_verification("I think this is probably a table, but I am not sure.")
    assert out["decision"] == "uncertain"
    assert out["confidence"] == 0.0


def test_empty_reply():
    out = parse_verification("")
    assert out["decision"] == "uncertain"


def test_confidence_absent_defaults_to_zero_not_crash():
    out = parse_verification('{"decision":"yes","predicted_object":"table"}')
    assert out["decision"] == "yes"
    assert out["confidence"] == 0.0
