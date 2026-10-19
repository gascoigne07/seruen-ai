"""Outdoor presentation must not invent hours or bypass known closing times."""

from datetime import datetime

import pytest

from seruen.core.generate import template_answer, validate_picks
from seruen.core.hours import UNKNOWN, hours_context, hours_status
from seruen.core.intent import parse_intent_regex
from seruen.core.retrieval import Candidate, load_pois, retrieve
from tests.test_retrieval import NEAR, ORIGIN, poi

NIGHT = datetime(2026, 10, 19, 23, 30)


@pytest.mark.parametrize("subtype", ["park", "garden", "playground", "memorial"])
def test_missing_outdoor_schedule_does_not_imply_closed_or_24_7(subtype):
    record = poi("node/1", "Прогулка", "park", *NEAR, indoor=False)
    record["osm_subtype"] = subtype
    intent = parse_intent_regex("погулять ночью час")
    result = retrieve(intent, *ORIGIN, NIGHT, pois=(record,))
    candidate = result.candidates[0]
    pick = template_answer(intent, result.candidates, result.radius_m).picks[0]
    assert candidate.card()["hours_context"] == "outdoor"
    assert candidate.hours.status == UNKNOWN
    assert candidate.hours.raw is None
    assert pick.hours["context"] == "outdoor"
    assert "на открытом воздухе" in pick.why
    assert "часы работы не подтверждены" not in pick.why
    assert "часы работы не подтверждены" not in pick.uncertainties
    assert "24/7" not in pick.why and "круглосуточно" not in pick.why


@pytest.mark.parametrize("subtype,indoor", [
    ("park", True), ("park", None), ("memorial", True),
    ("monument", False), ("artwork", False), ("viewpoint", False),
    ("museum", False), ("cafe", False),
])
def test_unknown_buildings_and_businesses_keep_schedule_warning(subtype, indoor):
    record = poi("node/2", "Объект", "landmark", *NEAR, indoor=indoor)
    record["osm_subtype"] = subtype
    assert hours_context(record, hours_status(None, NIGHT)) == "schedule"


@pytest.mark.parametrize("raw,expected", [
    ("24/7", "open"), ("Mo-Su 09:00-22:00", "closed"),
    ("sunrise-sunset", "unknown"), ("Mo-Su 09:00-22:00; PH off", "unknown"),
])
def test_explicit_outdoor_hours_keep_their_status_and_detail(raw, expected):
    record = poi("node/3", "Парк", "park", *NEAR, indoor=False, hours_raw=raw)
    status = hours_status(raw, NIGHT)
    assert status.status == expected
    assert hours_context(record, status) == "schedule"
    result = retrieve(parse_intent_regex("парк час"), *ORIGIN, NIGHT, pois=(record,))
    if expected == "closed":
        assert not result.candidates
        assert result.stats["excluded_closed"] == 1
    else:
        candidate = result.candidates[0]
        assert candidate.hours.raw == raw
        assert candidate.card()["hours_context"] == "schedule"
        assert ("часы работы не подтверждены" in candidate.uncertainties()) == (expected == "unknown")


@pytest.mark.parametrize("language,phrase", [("ru", "на открытом воздухе"), ("en", "outdoors"), ("kk", "ашық ауада")])
def test_real_memorial_and_ai_answer_use_the_same_access_policy(language, phrase):
    record = next(p for p in load_pois() if p["id"] == "node/5878503586")
    candidate = Candidate(record, 548, 10.2, hours_status(record["hours_raw"], NIGHT))
    intent = parse_intent_regex("памятник час")
    intent.language = language
    picks, dropped = validate_picks([{
        "poi_id": candidate.id, "why": "Круглосуточно, вход всегда свободный",
        "activity": "Место открыто круглосуточно", "visit_minutes": 20, "confidence": "high",
    }], [candidate], intent)
    assert dropped == 0
    assert picks[0].hours["context"] == "outdoor"
    assert picks[0].hours["status"] == "unknown"
    assert picks[0].confidence == "medium"
    assert phrase in picks[0].why
    assert "круглосуточно" not in picks[0].why + picks[0].activity
