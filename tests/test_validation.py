"""Generation guardrails: a model may only point at candidates we retrieved."""

from __future__ import annotations

from datetime import datetime

from seruen.core.generate import (
    MODE_TEMPLATE,
    SOURCE_IDEA,
    SOURCE_POI,
    generate,
    template_answer,
    validate_picks,
)
from seruen.core.hours import hours_status
from seruen.core.intent import parse_intent_regex
from seruen.core.retrieval import Candidate
from tests.test_retrieval import poi

MONDAY_12 = datetime(2026, 10, 19, 12, 0)


def make_candidate(poi_id: str, name: str, category: str, hours_raw: str | None) -> Candidate:
    record = poi(poi_id, name, category, 51.13, 71.43, hours_raw=hours_raw)
    return Candidate(
        poi=record,
        distance_m=400.0,
        walk_min=7.4,
        hours=hours_status(hours_raw, MONDAY_12),
    )


CANDIDATES = [
    make_candidate("node/1", "Парк", "park", "Mo-Su 08:00-22:00"),
    make_candidate("node/2", "Кафе", "cafe", None),
]
INTENT = parse_intent_regex("хочу погулять 60 минут")


def test_unknown_poi_id_is_dropped():
    picks, dropped = validate_picks(
        [
            {"poi_id": "node/999", "why": "выдуманное место", "activity": "x",
             "visit_minutes": 20, "confidence": "high", "source": SOURCE_POI},
            {"poi_id": "node/1", "why": "рядом и открыто", "activity": "пройтись",
             "visit_minutes": 20, "confidence": "high", "source": SOURCE_POI},
        ],
        CANDIDATES,
        INTENT,
    )
    assert [p.poi_id for p in picks] == ["node/1"]
    assert dropped == 1


def test_duplicate_ids_are_dropped():
    raw = {"poi_id": "node/1", "why": "ок", "activity": "a", "visit_minutes": 20,
           "confidence": "medium", "source": SOURCE_POI}
    picks, dropped = validate_picks([raw, dict(raw)], CANDIDATES, INTENT)
    assert len(picks) == 1
    assert dropped == 1


def test_all_invalid_ids_leaves_nothing():
    picks, dropped = validate_picks(
        [{"poi_id": "way/404", "why": "нет такого", "activity": "a", "visit_minutes": 10,
          "confidence": "high", "source": SOURCE_POI}],
        CANDIDATES,
        INTENT,
    )
    assert picks == []
    assert dropped == 1


def test_confidence_cannot_exceed_data_quality():
    # node/2 has no opening hours, so "high" must be downgraded
    picks, _ = validate_picks(
        [{"poi_id": "node/2", "why": "тихое место", "activity": "кофе", "visit_minutes": 25,
          "confidence": "high", "source": SOURCE_POI}],
        CANDIDATES,
        INTENT,
    )
    assert picks[0].confidence == "medium"
    assert picks[0].hours["status"] == "unknown"


def test_bogus_enum_values_are_repaired():
    picks, _ = validate_picks(
        [{"poi_id": "node/1", "why": "ок", "activity": "a", "visit_minutes": 9999,
          "confidence": "очень высокая", "source": "выдумка"}],
        CANDIDATES,
        INTENT,
    )
    assert picks[0].confidence == "medium"
    assert picks[0].source == SOURCE_POI  # displayed rationale is always assembled locally
    assert picks[0].visit_minutes <= INTENT.time_budget_min


def test_pick_without_why_is_dropped():
    picks, dropped = validate_picks(
        [{"poi_id": "node/1", "why": "   ", "activity": "a", "visit_minutes": 20,
          "confidence": "high", "source": SOURCE_POI}],
        CANDIDATES,
        INTENT,
    )
    assert picks == []
    assert dropped == 1


def test_no_more_than_three_picks():
    candidates = [make_candidate(f"node/{i}", f"Место {i}", "park", "24/7") for i in range(6)]
    raw = [
        {"poi_id": f"node/{i}", "why": "ок", "activity": "a", "visit_minutes": 15,
         "confidence": "high", "source": SOURCE_POI}
        for i in range(6)
    ]
    picks, _ = validate_picks(raw, candidates, INTENT)
    assert len(picks) == 3


def test_template_answer_is_labelled_and_sourced():
    answer = template_answer(INTENT, CANDIDATES, radius_m=800)
    assert answer.mode == MODE_TEMPLATE
    assert 1 <= len(answer.picks) <= 3
    for pick in answer.picks:
        assert pick.source == SOURCE_POI
        assert pick.activity_source == SOURCE_IDEA
        assert pick.why and pick.activity
        assert pick.hours["status"] in {"open", "unknown"}


def test_demo_mode_never_calls_the_model():
    answer = generate(INTENT, CANDIDATES, MONDAY_12, radius_m=800, demo=True)
    assert answer.mode == MODE_TEMPLATE
    assert answer.model is None
    assert answer.fallback_reason == "demo-режим"


def test_no_candidates_gives_an_honest_empty_answer():
    answer = generate(INTENT, [], MONDAY_12, radius_m=800, demo=True)
    assert answer.picks == []
    assert "не нашлось" in answer.note


def test_arbitrary_provider_claims_never_become_displayed_facts():
    picks, _ = validate_picks([
        {"poi_id": "node/2", "why": "OPEN NOW and FREE with 20 power sockets",
         "activity": "The pool and rooftop are open all night", "visit_minutes": 25,
         "confidence": "high", "source": SOURCE_POI},
    ], CANDIDATES, INTENT)
    assert len(picks) == 1
    assert "OPEN NOW" not in picks[0].why
    assert "20 power sockets" not in picks[0].why
    assert "pool" not in picks[0].activity
    assert "часы работы не подтверждены" in picks[0].why
    assert picks[0].source_url == "https://example.invalid"
    assert picks[0].source_license == "ODbL"


def test_visit_time_includes_return_walk_and_buffer():
    intent = parse_intent_regex("парк на 45 минут и обратно")
    picks, _ = validate_picks([
        {"poi_id": "node/1", "why": "valid", "visit_minutes": 44},
    ], CANDIDATES, intent)
    assert picks[0].visit_minutes + 2 * 8 + 5 <= 45


def test_displayed_walking_time_uses_same_ceiling_as_budget():
    answer = template_answer(INTENT, CANDIDATES, radius_m=800)
    assert answer.picks[0].walk_min == 8
    assert answer.picks[0].why.startswith("8 мин пешком")


def test_work_template_keeps_time_for_work_instead_of_a_drink_break():
    intent = parse_intent_regex("ищу кафе поработать за ноутбуком 60 минут")
    answer = template_answer(intent, [CANDIDATES[1]], radius_m=800)
    pick = answer.picks[0]
    assert "одну задачу" in pick.activity
    assert pick.visit_minutes == 47
    assert pick.walk_min + pick.visit_minutes + 5 == 60


def test_model_cannot_reduce_a_work_session_to_a_five_minute_stop():
    intent = parse_intent_regex("кафе поработать за ноутбуком 60 минут")
    picks, _ = validate_picks([{"poi_id": "node/2", "why": "valid", "visit_minutes": 5}], CANDIDATES, intent)
    assert picks[0].visit_minutes == 47


def test_malformed_picks_and_fields_never_crash():
    for payload in [None, "picks", 1, {}]:
        assert validate_picks(payload, CANDIDATES, INTENT) == ([], 1)
    picks, dropped = validate_picks([
        {"poi_id": "node/1", "why": ["bad type"]},
        {"poi_id": "node/1", "why": "ok", "confidence": [], "source": {}, "visit_minutes": True},
    ], CANDIDATES, INTENT)
    assert len(picks) == 1 and dropped == 1
    assert picks[0].visit_minutes > 5


def test_low_confidence_cannot_be_upgraded():
    raw = {"poi_id": "node/1", "why": "ok", "confidence": "low"}
    picks, _ = validate_picks([raw], CANDIDATES, INTENT)
    assert picks[0].confidence == "low"


def test_english_templates_do_not_leak_russian_description():
    intent = parse_intent_regex("A cafe for one hour")
    answer = template_answer(intent, CANDIDATES, 800)
    assert answer.language == "en"
    assert "Астане" not in answer.picks[0].why
    assert "розетки" not in " ".join(answer.picks[0].uncertainties)


def test_model_is_required_and_not_invented(monkeypatch):
    from seruen.core.generate import generate_with_gemini

    monkeypatch.setenv("GOOGLE_API_KEY", "test-placeholder")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    answer, reason = generate_with_gemini(INTENT, CANDIDATES, MONDAY_12)
    assert answer is None
    assert "GEMINI_MODEL" in reason


def test_provider_error_falls_back_without_leaking_raw_message(monkeypatch):
    from types import SimpleNamespace

    from google import genai

    def fail(**kwargs):
        raise RuntimeError("SECRET_TOKEN precise coordinates 51.123456 request body")

    monkeypatch.setenv("GOOGLE_API_KEY", "test-placeholder")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(models=SimpleNamespace(generate_content=fail)))
    answer = generate(INTENT, CANDIDATES, MONDAY_12, radius_m=800, demo=False)
    assert answer.mode == MODE_TEMPLATE
    assert answer.candidates_sent_to_model == len(CANDIDATES)
    assert "SECRET_TOKEN" not in str(answer.as_dict())
    assert "51.123456" not in str(answer.as_dict())
    assert "RuntimeError" in answer.fallback_reason


def test_provider_invalid_payload_falls_back_and_request_redacts_coordinates(monkeypatch):
    from types import SimpleNamespace

    from google import genai

    seen = {}

    def respond(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(text='["not an object"]')

    monkeypatch.setenv("GOOGLE_API_KEY", "test-placeholder")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(models=SimpleNamespace(generate_content=respond)))
    intent = parse_intent_regex("Я у 51.123456, 71.456789, хочу парк на час")
    answer = generate(intent, CANDIDATES, MONDAY_12, radius_m=800, demo=False)
    assert answer.mode == MODE_TEMPLATE
    assert answer.candidates_sent_to_model == len(CANDIDATES)
    assert "51.123456" not in seen["contents"] and "71.456789" not in seen["contents"]
    assert "max_visit_minutes" in seen["contents"]


def test_safe_creative_activity_is_used_and_labelled_as_an_idea():
    raw = {
        "poi_id": "node/1", "why": "arbitrary provider explanation is ignored",
        "activity": "Найдите интересную деталь и нарисуйте её по памяти.",
        "visit_minutes": 20, "confidence": "medium", "source": SOURCE_POI,
    }
    picks, _ = validate_picks([raw], CANDIDATES, INTENT)
    assert picks[0].activity.startswith("Если условия на месте позволяют: ")
    assert "нарисуйте" in picks[0].activity
    assert picks[0].activity_source == SOURCE_IDEA
    assert "arbitrary provider explanation" not in picks[0].why


def test_unsupported_model_activity_uses_grounded_fallback():
    from seruen.core.generate import _safe_generated_activity

    for activity in [
        "Проверьте бассейн на крыше и получите бесплатный билет.",
        "Look at the rooftop pool, which is open all night.",
        "Найдите кафе с гарантированной тишиной и розетками.",
        "Найдите пять видов напитков за 500 тенге.",
        "Қауіпсіз және тегін орынға барыңыз.",
    ]:
        assert _safe_generated_activity(activity, "ru") is None
    assert _safe_generated_activity("Compare shapes and sketch an interesting detail.", "en")
    assert _safe_generated_activity("Қызықты бөлшекті тауып, суретке түсіріңіз.", "kk")


def test_successful_provider_selection_has_grounded_reason_and_generated_action(monkeypatch):
    import json
    from types import SimpleNamespace

    from google import genai

    captured = {}
    payload = {"picks": [{
        "poi_id": "node/1", "why": "UNSUPPORTED FACT",
        "activity": "Найдите интересную деталь и нарисуйте её по памяти.",
        "visit_minutes": 20, "confidence": "medium", "source": SOURCE_POI,
    }], "note": "UNSUPPORTED NOTE"}

    def client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(text=json.dumps(payload)),
        ))

    monkeypatch.setenv("GOOGLE_API_KEY", "test-placeholder")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setattr(genai, "Client", client)
    answer = generate(INTENT, CANDIDATES, MONDAY_12, demo=False)
    assert answer.mode == "gemini" and answer.candidates_sent_to_model == 2
    assert captured["http_options"].timeout == 15000
    assert "UNSUPPORTED" not in str(answer.as_dict())
    assert "нарисуйте" in answer.picks[0].activity
