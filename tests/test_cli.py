from datetime import datetime, timezone

import pytest

from seruen.cli import PRESETS, advise, format_advice, resolve_origin


def test_text_location_detected_with_explicit_override_priority():
    assert resolve_origin("Я у Mega Silk Way")[1:] == PRESETS["mega"][1:]
    assert resolve_origin("I am at the airport")[1:] == PRESETS["airport"][1:]
    assert resolve_origin("Я у Mega", preset="khan")[1:] == PRESETS["khan"][1:]


@pytest.mark.parametrize("query,preset", [
    ("Я у Байтерека, до самолёта 2 часа", "baiterek"),
    ("Я у Хан Шатыра, до рейса 2 часа", "khan"),
    ("Я у Байтерека, хочу дойти до Mega Silk Way за 90 минут", "baiterek"),
    ("Хочу в Mega, сейчас у Байтерека", "baiterek"),
    ("Байтерек жанындамын, Мегаға барғым келеді", "baiterek"),
])
def test_starting_location_wins_over_destination_or_flight_context(query, preset):
    assert resolve_origin(query)[1:] == PRESETS[preset][1:]


@pytest.mark.parametrize("coords", [(51.1, None), (float("nan"), 71.4), (91, 71.4)])
def test_bad_coordinates_rejected(coords):
    with pytest.raises(ValueError):
        resolve_origin("query", lat=coords[0], lon=coords[1])


def test_astana_timezone_and_real_model_count():
    a = advise("прогулка 45 минут", *PRESETS["mega"][1:], demo=True, use_weather=False,
               when=datetime(2026, 10, 19, 7, tzinfo=timezone.utc), max_picks=1)
    assert a.when.hour == 12
    assert len(a.answer.picks) <= 1
    assert a.data_used()["candidates_shown_to_model"] == 0
    assert a.data_used()["timezone"] == "Asia/Almaty"


def test_empty_answer_still_includes_evidence_panel():
    a = advise("кафе 10 минут", 0, 0, demo=True, use_weather=False)
    assert not a.answer.picks
    assert "Какие данные использованы" in format_advice(a)


def test_night_request_uses_night_time_when_no_clock_given():
    a = advise("необычное ночью 90 минут", *PRESETS["baiterek"][1:], demo=True, use_weather=False)
    assert (a.when.hour, a.when.minute) == (23, 30)


def test_places_to_work_are_alternatives_not_cafe_hopping():
    a = advise("ищу кафе поработать за ноутбуком 60 минут", *PRESETS["baiterek"][1:],
               demo=True, use_weather=False, when=datetime(2026, 10, 19, 12))
    assert len(a.answer.picks) >= 2
    assert a.intent.wants_work
    assert a.route is None


def test_absent_origin_never_silently_defaults_to_baiterek():
    with pytest.raises(ValueError, match="где вы находитесь"):
        resolve_origin("куда могу пойти")
    with pytest.raises(ValueError, match="где вы находитесь"):
        resolve_origin("до рейса два часа")


def test_verified_origin_context_does_not_become_desired_category():
    query = "я вот в центральном парке. куда могу отсюда пойти 45 минут"
    metadata = {
        "status": "resolved", "place_query": "Центральный парк", "ai_used": True,
        "place": {"source": "visitastana", "source_url": "https://visitastana.kz/",
                  "lat": 51.152859, "lon": 71.41902},
    }
    a = advise(query, 51.152859, 71.41902, demo=True, use_weather=False,
               origin_label="Центральный парк", origin_resolution=metadata,
               intent_query="куда могу отсюда пойти", when=datetime(2026, 10, 19, 12))
    assert a.intent.categories_wanted == []
    assert a.intent.time_budget_min == 45
    assert a.intent.raw_query == query
    used = a.data_used()
    assert used["origin_resolution"]["ai_used"] is True
    assert used["origin_source"] == "visitastana"
    assert "51.152859" not in str(used)


def test_cli_missing_origin_returns_clarification_json_without_generation(monkeypatch, capsys):
    import json

    from seruen import cli

    monkeypatch.setenv("SERUEN_DEMO", "1")
    monkeypatch.setattr(cli, "advise", lambda *a, **kw: pytest.fail("must not recommend before origin"))
    assert cli.main(["куда пойти", "--demo", "--no-weather", "--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["origin_resolution"]["status"] == "missing"


def test_cli_provider_override_uses_its_own_model(monkeypatch, capsys):
    from seruen import cli

    monkeypatch.setenv("SERUEN_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-owner-key")
    monkeypatch.setenv("OPENAI_MODEL", "openai-model")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-model")
    monkeypatch.setenv("SERUEN_DEMO", "0")
    captured = {}
    local_advice = advise("кофе", *PRESETS["mega"][1:], demo=True, use_weather=False)

    def mocked_advise(*args, **kwargs):
        captured.update(kwargs)
        return local_advice

    monkeypatch.setattr(cli, "advise", mocked_advise)
    assert cli.main(["кофе", "--preset", "mega", "--provider", "gemini", "--no-weather", "--json"]) == 0
    assert captured["provider"] == "gemini"
    assert captured["model"] == "gemini-model"
    capsys.readouterr()


def test_explicit_24_hour_schedule_stays_a_schedule_in_cli():
    advice = advise("Я возле Ботанического сада, куда пойти, 60 минут", *PRESETS["botanical"][1:],
                    when=datetime(2026, 10, 19, 23, 30), demo=True, provider="local", use_weather=False)
    tary = next(pick for pick in advice.answer.picks if pick.poi_id == "node/11925892164")
    assert tary.hours["context"] == "schedule"
    assert tary.hours["raw"] == "24/7" and tary.hours["status"] == "open"
    text = format_advice(advice).split(tary.name + "  —", 1)[1].split("\n\n", 1)[0]
    assert "часы: открыто" in text
    assert "круглосуточно по данным OSM" in text
    assert "Под открытым небом" not in text
