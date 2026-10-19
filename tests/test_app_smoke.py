"""Offline UI workflows: input precedence, durable results, honest empty states."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "seruen" / "app" / "streamlit_app.py"
DATASET = Path(__file__).resolve().parents[1] / "data" / "pois.json"
pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="dataset not collected yet (run: make data)")


@pytest.fixture
def app(monkeypatch):
    # Fail loudly if the offline UI ever attempts a weather call; never expose a real model key.
    import requests

    from seruen.core import server_config, weather

    read_text = Path.read_text

    def read_without_private_env(path, *args, **kwargs):
        return "" if path.name == ".env" else read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_without_private_env)
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("SERUEN_DEMO", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(server_config, "get_server_config", lambda: {"provider": "local", "model": None, "ready": True})

    def no_inference(*args, **kwargs):
        pytest.fail("UI tests must not make real inference requests")

    monkeypatch.setattr(requests, "request", no_inference)
    monkeypatch.setattr(requests, "post", no_inference)
    monkeypatch.setattr(requests, "get", no_inference)

    def no_weather(*args, **kwargs):
        pytest.fail("The default UI must not request weather without opt-in")

    monkeypatch.setattr(weather, "get_weather", no_weather)
    result = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not result.exception, [item.value for item in result.exception]
    return result


def last_advice(app):
    return app.session_state["history"][-1]["advice"]


def origin_payload(status="resolved", *, place=None, candidates=None, context_query="куда пойти, 60 минут"):
    return {"status": status, "place": place, "candidates": candidates or [], "message": "",
            "place_query": "Центральный парк", "context_query": context_query, "ai_used": True}


def mock_origin(monkeypatch, payload):
    from seruen.core import origin_resolution

    calls = []

    def resolve(query, **kwargs):
        calls.append((query, kwargs))
        return SimpleNamespace(as_dict=lambda: deepcopy(payload))

    monkeypatch.setattr(origin_resolution, "resolve_query_origin", resolve)
    return calls


def place_candidate(name="Центральный парк", lat=51.1600, lon=71.4190, *, place_id="way/123", address="Астана, улица Туран"):
    return {"id": place_id, "name": name, "lat": lat, "lon": lon, "address": address,
            "source_url": f"https://www.openstreetmap.org/{place_id}", "source": "photon", "kind": "park"}


def test_app_starts_without_a_query(app):
    assert app.title[0].value == "Seruen AI"
    assert len([button for button in app.button if button.key.startswith("scenario_")]) == 5
    assert app.checkbox(key="weather").value is False
    assert app.checkbox(key="query_location").value is True
    manual_start = next(expander for expander in app.expander if expander.label == "Указать старт вручную")
    assert manual_start.proto.expanded is False
    assert not app.session_state["history"]


@pytest.mark.parametrize("index", range(5))
def test_every_preset_scenario_answers_or_explains_empty_result(app, index):
    app.button(key=f"scenario_{index}").click().run()
    assert not app.exception, [item.value for item in app.exception]
    advice = last_advice(app)
    texts = [block.value for block in app.markdown]
    if advice.answer.picks:
        assert any("Почему подходит" in text for text in texts)
        assert any("Что делать" in text for text in texts)
    else:
        assert app.warning, "An honest empty result must have an explanation"
        assert advice.answer.note
    metrics = {metric.label: metric.value for metric in app.metric}
    assert "Кандидатов после фильтров" in metrics
    assert metrics["Передано модели"] == "0"
    assert len(app.get("deck_gl_json_chart")) == 1


def test_typed_budget_location_and_group_are_not_overridden_by_defaults(app):
    app.chat_input[0].set_value("Я у Mega Silk Way, нас 4 человека, есть 45 минут, хочу погулять").run()
    assert not app.exception
    advice = last_advice(app)
    assert advice.intent.time_budget_min == 45
    assert advice.intent.group_size == 4
    assert advice.origin_label == "Mega Silk Way"


def test_result_survives_sidebar_changes_and_uses_explicit_overrides_next_time(app):
    app.chat_input[0].set_value("Я у Байтерека, хочу погулять 45 минут").run()
    original = last_advice(app).as_dict()
    app.checkbox(key="manual_budget").check().run()
    app.slider(key="budget").set_value(30)
    app.checkbox(key="query_location").uncheck()
    app.selectbox(key="preset").set_value("mega")
    app.run()
    assert not app.exception
    assert last_advice(app).as_dict() == original
    assert len(app.session_state["history"]) == 1
    app.chat_input[0].set_value("Я у Байтерека, хочу погулять 45 минут").run()
    assert last_advice(app).intent.time_budget_min == 30
    assert last_advice(app).origin_label == "Mega Silk Way"


def test_scenario_resets_stale_filters_and_clock(app):
    app.button(key="scenario_4").click().run()
    assert app.session_state["clock"] == "23:30"
    app.checkbox(key="wheelchair").check()
    app.checkbox(key="manual_budget").check()
    app.radio(key="indoor").set_value("В помещении")
    app.run()
    app.button(key="scenario_0").click().run()
    assert not app.exception
    assert app.session_state["clock"] == ""
    assert not app.session_state["wheelchair"]
    assert not app.session_state["manual_budget"]
    assert app.session_state["indoor"] == "Из запроса"
    assert last_advice(app).intent.time_budget_min == 45


def test_session_history_can_be_reopened_and_cleared(app):
    app.button(key="scenario_0").click().run()
    first = app.session_state["history"][0]
    app.button(key="scenario_1").click().run()
    app.selectbox(key="history_selection").set_value(first["id"]).run()
    assert not app.exception
    assert app.session_state["history_selection"] == first["id"]
    assert any(first["query"] in item.value for item in app.markdown)
    app.button(key="clear_history").click().run()
    assert not app.exception
    assert app.session_state["history"] == []
    assert not app.get("deck_gl_json_chart")


def test_invalid_clock_does_not_silently_run_at_current_time(app):
    app.text_input(key="clock").set_value("25:99").run()
    app.chat_input[0].set_value("хочу кафе, 30 минут").run()
    assert not app.exception
    assert app.error
    assert app.session_state["history"] == []


def test_map_path_and_stop_labels_match_computed_route(app):
    app.button(key="scenario_4").click().run()
    advice = last_advice(app)
    deck = json.loads(app.get("deck_gl_json_chart")[0].proto.json)
    text_layer = next(layer for layer in deck["layers"] if layer["@@type"] == "TextLayer")
    # Bare strings become field accessors in pydeck; anchors must be literals.
    assert text_layer["getTextAnchor"] == "middle"
    assert text_layer["getAlignmentBaseline"] == "center"
    paths = [layer for layer in deck["layers"] if layer["@@type"] == "PathLayer"]
    if not advice.route:
        assert not paths
        return
    by_id = {pick.poi_id: pick for pick in advice.answer.picks}
    expected = [[advice.origin[1], advice.origin[0]]]
    expected += [[by_id[poi_id].lon, by_id[poi_id].lat] for poi_id in advice.route.order]
    if advice.route.return_leg:
        expected.append(expected[0])
    assert paths[0]["data"][0]["path"] == expected
    labels = text_layer["data"]
    for number, poi_id in enumerate(advice.route.order, 1):
        marker = next(marker for marker in labels if marker["name"] == by_id[poi_id].name)
        assert marker["label"] == str(number)


def test_user_markdown_is_rendered_as_text(app):
    app.chat_input[0].set_value("Я у Байтерека, кафе, 30 минут ![external](https://example.com/pixel.png)").run()
    assert not app.exception
    query_block = next(item.value for item in app.markdown if "example" in item.value)
    assert r"\!\[external\]" in query_block


def rendered_text(app):
    values = []
    for kind in ("markdown", "caption", "info", "warning", "error", "success", "title", "subheader"):
        values.extend(str(item.value) for item in getattr(app, kind))
    for kind in ("selectbox", "button", "checkbox", "text_input", "radio"):
        values.extend(str(item.label) for item in getattr(app, kind))
    return "\n".join(values)


def test_customers_have_no_provider_account_or_model_controls(app):
    for stage in range(2):
        if stage:
            app.button(key="scenario_0").click().run()
        assert not app.exception
        widget_keys = {item.key for kind in ("selectbox", "button", "checkbox", "text_input") for item in getattr(app, kind)}
        assert "provider" not in widget_keys
        assert not any(key and any(part in key for part in ("chatgpt", "account", "model", "api_key", "signin")) for key in widget_keys)
        text = rendered_text(app).casefold()
        for forbidden in ("войти через", "подключить chatgpt", "моя подписка", "api-ключ", "openai_api_key", "аккаунт chatgpt"):
            assert forbidden not in text
        assert not any("auth.openai.com" in item.proto.url for item in app.get("link_button"))


def test_server_configuration_supplies_ai_without_customer_setup_or_secret_exposure(app, monkeypatch):
    from seruen import cli
    from seruen.core import server_config

    owner_key = "test-owner-key-never-render-this"
    monkeypatch.setenv("OPENAI_API_KEY", owner_key)
    monkeypatch.setattr(server_config, "get_server_config", lambda: {
        "provider": "openai", "model": "owner-selected-model", "ready": True,
    })
    actual_advise = cli.advise
    captured = []

    def server_advice(*args, **kwargs):
        captured.append(kwargs.copy())
        local = {**kwargs, "provider": "local", "demo": True, "model": None}
        advice = actual_advise(*args, **local)
        advice.answer.mode = "openai"
        advice.answer.model = kwargs["model"]
        advice.answer.candidates_sent_to_model = len(advice.retrieval.candidates)
        return advice

    monkeypatch.setattr(cli, "advise", server_advice)
    resolution_calls = mock_origin(monkeypatch, origin_payload(place=place_candidate("Байтерек", 51.1283, 71.4305)))
    app.chat_input[0].set_value("Я у Байтерека, кафе, 60 минут").run()
    assert not app.exception
    assert captured[0]["provider"] == "openai"
    assert captured[0]["model"] == "owner-selected-model"
    assert captured[0]["demo"] is False
    assert resolution_calls[0][1] == {"use_ai": True, "allow_online": True, "model": "owner-selected-model"}
    assert "api_key" not in captured[0]
    assert any("Ответ ИИ" in item.value for item in app.caption)
    assert "owner-selected-model" in rendered_text(app)
    assert owner_key not in rendered_text(app)
    assert owner_key not in repr(app.session_state)
    assert last_advice(app).answer.mode == "openai"


def test_customer_can_get_local_result_when_server_ai_is_not_ready(app, monkeypatch):
    from seruen import cli
    from seruen.core import server_config

    monkeypatch.setattr(server_config, "get_server_config", lambda: {"provider": "openai", "model": None, "ready": False})
    actual_advise = cli.advise
    called = []

    def fallback_advice(*args, **kwargs):
        called.append(kwargs["provider"])
        advice = actual_advise(*args, **{**kwargs, "provider": "local", "demo": True})
        advice.answer.fallback_reason = "OPENAI_API_KEY is not configured"
        return advice

    monkeypatch.setattr(cli, "advise", fallback_advice)
    mock_origin(monkeypatch, origin_payload(place=place_candidate("Байтерек", 51.1283, 71.4305)))
    app.chat_input[0].set_value("Я у Байтерека, кафе, 60 минут").run()
    assert not app.exception
    assert called == ["openai"], "Readiness must not become a customer setup blocker"
    assert len(app.session_state["history"]) == 1
    assert "OPENAI_API_KEY" not in rendered_text(app)
    assert any("ответ по шаблонам" in item.value.casefold() for item in app.caption)


def test_customer_histories_and_context_stay_isolated_between_sessions(app):
    second = AppTest.from_file(str(APP), default_timeout=60).run()
    first_query = "Я у Mega Silk Way, хочу погулять 45 минут"
    second_query = "Я у Байтерека, хочу тихое кафе на 60 минут"
    app.chat_input[0].set_value(first_query).run()
    second.chat_input[0].set_value(second_query).run()
    assert not app.exception and not second.exception
    assert [row["query"] for row in app.session_state["history"]] == [first_query]
    assert [row["query"] for row in second.session_state["history"]] == [second_query]
    assert last_advice(app).origin_label == "Mega Silk Way"
    assert last_advice(second).origin_label == "Байтерек"
    app.button(key="clear_history").click().run()
    assert app.session_state["history"] == []
    assert [row["query"] for row in second.session_state["history"]] == [second_query]

def test_alternative_and_route_times_include_return_and_buffer(app):
    app.text_input(key="clock").set_value("12:00").run()
    app.chat_input[0].set_value("Я у Байтерека, кафе, 90 минут, вернуться к старту").run()
    assert not app.exception
    advice = last_advice(app)
    assert advice.intent.return_to_start
    assert advice.answer.picks
    text = rendered_text(app)
    for pick in advice.answer.picks:
        expected = 2 * pick.walk_min + pick.visit_minutes + 5
        assert f"На этот вариант: ≈{expected} из 90 мин" in text
        assert f"{pick.walk_min} мин туда + {pick.walk_min} мин обратно + {pick.visit_minutes} мин на месте + 5 мин запаса" in text
    assert advice.route and advice.route.return_leg
    progress = app.get("progress")[0].proto
    assert f"{advice.route.as_dict()['planned_min']} из 90 мин" in progress.text
    assert "Возвращение к старту" in text
    assert advice.route.as_dict()["planned_min"] <= advice.intent.time_budget_min


def test_refresh_uses_current_settings_without_retyping_or_rerun_calls(app, monkeypatch):
    from seruen import cli

    actual_advise = cli.advise
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return actual_advise(*args, **kwargs)

    monkeypatch.setattr(cli, "advise", counted)
    query = "Я у Байтерека, хочу кафе, 60 минут"
    app.chat_input[0].set_value(query).run()
    app.checkbox(key="manual_budget").check().run()
    app.slider(key="budget").set_value(45).run()
    assert calls == [query]
    app.button(key="repeat_query").click().run()
    assert not app.exception
    assert calls == [query, query]
    assert last_advice(app).intent.time_budget_min == 45
    app.run()
    assert calls == [query, query]


def test_download_contains_usable_plan_but_not_raw_query_or_precise_start(app, monkeypatch):
    from streamlit.delta_generator import DeltaGenerator

    captured = []
    original_download = DeltaGenerator.download_button

    def download(self, label, data, **kwargs):
        captured.append((data, kwargs))
        return original_download(self, label, data, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "download_button", download)
    app.checkbox(key="manual_coordinates").check().run()
    app.number_input(key="latitude").set_value(51.1283789)
    app.number_input(key="longitude").set_value(71.4305123)
    app.text_input(key="clock").set_value("12:00")
    query = "кафе, 90 минут, вернуться к старту. Личная заметка: СЕКРЕТ_ДЛЯ_ТЕСТА 51.1283789 71.4305123"
    app.chat_input[0].set_value(query).run()
    assert not app.exception
    advice = last_advice(app)
    assert advice.answer.picks
    exported, options = captured[-1]
    assert options["on_click"] == "ignore", "Saving a result must not run another AI request"
    assert options["file_name"].endswith(".txt")
    for private in (query, "СЕКРЕТ_ДЛЯ_ТЕСТА", "51.1283789", "71.4305123"):
        assert private not in exported
    for pick in advice.answer.picks:
        assert pick.name in exported
        assert pick.activity in exported
    assert "ВАРИАНТЫ ДЛЯ ОТДЕЛЬНОГО ПОСЕЩЕНИЯ" in exported
    assert "OpenStreetMap contributors" in exported
    assert "сгенерированная идея" in exported
    if advice.route:
        assert "ГОТОВЫЙ МАРШРУТ" in exported
        for poi_id in advice.route.order:
            assert f"на месте {advice.route.stop_visit_min[poi_id]} мин" in exported
        assert "Возвращение к старту:" in exported


def test_stationary_work_has_a_useful_single_place_explanation(app):
    app.text_input(key="clock").set_value("12:00").run()
    app.chat_input[0].set_value("Я у Байтерека, ищу тихое кафе с розетками поработать за ноутбуком, 90 минут").run()
    assert not app.exception
    assert last_advice(app).answer.picks
    assert last_advice(app).intent.wants_work
    assert last_advice(app).route is None
    assert "Для работы выберите одно место" in rendered_text(app)
    assert not app.get("progress")


def test_botanical_garden_origin_is_taken_from_the_user_request(app):
    places = json.loads(DATASET.read_text(encoding="utf-8"))
    garden = next(poi for poi in places if poi.get("name", {}).get("ru") == "Ботанический сад")
    app.chat_input[0].set_value("я возле ботанического сада куда могу пойти").run()
    assert not app.exception
    advice = last_advice(app)
    assert advice.origin_label == "Ботанический сад"
    assert advice.origin == (garden["lat"], garden["lon"])
    assert "Ботанический сад" in rendered_text(app)


def test_unknown_explicit_origin_does_not_silently_generate_from_a_preset(app, monkeypatch):
    from seruen import cli

    actual_advise = cli.advise

    def no_advice(*args, **kwargs):
        pytest.fail("An unrecognized explicit starting point must be clarified before inference")

    monkeypatch.setattr(cli, "advise", no_advice)
    app.chat_input[0].set_value("я возле Несуществующего маяка, куда могу пойти").run()
    assert not app.exception
    assert app.warning
    assert app.session_state["origin_clarification"]
    assert not app.session_state["history"]
    # The visitor can deliberately choose a known start to recover from ambiguity.
    monkeypatch.setattr(cli, "advise", actual_advise)
    app.selectbox(key="preset").set_value("mega").run()
    assert app.checkbox(key="query_location").value is False
    app.button(key="retry_manual_origin").click().run()
    assert not app.exception
    assert not app.error
    assert last_advice(app).origin_label == "Mega Silk Way"


def test_preference_without_a_start_requests_clarification(app):
    app.chat_input[0].set_value("хочу погулять у воды, 45 минут").run()
    assert not app.exception
    assert not app.session_state["history"]
    assert app.session_state["origin_clarification"]["resolution"]["status"] == "missing"
    assert app.warning
    assert not app.get("deck_gl_json_chart")


def test_resolved_place_drives_map_and_context_without_default_landmark(app, monkeypatch):
    from seruen import cli

    place = place_candidate(lat=51.152859, lon=71.41902, address="Астана, Центральный парк")
    place.update(source="visitastana", source_url="https://visitastana.kz/ru/about-city/what-to-see/tsentralnyy-park/")
    original = "я вот в центральном парке. куда могу отсюда пойти"
    clean_context = "куда могу отсюда пойти"
    calls = mock_origin(monkeypatch, origin_payload(place=place, context_query=clean_context))
    actual_advise, advice_calls = cli.advise, []

    def observe(*args, **kwargs):
        advice_calls.append((args, kwargs))
        return actual_advise(*args, **kwargs)

    monkeypatch.setattr(cli, "advise", observe)
    app.chat_input[0].set_value(original).run()
    assert not app.exception
    assert calls == [(original, {"use_ai": False, "allow_online": False, "model": None})]
    assert advice_calls[0][0][0] == original
    assert advice_calls[0][1]["intent_query"] == clean_context
    assert advice_calls[0][1]["origin_resolution"]["place"] == place
    assert last_advice(app).origin == (place["lat"], place["lon"])
    assert "Старт определён:** Центральный парк" in rendered_text(app)
    assert place["address"] in rendered_text(app)
    assert place["source_url"] in rendered_text(app)
    assert "Источник: Visit Astana" in rendered_text(app)
    assert "Старт подборки:** Центральный парк" in rendered_text(app)
    deck = json.loads(app.get("deck_gl_json_chart")[0].proto.json)
    starts = [layer for layer in deck["layers"] if any(row.get("kind") == "Точка старта" for row in layer.get("data", []))]
    assert starts[0]["data"][0]["lat"] == place["lat"]
    assert starts[0]["data"][0]["lon"] == place["lon"]


@pytest.mark.parametrize("status", ["missing", "unresolved", "unavailable"])
def test_unconfirmed_origin_never_creates_recommendations(app, monkeypatch, status):
    from seruen import cli

    calls = mock_origin(monkeypatch, origin_payload(status))

    def no_advice(*args, **kwargs):
        pytest.fail("No confirmed location must mean no recommendation request")

    monkeypatch.setattr(cli, "advise", no_advice)
    app.chat_input[0].set_value("я здесь, куда пойти").run()
    assert not app.exception
    assert app.warning
    assert len(calls) == 1
    assert not app.session_state["history"]
    assert not app.get("deck_gl_json_chart")
    app.run()
    app.checkbox(key="with_child").check().run()
    assert len(calls) == 1
    assert app.session_state["origin_clarification"]["resolution"]["status"] == status


def test_ambiguous_choice_resumes_original_request_once_without_extracting_again(app, monkeypatch):
    from seruen import cli
    from seruen.core import server_config

    monkeypatch.setattr(server_config, "get_server_config", lambda: {"provider": "openai", "model": "test-model", "ready": True})
    places = [place_candidate(address="Астана, проспект Туран"),
              place_candidate("Другой центральный парк", 51.145, 71.410, place_id="way/456", address="Астана, другой район")]
    original = "я вот в центральном парке. куда могу отсюда пойти"
    resolution_calls = mock_origin(monkeypatch, origin_payload("ambiguous", candidates=places, context_query="куда могу отсюда пойти"))
    actual_advise, advice_calls = cli.advise, []

    def local_advice(*args, **kwargs):
        advice_calls.append((args, kwargs))
        return actual_advise(*args, **{**kwargs, "provider": "local", "demo": True, "model": None})

    monkeypatch.setattr(cli, "advise", local_advice)
    app.checkbox(key="manual_budget").check().run()
    app.slider(key="budget").set_value(45).run()
    app.chat_input[0].set_value(original).run()
    assert not app.exception
    assert len(resolution_calls) == 1
    assert not advice_calls
    assert not app.session_state["history"]
    assert len([button for button in app.button if button.label == "Начать отсюда"]) == 2
    assert all(place["address"] in rendered_text(app) for place in places)
    app.run()
    app.slider(key="budget").set_value(90).run()
    app.button(key="origin_candidate_1").click().run()
    assert not app.exception
    assert len(resolution_calls) == 1
    assert len(advice_calls) == 1
    assert advice_calls[0][0][0] == original
    assert advice_calls[0][1]["intent_query"] == "куда могу отсюда пойти"
    assert advice_calls[0][1]["origin_resolution"]["user_confirmed"] is True
    assert last_advice(app).origin == (places[1]["lat"], places[1]["lon"])
    assert last_advice(app).intent.time_budget_min == 45
    first_id = app.session_state["history"][0]["id"]
    app.run()
    app.checkbox(key="query_location").uncheck().run()
    app.chat_input[0].set_value("хочу кафе, 60 минут").run()
    assert len(resolution_calls) == 1
    assert len(advice_calls) == 2
    app.selectbox(key="history_selection").set_value(first_id).run()
    assert len(resolution_calls) == 1
    assert len(advice_calls) == 2
    assert places[1]["name"] in rendered_text(app)


@pytest.mark.parametrize("manual_mode", ["preset", "coordinates"])
def test_manual_start_never_invokes_automatic_resolution(app, monkeypatch, manual_mode):
    from seruen.core import origin_resolution
    from seruen.core.location_presets import PRESETS

    def no_resolution(*args, **kwargs):
        pytest.fail("An explicit manual starting point must bypass AI extraction and geocoding")

    monkeypatch.setattr(origin_resolution, "resolve_query_origin", no_resolution)
    if manual_mode == "preset":
        app.selectbox(key="preset").set_value("mega").run()
        expected = PRESETS["mega"][1:]
    else:
        app.checkbox(key="manual_coordinates").check().run()
        app.number_input(key="latitude").set_value(51.1283)
        app.number_input(key="longitude").set_value(71.4305).run()
        expected = (51.1283, 71.4305)
    app.chat_input[0].set_value("Я возле неизвестного места, хочу гулять 45 минут").run()
    assert not app.exception
    assert last_advice(app).origin == expected
    assert last_advice(app).origin_resolution["place"]["source"] == "manual"


def test_real_memorial_at_night_describes_outdoor_access_in_ui_and_download(app, monkeypatch):
    from streamlit.delta_generator import DeltaGenerator

    from seruen.cli import format_advice

    downloads = []
    original_download = DeltaGenerator.download_button

    def download(self, label, data, **kwargs):
        downloads.append(data)
        return original_download(self, label, data, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "download_button", download)
    app.text_input(key="clock").set_value("23:30").run()
    app.chat_input[0].set_value("Я возле Ботанического сада, хочу посмотреть памятник, 60 минут").run()
    assert not app.exception
    advice = last_advice(app)
    memorial = next(pick for pick in advice.answer.picks if pick.poi_id == "node/5878503586")
    assert (advice.when.hour, advice.when.minute) == (23, 30)
    assert memorial.hours["context"] == "outdoor"
    assert memorial.hours["status"] == "unknown" and memorial.hours["raw"] is None
    assert memorial.hours["confidence"] == "low", "Raw schedule evidence must remain unchanged"
    def card_for(pick):
        return next(container for container in app.container if any(
            child.type == "markdown" and child.value.startswith("### ") and pick.name in child.value
            for child in container.children.values()
        ))

    memorial_card = card_for(memorial)
    displayed = rendered_text(memorial_card)
    assert "Под открытым небом" in displayed
    assert "Учитывайте правила доступа на территорию." in displayed
    assert "Часы работы неизвестны" not in displayed
    assert "расписания: низкая" not in displayed
    assert [caption.value for caption in memorial_card.caption].count("Учитывайте правила доступа на территорию.") == 1
    arch = next(pick for pick in advice.answer.picks if pick.poi_id == "way/549307755")
    assert arch.hours["context"] == "schedule"
    assert "Часы работы неизвестны" in rendered_text(card_for(arch))
    alternatives = downloads[-1].split("ВАРИАНТЫ ДЛЯ ОТДЕЛЬНОГО ПОСЕЩЕНИЯ", 1)[1]
    exported = alternatives.split(memorial.name, 1)[1].split("\n\n", 1)[0]
    cli_card = format_advice(advice).split(memorial.name + "  —", 1)[1].split("\n\n", 1)[0]
    for text in (exported, cli_card):
        assert "Под открытым небом" in text
        assert "Учитывайте правила доступа на территорию." in text
        assert "Часы работы неизвестны" not in text
        assert "часы: неизвестно" not in text
        assert "круглосуточно" not in text and "24/7" not in text
