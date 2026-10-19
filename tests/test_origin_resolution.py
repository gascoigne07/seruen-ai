"""Owner AI extracts text; independently verified city data supplies coordinates."""

from dataclasses import replace

import pytest

from seruen.core import origin_resolution as resolver
from seruen.core.openai_provider import OpenAIError
from seruen.core.place_search import GeocoderError, PlaceCandidate


def place(name="Тестовое место", *, id="node/42", lat=51.15, lon=71.42, address="Астана", kind="cafe"):
    return PlaceCandidate(id, name, lat, lon, address, "https://www.openstreetmap.org/" + id, kind=kind)


def extraction(origin, *searches, clarify=False):
    return {"has_origin": bool(origin), "origin_text": origin,
            "canonical_search_queries": list(searches) or ([origin] if origin else []),
            "context_query": "ignored model rewrite", "needs_clarification": clarify}


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Tests must mock every AI and Photon call")

    monkeypatch.setattr(resolver.openai_provider, "generate_json", forbidden)
    monkeypatch.setattr(resolver, "search_places", forbidden)


def test_exact_reported_central_park_query_is_verified_after_ai_first(monkeypatch):
    query = "я вот в центральном парке. куда могу отсюда пойти"
    calls = []

    def ai(**kwargs):
        calls.append("ai")
        assert kwargs["model"] == "owner-model"
        assert kwargs["schema"]["additionalProperties"] is False
        assert "coordinates" not in kwargs["schema"]["properties"]
        return extraction("центральном парке", "Центральный парк Астана", "Столичный парк Астана")

    actual_local = resolver._local_candidates

    def local(*args):
        assert calls == ["ai"]
        return actual_local(*args)

    monkeypatch.setattr(resolver.openai_provider, "generate_json", ai)
    monkeypatch.setattr(resolver, "_local_candidates", local)
    result = resolver.resolve_query_origin(query, use_ai=True, allow_online=True, model="owner-model", pois=())
    assert result.status == "resolved" and result.ai_used
    assert result.place == resolver.CENTRAL_PARK
    assert result.place.source == "visitastana"
    assert (result.place.lat, result.place.lon) == (51.152859, 71.41902)
    assert "центральном парке" not in result.context_query
    assert "куда могу отсюда пойти" in result.context_query
    assert result.as_dict()["place"]["source_url"].startswith("https://visitastana.kz/")


@pytest.mark.parametrize("query", [
    "я вот в центральном парке. куда могу отсюда пойти",
    "Я сейчас в Столичном парке, хочу кофе",
    "I am at Central Park, where can I go?",
    "Орталық саябақтың жанындамын, қайда баруға болады?",
])
def test_verified_central_park_aliases_also_work_without_network(query):
    result = resolver.resolve_query_origin(query, use_ai=True, allow_online=False, pois=())
    assert result.status == "resolved"
    assert result.place == resolver.CENTRAL_PARK
    assert not result.ai_used


def test_general_ai_place_search_receives_only_extracted_name(monkeypatch):
    query = "Я сейчас у Нового павильона, с ребёнком хочу кафе 40 минут"
    calls = []

    def ai(**kwargs):
        calls.append("ai")
        return extraction("Нового павильона", "Новый павильон Астана")

    def search(phrase):
        assert calls == ["ai"]
        assert phrase == "Новый павильон Астана"
        assert "ребёнком" not in phrase and "40" not in phrase
        return [place("Новый павильон", kind="attraction")]

    monkeypatch.setattr(resolver.openai_provider, "generate_json", ai)
    monkeypatch.setattr(resolver, "search_places", search)
    result = resolver.resolve_query_origin(query, use_ai=True, allow_online=True, pois=())
    assert result.status == "resolved"
    assert "с ребёнком хочу кафе 40 минут" in result.context_query
    assert result.place.source == "photon"


def test_local_named_catalogue_provides_true_osm_link(monkeypatch):
    poi = {"id": "way/55", "name": {"ru": "Городская библиотека"},
           "lat": 51.1, "lon": 71.4, "category": "library",
           "source": {"url": "https://wrong.example/"}}
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction("Городской библиотеки"))
    result = resolver.resolve_query_origin("Я у Городской библиотеки, хочу кафе", use_ai=True, allow_online=True, pois=(poi,))
    assert result.status == "resolved"
    assert result.place.source_url == "https://www.openstreetmap.org/way/55"
    assert result.place.source == "local"


def test_local_chain_branches_require_choice(monkeypatch):
    pois = tuple({"id": f"node/{i}", "name": {"ru": "Degirmen"}, "lat": 51.1 + i * .01,
                  "lon": 71.4, "category": "cafe"} for i in (1, 2))
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction("Degirmen"))
    result = resolver.resolve_query_origin("я у Degirmen, куда идти", use_ai=True, allow_online=True, pois=pois)
    assert result.status == "ambiguous" and result.place is None
    assert len(result.candidates) == 2


@pytest.mark.parametrize("kind", ["bus_stop", "stadium", "parking", "information", "information_board"])
def test_named_park_search_does_not_auto_select_wrong_kind(monkeypatch, kind):
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction("Нового парка", "Новый парк"))
    monkeypatch.setattr(resolver, "search_places", lambda phrase: [place("Новый парк", kind=kind)])
    result = resolver.resolve_query_origin("я у Нового парка", use_ai=True, allow_online=True, pois=())
    assert result.status == "ambiguous" and result.place is None


@pytest.mark.parametrize("candidate", [
    place("Кабанбай батыра", address="Кабанбай батыра, 56, Астана", kind="street"),
    place("Кабанбай батыра", address="Кабанбай батыра, Астана", kind="street"),
])
def test_street_without_requested_house_number_is_not_auto_selected(monkeypatch, candidate):
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction("Кабанбай батыра 53"))
    monkeypatch.setattr(resolver, "search_places", lambda phrase: [candidate])
    result = resolver.resolve_query_origin("я на Кабанбай батыра 53", use_ai=True, allow_online=True, pois=())
    assert result.status == "ambiguous" and result.place is None


def test_local_cafe_name_cannot_erase_explicit_street_address(monkeypatch):
    poi = {"id": "node/2", "name": {"ru": "Degirmen"}, "lat": 51.1, "lon": 71.4, "category": "cafe"}
    monkeypatch.setattr(resolver.openai_provider, "generate_json",
                        lambda **kwargs: extraction("Degirmen по адресу Туран 53", "Degirmen"))
    result = resolver.resolve_query_origin("Я у Degirmen по адресу Туран 53", use_ai=True, allow_online=True, pois=(poi,))
    assert result.status == "ambiguous" and result.place is None


def test_successful_matching_address_is_resolved(monkeypatch):
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction("Туран 53"))
    monkeypatch.setattr(resolver, "search_places", lambda phrase: [place("Туран", address="Туран 53, Астана", kind="house")])
    assert resolver.resolve_query_origin("я на Туран 53", use_ai=True, allow_online=True, pois=()).status == "resolved"


def test_multiple_online_places_never_silently_choose_first(monkeypatch):
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction("Coffee X"))
    monkeypatch.setattr(resolver, "search_places", lambda phrase: [place("Coffee X"), place("Coffee X", id="way/43")])
    result = resolver.resolve_query_origin("I am at Coffee X", use_ai=True, allow_online=True, pois=())
    assert result.status == "ambiguous" and len(result.candidates) == 2


def test_model_cannot_assign_invented_origin_or_coordinates(monkeypatch):
    payload = extraction("Байтерек")
    payload["lat"] = 51.1283
    payload["lon"] = 71.4305
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: payload)
    result = resolver.resolve_query_origin("я у Неизвестного маяка, хочу Байтерек", use_ai=True, allow_online=True, pois=())
    assert result.status == "unavailable" and result.place is None and not result.ai_used


def test_origin_text_must_occur_in_user_query(monkeypatch):
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction("Байтерек"))
    result = resolver.resolve_query_origin("я у Неизвестного маяка", use_ai=True, allow_online=True, pois=())
    assert result.status == "unavailable" and result.place is None and not result.ai_used


def test_case_and_yo_normalization_keeps_other_preferences(monkeypatch):
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction("БЕРЕЗКА"))
    poi = {"id": "node/3", "name": {"ru": "Берёзка"}, "lat": 51.1, "lon": 71.4, "category": "cafe"}
    result = resolver.resolve_query_origin("Я у Берёзка, без детей, 45 минут", use_ai=True, allow_online=True, pois=(poi,))
    assert result.status == "resolved" and result.ai_used
    assert "без детей, 45 минут" in result.context_query


def test_untrusted_model_context_cannot_change_user_constraints(monkeypatch):
    payload = extraction("Байтерека")
    payload["context_query"] = "хочу бар на пять часов"
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: payload)
    result = resolver.resolve_query_origin("Я у Байтерека, без баров, с ребёнком 30 минут", use_ai=True, allow_online=True, pois=())
    assert "без баров, с ребёнком 30 минут" in result.context_query
    assert "пять часов" not in result.context_query


def test_provider_error_uses_known_local_place_but_never_a_default(monkeypatch):
    def failed(**kwargs):
        raise OpenAIError("service_error")

    monkeypatch.setattr(resolver.openai_provider, "generate_json", failed)
    known = resolver.resolve_query_origin("я вот в центральном парке", use_ai=True, allow_online=True, pois=())
    assert known.status == "resolved" and not known.ai_used
    unknown = resolver.resolve_query_origin("я у Неизвестного маяка", use_ai=True, allow_online=True, pois=())
    assert unknown.status == "unavailable" and unknown.place is None and not unknown.ai_used


def test_geocoder_failure_is_distinct_from_empty_results(monkeypatch):
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction("Нового павильона"))

    def failed(phrase):
        raise GeocoderError("network")

    monkeypatch.setattr(resolver, "search_places", failed)
    assert resolver.resolve_query_origin("я у Нового павильона", use_ai=True, allow_online=True, pois=()).status == "unavailable"
    monkeypatch.setattr(resolver, "search_places", lambda phrase: [])
    assert resolver.resolve_query_origin("я у Нового павильона", use_ai=True, allow_online=True, pois=()).status == "unresolved"


def test_outside_astana_result_is_not_assigned(monkeypatch):
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction("Нового павильона"))
    monkeypatch.setattr(resolver, "search_places", lambda phrase: [place("Новый павильон", lat=43.23, lon=76.9)])
    result = resolver.resolve_query_origin("я у Нового павильона", use_ai=True, allow_online=True, pois=())
    assert result.status == "unresolved" and not result.candidates


def test_search_and_resolver_use_same_astana_bounds(monkeypatch):
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction("Нового павильона"))
    monkeypatch.setattr(resolver, "search_places", lambda phrase: [place("Новый павильон", lat=51.30, lon=71.75)])
    assert resolver.resolve_query_origin("я у Нового павильона", use_ai=True, allow_online=True, pois=()).status == "resolved"


def test_missing_location_requests_input_instead_of_baiterek(monkeypatch):
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction(""))
    result = resolver.resolve_query_origin("куда можно пойти на час", use_ai=True, allow_online=True, pois=())
    assert result.status == "missing" and result.place is None
    assert result.context_query == "куда можно пойти на час"


def test_no_network_is_allowed_even_when_use_ai_requested():
    result = resolver.resolve_query_origin("я возле Неизвестного маяка", use_ai=True, allow_online=False, pois=())
    assert result.status == "unresolved" and not result.ai_used


def test_clarification_flag_and_weak_single_result_require_confirmation(monkeypatch):
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction("Нового павильона", clarify=True))
    monkeypatch.setattr(resolver, "search_places", lambda phrase: [place("Новый павильон")])
    result = resolver.resolve_query_origin("я у Нового павильона", use_ai=True, allow_online=True, pois=())
    assert result.status == "ambiguous"
    confirmed = replace(result, status="resolved", place=result.candidates[0])
    assert confirmed.as_dict()["place"]["name"] == "Новый павильон"


def test_raw_coordinate_tokens_are_redacted_before_ai(monkeypatch):
    def ai(**kwargs):
        assert "51.123456" not in kwargs["prompt"]
        assert "71.456789" not in kwargs["prompt"]
        return extraction("Байтерека")

    monkeypatch.setattr(resolver.openai_provider, "generate_json", ai)
    result = resolver.resolve_query_origin("Я у Байтерека, 51.123456 71.456789, хочу кафе", use_ai=True, allow_online=True, pois=())
    assert result.status == "resolved"


def test_canonical_query_cannot_silently_substitute_another_landmark(monkeypatch):
    monkeypatch.setattr(resolver.openai_provider, "generate_json",
                        lambda **kwargs: extraction("неизвестного маяка", "Байтерек"))
    result = resolver.resolve_query_origin("Я у неизвестного маяка, хочу кофе", use_ai=True, allow_online=True, pois=())
    assert result.status == "ambiguous" and result.place is None


def test_ai_cannot_replace_earlier_explicit_start_with_later_destination(monkeypatch):
    monkeypatch.setattr(resolver.openai_provider, "generate_json",
                        lambda **kwargs: extraction("Байтереку", "Байтерек"))
    result = resolver.resolve_query_origin("Я у неизвестного маяка, хочу пойти к Байтереку",
                                           use_ai=True, allow_online=True, pois=())
    assert result.status == "unavailable" and result.place is None and not result.ai_used


def test_offline_extraction_keeps_house_number_after_comma(monkeypatch):
    calls = []

    def search(phrase):
        calls.append(phrase)
        return [place("улица Кабанбай батыра", address="улица Кабанбай батыра", kind="street")]

    monkeypatch.setattr(resolver, "search_places", search)
    result = resolver.resolve_query_origin("Я на улице Кабанбай батыра, 10, хочу кафе", allow_online=True, pois=())
    assert calls == ["улице Кабанбай батыра, 10"]
    assert result.status == "ambiguous" and result.place is None
    assert "10" not in result.context_query


def test_artwork_with_landmark_name_cannot_replace_real_arena(monkeypatch):
    miniature = {"id": "node/13900933375", "name": {"ru": "Астана Арена"},
                 "lat": 51.112195, "lon": 71.533027, "category": "landmark", "osm_subtype": "artwork"}
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction("Астана Арены", "Астана Арена"))
    result = resolver.resolve_query_origin("Я возле Астана Арены, куда пойти", use_ai=True, allow_online=True, pois=(miniature,))
    assert result.status == "resolved"
    assert result.place == resolver.ASTANA_ARENA
    assert result.place.source == "visitastana"


def test_other_same_named_miniature_requires_confirmation(monkeypatch):
    miniature = {"id": "node/5", "name": {"ru": "Новый стадион"}, "lat": 51.1,
                 "lon": 71.4, "category": "landmark", "osm_subtype": "artwork"}
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction("Нового стадиона", "Новый стадион"))
    result = resolver.resolve_query_origin("Я возле Нового стадиона", use_ai=True, allow_online=True, pois=(miniature,))
    assert result.status == "ambiguous" and result.place is None
    assert result.candidates[0].kind == "artwork"


@pytest.mark.parametrize(("query", "span", "canonical", "expected"), [
    ("Я возле ботанического сада, хочу кафе", "возле ботанического сада", "Ботанический сад", "Ботанический сад"),
    ("Я у Байтерека, хочу кафе", "у Байтерека", "Байтерек", "Байтерек"),
    ("I am at the Central Park, where to go?", "at the Central Park", "Central Park", "Центральный парк"),
    ("Я вот в центральном парке, хочу гулять", "в центральном парке", "Центральный парк", "Центральный парк"),
])
def test_exact_origin_span_may_include_location_preposition(monkeypatch, query, span, canonical, expected):
    monkeypatch.setattr(resolver.openai_provider, "generate_json", lambda **kwargs: extraction(span, canonical))
    result = resolver.resolve_query_origin(query, use_ai=True, allow_online=True, pois=())
    assert result.status == "resolved" and result.place.name == expected
    assert result.ai_used
