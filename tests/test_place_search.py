"""Photon boundary tests: mocked HTTP only, no credentials or live geocoding."""

from collections import OrderedDict, deque
from dataclasses import FrozenInstanceError
from threading import Lock
from types import SimpleNamespace

import pytest
import requests

from seruen.core import place_search as search


class Response:
    def __init__(self, payload=None, status=200, error=None):
        self.payload, self.status_code, self.error = payload, status, error
        self.closed = False

    def json(self):
        if self.error:
            raise self.error
        return self.payload

    def close(self):
        self.closed = True


def feature(**overrides):
    properties = {"osm_type": "W", "osm_id": 123, "name": "Центральный парк",
                  "osm_key": "leisure", "osm_value": "park", "city": "Астана", "countrycode": "KZ",
                  "street": "проспект Туран", "housenumber": "1", "country": "Казахстан"}
    properties.update(overrides)
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [71.42, 51.17]},
            "properties": properties}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.delenv("SERUEN_GEOCODER_URL", raising=False)
    monkeypatch.delenv("SERUEN_GEOCODING", raising=False)
    monkeypatch.setattr(search, "_CACHE", OrderedDict())
    monkeypatch.setattr(search, "_REQUESTS", deque())
    monkeypatch.setattr(search, "_LAST_REQUEST", None)
    monkeypatch.setattr(search, "_RATE_LOCK", Lock())
    clock = SimpleNamespace(now=100_000.0)
    monkeypatch.setattr(search.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(search.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("Unexpected real HTTP request"))
    return clock


def install(monkeypatch, payload=None, *, status=200, error=None):
    response = Response(payload if payload is not None else {"features": [feature()]}, status, error)
    calls = []
    monkeypatch.setattr(requests, "get", lambda *a, **k: calls.append((a, k)) or response)
    return response, calls


def test_bounded_request_and_verified_candidate(monkeypatch):
    response, calls = install(monkeypatch)
    places = search.search_places("  Центральный   парк  ")
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (search.DEFAULT_URL,)
    assert kwargs["params"] == {"q": "Центральный парк", "bbox": "71.2,50.95,71.8,51.35",
                                "countrycode": "KZ", "limit": 8}
    assert kwargs["timeout"] == (3, 12) and kwargs["allow_redirects"] is False
    assert kwargs["headers"]["User-Agent"].startswith("Seruen-AI/")
    assert places[0].as_dict() == {
        "id": "way/123", "name": "Центральный парк", "lat": 51.17, "lon": 71.42,
        "address": "проспект Туран 1, Астана, Казахстан",
        "source_url": "https://www.openstreetmap.org/way/123", "source": "photon", "kind": "park",
    }
    assert response.closed


@pytest.mark.parametrize("osm_type,expected", [("N", "node"), ("W", "way"), ("R", "relation")])
def test_osm_link_is_built_not_copied(monkeypatch, osm_type, expected):
    install(monkeypatch, {"features": [feature(osm_type=osm_type, osm_id="123", url="https://evil.example/",
                                              source_url="javascript:evil()")]})
    place = search.search_places("парк")[0]
    assert place.id == f"{expected}/123"
    assert place.source_url == f"https://www.openstreetmap.org/{expected}/123"


@pytest.mark.parametrize("osm_id", [None, False, 0, -1, 1.5, "1.5", "../123", "１２３", {}, 2**100])
def test_invalid_osm_id_is_discarded(monkeypatch, osm_id):
    install(monkeypatch, {"features": [feature(osm_id=osm_id)]})
    assert search.search_places("парк") == []


@pytest.mark.parametrize("osm_type", [None, {}, "x", "node/1", "../way"])
def test_invalid_osm_type_is_discarded(monkeypatch, osm_type):
    install(monkeypatch, {"features": [feature(osm_type=osm_type)]})
    assert search.search_places("парк") == []


@pytest.mark.parametrize("coordinates", [
    [71.42, float("nan")], [float("inf"), 51.17], [True, 51.17], ["71.42", 51.17],
    [71.42], [71.42, 51.17, 1], [76.95, 43.24], [71.19, 51.17], [71.42, 51.36],
    [10**400, 51.17], None,
])
def test_malformed_or_outside_astana_coordinates_are_discarded(monkeypatch, coordinates):
    bad = feature()
    bad["geometry"]["coordinates"] = coordinates
    install(monkeypatch, {"features": [bad]})
    assert search.search_places("парк") == []


def test_wrong_country_is_discarded_even_inside_bbox(monkeypatch):
    install(monkeypatch, {"features": [feature(countrycode="RU")]})
    assert search.search_places("парк") == []


@pytest.mark.parametrize("city", ["Қоянды", "Косшы", "Алматы", "", {}, 123])
def test_explicit_other_city_inside_large_rectangle_is_discarded(monkeypatch, city):
    install(monkeypatch, {"features": [feature(city=city)]})
    assert search.search_places("Кабанбай батыра 53") == []


@pytest.mark.parametrize("city", ["Астана", "Астана қаласы", "город Астана", "Astana", "Nur-Sultan",
                                  "Нур-Султан", "Нұр-Сұлтан", None])
def test_city_aliases_or_missing_city_preserve_bbox_valid_result(monkeypatch, city):
    install(monkeypatch, {"features": [feature(city=city)]})
    assert len(search.search_places("Туран 1")) == 1


def test_bounds_helper_rejects_nonfinite_and_accepts_boundary():
    assert search.is_in_astana(50.95, 71.20)
    assert search.is_in_astana(51.35, 71.80)
    assert not search.is_in_astana("51.1", 71.4)
    assert not search.is_in_astana(51.1, True)
    assert not search.is_in_astana(float("inf"), 71.4)
    assert not search.is_in_astana(10**400, 71.4)


def test_bad_features_do_not_hide_later_valid_match(monkeypatch):
    bad = feature(osm_id=999)
    bad["geometry"]["type"] = "Polygon"
    install(monkeypatch, {"features": [None, [], {}, bad, feature()]})
    assert [place.id for place in search.search_places("парк")] == ["way/123"]


def test_duplicates_and_excess_results_are_bounded(monkeypatch):
    install(monkeypatch, {"features": [feature(), feature()] + [feature(osm_id=i) for i in range(1, 30)]})
    places = search.search_places("парк")
    assert len(places) == 8
    assert len({place.id for place in places}) == 8


def test_address_search_can_use_street_when_name_is_missing(monkeypatch):
    install(monkeypatch, {"features": [feature(name=None, osm_value="house")]})
    place = search.search_places("Туран 1")[0]
    assert place.name == "проспект Туран 1" and place.kind == "house"


def test_nameless_feature_without_address_is_discarded(monkeypatch):
    install(monkeypatch, {"features": [feature(name=None, street=None, housenumber=None)]})
    assert search.search_places("неизвестный объект") == []


@pytest.mark.parametrize("payload", [[], "upstream text", {}, {"features": None}, {"features": {}}])
def test_invalid_top_level_response_is_classified(monkeypatch, payload):
    response, _ = install(monkeypatch, payload)
    with pytest.raises(search.GeocoderError) as error:
        search.search_places("парк")
    assert error.value.reason == "malformed"
    assert response.closed and not search._CACHE


def test_json_error_does_not_leak_upstream_data(monkeypatch):
    response, _ = install(monkeypatch, error=ValueError("private raw upstream text"))
    with pytest.raises(search.GeocoderError) as error:
        search.search_places("личный адрес")
    assert error.value.reason == "malformed"
    assert "private" not in str(error.value) and "личный" not in str(error.value)
    assert error.value.__cause__ is None and response.closed


@pytest.mark.parametrize("status", [301, 401, 403, 429, 500])
def test_http_error_is_safe_no_retry(monkeypatch, status):
    response, calls = install(monkeypatch, {"error": "private upstream body"}, status=status)
    with pytest.raises(search.GeocoderError) as error:
        search.search_places("парк")
    assert error.value.reason == ("rate_limited" if status == 429 else "unavailable")
    assert "private" not in str(error.value)
    assert len(calls) == 1 and response.closed and not search._CACHE


@pytest.mark.parametrize("exception", [requests.Timeout, requests.ConnectionError, requests.RequestException])
def test_network_error_is_safe_and_not_cached(monkeypatch, exception):
    def fail(*args, **kwargs):
        raise exception("private upstream query text")
    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(search.GeocoderError) as error:
        search.search_places("парк")
    assert error.value.reason == "network"
    assert "private" not in str(error.value) and error.value.__cause__ is None
    assert not search._CACHE and len(search._REQUESTS) == 1


@pytest.mark.parametrize("query", [None, 123, "", " \n ", "а" * 241, "а" * 1001, "парк\x00сад"])
def test_invalid_query_does_not_use_network_or_rate_slot(query):
    with pytest.raises(search.GeocoderError) as error:
        search.search_places(query)
    assert error.value.reason == "invalid_query" and not search._REQUESTS


def test_disabled_search_does_not_even_return_cached_places(monkeypatch):
    install(monkeypatch)
    search.search_places("парк")
    monkeypatch.setenv("SERUEN_GEOCODING", "0")
    with pytest.raises(search.GeocoderError) as error:
        search.search_places("парк")
    assert error.value.reason == "disabled" and len(search._REQUESTS) == 1


@pytest.mark.parametrize("url", ["", "file:///tmp/local", "https://user:secret@example.org/api/",
                                 "https://example.org/api/?token=secret", "https://example.org:bad/api/"])
def test_invalid_endpoint_is_rejected_without_disclosing_it(monkeypatch, url):
    monkeypatch.setenv("SERUEN_GEOCODER_URL", url)
    with pytest.raises(search.GeocoderError) as error:
        search.search_places("парк")
    assert error.value.reason == "invalid_config"
    assert "secret" not in str(error.value) and not search._REQUESTS


def test_owner_can_configure_separate_photon_endpoint(monkeypatch):
    monkeypatch.setenv("SERUEN_GEOCODER_URL", "http://127.0.0.1:2322/api/")
    _, calls = install(monkeypatch)
    search.search_places("парк")
    assert calls[0][0] == ("http://127.0.0.1:2322/api/",)


def test_normalized_cache_key_and_defensive_result_copy(monkeypatch):
    _, calls = install(monkeypatch)
    first = search.search_places("  Центральный   Парк ")
    first.clear()
    second = search.search_places("центральный парк")
    assert len(second) == 1 and len(calls) == 1
    assert next(iter(search._CACHE))[1] == "центральный парк"
    with pytest.raises(FrozenInstanceError):
        second[0].name = "changed"


def test_empty_success_is_cached(monkeypatch):
    _, calls = install(monkeypatch, {"features": []})
    assert search.search_places("парк") == search.search_places("парк") == []
    assert len(calls) == 1


def test_cache_expires_at_24_hours(monkeypatch, isolated):
    _, calls = install(monkeypatch)
    search.search_places("парк")
    isolated.now += 24 * 60 * 60 - 1
    search.search_places("парк")
    assert len(calls) == 1
    isolated.now += 1
    search.search_places("парк")
    assert len(calls) == 2


def test_cache_is_bounded_and_evicts_oldest(monkeypatch, isolated):
    install(monkeypatch)
    for index in range(514):
        isolated.now += 2
        # Spread search starts over the minute cap without real waiting.
        if index % 30 == 0:
            isolated.now += 61
        search.search_places(f"Место {index}")
    assert len(search._CACHE) == 512
    assert (search.DEFAULT_URL, "место 0") not in search._CACHE
    assert (search.DEFAULT_URL, "место 513") in search._CACHE


def test_sequential_search_waits_at_most_one_second_and_cache_does_not_wait(monkeypatch, isolated):
    _, calls = install(monkeypatch)
    start = isolated.now
    search.search_places("парк")
    search.search_places("парк")
    assert isolated.now == start
    search.search_places("музей")
    assert isolated.now == start + 1 and len(calls) == 2
    assert list(search._REQUESTS) == [start, start + 1]


def test_request_queue_is_bounded(monkeypatch):
    class BusyLock:
        def acquire(self, *, timeout):
            assert timeout == 1.0
            return False

    monkeypatch.setattr(search, "_RATE_LOCK", BusyLock())
    with pytest.raises(search.GeocoderError) as error:
        search.search_places("парк")
    assert error.value.reason == "rate_limited" and not search._REQUESTS


def test_minute_cap_does_not_wait(monkeypatch, isolated):
    install(monkeypatch)
    search._REQUESTS.extend([isolated.now] * 30)
    start = isolated.now
    with pytest.raises(search.GeocoderError) as error:
        search.search_places("парк")
    assert error.value.reason == "rate_limited" and isolated.now == start


def test_rolling_minute_cap_and_expiration(monkeypatch, isolated):
    _, calls = install(monkeypatch)
    for index in range(30):
        search.search_places(f"Место {index}")
        isolated.now += 1
    with pytest.raises(search.GeocoderError) as error:
        search.search_places("ещё место")
    assert error.value.reason == "rate_limited" and len(calls) == 30
    isolated.now += 30
    search.search_places("ещё место")
    assert len(calls) == 31
