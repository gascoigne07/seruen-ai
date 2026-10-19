"""Bounded, read-only Photon search for a place name/address supplied by a user.

Callers must pass only the extracted place phrase, not the full chat message.
Results are map points/area centres, not GPS or verified entrances. The caller
must resolve ambiguity; Photon results do not imply confidence or opening hours.
No credentials, files, raw chat messages or persistent history are read/written.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

import requests

DEFAULT_URL = "https://photon.komoot.io/api/"
ASTANA_BBOX = (71.20, 50.95, 71.80, 51.35)  # minLon, minLat, maxLon, maxLat
RESULT_LIMIT = 8
CACHE_LIMIT = 512
CACHE_TTL_SECONDS = 24 * 60 * 60
MIN_REQUEST_INTERVAL = 1.0
MAX_RATE_WAIT_SECONDS = 1.0
MAX_REQUESTS_PER_MINUTE = 30
MAX_QUERY_LENGTH = 240
TIMEOUT = (3, 12)
_CITY_NAMES = {"астана", "астанақаласы", "городастана", "astana", "astanacity",
               "нурсултан", "нұрсұлтан", "nursultan"}


def is_in_astana(lat, lon) -> bool:
    """Validate numeric finite coordinates against the configured project bbox."""
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (lat, lon)):
        return False
    try:
        if not math.isfinite(lat) or not math.isfinite(lon):
            return False
    except (ValueError, OverflowError):
        return False
    min_lon, min_lat, max_lon, max_lat = ASTANA_BBOX
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


@dataclass(frozen=True)
class PlaceCandidate:
    id: str
    name: str
    lat: float
    lon: float
    address: str
    source_url: str
    source: str = "photon"
    kind: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class GeocoderError(RuntimeError):
    """Safe classified error, without upstream response text, URL or user input."""

    def __init__(self, reason: str):
        self.reason = reason
        messages = {
            "disabled": "Внешний поиск мест отключён. Укажите точку на карте или координаты.",
            "invalid_query": "Укажите короткое название места или адрес для поиска.",
            "invalid_config": "Поиск мест временно недоступен из-за настройки сервера.",
            "rate_limited": "Поиск мест занят. Подождите немного и повторите запрос.",
            "network": "Не удалось связаться с поиском мест. Повторите позже или выберите точку вручную.",
            "unavailable": "Сервис поиска мест временно недоступен. Выберите точку вручную.",
            "malformed": "Поиск мест вернул неподходящий ответ. Уточните место или выберите точку вручную.",
        }
        super().__init__(messages.get(reason, messages["unavailable"]))


# Keys contain only the configured endpoint and normalized place phrase, never a
# full chat. Cached tuples and frozen candidates cannot be changed by callers.
_CACHE: OrderedDict[tuple[str, str], tuple[float, tuple[PlaceCandidate, ...]]] = OrderedDict()
_REQUESTS: deque[float] = deque()
_LAST_REQUEST: float | None = None
_LOCK = threading.Lock()
_RATE_LOCK = threading.Lock()


def _endpoint() -> str:
    endpoint = os.environ.get("SERUEN_GEOCODER_URL", DEFAULT_URL).strip()
    try:
        parsed = urlsplit(endpoint)
        valid = (parsed.scheme in {"http", "https"} and bool(parsed.hostname)
                 and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment)
        # Accessing port rejects malformed ports without disclosing the value.
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise GeocoderError("invalid_config")
    return endpoint


def _phrase(query: str) -> str:
    if not isinstance(query, str) or len(query) > 1000:
        raise GeocoderError("invalid_query")
    phrase = " ".join(query.split())
    if not phrase or len(phrase) > MAX_QUERY_LENGTH or any(ord(char) < 32 for char in phrase):
        raise GeocoderError("invalid_query")
    return phrase


def _cached(key: tuple[str, str]) -> tuple[PlaceCandidate, ...] | None:
    with _LOCK:
        now = time.monotonic()
        for expired in [item for item, (saved, _) in _CACHE.items() if now - saved >= CACHE_TTL_SECONDS]:
            del _CACHE[expired]
        entry = _CACHE.get(key)
        if entry is not None:
            _CACHE.move_to_end(key)
            return entry[1]
    return None


def _reserve_request() -> None:
    global _LAST_REQUEST
    # Cache lookups have a separate lock. Only a short queue for starting a
    # request is allowed; reaching the minute cap never sleeps for that window.
    deadline = time.monotonic() + MAX_RATE_WAIT_SECONDS
    if not _RATE_LOCK.acquire(timeout=MAX_RATE_WAIT_SECONDS):
        raise GeocoderError("rate_limited")
    try:
        now = time.monotonic()
        while _REQUESTS and _REQUESTS[0] <= now - 60:
            _REQUESTS.popleft()
        if len(_REQUESTS) >= MAX_REQUESTS_PER_MINUTE:
            raise GeocoderError("rate_limited")
        wait = max(0.0, _LAST_REQUEST + MIN_REQUEST_INTERVAL - now) if _LAST_REQUEST is not None else 0.0
        if wait > max(0.0, deadline - now):
            raise GeocoderError("rate_limited")
        if wait:
            time.sleep(wait)
        now = time.monotonic()
        _LAST_REQUEST = now
        _REQUESTS.append(now)
    finally:
        _RATE_LOCK.release()


def _text(value, *, limit: int = 240) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = "".join(" " if ord(char) < 32 or 127 <= ord(char) <= 159 else char for char in value[:2000])
    return " ".join(cleaned.split())[:limit]


def _candidate(feature) -> PlaceCandidate | None:
    if not isinstance(feature, dict):
        return None
    geometry, properties = feature.get("geometry"), feature.get("properties")
    if not isinstance(geometry, dict) or geometry.get("type") != "Point" or not isinstance(properties, dict):
        return None
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) != 2:
        return None
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in coordinates):
        return None
    lon, lat = coordinates
    if not is_in_astana(lat, lon):
        return None
    country = properties.get("countrycode")
    if country is not None and (not isinstance(country, str) or country.upper() != "KZ"):
        return None
    # The search rectangle also covers nearby settlements. A result explicitly
    # assigned to another city is not an Astana match, even inside the bbox.
    city = properties.get("city")
    if city is not None and (not isinstance(city, str) or
                             re.sub(r"[\W_]+", "", city.casefold()) not in _CITY_NAMES):
        return None
    osm_type = properties.get("osm_type")
    kind = {"N": "node", "W": "way", "R": "relation", "node": "node", "way": "way", "relation": "relation"}.get(
        osm_type if isinstance(osm_type, str) else ""
    )
    osm_id = properties.get("osm_id")
    if isinstance(osm_id, str) and osm_id.isascii() and osm_id.isdecimal() and len(osm_id) <= 20:
        osm_id = int(osm_id)
    if kind is None or not isinstance(osm_id, int) or isinstance(osm_id, bool) or not 0 < osm_id <= 2**63 - 1:
        return None
    street = " ".join(filter(None, [_text(properties.get("street")), _text(properties.get("housenumber"))]))
    address_parts = [street, *[_text(properties.get(field)) for field in ("district", "city", "state", "country")]]
    address = ", ".join(dict.fromkeys(part for part in address_parts if part))[:500]
    name = _text(properties.get("name")) or street
    if not name:
        return None
    identity = f"{kind}/{osm_id}"
    return PlaceCandidate(
        id=identity, name=name, lat=float(lat), lon=float(lon), address=address,
        source_url=f"https://www.openstreetmap.org/{identity}",
        kind=_text(properties.get("osm_value") or properties.get("type"), limit=80),
    )


def search_places(query: str) -> list[PlaceCandidate]:
    """Search one place phrase; [] is no valid match, GeocoderError is unavailable.

    Only a user-initiated search may call this function. No bulk collection,
    retries, automatic query expansion or model-provided URLs are supported.
    A request may wait up to one second for a rate slot; minute limits fail fast.
    Shared deployments need a shared limiter.
    """
    if os.environ.get("SERUEN_GEOCODING", "1").strip() == "0":
        raise GeocoderError("disabled")
    phrase, endpoint = _phrase(query), _endpoint()
    key = endpoint, phrase.casefold()
    cached = _cached(key)
    if cached is not None:
        return list(cached)
    _reserve_request()
    try:
        response = requests.get(
            endpoint,
            # Do not force lang=ru: the public server supports default/de/en/fr.
            # Omitting lang uses server defaults and the local OSM name.
            params={"q": phrase, "bbox": ",".join(str(value) for value in ASTANA_BBOX),
                    "countrycode": "KZ", "limit": RESULT_LIMIT},
            headers={"User-Agent": "Seruen-AI/0.1 (interactive place search)", "Accept": "application/json"},
            timeout=TIMEOUT, allow_redirects=False,
        )
    except requests.RequestException:
        raise GeocoderError("network") from None
    try:
        if response.status_code != 200:
            raise GeocoderError("rate_limited" if response.status_code == 429 else "unavailable")
        try:
            payload = response.json()
        except (ValueError, requests.RequestException):
            raise GeocoderError("malformed") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("features"), list):
            raise GeocoderError("malformed")
        candidates: dict[str, PlaceCandidate] = {}
        for feature in payload["features"][:64]:
            candidate = _candidate(feature)
            if candidate is not None:
                candidates.setdefault(candidate.id, candidate)
            if len(candidates) >= RESULT_LIMIT:
                break
        results = tuple(candidates.values())
    finally:
        response.close()
    with _LOCK:
        _CACHE[key] = time.monotonic(), results
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_LIMIT:
            _CACHE.popitem(last=False)
    return list(results)
