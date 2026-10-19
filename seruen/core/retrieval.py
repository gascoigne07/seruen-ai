"""Retrieval: the R in RAG.

Pipeline (contract from CLAUDE.md):
    intent -> radius from time budget -> hard filters -> scoring -> top 8 cards.

Only the cards produced here ever reach the language model.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from seruen.core import geo
from seruen.core.hours import CLOSED, OPEN, UNKNOWN, HoursResult, hours_context, hours_status
from seruen.core.intent import Intent

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "data" / "pois.json"
TOP_K = 8

# scoring weights — kept in one place so the UI can explain a ranking
W_CATEGORY = 3.0
W_BM25 = 1.0
W_PROXIMITY = 1.0
W_OPEN = 0.5
W_WEATHER = 0.8
W_CONTEXT = 1.0

CHILD_FRIENDLY = {"playground": 1.0, "park": 0.8, "mall": 0.5, "museum": 0.3, "sport": 0.3}
QUIET_FRIENDLY = {"library": 1.0, "park": 0.5, "museum": 0.4, "cafe": 0.2}
UNUSUAL_FRIENDLY = {"viewpoint": 0.8, "museum": 0.6, "landmark": 0.5, "sport": 0.4, "theatre": 0.3}
NIGHT_FRIENDLY = {"restaurant": 0.6, "cafe": 0.4, "mall": 0.3, "viewpoint": 0.3, "theatre": 0.2}
GROUP_FRIENDLY = {"mall": 0.3, "park": 0.3, "restaurant": 0.3, "theatre": 0.2}

ADULT_ONLY_TAGS = {"бар", "паб", "пиво", "bar", "pub", "nightclub", "казино"}


@dataclass
class Candidate:
    """A POI that survived the filters, with everything the model may see."""

    poi: dict
    distance_m: float
    walk_min: float
    hours: HoursResult
    score: float = 0.0
    score_parts: dict = field(default_factory=dict)
    poi_index: int = -1  # position in the dataset, to look up its BM25 score

    @property
    def id(self) -> str:
        return self.poi["id"]

    def name(self, language: str = "ru") -> str:
        names = self.poi.get("name") or {}
        return names.get(language) or names.get("ru") or names.get("en") or "—"

    def card(self, language: str = "ru") -> dict:
        """The only representation handed to the generator. No user coordinates."""
        return {
            "poi_id": self.poi["id"],
            "name": self.name(language),
            "category": self.poi["category"],
            "description": (self.poi.get("description") or "") if self.poi.get("description_source") != "generated" else "",
            "tags": self.poi.get("tags") or [],
            "hours_status": self.hours.status,
            "hours_raw": self.hours.raw,
            "hours_context": hours_context(self.poi, self.hours),
            "distance_m": round(self.distance_m),
            "walk_min": math.ceil(self.walk_min),
            "indoor": self.poi.get("indoor"),
            "wheelchair": self.poi.get("wheelchair", "unknown"),
        }

    def uncertainties(self, language: str = "ru") -> list[str]:
        """Unverified amenities never become factual promises in a card."""
        messages = {
            "generated_name": ("название дано нами: в OSM объект без имени", "name assigned by us: unnamed OSM object", "атауды біз бердік: OSM-де атауы жоқ"),
            "hours": ("часы работы не подтверждены", "opening hours not confirmed", "жұмыс уақыты расталмаған"),
            "access": ("доступность на коляске не размечена", "wheelchair access is not documented", "арбамен кіру мүмкіндігі белгіленбеген"),
            "limited": ("доступность на коляске ограничена", "wheelchair access is limited", "арбамен кіру мүмкіндігі шектеулі"),
            "amenities": ("розетки и уровень шума в данных не размечены", "power sockets and noise level are not documented", "розеткалар мен шу деңгейі белгіленбеген"),
            "indoor": ("размещение в помещении или на улице не подтверждено", "indoor/outdoor setting is not confirmed", "орынның іште не сыртта екені расталмаған"),
            "prices": ("цены и наличие свободных мест не подтверждены", "prices and availability are not confirmed", "бағалар мен бос орындар расталмаған"),
        }
        keys = []
        if self.poi.get("name_generated"):
            keys.append("generated_name")
        if self.hours.status == UNKNOWN and hours_context(self.poi, self.hours) != "outdoor":
            keys.append("hours")
        if self.poi.get("wheelchair", "unknown") == "unknown":
            keys.append("access")
        elif self.poi.get("wheelchair") == "limited":
            keys.append("limited")
        if self.poi["category"] in {"cafe", "restaurant", "library"}:
            keys.append("amenities")
        if self.poi.get("indoor") is None:
            keys.append("indoor")
        if self.poi["category"] in {"cafe", "restaurant", "museum", "theatre", "sport", "viewpoint"}:
            keys.append("prices")
        index = {"ru": 0, "en": 1, "kk": 2}.get(language, 0)
        return [messages[key][index] for key in keys]


@lru_cache(maxsize=4)
def load_pois(path: str | Path = DEFAULT_DATASET) -> tuple[dict, ...]:
    """Load the dataset once and cache it (tuple so the cache key is hashable)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("POI dataset must be a JSON list")
    return tuple(data)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\-]{2,}", text.lower(), flags=re.UNICODE)


def _poi_document(poi: dict) -> list[str]:
    names = poi.get("name") or {}
    parts = [
        names.get("ru", ""),
        names.get("en", ""),
        names.get("kk", ""),
        poi.get("description", ""),
        " ".join(poi.get("tags") or []),
        poi.get("category", ""),
        poi.get("osm_subtype") or "",
    ]
    return _tokenize(" ".join(parts))


class _Bm25Index:
    """Thin wrapper over rank_bm25 with a pure-python fallback."""

    def __init__(self, pois: tuple[dict, ...]):
        self.docs = [_poi_document(poi) for poi in pois]
        self.index = None
        try:
            from rank_bm25 import BM25Okapi

            self.index = BM25Okapi(self.docs)
        except Exception:  # noqa: BLE001 — degrade to term overlap, never crash retrieval
            self.index = None

    def scores(self, query_tokens: list[str]) -> list[float]:
        if not query_tokens:
            return [0.0] * len(self.docs)
        if self.index is not None:
            return list(self.index.get_scores(query_tokens))
        # fallback: fraction of query terms present in the document
        result = []
        for doc in self.docs:
            doc_set = set(doc)
            hits = sum(1 for token in query_tokens if token in doc_set)
            result.append(hits / len(query_tokens))
        return result


_INDEX_CACHE: dict[int, tuple[tuple[dict, ...], _Bm25Index]] = {}


def _get_index(pois: tuple[dict, ...]) -> _Bm25Index:
    """Cache by object identity — a tuple of dicts is not hashable."""
    key = id(pois)
    entry = _INDEX_CACHE.get(key)
    if entry is not None and entry[0] is pois:
        return entry[1]
    index = _Bm25Index(pois)
    if len(_INDEX_CACHE) >= 4:
        _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)))
    # Retain the tuple itself, so a recycled Python id can never address old data.
    _INDEX_CACHE[key] = (pois, index)
    return index


@dataclass
class RetrievalResult:
    candidates: list[Candidate]
    radius_m: float
    stats: dict

    @property
    def unknown_hours_count(self) -> int:
        return sum(1 for c in self.candidates if c.hours.status == UNKNOWN)


def _context_bonus(poi: dict, intent: Intent) -> float:
    """Soft preferences: child, quiet, sockets, night, group, unusual, wheelchair."""
    category = poi["category"]
    tags = {t.casefold() for t in (poi.get("tags") or [])}
    bonus = 0.0

    if intent.with_child:
        bonus += CHILD_FRIENDLY.get(category, 0.0)
        if tags & ADULT_ONLY_TAGS:
            bonus -= 1.5
        if "для детей" in tags:
            bonus += 0.5
    if intent.wants_quiet:
        bonus += QUIET_FRIENDLY.get(category, 0.0)
        if category == "mall" or "фастфуд" in tags:
            bonus -= 0.5
        if "без курения" in tags:
            bonus += 0.2
    if intent.wants_sockets:
        if "wi-fi" in tags:
            bonus += 0.8
        if category in {"library", "cafe"}:
            bonus += 0.3
        if "места внутри" in tags:
            bonus += 0.2
    if intent.wants_unusual:
        bonus += 2.5 * UNUSUAL_FRIENDLY.get(category, 0.0)
    if intent.night:
        if not intent.wants_unusual:
            bonus += NIGHT_FRIENDLY.get(category, 0.0)
        if "круглосуточно" in tags:
            bonus += 1.0
    if intent.group_size >= 4:
        bonus += GROUP_FRIENDLY.get(category, 0.0)
    if intent.needs_wheelchair:
        wheelchair = poi.get("wheelchair", "unknown")
        if wheelchair == "yes":
            bonus += 2.0
        elif wheelchair == "limited":
            bonus += 0.5
    indoor = poi.get("indoor")
    if intent.indoor_pref and indoor is not None:
        wanted_indoor = intent.indoor_pref == "indoor"
        bonus += 1.2 if wanted_indoor is bool(indoor) else -0.8
    return bonus


def _weather_bonus(poi: dict, weather: dict | None) -> float:
    if not weather or not weather.get("used"):
        return 0.0
    indoor = poi.get("indoor")
    if indoor is None:
        return 0.0
    if weather.get("indoor_preferred"):
        return 1.0 if indoor else -0.6
    # pleasant weather: nudge towards being outside
    return 0.3 if indoor is False else 0.0


def retrieve(
    intent: Intent,
    lat: float,
    lon: float,
    when: datetime,
    pois: tuple[dict, ...] | None = None,
    weather: dict | None = None,
    top_k: int = TOP_K,
    dataset_path: str | Path = DEFAULT_DATASET,
) -> RetrievalResult:
    """Geofilter + hard constraints + BM25 + category score -> top candidates."""
    all_pois = pois if pois is not None else load_pois(dataset_path)
    index = _get_index(all_pois)

    query_tokens = _tokenize(" ".join(intent.keywords) or intent.raw_query)
    bm25_scores = index.scores(query_tokens)
    bm25_max = max([0.0, *bm25_scores])

    visit_min = geo.typical_visit_minutes(intent.categories_wanted)
    minimum_visit = min(geo.minimum_visit_minutes(category, wants_work=intent.wants_work)
                        for category in (intent.categories_wanted or ["other"]))
    radius_m = geo.radius_from_budget(
        intent.time_budget_min, minimum_visit, return_to_start=intent.return_to_start
    )

    stats = {
        "dataset_size": len(all_pois),
        "radius_m": round(radius_m),
        "typical_visit_min": visit_min,
        "minimum_visit_min": minimum_visit,
        "excluded_too_far": 0,
        "excluded_closed": 0,
        "excluded_no_time": 0,
        "excluded_category": 0,
        "excluded_wheelchair": 0,
        "excluded_indoor": 0,
        "excluded_child": 0,
        "excluded_closing": 0,
        "unknown_hours_kept": 0,
        "passed_filters": 0,
        "relaxed": False,
    }

    def collect(current_radius: float, enforce_category: bool) -> list[Candidate]:
        found: list[Candidate] = []
        for poi_index, poi in enumerate(all_pois):
            if poi["category"] in intent.categories_avoided or (
                enforce_category and intent.categories_wanted and poi["category"] not in intent.categories_wanted
            ):
                stats["excluded_category"] += 1
                continue
            if intent.indoor_pref and poi.get("indoor") is not (intent.indoor_pref == "indoor"):
                stats["excluded_indoor"] += 1
                continue
            tags = {str(tag).casefold() for tag in poi.get("tags", [])}
            if intent.with_child and (tags & ADULT_ONLY_TAGS or poi.get("osm_subtype") in {"bar", "pub", "nightclub"}):
                stats["excluded_child"] += 1
                continue
            distance = geo.haversine_m(lat, lon, poi["lat"], poi["lon"])
            if distance > current_radius:
                stats["excluded_too_far"] += 1
                continue
            status = hours_status(poi.get("hours_raw"), when)
            if status.status == CLOSED:
                stats["excluded_closed"] += 1
                continue
            if intent.needs_wheelchair and poi.get("wheelchair") != "yes":
                stats["excluded_wheelchair"] += 1
                continue
            poi_visit = geo.minimum_visit_minutes(poi["category"], wants_work=intent.wants_work)
            if not geo.fits_in_budget(
                distance, poi_visit, intent.time_budget_min, intent.return_to_start
            ):
                stats["excluded_no_time"] += 1
                continue
            walk_min = geo.walking_minutes(distance)
            arrival = when + timedelta(minutes=math.ceil(walk_min))
            arrival_status = hours_status(poi.get("hours_raw"), arrival)
            if arrival_status.status == CLOSED or (
                arrival_status.minutes_until_close is not None
                and arrival_status.minutes_until_close < poi_visit + geo.BUFFER_MIN
            ):
                stats["excluded_closing"] += 1
                continue
            if status.status == UNKNOWN:
                stats["unknown_hours_kept"] += 1
            found.append(
                Candidate(
                    poi=poi,
                    distance_m=distance,
                    walk_min=geo.walking_minutes(distance),
                    hours=status,
                    poi_index=poi_index,
                )
            )
        return found

    passed = collect(radius_m, enforce_category=True)
    if len(passed) < 3:
        # Expand only the distance window; all time and explicit context filters stay hard.
        stats["relaxed"] = True
        radius_m = min(geo.MAX_RADIUS_M, radius_m * 1.8)
        stats["radius_m"] = round(radius_m)
        for counter in (
            "excluded_too_far",
            "excluded_closed",
            "excluded_no_time",
            "excluded_category",
            "excluded_wheelchair",
            "excluded_indoor",
            "excluded_child",
            "excluded_closing",
            "unknown_hours_kept",
        ):
            stats[counter] = 0  # the second pass replaces the first, not adds to it
        passed = collect(radius_m, enforce_category=True)

    for candidate in passed:
        poi = candidate.poi
        category_match = 0.0
        if intent.categories_wanted:
            category_match = 1.0 if poi["category"] in intent.categories_wanted else 0.0
        bm25_raw = bm25_scores[candidate.poi_index] if bm25_max else 0.0
        bm25_norm = max(0.0, bm25_raw / bm25_max) if bm25_max else 0.0
        proximity = max(0.0, 1.0 - candidate.distance_m / max(1.0, radius_m))
        open_bonus = 1.0 if candidate.hours.status == OPEN else 0.0
        context = _context_bonus(poi, intent)
        if intent.night and candidate.hours.status == UNKNOWN and poi.get("indoor") is True:
            context -= 2.0  # an unconfirmed indoor venue is a weak late-night suggestion
        weather_fit = _weather_bonus(poi, weather)

        candidate.score_parts = {
            "category": round(W_CATEGORY * category_match, 3),
            "bm25": round(W_BM25 * bm25_norm, 3),
            "proximity": round(W_PROXIMITY * proximity, 3),
            "open": round(W_OPEN * open_bonus, 3),
            "context": round(W_CONTEXT * context, 3),
            "weather": round(W_WEATHER * weather_fit, 3),
        }
        candidate.score = round(sum(candidate.score_parts.values()), 3)

    passed.sort(key=lambda c: (-c.score, c.distance_m))
    stats["passed_filters"] = len(passed)
    limit = max(0, min(TOP_K, top_k))
    # A dense restaurant dataset must not erase the other matching categories
    # before generation gets a chance to choose a varied answer.
    representatives = diversify(passed, limit=min(3, limit), max_per_category=1)
    selected_ids = {candidate.id for candidate in representatives}
    top = (representatives + [candidate for candidate in passed if candidate.id not in selected_ids])[:limit]
    top.sort(key=lambda c: (-c.score, c.distance_m))
    stats["returned"] = len(top)
    return RetrievalResult(candidates=top, radius_m=radius_m, stats=stats)


def diversify(candidates: list[Candidate], limit: int = 3, max_per_category: int = 2) -> list[Candidate]:
    """Pick up to `limit` candidates without repeating a category or a name.

    The name check matters for objects we labelled ourselves: two cards both
    reading "Детская площадка" are useless to the user.
    """
    if limit <= 0 or max_per_category <= 0:
        return []
    chosen: list[Candidate] = []
    used: dict[str, int] = {}
    seen_names: set[str] = set()
    # First use distinct categories; only repeat when alternatives run out.
    for category_cap in range(1, max_per_category + 1):
        for candidate in candidates:
            category = candidate.poi["category"]
            name = candidate.name().casefold()
            if used.get(category, 0) >= category_cap or name in seen_names:
                continue
            chosen.append(candidate)
            seen_names.add(name)
            used[category] = used.get(category, 0) + 1
            if len(chosen) >= limit:
                return chosen

    return chosen


def summarize_hours(candidates: list[Candidate]) -> dict:
    counts = {OPEN: 0, CLOSED: 0, UNKNOWN: 0}
    for candidate in candidates:
        counts[candidate.hours.status] = counts.get(candidate.hours.status, 0) + 1
    return counts


def bearing_label(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """Rough compass direction, used in templated explanations."""
    d_lon = math.radians(lon2 - lon1)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    y = math.sin(d_lon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lon)
    degrees = (math.degrees(math.atan2(y, x)) + 360) % 360
    labels = ["к северу", "к северо-востоку", "к востоку", "к юго-востоку",
              "к югу", "к юго-западу", "к западу", "к северо-западу"]
    return labels[int((degrees + 22.5) % 360 // 45)]
