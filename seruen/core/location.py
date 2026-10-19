"""Resolve textual starting landmarks from the local POI dataset only.

Coordinates are OSM point/area centres, not GPS readings or entrance locations.
An explicit unknown or ambiguous origin must never silently become Baiterek.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from seruen.core.retrieval import load_pois

Origin = tuple[str, float, float]

ALIASES = {
    "mega": r"mega(?:\s+silk\s+way)?|мега|silk\s*way|силк(?:\s+вей)?",
    "airport": r"аэропорт(?:а|е|у)?|әуежай(?:да|дың)?|airport",
    "expo": r"expo|экспо|н[ұу]р\s*алем",
    "khan": r"хан\s*шат[ыи]р(?:а|е)?|khan\s*shatyr",
    "nu": r"назарбаев\s*университет(?:а|е)?|nazarbayev\s*university|nu",
    "baiterek": r"байтерек(?:а|е|ом)?|бәйтерек(?:тің|те)?|baiterek|bayterek",
    "botanical": r"ботаническ(?:ий|ого|ом|им)\s+сад(?:а|у|е|ом)?|ботаникалық\s+бақ(?:тың|та)?|botanical\s+garden",
}

_ORIGIN_PREFIX = re.compile(
    r"(?<!\w)(?:возле|около|рядом\s+(?:с|со)|у|"
    r"(?:я|мы)\s+(?:(?:вот|сейчас|тут|уже)\s+){0,3}(?:в|на)|сейчас\s+(?:в|на)|"
    r"нахожусь\s+(?:в|на)|находимся\s+(?:в|на)|near|at|by|from)\s+",
    re.IGNORECASE,
)
_KAZAKH_ORIGIN = re.compile(r"([^,;.!?]+?)\s+(?:жанында\w*|қасында\w*)", re.IGNORECASE)
_NON_LOCATION_START = re.compile(
    r"(?:меня|нас|него|нее|них|вас|тебя|мной|нами|дома|воды|водой|реки|рекой|озера|озером|"
    r"me|us|here|home|night|noon|midnight|least|most|foot|bike|bus|car|now|today|water|"
    r"(?:about\s+)?(?:один|одного|одну|два|две|двух|три|трех|трёх|четыре|четырех|пять|пяти|"
    r"десять|десяти|двадцать|двадцати|тридцать|тридцати|сорок|сорока|"
    r"полтора|полутора|пары|несколько|нескольких|one|two|three|four|five|ten|twenty|thirty|forty|half\s+an|an|a)\s+"
    r"(?:час\w*|минут\w*|hours?|minutes?)|полчаса|получаса|"
    r"\d+(?:[.,]\d+)?\s*(?:мин|час|minutes?|hours?)|\d{1,2}:\d{2}|часа?)(?!\w)",
    re.IGNORECASE,
)
_GENERIC_PREFIXES = {"the", "a", "кафе", "кофейня", "кофейн", "ресторан", "парк", "музей", "музе", "тц", "трц", "cafe", "restaurant"}
_REQUEST_BOUNDARY = re.compile(r"[,;.!?]|\b(?:куда|где|что|хочу|хотим|хочется|можно|where|what|want|wants)\b", re.IGNORECASE)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[^\W_]+", value.casefold().replace("ё", "е"), re.UNICODE)


def _stem(token: str) -> str:
    if not re.fullmatch(r"[а-я]+", token) or len(token) < 4:
        return token
    # Conservative grammatical endings, not fuzzy spelling correction.
    return re.sub(r"(?:ого|его|ому|ему|ыми|ими|ых|их|ая|яя|ое|ее|ые|ие|ый|ий|ой|ую|юю|ом|ем|ым|им|а|я|у|ю|е|ы|и|й|ь)$", "", token)


def _same_token(query: str, name: str) -> bool:
    if query == name or _stem(query) == _stem(name):
        return True
    # Kazakh postposition phrases commonly put the landmark in the genitive.
    return len(name) >= 3 and query in {name + ending for ending in ("ның", "нің", "дың", "дің", "тың", "тің", "да", "де", "та", "те")}


def _name_match(fragment: str, name: str) -> tuple[int, int]:
    token_spans = list(re.finditer(r"[^\W_]+", fragment, re.UNICODE))
    query_tokens, name_tokens = _tokens(fragment), _tokens(name)
    if not name_tokens:
        return 0, 0
    starts = [0]
    for index, token in enumerate(query_tokens[:2]):
        if token in _GENERIC_PREFIXES or _stem(token) in _GENERIC_PREFIXES:
            starts.append(index + 1)
        else:
            break
    matched = (0, 0)
    for start in starts:
        tokens = query_tokens[start:]
        count = min(len(tokens), len(name_tokens))
        if not count or not all(_same_token(tokens[i], name_tokens[i]) for i in range(count)):
            continue
        # A complete short name or a distinctive two-word prefix of a longer
        # landmark is useful. A single generic “музей” is not a unique address.
        if count == len(name_tokens) or (count >= 2 and count == len(tokens)):
            matched = max(matched, (count, token_spans[start + count - 1].end()))
    return matched


def _resolve_fragment(fragment: str, presets: Mapping[str, Origin], pois: Sequence[dict]) -> tuple[Origin, int]:
    stripped = fragment.lstrip(' \t\n"«»')
    leading = len(fragment) - len(stripped)
    fragment = stripped
    for key, pattern in ALIASES.items():
        match = re.match(r"(?:the\s+)?(?:" + pattern + r")(?!\w)", fragment, re.IGNORECASE)
        if key in presets and match:
            return presets[key], leading + match.end()

    fragment = _REQUEST_BOUNDARY.split(fragment, maxsplit=1)[0].strip()
    matches: dict[str, tuple[dict, int]] = {}
    longest = 0
    for poi in pois:
        if poi.get("name_generated"):
            continue
        for name in (poi.get("name") or {}).values():
            if not isinstance(name, str):
                continue
            length, end = _name_match(fragment, name)
            if not length:
                continue
            if length > longest:
                longest, matches = length, {}
            if length == longest:
                matches[poi["id"]] = poi, leading + end
    if len(matches) == 1:
        poi, end = next(iter(matches.values()))
        names = poi.get("name") or {}
        return (names.get("ru") or names.get("kk") or names.get("en"), poi["lat"], poi["lon"]), end
    if matches:
        raise ValueError("Найдено несколько мест с таким названием. Уточните точку старта: выберите ориентир или укажите координаты.")
    raise ValueError("Не удалось определить указанное место по нашему каталогу Астаны. Выберите точку старта или укажите координаты.")


def _origin_fragments(query: str) -> list[tuple[int, int, str, int | None]]:
    origins = []
    for match in _ORIGIN_PREFIX.finditer(query):
        fragment = query[match.end():]
        if not fragment or _NON_LOCATION_START.match(fragment):
            continue
        origins.append((match.start(), match.end(), fragment, None))
    for match in _KAZAKH_ORIGIN.finditer(query):
        fragment = re.sub(r"^\s*(?:мен|біз)\s+", "", match.group(1), flags=re.IGNORECASE)
        start = match.start(1) + len(match.group(1)) - len(fragment)
        origins.append((match.start(), start, fragment, match.end()))
    return origins


def strip_origin_for_intent(query: str) -> str:
    """Hide only a recognized starting landmark from category/preference parsing."""
    origins = _origin_fragments(query)
    if not origins:
        return query
    start, name_start, fragment, suffix_end = min(origins)
    try:
        _, name_end = _resolve_fragment(fragment, {key: (key, 0, 0) for key in ALIASES}, load_pois())
    except (ValueError, OSError):
        return query
    end = suffix_end if suffix_end is not None else name_start + name_end
    return query[:start] + " " + query[end:]


def infer_origin(query: str, presets: Mapping[str, Origin], *, pois: Sequence[dict] | None = None) -> Origin | None:
    """Return a named origin, None without a location, or a clear selection error."""
    origins = _origin_fragments(query)
    if origins:
        return _resolve_fragment(min(origins)[2], presets, pois if pois is not None else load_pois())[0]

    # Preserve the established concise landmark input when no explicit origin
    # exists. A destination mentioned after an unknown explicit origin never
    # reaches this fallback.
    mentions = [(match.start(), key) for key, pattern in ALIASES.items() if key in presets
                for match in re.finditer(r"(?<!\w)(?:" + pattern + r")(?!\w)", query, re.IGNORECASE)]
    if mentions:
        return presets[min(mentions)[1]]
    return None
