"""Add real descriptions from Wikidata / Wikipedia to POIs that have one.

Two passes, because only a dozen POIs in Astana carry a `wikidata` tag:

1. direct: POI has `wikidata` -> entity -> its Wikipedia sitelink -> extract;
2. geosearch: ask ru.wikipedia for articles with coordinates inside the Astana
   bbox, then match an article to a POI by name and distance (<= 250 m).

Only `description` and `description_source` change; a POI never loses its OSM
description unless a better sourced one was found. Stdlib only.

Usage:
    python -m seruen.collect.enrich_wiki [--limit 400] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "SeruenAI/0.1 (hackathon POI enrichment; local use only)"

ASTANA_CENTER = (51.1283, 71.4305)
GEOSEARCH_RADIUS_M = 10000  # API maximum
MATCH_RADIUS_M = 250
LANGUAGES = ["ru", "kk", "en"]
MAX_DESCRIPTION_CHARS = 420
REQUEST_PAUSE_S = 0.5


def _get_json(url: str, params: dict, timeout: int = 20, attempts: int = 3) -> dict | None:
    """GET + parse JSON, backing off on 429 (anonymous Wikipedia rate limit)."""
    query = urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < attempts - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            return None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return None
    return None


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * 6_371_008.8 * math.asin(math.sqrt(a))


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").casefold()
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)


def _trim(text: str, limit: int = MAX_DESCRIPTION_CHARS) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if "." in cut:
        cut = cut[: cut.rfind(".") + 1]
    return cut.rstrip() + ("" if cut.endswith(".") else "…")


def fetch_extract(title: str, language: str) -> str | None:
    """Lead paragraph of a Wikipedia article."""
    payload = _get_json(
        WIKI_API.format(lang=language),
        {
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "redirects": 1,
            "format": "json",
            "titles": title,
        },
    )
    if not payload:
        return None
    pages = (payload.get("query") or {}).get("pages") or {}
    for page in pages.values():
        extract = (page.get("extract") or "").strip()
        if extract:
            return extract
    return None


def wikidata_sitelinks(entity_ids: list[str]) -> dict[str, dict[str, str]]:
    """{entity id: {language: article title}} for up to 50 entities per call."""
    result: dict[str, dict[str, str]] = {}
    for start in range(0, len(entity_ids), 50):
        chunk = entity_ids[start : start + 50]
        payload = _get_json(
            WIKIDATA_API,
            {
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "sitelinks",
                "format": "json",
            },
        )
        if not payload:
            continue
        for entity_id, entity in (payload.get("entities") or {}).items():
            links = {}
            for language in LANGUAGES:
                site = entity.get("sitelinks", {}).get(f"{language}wiki")
                if site and site.get("title"):
                    links[language] = site["title"]
            if links:
                result[entity_id] = links
        time.sleep(REQUEST_PAUSE_S)
    return result


def geosearch(language: str = "ru", limit: int = 500) -> list[dict]:
    """Wikipedia articles with coordinates around the centre of Astana."""
    payload = _get_json(
        WIKI_API.format(lang=language),
        {
            "action": "query",
            "list": "geosearch",
            "gscoord": f"{ASTANA_CENTER[0]}|{ASTANA_CENTER[1]}",
            "gsradius": GEOSEARCH_RADIUS_M,
            "gslimit": min(500, limit),
            "format": "json",
        },
    )
    if not payload:
        return []
    return (payload.get("query") or {}).get("geosearch") or []


def _set_description(poi: dict, extract: str, title: str, language: str) -> None:
    """Keep attribution and describe the transformation alongside the excerpt."""
    url = f"https://{language}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
    poi["description"] = _trim(extract)
    poi["description_source"] = "wikipedia"
    poi.setdefault("evidence", []).append({
        "field": "description",
        "source_url": url,
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors",
        "transformation": "plain text excerpt; whitespace normalized; possibly shortened",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def enrich(pois: list[dict], limit: int = 400, verbose: bool = True) -> dict:
    """Mutates `pois` in place. Returns a small report."""
    stats = {"by_wikidata": 0, "by_geosearch": 0, "attempted": 0, "articles_found": 0}

    # pass 1 — POIs that already point at a Wikidata entity
    tagged: dict[str, list[dict]] = {}
    for poi in pois:
        if poi.get("wikidata") and poi.get("description_source") != "wikipedia":
            tagged.setdefault(poi["wikidata"], []).append(poi)
    if tagged:
        sitelinks = wikidata_sitelinks(list(tagged))
        for entity_id, links in sitelinks.items():
            matching = tagged.get(entity_id)
            if not matching:
                continue
            for language in LANGUAGES:
                title = links.get(language)
                if not title:
                    continue
                stats["attempted"] += 1
                extract = fetch_extract(title, language)
                time.sleep(REQUEST_PAUSE_S)
                if not extract:
                    continue
                for poi in matching:
                    _set_description(poi, extract, title, language)
                    stats["by_wikidata"] += 1
                    if verbose:
                        print(f"[wiki] {poi['name']['ru']} <- {language}wiki/{title}")
                break

    # pass 2 — match nearby articles to POIs by name
    articles = geosearch("ru", limit=limit)
    stats["articles_found"] = len(articles)
    by_name: dict[str, list[dict]] = {}
    for poi in pois:
        if poi.get("description_source") == "wikipedia" or poi.get("name_generated"):
            continue
        by_name.setdefault(_norm(poi["name"]["ru"]), []).append(poi)
        english = _norm(poi["name"]["en"])
        if english and english != _norm(poi["name"]["ru"]):
            by_name.setdefault(english, []).append(poi)

    for article in articles:
        title = article.get("title") or ""
        candidates = [p for p in by_name.get(_norm(title), []) if p.get("description_source") != "wikipedia"]
        if not candidates:
            continue
        nearest = min(
            candidates,
            key=lambda poi: _haversine_m(poi["lat"], poi["lon"], article["lat"], article["lon"]),
        )
        distance = _haversine_m(nearest["lat"], nearest["lon"], article["lat"], article["lon"])
        if distance > MATCH_RADIUS_M:
            continue  # same name, different object
        stats["attempted"] += 1
        extract = fetch_extract(title, "ru")
        time.sleep(REQUEST_PAUSE_S)
        if not extract:
            continue
        _set_description(nearest, extract, title, "ru")
        stats["by_geosearch"] += 1
        if verbose:
            print(f"[geo ] {nearest['name']['ru']} <- ruwiki/{title} ({distance:.0f} м)")

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enrich POI descriptions from Wikipedia/Wikidata")
    parser.add_argument("--pois", default="data/pois.json")
    parser.add_argument("--report", default="data/quality_report.json")
    parser.add_argument("--limit", type=int, default=400, help="geosearch article limit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.pois)
    pois = json.loads(path.read_text(encoding="utf-8"))
    stats = enrich(pois, limit=args.limit)

    enriched = sum(1 for poi in pois if poi.get("description_source") == "wikipedia")
    print(
        f"[enrich] описаний из Wikipedia: {enriched} "
        f"(по wikidata: {stats['by_wikidata']}, по геопоиску: {stats['by_geosearch']}; "
        f"статей рядом: {stats['articles_found']})"
    )

    if args.dry_run:
        print("[enrich] dry-run: файл не изменён")
        return 0

    path.write_text(json.dumps(pois, ensure_ascii=False, indent=1), encoding="utf-8")

    report_path = Path(args.report)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["with_wikipedia_description"] = enriched
        report["enrichment"] = stats
        report["licenses"] = [
            "OpenStreetMap — ODbL 1.0, © OpenStreetMap contributors",
            "Wikidata — CC0",
            "Wikipedia extracts — CC BY-SA 4.0",
        ]
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
