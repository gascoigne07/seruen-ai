"""Collect raw POI data for Astana from OpenStreetMap via the Overpass API.

Stdlib only on purpose: the collection step must be reproducible with a bare
Python install, without first building a virtualenv.

Usage:
    python -m seruen.collect.overpass [--out data/raw] [--timeout 180]

Output:
    data/raw/overpass_<UTC timestamp>.json   raw Overpass response
    data/metadata.json                      provenance (mirror, sha256, counts)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# bbox (South, West, North, East) — Astana
BBOX = (51.02, 71.25, 51.25, 71.65)

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# key -> accepted values; only elements carrying a `name` tag are collected
TAG_GROUPS: dict[str, list[str]] = {
    "amenity": [
        "cafe",
        "restaurant",
        "fast_food",
        "bar",
        "pub",
        "library",
        "theatre",
        "cinema",
        "arts_centre",
        "ice_cream",
    ],
    "tourism": [
        "museum",
        "gallery",
        "attraction",
        "viewpoint",
        "artwork",
        "zoo",
        "theme_park",
    ],
    "leisure": [
        "park",
        "garden",
        "playground",
        "sports_centre",
        "ice_rink",
        "water_park",
    ],
    "historic": ["monument", "memorial"],
    "shop": ["mall"],
}

# Children's playgrounds are the one category OSM almost never names: filtering
# on `name` leaves a single playground in all of Astana. They are collected
# without that filter and labelled by street in normalize.py.
UNNAMED_GROUPS: dict[str, list[str]] = {"leisure": ["playground"]}

USER_AGENT = "SeruenAI/0.1 (hackathon POI collector; contact: local use only)"


def build_query(timeout: int = 180) -> str:
    """Overpass QL for every POI of interest inside the Astana bbox."""
    s, w, n, e = BBOX
    bbox = f"({s},{w},{n},{e})"
    parts = []
    for key, values in TAG_GROUPS.items():
        # the regex form also covers single-value groups, so there is no branch here
        selector = f'["{key}"~"^({"|".join(values)})$"]'
        parts.append(f'  nwr["name"]{selector}{bbox};')
    for key, values in UNNAMED_GROUPS.items():
        selector = f'["{key}"~"^({"|".join(values)})$"]'
        parts.append(f'  nwr{selector}{bbox};')
    body = "\n".join(parts)
    # `out center tags` gives a single representative point for ways/relations
    return f"[out:json][timeout:{timeout}];\n(\n{body}\n);\nout center tags;\n"


def parse_response(raw: bytes) -> dict:
    """Reject error/partial replies, including Overpass errors sent with HTTP 200."""
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Overpass response must be a JSON object")
    if payload.get("remark"):
        raise ValueError(f"Overpass returned an error or incomplete result: {payload['remark']}")
    elements = payload.get("elements")
    if not isinstance(elements, list) or not all(isinstance(e, dict) for e in elements):
        raise ValueError("Overpass response must contain an elements array of objects")
    if not elements:
        raise ValueError("Overpass returned no POIs; refusing to replace the Astana dataset")
    return payload


def fetch(query: str, timeout: int = 180, attempts: int = 3) -> tuple[str, bytes]:
    """POST the query to each mirror in turn. Returns (mirror_url, raw body)."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        for mirror in MIRRORS:
            print(f"[overpass] attempt {attempt}/{attempts} -> {mirror}", flush=True)
            request = urllib.request.Request(
                mirror,
                data=urllib.parse.urlencode({"data": query}).encode("utf-8"),
                headers={
                    "User-Agent": USER_AGENT,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=timeout + 30) as response:
                    raw = response.read()
                parse_response(raw)
                took = time.time() - started
                print(f"[overpass] ok: {len(raw) / 1024:.0f} KiB in {took:.0f}s", flush=True)
                return mirror, raw
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
                last_error = exc
                print(f"[overpass] failed after {time.time() - started:.0f}s: {exc}", flush=True)
                time.sleep(3)
    raise RuntimeError(f"all Overpass mirrors failed: {last_error}")


def write_metadata(meta_path: Path, entry: dict) -> None:
    """Append one collection run to data/metadata.json."""
    data: dict = {"runs": []}
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("runs", []), list):
            raise ValueError("metadata must contain a runs array; refusing to discard provenance")
    data.setdefault("runs", []).append(entry)
    data["latest"] = entry
    data["license"] = "ODbL 1.0 — © OpenStreetMap contributors"
    data["source"] = "OpenStreetMap via Overpass API"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect Astana POIs from Overpass")
    parser.add_argument("--out", default="data/raw", help="directory for raw dumps")
    parser.add_argument("--meta", default="data/metadata.json")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--print-query", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout < 1 or args.attempts < 1:
        parser.error("--timeout and --attempts must be positive")

    query = build_query(args.timeout)
    if args.print_query:
        print(query)
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mirror, raw = fetch(query, timeout=args.timeout, attempts=args.attempts)

    try:
        parsed = parse_response(raw)
    except ValueError as exc:
        print(f"[overpass] invalid response: {exc}", file=sys.stderr)
        return 1
    elements = parsed["elements"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = out_dir / f"overpass_{stamp}.json"
    raw_path.write_bytes(raw)

    entry = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mirror": mirror,
        "file": str(raw_path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "elements": len(elements),
        "bbox": {"south": BBOX[0], "west": BBOX[1], "north": BBOX[2], "east": BBOX[3]},
        "query": query,
    }
    write_metadata(Path(args.meta), entry)

    print(f"[overpass] {len(elements)} elements -> {raw_path}")
    print(f"[overpass] sha256 {entry['sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
