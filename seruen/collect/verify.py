"""Verify the committed dataset and its recorded raw inputs without network access.

Usage: python -m seruen.collect.verify [--data data] [--json]
Wikipedia enrichment is checked for attribution, not silently re-fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from seruen.collect.normalize import dedup, normalize_element
from seruen.collect.overpass import parse_response


def _resolve(data_dir: Path, recorded: str) -> Path:
    path = Path(recorded)
    return path if path.is_absolute() else data_dir.parent / path


def verify(data_dir: Path) -> dict:
    """Return errors and measurable counts; do not mutate the dataset."""
    errors: list[str] = []
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    report = json.loads((data_dir / "quality_report.json").read_text(encoding="utf-8"))
    pois = json.loads((data_dir / "pois.json").read_text(encoding="utf-8"))
    entries = metadata.get("runs") or [metadata.get("latest", {})]
    for run in entries:
        path = _resolve(data_dir, run["file"])
        if not path.is_file():
            errors.append(f"Missing raw dump: {path}")
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != run.get("sha256"):
            errors.append(f"SHA-256 mismatch: {path}")
        if len(raw) != run.get("bytes", len(raw)):
            errors.append(f"Byte count mismatch: {path}")
        payload = parse_response(raw)
        if len(payload["elements"]) != run.get("elements", len(payload["elements"])):
            errors.append(f"Element count mismatch: {path}")

    ids = [poi["id"] for poi in pois]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate POI identifiers")

    raw_path = _resolve(data_dir, report["raw_file"])
    raw_bytes = raw_path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    matching = [r for r in entries if r.get("sha256") == digest]
    if not matching:
        errors.append("Dataset raw input has no matching SHA-256 in metadata")
    raw_payload = parse_response(raw_bytes)
    rebuilt, removed = dedup([
        record for element in raw_payload["elements"]
        if (record := normalize_element(element)) is not None
    ])
    current = {poi["id"]: poi for poi in pois}
    expected = {poi["id"]: poi for poi in rebuilt}
    if current.keys() != expected.keys():
        errors.append("POI IDs differ from normalization of the recorded raw input")
    enrichment_fields = {"description", "description_source", "evidence", "source"}
    for poi_id in current.keys() & expected.keys():
        poi, baseline = current[poi_id], expected[poi_id]
        changed = [key for key, value in baseline.items()
                   if key not in enrichment_fields and poi.get(key) != value]
        if changed:
            errors.append(f"{poi_id}: normalized fields changed: {', '.join(changed)}")
        source = poi.get("source", {})
        if source.get("url") != baseline["source"]["url"] or source.get("license") != "ODbL":
            errors.append(f"{poi_id}: missing or inconsistent OSM attribution")
        if matching and source.get("fetched_at") != matching[-1].get("fetched_at"):
            errors.append(f"{poi_id}: acquisition time differs from raw provenance")
        if poi.get("description_source") == "wikipedia":
            if not any(e.get("field") == "description" and ".wikipedia.org/wiki/" in e.get("source_url", "")
                       for e in poi.get("evidence", [])):
                errors.append(f"{poi_id}: Wikipedia description has no article attribution")
        elif poi.get("description") != baseline["description"] or poi.get("description_source") != "osm_tags":
            errors.append(f"{poi_id}: unaccounted description change")

    counts = {
        "raw_elements": len(raw_payload["elements"]),
        "pois_total": len(pois),
        "dropped_duplicates": removed,
        "with_hours": sum(bool(poi.get("hours_raw")) for poi in pois),
        "with_hours_pct": round(100 * sum(bool(poi.get("hours_raw")) for poi in pois) / max(1, len(pois)), 1),
        "with_website": sum(bool(poi.get("website")) for poi in pois),
        "with_phone": sum(bool(poi.get("phone")) for poi in pois),
        "with_wikidata": sum(bool(poi.get("wikidata")) for poi in pois),
        "wheelchair_known": sum(poi.get("wheelchair") != "unknown" for poi in pois),
        "wheelchair_yes": sum(poi.get("wheelchair") == "yes" for poi in pois),
        "name_generated": sum(bool(poi.get("name_generated")) for poi in pois),
        "indoor_known": sum(poi.get("indoor") is not None for poi in pois),
        "by_category": dict(Counter(poi["category"] for poi in pois)),
        "with_wikipedia_description": sum(poi.get("description_source") == "wikipedia" for poi in pois),
    }
    for key, value in counts.items():
        if key in report and report[key] != value:
            errors.append(f"Quality report mismatch: {key}: recorded {report[key]!r}, actual {value!r}")
    return {
        "ok": not errors,
        "raw_dumps_checked": len(entries),
        "raw_sha256": digest,
        "counts": counts,
        "errors": errors,
        "limitations": [
            "Normalization is reproducible from saved OSM input; a new live fetch may differ.",
            "Existing Wikipedia excerpts retain source links, but not revision snapshots.",
            "Hashes verify file integrity, not the accuracy or freshness of place information.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline dataset integrity and provenance check")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = verify(args.data)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["ok"]:
        print(f"Dataset verified: {result['counts']['pois_total']} POIs; {result['raw_dumps_checked']} raw dumps; SHA-256 OK")
        for limitation in result["limitations"]:
            print(f"- {limitation}")
    else:
        for error in result["errors"]:
            print(f"ERROR: {error}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
