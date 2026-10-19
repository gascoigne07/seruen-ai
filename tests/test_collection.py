"""Offline regression checks for acquisition integrity and evidence provenance."""

import hashlib
import io
import json
from pathlib import Path

import pytest

from seruen.collect import enrich_wiki, normalize, overpass, verify


def raw_dump(elements=None):
    return json.dumps({"elements": elements or [
        {"type": "node", "id": 123, "lat": 51.1, "lon": 71.4,
         "tags": {"name": "Test café", "amenity": "cafe"}},
    ]}).encode()


@pytest.mark.parametrize("payload", [
    {"remark": "runtime error: Query timed out", "elements": [{"type": "node"}]},
    {"remark": "out of memory", "elements": []},
    {"elements": []}, {"elements": {}}, {"elements": [None]}, [], {},
])
def test_overpass_rejects_incomplete_or_invalid_reply(payload):
    with pytest.raises(ValueError):
        overpass.parse_response(json.dumps(payload).encode())


def test_fetch_tries_next_mirror_after_http_200_error(monkeypatch):
    calls = []
    responses = iter([b'{"remark":"runtime error","elements":[]}', raw_dump()])

    def urlopen(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(next(responses))

    monkeypatch.setattr(overpass.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(overpass.time, "sleep", lambda _: None)
    mirror, body = overpass.fetch("query", attempts=1)
    assert mirror == overpass.MIRRORS[1]
    assert calls == overpass.MIRRORS[:2]
    assert body == raw_dump()


@pytest.mark.parametrize("lat,lon", [(float("nan"), 71), (51, float("inf")),
                                     (91, 71), (51, 181), (None, 71), (True, 71), ("bad", 71)])
def test_invalid_coordinates_do_not_enter_dataset(lat, lon):
    element = {"type": "node", "id": 1, "lat": lat, "lon": lon,
               "tags": {"name": "Test", "amenity": "cafe"}}
    assert normalize.normalize_element(element) is None


def test_selected_raw_gets_its_own_metadata_timestamp(tmp_path):
    raw = raw_dump()
    path = tmp_path / "overpass_old.json"
    meta = tmp_path / "metadata.json"
    old = {"file": str(path), "fetched_at": "2026-10-01T10:00:00Z",
           "sha256": hashlib.sha256(raw).hexdigest()}
    meta.write_text(json.dumps({"runs": [old], "latest": {
        "file": "overpass_new.json", "fetched_at": "2026-10-19T10:00:00Z", "sha256": "other"}}))
    fetched_at, _, verified = normalize.raw_provenance(path, raw, meta)
    assert fetched_at == old["fetched_at"]
    assert verified


def test_unknown_acquisition_time_is_not_invented(tmp_path):
    fetched_at, _, verified = normalize.raw_provenance(Path("unrecorded.json"), raw_dump(), tmp_path / "absent.json")
    assert fetched_at is None
    assert not verified


def test_hash_mismatch_leaves_dataset_untouched(tmp_path):
    path = tmp_path / "overpass_test.json"
    path.write_bytes(raw_dump())
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps({"latest": {"file": str(path), "sha256": "wrong"}}))
    out = tmp_path / "pois.json"
    out.write_text("existing dataset")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        normalize.main(["--raw", str(path), "--meta", str(meta), "--out", str(out),
                        "--report", str(tmp_path / "report.json")])
    assert out.read_text() == "existing dataset"


def test_normalize_records_verified_provenance_and_drops_invalid_tags(tmp_path):
    raw = raw_dump([
        {"type": "way", "id": 1, "center": {"lat": 51.1, "lon": 71.4},
         "tags": {"leisure": "playground"}},
        {"type": "node", "id": 2, "lat": 51.1, "lon": 71.4, "tags": {"name": None}},
    ])
    path = tmp_path / "overpass_test.json"
    path.write_bytes(raw)
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps({"latest": {"file": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                                          "fetched_at": "2026-10-19T10:00:00Z"}}))
    out, report = tmp_path / "result" / "pois.json", tmp_path / "quality" / "report.json"
    assert normalize.main(["--raw", str(path), "--meta", str(meta), "--out", str(out),
                           "--report", str(report)]) == 0
    pois, quality = json.loads(out.read_text()), json.loads(report.read_text())
    assert len(pois) == 1 and pois[0]["name_generated"]
    assert quality["raw_hash_verified"] and quality["dropped_invalid_elements"] == 1
    assert pois[0]["source"]["fetched_at"] == "2026-10-19T10:00:00Z"


def test_wikidata_enriches_all_matching_records_with_attribution(monkeypatch):
    pois = [{"id": f"node/{i}", "wikidata": "Q1", "name": {"ru": f"Place {i}"},
             "description_source": "osm_tags", "evidence": []} for i in (1, 2)]
    monkeypatch.setattr(enrich_wiki, "wikidata_sitelinks", lambda _: {"Q1": {"ru": "Test"}})
    monkeypatch.setattr(enrich_wiki, "fetch_extract", lambda *_: "Real sourced description.")
    monkeypatch.setattr(enrich_wiki, "geosearch", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(enrich_wiki.time, "sleep", lambda _: None)
    result = enrich_wiki.enrich(pois, verbose=False)
    assert result["by_wikidata"] == 2
    for poi in pois:
        assert poi["description_source"] == "wikipedia"
        assert poi["evidence"][0]["license"] == "CC BY-SA 4.0"
        assert poi["evidence"][0]["source_url"].endswith("/Test")


def test_integrity_command_detects_report_and_attribution_damage(tmp_path):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    raw = raw_dump()
    path = raw_dir / "overpass_test.json"
    path.write_bytes(raw)
    meta = data_dir / "metadata.json"
    overpass.write_metadata(meta, {
        "file": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
        "fetched_at": "2026-10-19T10:00:00Z", "bytes": len(raw), "elements": 1,
    })
    out, report = data_dir / "pois.json", data_dir / "quality_report.json"
    normalize.main(["--raw", str(path), "--meta", str(meta), "--out", str(out), "--report", str(report)])
    assert verify.verify(data_dir)["ok"]
    pois = json.loads(out.read_text())
    pois[0]["source"]["url"] = "https://example.com/invented-source"
    out.write_text(json.dumps(pois))
    quality = json.loads(report.read_text())
    quality["pois_total"] = 999
    report.write_text(json.dumps(quality))
    result = verify.verify(data_dir)
    assert not result["ok"]
    assert any("attribution" in e for e in result["errors"])
    assert any("pois_total" in e for e in result["errors"])
