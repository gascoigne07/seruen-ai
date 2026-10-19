# Seruen AI — project context

Seruen AI is a hackathon prototype for case 1, НИЦ Smart City, in the supplied
Build with AI brief. It recommends nearby Astana POIs and concrete activities using
our automatically collected dataset and a RAG pipeline. UI language is Russian;
answer content can be RU, EN or KK. This file describes the current project, not a
past deadline or an instruction to stop improving it after a fixed time.

## Product and submission

Users choose a starting point, time budget and context, then ask where to go.
They do not sign in, provide keys or select the model. The owner configures one
server connection. Recommendations include why, visit time, activity, source,
hours status and uncertainty. An empty result is valid when no POI meets the
constraints; never invent a place to reach the brief's expected 1–3 suggestions.

The brief requires automated data collection, interactive RAG, visible input data,
honest uncertainty, test scenarios, privacy notes, a 5–7 minute presentation and a
YouTube demo. Gemini, MCP and GCP are requirements of the separate CampusPulse
case 2, not this project. Requirements and current gaps: docs/REQUIREMENTS.md.
YouTube recording/upload remains a human submission step.

## Architecture

- `seruen/cli.py`: shared `advise()` pipeline and CLI.
- `seruen/app/streamlit_app.py`: visitor UI, map, presets, session history, data panel.
- `core/origin_resolution.py`: OpenAI first extracts the exact current-place span;
  coordinates come only from local POIs, verified references or Photon. Multiple or
  weak matches require confirmation. Missing/unknown starts never default to Baiterek.
  Context is rebuilt from the original query by removing the validated origin span,
  not by trusting a model rewrite. Desired destinations remain preferences.
- `core/location.py`, `location_presets.py`: local name matching and manual presets.
  Common RU inflections and EN/KK forms are supported. Points can be area centres,
  not GPS or verified entrances. «Около двух часов» is time, not an address.
- `core/place_search.py`: user-initiated Photon forward search of at most two short
  extracted names/addresses. Validate bbox, explicit city/country and OSM ID; construct
  source links locally. No model coordinates/URLs. In-memory phrase cache: 512 entries,
  24 hours; one request/second and 30/minute/process, queue <=1 second, timeout 3/12.
  No retries/bulk collection. SERUEN_GEOCODING=0 disables it; local mode is offline.
- `core/intent.py`: structured preferences from text; local parser for OpenAI/local,
  optional model parsing only for Gemini. Explicit constraints are preserved.
- `core/retrieval.py`: geometric limits, category/access filters, BM25 and context
  scoring; up to eight candidate cards. Dataset is local JSON, no vector database.
- `core/generate.py`: candidate selection and activity generation, local validation,
  factual rationale assembled from POI data, template fallback on errors.
- `core/geo.py`, `hours.py`, `route.py`: estimated walking time, partial OSM schedule
  support, optional route order, visits, return leg and five-minute buffer.
- `core/server_config.py`: `SERUEN_PROVIDER=openai|local|gemini`; absent provider
  chooses OpenAI when its key exists, else Gemini when its key exists, else local.
  `SERUEN_DEMO=1` forces local operation.
- `core/openai_provider.py`: owner's `OPENAI_API_KEY` and explicit `OPENAI_MODEL`;
  Responses API, strict JSON schema, `store=false`, `stream=false`, completed status
  mandatory, output cap 3000 tokens, no automatic retry, two concurrent calls and
  30 call starts per minute per process. Queries are limited to 2000 characters.
- `core/weather.py`: optional Open-Meteo; disabled by default.

Model IDs are not hardcoded: `make models` lists API IDs; the owner chooses a compatible model.
Gemini is optional (`GOOGLE_API_KEY`, `GEMINI_MODEL`, `make gemini-models`).
The owner has privately configured and successfully tested OpenAI `gpt-6-luna`:
HTTP 200, completed response, one accepted place from two candidates, no fallback.
That observation is evidence of one successful request, not a default model or an
availability guarantee. Saved ten-case live run before the new origin pipeline: eight openai answers, two empty
candidate sets without inference (airport and strict park 40), zero invariant failures.
Explicit route: three stops plus return in 72 minutes including buffer, budget 90.
Current six-case live origin test: Central Park/Visit Astana, Botanical Garden/local,
Astana Arena/Visit Astana resolved; Kenesary 32 has seven Photon choices; fictional
origin stays unresolved even with Baiterek as destination; missing start asks for it.
All six used valid AI extraction. Current make check: 467 tests, Ruff and dataset
verification pass; ten offline scenarios pass. Evidence: docs/LIVE_VALIDATION.md.

## Data and grounding

Saved snapshot: 1171 normalized POIs from 1232 OSM elements; 61 duplicates removed.
317 have hours, 41 wheelchair=yes, 14 Wikipedia extracts, 261 generated names.
The collector intentionally includes unnamed playgrounds and marks generated names.
Scripts: `collect/overpass.py`, `normalize.py`, `enrich_wiki.py`, `verify.py`.
Overpass uses an Astana bbox, three mirrors and retries; partial/error responses
are rejected. Raw JSON, query, timestamp and SHA-256 are retained. Normalization
checks the selected dump's hash and timestamp, finite coordinates and valid IDs.
`make verify-data` reproduces main OSM fields offline and checks both raw files.
A fresh network collection can differ; Wikipedia revisions are not snapshotted.

POIs carry OSM IDs, names, coordinates, category/subtype/tags, descriptions,
hours_raw, wheelchair, indoor, contact links, provenance and field evidence.
OSM is ODbL with attribution; Wikidata is CC0; Wikipedia is CC BY-SA 4.0;
Open-Meteo weather data is CC BY 4.0. Google Maps/2GIS content is not collected.
`indoor` may be inferred from category and is not a field inspection.

Explicit category, indoor/outdoor and requested wheelchair=yes are hard filters.
Child scenarios exclude bars/pubs. Known closing is checked against arrival and
visit time; missing/unsupported hours remain unknown. PH/SH needs a holiday
calendar and currently remains unknown. Walking is an estimate (4.2 km/h, 1.3
factor on straight-line distance); map lines do not route through actual streets.
Flight check-in, security, queues and gate deadlines are not calculated.

For missing hours only, a narrow display heuristic may use access-condition wording
for indoor=false plus park/garden/playground/memorial. Keep status unknown and all
filters unchanged; never claim public, safe or 24/7 access. Do not apply this to a
filled but unsupported schedule, monument/artwork/viewpoint/attraction. Normalized
indoor may be subtype-derived: Baiterek has raw building=yes yet normalized false.
Raw access/private/customer restrictions are currently not carried into POI schema;
missing normalized access is not permission to enter.

The origin model receives the coordinate-redacted query. The generation model receives
retrieved cards plus request/context, without a separate user coordinate field. Returned IDs, types and visit duration are validated. Factual
why text is locally grounded; model activity prose is conservatively screened and
marked as an idea with an on-site condition. This is a heuristic, not semantic
proof. POI descriptions are untrusted data, not instructions. Unknown sockets,
noise levels, entry conditions and safety must never become factual promises.

## Privacy and development

- Do not expose keys, `.env`, tokens or personal data in logs, screenshots,
  archives or Git. Public UI never accepts credentials or controls owner accounts.
- Query/history state is separate for each Streamlit session, retained only in
  process memory by this application. Coordinates are needed on the server for
  calculation; display rounds to three decimals, weather receives two decimals.
- Model-bound decimal coordinate tokens are filtered as a best effort. Names,
  addresses and distances can still reveal location. `store=false` does not mean
  that provider/infrastructure logging or retention is zero. See docs/PRIVACY.md.
- Local mode disables model and geocoder calls; CARTO tiles and optional weather need
  the network. Photon receives extracted short place phrases, not the full chat.
  Its process cache is separate from per-session history; clearing history does not
  clear the cache. A process restart clears both. Full city coverage is not promised.
- Python 3.13 and pinned requirements are used. `make check` runs offline tests,
  Ruff, saved-data verification and a configuration-presence check, not inference.
- `make scenarios` always runs locally. `--live` on the scenario generator is an
  explicit network action; use `--output-dir docs/live` to keep both
  reports. Do not silently enable inference from an environment variable.
- After logic/data changes, refresh scenarios, verify actual outputs and update
  documentation. Keep historical phase logs clearly marked as historical.
- Release excludes secrets, `.env`, virtual environments and temporary files.
  Recording/upload of the required demo remains a human submission step.
