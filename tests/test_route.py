from datetime import datetime

from seruen.core.generate import Pick
from seruen.core.route import build_route


def pick(id, lat=51.1283, lon=71.4305, minutes=20, raw="24/7"):
    return Pick(id, id, "why", "activity", minutes, "medium", "poi_data",
                lat=lat, lon=lon, hours={"raw": raw})


def test_route_reserves_buffer_and_never_uses_old_five_minute_overrun():
    route = build_route([pick("a"), pick("b")], 51.1283, 71.4305, budget_min=40)
    assert route is not None
    assert route.total_min + 5 <= 40
    assert route.trimmed


def test_route_can_use_feasible_pair_without_distant_top_pick():
    route = build_route([pick("far", lat=51.22), pick("a"), pick("b")],
                        51.1283, 71.4305, budget_min=30)
    assert route is not None
    assert set(route.order) == {"a", "b"}
    assert route.total_min + 5 <= 30


def test_route_avoids_place_closing_during_visit():
    assert build_route([pick("a", raw="Mo 09:00-18:00"), pick("b")],
                       51.1283, 71.4305, budget_min=60,
                       when=datetime(2026, 10, 19, 17, 55)) is None


def test_outdoor_presentation_cannot_override_closure_during_a_route():
    park = pick("park", raw="Mo 09:00-18:00")
    park.category = "park"
    park.hours["context"] = "outdoor"  # Even a stale presentation hint cannot override the raw schedule.
    assert build_route([park, pick("memorial", raw=None)],
                       51.1283, 71.4305, budget_min=60,
                       when=datetime(2026, 10, 19, 17, 55)) is None


def test_round_trip_includes_return_in_budget():
    route = build_route([pick("a", lat=51.129), pick("b", lat=51.130)],
                        51.1283, 71.4305, return_to_start=True, budget_min=35)
    assert route is not None
    assert route.return_leg is not None
    assert route.total_min + 5 <= 35


def test_no_route_when_even_minimal_visits_cannot_fit():
    assert build_route([pick("a"), pick("b")], 51.1283, 71.4305, budget_min=20) is None


def test_route_does_not_trim_a_museum_visit_to_ten_minutes():
    museum = pick("museum", minutes=45)
    museum.category = "museum"
    cafe = pick("cafe", minutes=30)
    cafe.category = "cafe"
    route = build_route([museum, cafe], 51.1283, 71.4305, budget_min=50)
    assert route is not None and route.trimmed
    assert route.stop_visit_min["museum"] >= 30
    assert route.stop_visit_min["cafe"] >= 10
    assert route.total_min + 5 <= 50


def test_short_route_cannot_squeeze_in_a_theatre_performance():
    theatre = pick("theatre", minutes=90)
    theatre.category = "theatre"
    assert build_route([theatre, pick("cafe")], 51.1283, 71.4305, budget_min=60) is None


def test_explicit_work_route_preserves_time_for_work_at_each_cafe():
    stops = [pick("a", minutes=45), pick("b", minutes=45)]
    for stop in stops:
        stop.category = "cafe"
    assert build_route(stops, 51.1283, 71.4305, budget_min=30, wants_work=True) is None
    route = build_route(stops, 51.1283, 71.4305, budget_min=35, wants_work=True)
    assert route is not None and route.trimmed
    assert route.stop_visit_min == {"a": 15, "b": 15}
