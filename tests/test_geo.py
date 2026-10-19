"""Distance, walking time and the time-budget -> radius conversion."""

from __future__ import annotations

from seruen.core import geo

BAITEREK = (51.1283, 71.4305)
KHAN_SHATYR = (51.1325, 71.4039)


def test_haversine_known_distance():
    # Baiterek -> Khan Shatyr is roughly 1.9 km in a straight line
    distance = geo.haversine_m(*BAITEREK, *KHAN_SHATYR)
    assert 1700 < distance < 2100


def test_haversine_is_symmetric_and_zero_on_self():
    assert geo.haversine_m(*BAITEREK, *BAITEREK) == 0
    there = geo.haversine_m(*BAITEREK, *KHAN_SHATYR)
    back = geo.haversine_m(*KHAN_SHATYR, *BAITEREK)
    assert abs(there - back) < 1e-6


def test_walking_minutes_uses_detour_factor():
    # 1 km straight line = 1.3 km of walking at 4.2 km/h ≈ 18.6 min
    minutes = geo.walking_minutes(1000)
    assert 18 < minutes < 19.5
    assert geo.walking_minutes(0) == 0


def test_distance_for_minutes_is_inverse_of_walking_minutes():
    for metres in (300, 1000, 2500):
        assert abs(geo.distance_for_minutes(geo.walking_minutes(metres)) - metres) < 1e-6


def test_radius_grows_with_budget():
    small = geo.radius_from_budget(30, typical_visit_min=20)
    large = geo.radius_from_budget(120, typical_visit_min=20)
    assert small < large
    assert large <= geo.MAX_RADIUS_M


def test_radius_subtracts_visit_and_buffer():
    # 60 min budget, 25 min in a park, 5 min buffer -> 30 min of walking
    radius = geo.radius_from_budget(60, typical_visit_min=25)
    assert abs(radius - geo.distance_for_minutes(30)) < 1.0


def test_return_to_start_halves_the_radius():
    one_way = geo.radius_from_budget(120, typical_visit_min=20)
    round_trip = geo.radius_from_budget(120, typical_visit_min=20, return_to_start=True)
    assert round_trip < one_way
    assert abs(round_trip - one_way / 2) < 1.0


def test_tiny_budget_still_returns_minimum_radius():
    assert geo.radius_from_budget(10, typical_visit_min=50) == geo.MIN_RADIUS_M


def test_fits_in_budget():
    # 500 m away (~9 min) + 25 min visit fits into 45 min
    assert geo.fits_in_budget(500, 25, 45)
    # the same place does not fit into 20 minutes
    assert not geo.fits_in_budget(500, 25, 20)
    # a round trip doubles the walking
    assert not geo.fits_in_budget(1500, 30, 45, return_to_start=True)


def test_walking_cannot_eat_most_of_the_budget():
    # 5.6 km is ~104 min on foot: it technically leaves 16 min of a 120 min
    # budget, but nobody wants a recommendation like that
    assert not geo.fits_in_budget(5600, 10, 120)
    # under the cap the same budget is fine
    assert geo.fits_in_budget(2500, 10, 120)


def test_typical_visit_picks_shortest_wanted_category():
    assert geo.typical_visit_minutes(["theatre", "landmark"]) == geo.TYPICAL_VISIT_MIN["landmark"]
    assert geo.typical_visit_minutes([]) == geo.DEFAULT_VISIT_MIN
    assert geo.typical_visit_minutes(["nonsense"]) == geo.DEFAULT_VISIT_MIN


def test_round_coords_keeps_three_decimals():
    assert geo.round_coords(51.128312, 71.430577) == (51.128, 71.431)


def test_round_trip_budget_uses_conservative_whole_minute_legs():
    distance = geo.distance_for_minutes(10.2)
    assert not geo.fits_in_budget(distance, 30, 56, return_to_start=True)
    assert geo.fits_in_budget(distance, 30, 57, return_to_start=True)


def test_minimum_visits_preserve_short_coffee_and_meaningful_work():
    assert geo.minimum_visit_minutes("cafe") == 10
    assert geo.minimum_visit_minutes("cafe", wants_work=True) == 15
    assert geo.minimum_visit_minutes("theatre") == 90
