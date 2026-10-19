"""Distances, walking time and the budget -> radius conversion."""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_008.8

WALK_SPEED_KMH = 4.2
# streets are not straight lines: inflate the straight-line distance
DETOUR_FACTOR = 1.3
# minutes kept aside for "finding the door", crossings, ordering at the counter
BUFFER_MIN = 5

MIN_RADIUS_M = 300
MAX_RADIUS_M = 6000

# Walking must not eat the whole budget: a place 104 minutes away technically
# "fits" into two hours, but there is then no time to actually be there.
MAX_WALK_SHARE = 0.45

# how long a visit to each category typically takes, minutes
TYPICAL_VISIT_MIN = {
    "park": 25,
    "cafe": 30,
    "restaurant": 50,
    "museum": 50,
    "landmark": 10,
    "viewpoint": 10,
    "playground": 30,
    "mall": 40,
    "theatre": 90,
    "library": 30,
    "sport": 60,
    "other": 20,
}
DEFAULT_VISIT_MIN = 20

# These are planning estimates, not opening guarantees or booked activities.
# Short coffee/walk requests should not require the full typical visit, while
# a theatre performance must not be reduced to a ten-minute stop.
MIN_VISIT_MIN = {
    "park": 10, "cafe": 10, "restaurant": 30, "museum": 30,
    "landmark": 5, "viewpoint": 5, "playground": 15, "mall": 15,
    "theatre": 90, "library": 15, "sport": 30, "other": 10,
}
DEFAULT_MIN_VISIT_MIN = 10


def minimum_visit_minutes(category: str, *, wants_work: bool = False) -> int:
    minimum = MIN_VISIT_MIN.get(category, DEFAULT_MIN_VISIT_MIN)
    return max(minimum, 15) if wants_work else minimum


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def _metres_per_minute() -> float:
    return WALK_SPEED_KMH * 1000 / 60


def walking_minutes(distance_m: float) -> float:
    """Walking time for a straight-line distance, detour included."""
    return distance_m * DETOUR_FACTOR / _metres_per_minute()


def distance_for_minutes(minutes: float) -> float:
    """Straight-line distance reachable on foot in `minutes`."""
    return max(0.0, minutes) * _metres_per_minute() / DETOUR_FACTOR


def typical_visit_minutes(categories: list[str] | None) -> int:
    """Visit length to plan for, given the categories the user asked about."""
    if not categories:
        return DEFAULT_VISIT_MIN
    known = [TYPICAL_VISIT_MIN[c] for c in categories if c in TYPICAL_VISIT_MIN]
    if not known:
        return DEFAULT_VISIT_MIN
    # plan for the shortest fitting option, so a tight budget still returns places
    return min(known)


def radius_from_budget(
    budget_min: float,
    typical_visit_min: int = DEFAULT_VISIT_MIN,
    return_to_start: bool = False,
) -> float:
    """Search radius in metres for a time budget.

    walk_min = budget - typical visit - buffer; halved when the user wants to
    come back to the starting point. Clamped to [MIN_RADIUS_M, MAX_RADIUS_M] so
    a tiny budget still yields something nearby instead of an empty list.
    """
    walk_min = budget_min - typical_visit_min - BUFFER_MIN
    if return_to_start:
        walk_min /= 2
    radius = distance_for_minutes(walk_min)
    return float(min(MAX_RADIUS_M, max(MIN_RADIUS_M, radius)))


def fits_in_budget(
    distance_m: float,
    visit_min: float,
    budget_min: float,
    return_to_start: bool = False,
    slack_min: float = 0.0,
) -> bool:
    """Can the user walk there, spend `visit_min`, and stay inside the budget?"""
    legs = 2 if return_to_start else 1
    walk_total = math.ceil(walking_minutes(distance_m)) * legs
    if walk_total > budget_min * MAX_WALK_SHARE:
        return False
    return walk_total + visit_min + BUFFER_MIN <= budget_min + slack_min


def round_coords(lat: float, lon: float, digits: int = 3) -> tuple[float, float]:
    """Privacy: user coordinates are only ever logged/shown rounded."""
    return round(lat, digits), round(lon, digits)
