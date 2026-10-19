"""Mini-route: order 2-3 picks and count the walking time between them."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from itertools import combinations, permutations
from math import ceil

from seruen.core import geo
from seruen.core.generate import Pick
from seruen.core.hours import CLOSED, hours_status


@dataclass
class Leg:
    from_name: str
    to_name: str
    distance_m: int
    walk_min: int

    def as_dict(self) -> dict:
        return asdict(self)


ROUTE_BUFFER_MIN = 5


@dataclass
class Route:
    order: list[str] = field(default_factory=list)  # poi ids, in walking order
    legs: list[Leg] = field(default_factory=list)
    total_walk_min: int = 0
    total_visit_min: int = 0
    return_leg: Leg | None = None
    stop_visit_min: dict[str, int] = field(default_factory=dict)
    trimmed: bool = False  # visits were shortened to fit the budget

    @property
    def total_min(self) -> int:
        return self.total_walk_min + self.total_visit_min

    def fits(self, budget_min: int, slack_min: int = 0) -> bool:
        return self.total_min + ROUTE_BUFFER_MIN <= budget_min + slack_min

    def as_dict(self) -> dict:
        return {
            "order": self.order,
            "legs": [leg.as_dict() for leg in self.legs],
            "total_walk_min": self.total_walk_min,
            "total_visit_min": self.total_visit_min,
            "total_min": self.total_min,
            "return_leg": self.return_leg.as_dict() if self.return_leg else None,
            "stop_visit_min": self.stop_visit_min,
            "trimmed": self.trimmed,
            "buffer_min": ROUTE_BUFFER_MIN,
            "planned_min": self.total_min + ROUTE_BUFFER_MIN,
        }


def build_route(
    picks: list[Pick],
    start_lat: float,
    start_lon: float,
    return_to_start: bool = False,
    budget_min: int | None = None,
    when: datetime | None = None,
    wants_work: bool = False,
) -> Route | None:
    """Order the stops and make the result fit the time budget.

    With a tight budget we first shorten the stops to each category's minimum,
    then drop the last stop. If even two stops cannot fit, there is no honest
    mini-route to offer and we return None.
    """
    # At most three stops: test all orders/subsets, so a distant top pick does
    # not hide a feasible pair. Prefer more stops, then shorter walking.
    for count in range(min(3, len(picks)), 1, -1):
        feasible = []
        for subset in combinations(picks[:3], count):
            for ordered in permutations(subset):
                route = _ordered_route(list(ordered), start_lat, start_lon, return_to_start)
                if (budget_min is not None and not route.fits(budget_min)
                        and not _trim_to_budget(route, list(ordered), budget_min, wants_work=wants_work)):
                    continue
                if when is not None and not _hours_fit(route, list(ordered), when):
                    continue
                feasible.append(route)
        if feasible:
            return min(feasible, key=lambda route: (route.trimmed, route.total_walk_min))
    return None


def _hours_fit(route: Route, picks: list[Pick], when: datetime) -> bool:
    by_id = {p.poi_id: p for p in picks}
    elapsed = 0
    for poi_id, leg in zip(route.order, route.legs, strict=True):
        elapsed += leg.walk_min
        visit = route.stop_visit_min[poi_id]
        raw = by_id[poi_id].hours.get("raw")
        for offset in range(visit):
            if hours_status(raw, when + timedelta(minutes=elapsed + offset)).status == CLOSED:
                return False
        elapsed += visit
    return True


def _trim_to_budget(route: Route, picks: list[Pick], budget_min: int, *, wants_work: bool = False) -> bool:
    """Preserve meaningful visits, then share any remaining minutes equally."""
    available = budget_min - route.total_walk_min - ROUTE_BUFFER_MIN
    minimums = {
        pick.poi_id: min(pick.visit_minutes, geo.minimum_visit_minutes(pick.category, wants_work=wants_work))
        for pick in picks
    }
    if sum(minimums.values()) > available:
        return False
    route.stop_visit_min = minimums
    remaining = available - sum(minimums.values())
    while remaining > 0:
        extended = False
        for pick in picks:
            if remaining and route.stop_visit_min[pick.poi_id] < pick.visit_minutes:
                route.stop_visit_min[pick.poi_id] += 1
                remaining -= 1
                extended = True
        if not extended:
            break
    route.total_visit_min = sum(route.stop_visit_min.values())
    route.trimmed = True
    return True


def _ordered_route(
    picks: list[Pick],
    start_lat: float,
    start_lon: float,
    return_to_start: bool,
) -> Route:
    """Count conservative whole-minute legs in the supplied order."""
    remaining = list(picks)
    current = (start_lat, start_lon)
    current_name = "старт"
    route = Route()

    while remaining:
        nearest = remaining[0]
        distance = geo.haversine_m(current[0], current[1], nearest.lat, nearest.lon)
        walk = geo.walking_minutes(distance)
        route.legs.append(
            Leg(
                from_name=current_name,
                to_name=nearest.name,
                distance_m=round(distance),
                walk_min=ceil(walk),
            )
        )
        route.total_walk_min += ceil(walk)
        route.total_visit_min += nearest.visit_minutes
        route.stop_visit_min[nearest.poi_id] = nearest.visit_minutes
        route.order.append(nearest.poi_id)
        current = (nearest.lat, nearest.lon)
        current_name = nearest.name
        remaining.remove(nearest)

    if return_to_start:
        distance = geo.haversine_m(current[0], current[1], start_lat, start_lon)
        walk = geo.walking_minutes(distance)
        route.return_leg = Leg(
            from_name=current_name,
            to_name="старт",
            distance_m=round(distance),
            walk_min=ceil(walk),
        )
        route.total_walk_min += ceil(walk)

    return route


def describe_route(route: Route, language: str = "ru") -> str:
    """One-line route summary for the card list."""
    if not route.legs:
        return ""
    stops = " → ".join([route.legs[0].from_name] + [leg.to_name for leg in route.legs])
    if route.return_leg:
        stops += " → старт"
    templates = {
        "ru": "Маршрут: {stops}. Пешком {walk} мин + на месте {visit} мин + запас 5 мин, итого ≈{total} мин.",
        "en": "Route: {stops}. {walk} min walking + {visit} min at stops + 5 min buffer, ≈{total} min total.",
        "kk": "Маршрут: {stops}. Жаяу {walk} мин + орында {visit} мин + қор 5 мин, жиыны ≈{total} мин.",
    }
    template = templates.get(language, templates["ru"])
    text = template.format(
        stops=stops,
        walk=route.total_walk_min,
        visit=route.total_visit_min,
        total=route.total_min + ROUTE_BUFFER_MIN,
    )
    if route.trimmed:
        trimmed_note = {
            "ru": " Время на остановках сокращено, чтобы уложиться в бюджет.",
            "en": " Stop durations were shortened to fit the budget.",
            "kk": " Бюджетке сыю үшін аялдамадағы уақыт қысқартылды.",
        }
        text += trimmed_note.get(language, trimmed_note["ru"])
    return text
