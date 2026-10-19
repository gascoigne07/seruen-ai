"""A deliberately small subset of the OSM `opening_hours` syntax.

Design rule from the brief: never claim "open now" without data. Anything this
parser does not fully understand becomes `unknown`, never `open`.

Supported:
    24/7
    Mo-Fr 09:00-18:00
    Mo-Fr 09:00-18:00; Sa 10:00-16:00
    Mo-Su 10:00-22:00
    Mo-Su 20:00-02:00            (crosses midnight)
    Mo-Fr 09:00-12:00,13:00-18:00
    Mo,We,Fr 10:00-19:00
    Mo-Fr 09:00-18:00; Su off
    PH/SH rules remain unknown (no holiday calendar is bundled)
Everything else -> unknown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

OPEN = "open"
CLOSED = "closed"
UNKNOWN = "unknown"

DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
DAY_INDEX = {day.lower(): i for i, day in enumerate(DAYS)}
HOLIDAY_SELECTORS = {"ph", "sh", "ph off", "sh off"}

_TIME_RANGE_RE = re.compile(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$")
_DAY_PART_RE = re.compile(r"^[A-Za-z]{2}(\s*-\s*[A-Za-z]{2})?$")
# OSM also uses "," between whole rules: "Mo-Th 11:00-23:00, Fr,Sa 12:00-24:00".
# A comma right after a time (digit) and before a day name starts a new rule;
# a comma between two day names ("Fr,Sa") belongs to a day list.
_RULE_SEPARATOR_RE = re.compile(r"(?<=\d)\s*,\s*(?=(?:Mo|Tu|We|Th|Fr|Sa|Su|PH|SH)\b)", re.IGNORECASE)

STATUS_LABEL_RU = {
    OPEN: "открыто",
    CLOSED: "закрыто",
    UNKNOWN: "часы работы неизвестны",
}


@dataclass
class HoursResult:
    """Opening status plus the confidence we are allowed to show."""

    status: str  # open | closed | unknown
    confidence: str  # high | medium | low
    raw: str | None
    detail: str
    minutes_until_close: int | None = None

    @property
    def label_ru(self) -> str:
        return STATUS_LABEL_RU[self.status]

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "raw": self.raw,
            "detail": self.detail,
            "minutes_until_close": self.minutes_until_close,
        }


def hours_context(poi: dict, hours: HoursResult) -> str:
    """Choose how to describe missing hours, without changing opening status.

    A park or outdoor memorial is a different visit from a business with a
    missing timetable. This is a category-based presentation hint, not evidence
    of public access or 24/7 operation. Generic landmarks, monuments, viewpoints
    and artworks can be buildings or indoor exhibits, so are not inferred here.
    An explicit timetable, even an unsupported one, always keeps its own status.
    """
    if (
        hours.status == UNKNOWN
        and hours.raw is None
        and poi.get("indoor") is False
        and poi.get("osm_subtype") in {"park", "garden", "playground", "memorial"}
    ):
        return "outdoor"
    return "schedule"


def _parse_days(part: str) -> list[int] | None:
    """`Mo-Fr` / `Mo,We` / `Sa,Su,PH` -> weekday indices (Mo=0).

    Holiday selectors (PH/SH) are dropped from the list: we do not model
    holidays, but their presence must not make the whole rule unreadable.
    Returns None if unparsable, [] if the rule only covers holidays.
    """
    result: list[int] = []
    for chunk in part.split(","):
        chunk = chunk.strip()
        if chunk.lower() in {"ph", "sh"}:
            continue
        if not chunk or not _DAY_PART_RE.match(chunk):
            return None
        if "-" in chunk:
            start_raw, end_raw = (piece.strip().lower() for piece in chunk.split("-", 1))
            if start_raw not in DAY_INDEX or end_raw not in DAY_INDEX:
                return None
            start, end = DAY_INDEX[start_raw], DAY_INDEX[end_raw]
            if start <= end:
                result.extend(range(start, end + 1))
            else:  # wraps the week, e.g. Sa-Mo
                result.extend(list(range(start, 7)) + list(range(0, end + 1)))
        else:
            if chunk.lower() not in DAY_INDEX:
                return None
            result.append(DAY_INDEX[chunk.lower()])
    return sorted(set(result))


def _parse_time_ranges(part: str) -> list[tuple[int, int]] | None:
    """`09:00-18:00,19:00-22:00` -> [(540, 1080), (1140, 1320)] in minutes."""
    ranges: list[tuple[int, int]] = []
    for chunk in part.split(","):
        match = _TIME_RANGE_RE.match(chunk.strip())
        if not match:
            return None
        h1, m1, h2, m2 = (int(group) for group in match.groups())
        if not (0 <= h1 < 24 and 0 <= h2 <= 24 and m1 < 60 and m2 < 60):
            return None
        if (h2 == 24 and m2 != 0) or (h1 == h2 and m1 == m2):
            return None
        ranges.append((h1 * 60 + m1, h2 * 60 + m2))
    return ranges


def _split_rule(rule: str) -> tuple[list[int], list[tuple[int, int]] | str] | None:
    """One `;`-separated rule -> (weekdays, time ranges) or (weekdays, "off")."""
    # "Su, Tu-Fr 11:00-19:00" — a space after a day-list comma is still one list
    rule = re.sub(r",\s+", ",", rule.strip())
    if not rule:
        return None
    lowered = rule.lower()
    if lowered in {"24/7", "24x7", "00:00-24:00"}:
        return list(range(7)), [(0, 24 * 60)]

    tokens = rule.split()
    if not tokens:
        return None

    # a rule may start with a day selector; without one it applies to every day
    days = _parse_days(tokens[0])
    rest = tokens[1:] if days is not None else tokens
    if days is None:
        days = list(range(7))

    if not rest:
        return None
    rest_joined = " ".join(rest).strip().lower()
    if rest_joined in {"off", "closed"}:
        return days, "off"
    if rest_joined in {"24/7", "open", "00:00-24:00"}:
        return days, [(0, 24 * 60)]

    ranges = _parse_time_ranges(" ".join(rest).replace(" ", ""))
    if ranges is None:
        return None
    return days, ranges


def hours_status(raw: str | None, when: datetime) -> HoursResult:
    """Opening status of a POI at `when`, from its raw `opening_hours` tag."""
    if raw is None or not str(raw).strip():
        return HoursResult(
            status=UNKNOWN,
            confidence="low",
            raw=None,
            detail="в данных OSM нет часов работы",
        )

    raw_text = str(raw).strip()
    if "|" in raw_text:
        return HoursResult(UNKNOWN, "low", raw_text, "резервные правила расписания не поддерживаются")
    if when.tzinfo is not None:
        when = when.astimezone(ZoneInfo("Asia/Almaty"))
    if re.search(r"\b(?:PH|SH)\b", raw_text, re.IGNORECASE):
        return HoursResult(UNKNOWN, "low", raw_text, "праздничное расписание не проверено: " + raw_text)
    if raw_text in {"24/7", "00:00-24:00"}:
        return HoursResult(OPEN, "high", raw_text, "круглосуточно по данным OSM")
    # Preserve additional-rule commas, distinct from overriding semicolons.
    rules_text = _RULE_SEPARATOR_RE.sub(" | ", raw_text)
    weekday = when.weekday()  # Mo = 0
    minute_of_day = when.hour * 60 + when.minute

    ranges_by_day: list[list[tuple[int, int]]] = [[] for _ in DAYS]
    explicit_off = [False] * 7
    saw_any_rule = False

    rules = re.split(r"([;|])", rules_text)
    separator = ";"
    for rule in rules:
        if rule in {";", "|"}:
            separator = rule
            continue
        rule = rule.strip()
        if not rule:
            continue
        if rule.lower() in HOLIDAY_SELECTORS or rule.lower().startswith(("ph ", "sh ")):
            continue  # holidays: out of scope, but not a parse failure
        parsed = _split_rule(rule)
        if parsed is None:
            return HoursResult(
                status=UNKNOWN,
                confidence="low",
                raw=raw_text,
                detail=f"формат часов не распознан: {raw_text}",
            )
        saw_any_rule = True
        days, spec = parsed
        for day in days:
            if spec == "off":
                ranges_by_day[day] = []
                explicit_off[day] = True
            elif separator == ";":
                ranges_by_day[day] = list(spec)
                explicit_off[day] = False
            else:
                ranges_by_day[day].extend(spec)
                explicit_off[day] = False

    if not saw_any_rule:
        return HoursResult(
            status=UNKNOWN,
            confidence="low",
            raw=raw_text,
            detail=f"формат часов не распознан: {raw_text}",
        )

    # Build a full weekly horizon so “until close” agrees with the status after
    # midnight, including tomorrow's explicit off rule. Adjacent/overlapping
    # intervals form one uninterrupted visit window, not an artificial closure.
    intervals: list[tuple[int, int]] = []
    for offset in range(8):
        day = (weekday + offset) % 7
        base = offset * 1440
        for start, end in ranges_by_day[day]:
            intervals.append((base + start, base + (1440 if end < start else end)))
        if not explicit_off[day]:
            for start, end in ranges_by_day[(day - 1) % 7]:
                if end < start:
                    intervals.append((base, base + end))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    for start, end in merged:
        if start <= minute_of_day < end:
            until_close = end - minute_of_day
            return HoursResult(
                OPEN,
                "high",
                raw_text,
                f"по данным OSM: {raw_text}",
                minutes_until_close=None if until_close >= 7 * 1440 else until_close,
            )

    detail = f"по данным OSM: {raw_text}"
    if explicit_off[weekday]:
        detail = f"сегодня выходной по данным OSM: {raw_text}"
    return HoursResult(CLOSED, "high", raw_text, detail)
