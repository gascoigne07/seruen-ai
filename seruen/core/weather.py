"""Current weather from Open-Meteo (no API key).

Used for one decision only: should we prefer indoor places right now. Astana is
cold half the year, so this is the single most useful piece of live context.
Any failure degrades to `used: False` and retrieval simply ignores weather.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S = 6

# WMO weather codes -> short russian summary
WMO_RU = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "облачно",
    45: "туман",
    48: "туман с инеем",
    51: "слабая морось",
    53: "морось",
    55: "сильная морось",
    61: "слабый дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "ледяной дождь",
    67: "сильный ледяной дождь",
    71: "слабый снег",
    73: "снег",
    75: "сильный снег",
    77: "снежные зёрна",
    80: "ливни",
    81: "ливни",
    82: "сильные ливни",
    85: "снежные ливни",
    86: "сильные снежные ливни",
    95: "гроза",
    96: "гроза с градом",
    99: "гроза с градом",
}

BAD_WEATHER_CODES = set(WMO_RU) - {0, 1, 2, 3}
COLD_C = 2.0
HOT_C = 33.0
WINDY_KMH = 35.0


def _indoor_preferred(temperature_c: float | None, precipitation: float, code: int, wind_kmh: float) -> bool:
    if code in BAD_WEATHER_CODES or precipitation > 0.2:
        return True
    if temperature_c is not None and (temperature_c <= COLD_C or temperature_c >= HOT_C):
        return True
    return wind_kmh >= WINDY_KMH


def get_weather(lat: float, lon: float, timeout: int = TIMEOUT_S) -> dict:
    """Current conditions near (lat, lon). Never raises."""
    query = urllib.parse.urlencode(
        {
            "latitude": round(lat, 2),  # privacy: coarse coordinates leave the machine
            "longitude": round(lon, 2),
            "current": "temperature_2m,precipitation,weather_code,wind_speed_10m",
            "timezone": "Asia/Almaty",
        }
    )
    try:
        with urllib.request.urlopen(f"{ENDPOINT}?{query}", timeout=timeout) as response:
            payload = json.loads(response.read())
        current = payload["current"]
        temperature = float(current["temperature_2m"])
        precipitation = float(current["precipitation"])
        code = int(current["weather_code"])
        wind = float(current["wind_speed_10m"])
        if not all(math.isfinite(n) for n in (temperature, precipitation, wind)) or code not in WMO_RU:
            raise ValueError("invalid weather values")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
        return {"used": False, "error": f"{type(exc).__name__}", "indoor_preferred": False}

    return {
        "used": True,
        "temperature_c": temperature,
        "precipitation_mm": precipitation,
        "wind_kmh": wind,
        "code": code,
        "summary_ru": WMO_RU.get(code, "неизвестно"),
        "indoor_preferred": _indoor_preferred(temperature, precipitation, code, wind),
        "source": "Open-Meteo",
        "observed_at": current.get("time"),
        "requested_coords": [round(lat, 2), round(lon, 2)],
    }


def describe(weather: dict | None) -> str:
    if not weather or not weather.get("used"):
        return "погода не учитывалась (нет данных)"
    text = f"{weather.get('temperature_c')}°C, {weather.get('summary_ru')}"
    if weather.get("indoor_preferred"):
        text += " → приоритет местам в помещении"
    return text
