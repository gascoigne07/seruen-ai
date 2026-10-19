import io
import json

import pytest

from seruen.core.weather import get_weather


@pytest.mark.parametrize("payload", [{}, {"current": {}}, {"current": None}, {"current": {"temperature_2m": "bad"}}])
def test_missing_weather_is_not_sunny(monkeypatch, payload):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(json.dumps(payload).encode()))
    assert get_weather(51.1, 71.4)["used"] is False


def test_cold_weather_and_coarse_coordinates(monkeypatch):
    urls = []

    def request(url, **kwargs):
        urls.append(url)
        return io.BytesIO(json.dumps({"current": {
            "temperature_2m": -5, "precipitation": 0, "weather_code": 0,
            "wind_speed_10m": 10, "time": "2026-10-19T12:00",
        }}).encode())

    monkeypatch.setattr("urllib.request.urlopen", request)
    result = get_weather(51.123456, 71.412345)
    assert result["used"] and result["indoor_preferred"]
    assert "51.123456" not in urls[0]
    assert result["requested_coords"] == [51.12, 71.41]
