"""A shared service uses only its owner's API key, never visitor credentials."""

import json
import threading
from collections import deque

import pytest
import requests

from seruen.cli import PRESETS, advise
from seruen.core import openai_provider as provider
from seruen.core.generate import MODE_OPENAI, MODE_TEMPLATE, generate
from seruen.core.server_config import get_server_config
from tests.test_validation import CANDIDATES, INTENT, MONDAY_12


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code, self.closed = body, status, False
        self.headers = {"x-request-id": "request-123"}

    def json(self):
        return self.body

    def close(self):
        self.closed = True


def completed(value):
    return {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps(value)}]}]}


@pytest.fixture(autouse=True)
def isolated_server(monkeypatch):
    for name in ["OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_REASONING_EFFORT", "GOOGLE_API_KEY", "GEMINI_MODEL", "SERUEN_PROVIDER", "SERUEN_DEMO"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(provider, "_REQUESTS", deque())
    monkeypatch.setattr(provider, "_SLOTS", threading.BoundedSemaphore(2))
    def no_network(*args, **kwargs):
        pytest.fail("No real HTTP requests in tests")
    monkeypatch.setattr(requests, "post", no_network)
    monkeypatch.setattr(requests, "get", no_network)


def configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "owner-private-key")
    monkeypatch.setenv("OPENAI_MODEL", "owner-selected-model")


def call():
    return provider.generate_json(model=None, instructions="rules", prompt="query",
                                  schema={"type": "object", "properties": {}, "required": []})


def test_server_defaults_local_and_never_returns_secrets(monkeypatch):
    assert get_server_config() == {"provider": "local", "model": None, "ready": True}
    configured(monkeypatch)
    assert get_server_config() == {"provider": "openai", "model": "owner-selected-model", "ready": True}
    assert "owner-private-key" not in repr(get_server_config())


def test_missing_openai_model_does_not_switch_to_another_billing_path(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "owner-private-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "other-key")
    monkeypatch.setenv("GEMINI_MODEL", "other-model")
    assert get_server_config() == {"provider": "openai", "model": None, "ready": False}
    with pytest.raises(provider.OpenAIError) as error:
        call()
    assert error.value.reason == "not_configured"
    assert len(provider._REQUESTS) == 0


def test_server_cannot_select_personal_subscription_pool(monkeypatch):
    monkeypatch.setenv("SERUEN_PROVIDER", "chatgpt")
    with pytest.raises(ValueError):
        get_server_config()


def test_owner_can_force_local_without_charging_key(monkeypatch):
    configured(monkeypatch)
    monkeypatch.setenv("SERUEN_DEMO", "1")
    assert get_server_config()["provider"] == "local"
    answer = advise("прогулка 45 минут", *PRESETS["mega"][1:], use_weather=False)
    assert answer.answer.mode == MODE_TEMPLATE


def test_api_uses_owner_key_with_output_cap_and_no_storage(monkeypatch):
    configured(monkeypatch)
    captured = []
    response = Response(completed({"picks": []}))
    monkeypatch.setattr(requests, "post", lambda *a, **k: captured.append((a, k)) or response)
    assert call() == {"picks": []}
    args, kwargs = captured[0]
    assert args == ("https://api.openai.com/v1/responses",)
    assert kwargs["headers"]["Authorization"] == "Bearer owner-private-key"
    body = kwargs["json"]
    assert body["model"] == "owner-selected-model"
    assert body["store"] is False and body["stream"] is False
    assert body["max_output_tokens"] == 3000
    assert "reasoning" not in body
    assert body["text"]["format"]["strict"] is True
    assert body["text"]["format"]["schema"]["additionalProperties"] is False
    assert "owner-private-key" not in json.dumps(body)
    assert kwargs["allow_redirects"] is False
    assert kwargs["timeout"] == (10, 45)
    assert response.closed


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max", " LOW "])
def test_owner_can_opt_into_reasoning_effort(monkeypatch, effort):
    configured(monkeypatch)
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", effort)
    captured = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: captured.append(k) or Response(completed({})))
    assert call() == {}
    assert captured[0]["json"]["reasoning"] == {"effort": effort.strip().lower()}
    assert captured[0]["timeout"] == (10, 45)
    assert captured[0]["json"]["max_output_tokens"] == 3000


def test_blank_reasoning_effort_preserves_model_default(monkeypatch):
    configured(monkeypatch)
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "  ")
    captured = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: captured.append(k) or Response(completed({})))
    assert call() == {}
    assert "reasoning" not in captured[0]["json"]


@pytest.mark.parametrize("effort", ["ultra", "low,high", '{"effort":"low"}'])
def test_invalid_reasoning_effort_is_rejected_before_request(monkeypatch, effort):
    configured(monkeypatch)
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", effort)
    with pytest.raises(provider.OpenAIError) as error:
        call()
    assert error.value.reason == "not_configured"
    assert len(provider._REQUESTS) == 0


@pytest.mark.parametrize("status,code,reason", [
    (401, "invalid_api_key", "invalid_api_key"),
    (429, "insufficient_quota", "insufficient_quota"),
    (429, "rate_limit_exceeded", "rate_limit_exceeded"),
    (500, "unknown", "service_error"),
])
def test_failures_are_safe_and_not_retried(monkeypatch, status, code, reason):
    configured(monkeypatch)
    calls = []
    response = Response({"error": {"code": code, "message": "owner-private-key"}}, status)
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1) or response)
    with pytest.raises(provider.OpenAIError) as error:
        call()
    assert len(calls) == 1
    assert error.value.reason == reason
    assert "owner-private-key" not in str(error.value)
    assert error.value.request_id == "request-123"
    assert response.closed


@pytest.mark.parametrize("body,reason", [
    ({"status": "incomplete", "output": []}, "response_incomplete"),
    ({"status": "completed", "output": []}, "response_invalid"),
    (completed([]), "response_invalid"),
])
def test_incomplete_or_invalid_response_is_not_success(monkeypatch, body, reason):
    configured(monkeypatch)
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response(body))
    with pytest.raises(provider.OpenAIError) as error:
        call()
    assert error.value.reason == reason


def test_global_limit_blocks_requests_before_transmitting(monkeypatch):
    configured(monkeypatch)
    monkeypatch.setattr(provider, "MAX_REQUESTS_PER_MINUTE", 1)
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response(completed({})))
    call()
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("Throttled request must not be sent"))
    with pytest.raises(provider.OpenAIError) as error:
        call()
    assert error.value.reason == "busy"
    assert provider._SLOTS.acquire(blocking=False)
    provider._SLOTS.release()


def test_concurrency_limit_does_not_queue_paid_requests(monkeypatch):
    configured(monkeypatch)
    assert provider._SLOTS.acquire(blocking=False)
    assert provider._SLOTS.acquire(blocking=False)
    with pytest.raises(provider.OpenAIError) as error:
        call()
    assert error.value.reason == "busy"


def test_catalog_uses_standard_api_shape(monkeypatch):
    configured(monkeypatch)
    monkeypatch.setattr(requests, "get", lambda *a, **k: Response({"data": [{"id": "b"}, {"id": "a"}]}))
    assert provider.list_models() == ["a", "b"]


def test_api_result_passes_existing_rag_validation(monkeypatch):
    configured(monkeypatch)
    raw = {"picks": [{"poi_id": "invented", "why": "wrong"},
                     {"poi_id": "node/2", "why": "unverified assertion", "activity": "Найдите интересную деталь и обсудите её.",
                      "visit_minutes": 10, "confidence": "high", "source": "poi_data"}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response(completed(raw)))
    answer = generate(INTENT, CANDIDATES, MONDAY_12, provider="openai", demo=False)
    assert answer.mode == MODE_OPENAI
    assert answer.model == "owner-selected-model"
    assert [p.poi_id for p in answer.picks] == ["node/2"]
    assert answer.picks[0].confidence == "medium"
    assert "unverified assertion" not in answer.picks[0].why
    assert answer.candidates_sent_to_model == 2


def test_missing_owner_key_returns_template_without_authentication(monkeypatch):
    monkeypatch.setenv("SERUEN_PROVIDER", "openai")
    result = advise("прогулка 45 минут", *PRESETS["mega"][1:], use_weather=False)
    assert result.answer.mode == MODE_TEMPLATE
    assert result.answer.candidates_sent_to_model == 0
    assert result.intent.source == "regex"


def test_query_length_is_checked_before_weather_or_model():
    with pytest.raises(ValueError, match="2000"):
        advise("x" * 2001, *PRESETS["mega"][1:])
