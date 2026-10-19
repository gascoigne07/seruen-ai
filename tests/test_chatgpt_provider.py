"""Offline transport tests: subscription usage is explicit and terminal-event gated."""

import json

import pytest
import requests

from seruen.core import chatgpt_provider as provider
from seruen.core.generate import MODE_CHATGPT, MODE_TEMPLATE, generate
from tests.test_validation import CANDIDATES, INTENT, MONDAY_12

REAL_TOKEN = provider._token


class Response:
    def __init__(self, body=None, events=(), status=200, lines=None):
        self.body = body
        self.events = events
        self.status_code = status
        self.headers = {"x-request-id": "test-request"}
        self.closed = False
        self.lines = lines

    def json(self):
        return self.body

    def iter_lines(self):
        if self.lines is not None:
            yield from self.lines
            return
        for event in self.events:
            yield b"event: message"
            yield ("data: " + json.dumps(event)).encode()
            yield b""

    def close(self):
        self.closed = True


def completed(value):
    return {"type": "response.completed", "response": {
        "status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(value)}]}]}}


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    monkeypatch.setattr(provider, "_token", lambda **_: "test-token")
    monkeypatch.setattr(provider.requests, "request", lambda *a, **k: pytest.fail("Unexpected network request"))


def call(**kwargs):
    return provider.generate_json(model="account-model", instructions="rules", prompt="hello",
                                  schema={"type": "object", "properties": {}, "required": []}, **kwargs)


def test_catalog_uses_visible_models_and_server_order(monkeypatch):
    response = Response({"models": [
        {"slug": "b", "display_name": "Second", "visibility": "list"},
        {"slug": "hidden", "display_name": "Hidden", "visibility": "hide"},
        {"slug": "a", "display_name": "First", "visibility": "list"},
        {"slug": "a", "display_name": "Duplicate", "visibility": "list"},
    ]})
    monkeypatch.setattr(provider.requests, "request", lambda *a, **k: response)
    assert provider.list_models() == [{"id": "b", "name": "Second"}, {"id": "a", "name": "First"}]
    assert response.closed


def test_bad_catalog_does_not_invent_a_model(monkeypatch):
    monkeypatch.setattr(provider.requests, "request", lambda *a, **k: Response({"data": [{"id": "other"}]}))
    with pytest.raises(provider.ChatGPTError, match="список"):
        provider.list_models()


def test_request_obeys_subscription_route_and_requires_completed(monkeypatch):
    calls = []
    response = Response(events=[{"type": "response.output_text.delta", "delta": "untrusted partial"}, completed({"picks": []})])
    monkeypatch.setattr(provider.requests, "request", lambda *a, **k: calls.append((a, k)) or response)
    assert call() == {"picks": []}
    args, kwargs = calls[0]
    assert args == ("POST", "https://api.openai.com/v1/responses")
    assert kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert kwargs["allow_redirects"] is False
    payload = kwargs["json"]
    assert payload["store"] is False and payload["stream"] is True
    assert isinstance(payload["input"], list)
    assert payload["text"]["format"]["schema"]["additionalProperties"] is False
    assert not {"temperature", "max_output_tokens", "previous_response_id", "metadata"} & payload.keys()
    assert response.closed


@pytest.mark.parametrize("events,reason", [
    ([], "stream_incomplete"),
    ([{"type": "response.output_text.delta", "delta": '{"picks": []}'}], "stream_incomplete"),
    ([{"type": "response.incomplete"}], "stream_incomplete"),
    ([{"type": "response.failed", "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}}], "subscription_sharing_usage_limit_exceeded"),
    ([completed([])], "response_invalid"),
    ([{"type": "response.completed", "response": {"status": "failed"}}], "stream_incomplete"),
])
def test_failed_or_partial_stream_never_becomes_success(monkeypatch, events, reason):
    response = Response(events=events)
    monkeypatch.setattr(provider.requests, "request", lambda *a, **k: response)
    with pytest.raises(provider.ChatGPTError) as error:
        call()
    assert error.value.reason == reason
    assert response.closed


def test_http_error_never_leaks_body_or_credentials(monkeypatch):
    response = Response({"detail": "sensitive test-token"}, status=403)
    monkeypatch.setattr(provider.requests, "request", lambda *a, **k: response)
    with pytest.raises(provider.ChatGPTError) as error:
        call()
    assert error.value.status == 403
    assert error.value.request_id == "test-request"
    assert "sensitive" not in str(error.value)
    assert "test-token" not in str(error.value)
    assert response.closed


def test_401_refreshes_only_once(monkeypatch):
    refreshes, calls = [], []
    monkeypatch.setattr(provider, "_token", lambda **kwargs: refreshes.append(kwargs) or "test-token")
    monkeypatch.setattr(provider.requests, "request", lambda *a, **k: calls.append(1) or Response({}, status=401))
    with pytest.raises(provider.ChatGPTError):
        call()
    assert len(calls) == 2
    assert refreshes == [{}, {"force_refresh": True}]


def test_network_failure_is_safe(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("secret")
    monkeypatch.setattr(provider.requests, "request", fail)
    with pytest.raises(provider.ChatGPTError) as error:
        call()
    assert error.value.reason == "network_error"
    assert "secret" not in str(error.value)


def test_missing_model_does_not_send_request():
    with pytest.raises(provider.ChatGPTError) as error:
        provider.generate_json(model=None, instructions="", prompt="", schema={})
    assert error.value.reason == "model_required"


def test_rag_validates_model_picks_and_reports_actual_provider(monkeypatch):
    value = {"picks": [
        {"poi_id": "invented", "why": "fake"},
        {"poi_id": "node/2", "why": "free and open", "activity": "Найдите интересную деталь и обсудите её.",
         "visit_minutes": 15, "confidence": "high", "source": "generated_idea"},
    ]}
    monkeypatch.setattr(provider.requests, "request", lambda *a, **k: Response(events=[completed(value)]))
    answer = generate(INTENT, CANDIDATES, MONDAY_12, provider="chatgpt", model="account-model", demo=False)
    assert answer.mode == MODE_CHATGPT and answer.model == "account-model"
    assert [pick.poi_id for pick in answer.picks] == ["node/2"]
    assert answer.picks[0].confidence == "medium"
    assert "free and open" not in answer.picks[0].why
    assert answer.candidates_sent_to_model == 2


def test_rag_falls_back_without_claiming_llm_success(monkeypatch):
    monkeypatch.setattr(provider.requests, "request", lambda *a, **k: Response(events=[]))
    answer = generate(INTENT, CANDIDATES, MONDAY_12, provider="chatgpt", model="account-model", demo=False)
    assert answer.mode == MODE_TEMPLATE and answer.model is None
    assert answer.fallback_reason
    assert answer.candidates_sent_to_model == 2


def test_local_choice_never_uses_subscription():
    answer = generate(INTENT, CANDIDATES, MONDAY_12, provider="local", demo=False)
    assert answer.mode == MODE_TEMPLATE and answer.candidates_sent_to_model == 0


def test_cli_chatgpt_keeps_intent_local(monkeypatch):
    from seruen.cli import advise
    from seruen.core import intent
    monkeypatch.setattr(intent, "parse_intent_llm", lambda *a, **k: pytest.fail("Gemini must not be called"))
    advice = advise("Я у Mega, хочу погулять 45 минут", 51.0891, 71.4073,
                    provider="chatgpt", model=None, use_weather=False)
    assert advice.intent.source == "regex"
    assert advice.answer.mode == MODE_TEMPLATE
    assert advice.answer.candidates_sent_to_model == 0


def test_stream_error_retains_top_level_limit_code(monkeypatch):
    response = Response(events=[{"type": "error", "code": "subscription_sharing_usage_limit_exceeded", "message": "private"}])
    monkeypatch.setattr(provider.requests, "request", lambda *a, **k: response)
    with pytest.raises(provider.ChatGPTError) as error:
        call()
    assert error.value.reason == "subscription_sharing_usage_limit_exceeded"
    assert "private" not in str(error.value)


def test_auth_errors_distinguish_permission_and_temporary_failure(monkeypatch):
    from seruen.core import chatgpt_auth

    for code, expected in [("plan_disabled", "plan_disabled"), ("auth_error", "auth_unavailable"), ("invalid_grant", "not_signed_in")]:
        def fail(_code=code, **kwargs):
            raise chatgpt_auth.AuthError("secret", _code)
        monkeypatch.setattr(chatgpt_auth, "get_access_token", fail)
        with pytest.raises(provider.ChatGPTError) as error:
            REAL_TOKEN()
        assert error.value.reason == expected
        assert "secret" not in str(error.value)
