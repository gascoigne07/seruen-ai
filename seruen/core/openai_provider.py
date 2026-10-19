"""Responses API using one owner-managed server API key for all visitors.

No ChatGPT OAuth credentials, browser cookies or per-visitor API keys are read.
Limits protect a single hackathon process; production needs a shared gateway.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from collections import deque
from contextlib import contextmanager

import requests

API_URL = "https://api.openai.com/v1"
MAX_OUTPUT_TOKENS = 3000
MAX_REQUESTS_PER_MINUTE = 30
# Responses API values; the configured model must support the selected effort.
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_SLOTS = threading.BoundedSemaphore(2)
_LOCK = threading.Lock()
_REQUESTS: deque[float] = deque()


class OpenAIError(RuntimeError):
    def __init__(self, reason: str, *, status=None, request_id=None):
        self.reason = reason
        self.status = status
        self.request_id = request_id
        messages = {
            "not_configured": "Ответ ИИ пока недоступен; использован локальный подбор.",
            "busy": "Сервис ИИ занят. Использован локальный подбор; попробуйте позже.",
            "network_error": "Не удалось получить ответ ИИ. Использован локальный подбор.",
            "invalid_api_key": "Подключение ИИ временно недоступно. Использован локальный подбор.",
            "permission_denied": "Модель ИИ временно недоступна. Использован локальный подбор.",
            "rate_limit_exceeded": "Сервис ИИ достиг лимита запросов. Использован локальный подбор.",
            "insufficient_quota": "Лимит сервиса ИИ исчерпан. Использован локальный подбор.",
            "response_incomplete": "Ответ ИИ не завершён. Использован локальный подбор.",
            "response_invalid": "Ответ ИИ не прошёл проверку. Использован локальный подбор.",
            "catalog_invalid": "Не удалось прочитать каталог моделей OpenAI.",
            "service_error": "Сервис ИИ временно недоступен. Использован локальный подбор.",
        }
        super().__init__(messages.get(reason, messages["service_error"]))


def _key() -> str:
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if not value:
        raise OpenAIError("not_configured")
    return value


def _http_error(response) -> OpenAIError:
    try:
        body = response.json()
        error = body.get("error", {}) if isinstance(body, dict) else {}
        code = error.get("code") if isinstance(error, dict) else None
    except (ValueError, requests.RequestException):
        code = None
    if code not in {"invalid_api_key", "insufficient_quota", "rate_limit_exceeded"}:
        code = {401: "invalid_api_key", 403: "permission_denied", 404: "permission_denied", 429: "rate_limit_exceeded"}.get(response.status_code, "service_error")
    return OpenAIError(code, status=response.status_code, request_id=response.headers.get("x-request-id"))


@contextmanager
def _request_slot():
    if not _SLOTS.acquire(blocking=False):
        raise OpenAIError("busy")
    try:
        with _LOCK:
            now = time.monotonic()
            while _REQUESTS and _REQUESTS[0] <= now - 60:
                _REQUESTS.popleft()
            if len(_REQUESTS) >= MAX_REQUESTS_PER_MINUTE:
                raise OpenAIError("busy")
            _REQUESTS.append(now)
        yield
    finally:
        _SLOTS.release()


def _strict_schema(schema: dict) -> dict:
    result = copy.deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result


def generate_json(*, model: str | None, instructions: str, prompt: str, schema: dict, on_request=None) -> dict:
    key = _key()
    model = (model or os.environ.get("OPENAI_MODEL", "")).strip()
    if not model:
        raise OpenAIError("not_configured")
    effort = os.environ.get("OPENAI_REASONING_EFFORT", "").strip().lower()
    if effort and effort not in REASONING_EFFORTS:
        raise OpenAIError("not_configured")
    payload = {
        "model": model, "instructions": instructions,
        "input": [{"role": "user", "content": prompt}],
        "store": False, "stream": False, "max_output_tokens": MAX_OUTPUT_TOKENS,
        "text": {"format": {"type": "json_schema", "name": "seruen_recommendations",
                            "strict": True, "schema": _strict_schema(schema)}},
    }
    if effort:
        payload["reasoning"] = {"effort": effort}
    with _request_slot():
        try:
            if on_request:
                on_request()
            response = requests.post(
                API_URL + "/responses", json=payload,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                timeout=(10, 45), allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise OpenAIError("network_error") from exc
        try:
            if response.status_code != 200:
                raise _http_error(response)
            body = response.json()
            if not isinstance(body, dict) or body.get("status") != "completed":
                raise OpenAIError("response_incomplete")
            texts = [part["text"] for item in body.get("output", []) if item.get("type") == "message"
                     for part in item.get("content", []) if part.get("type") == "output_text"]
            result = json.loads("".join(texts))
            if not isinstance(result, dict):
                raise OpenAIError("response_invalid")
            return result
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise OpenAIError("response_invalid") from exc
        finally:
            response.close()


def list_models() -> list[str]:
    """Owner-only CLI helper. Listing models does not perform inference."""
    try:
        response = requests.get(API_URL + "/models", headers={"Authorization": f"Bearer {_key()}"},
                                timeout=(10, 20), allow_redirects=False)
    except requests.RequestException as exc:
        raise OpenAIError("network_error") from exc
    try:
        if response.status_code != 200:
            raise _http_error(response)
        body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise OpenAIError("catalog_invalid")
        return sorted({item["id"] for item in body["data"] if isinstance(item, dict) and isinstance(item.get("id"), str)})
    except (ValueError, TypeError) as exc:
        raise OpenAIError("catalog_invalid") from exc
    finally:
        response.close()
