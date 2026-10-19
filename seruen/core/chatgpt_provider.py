"""Personal ChatGPT plan usage via the documented public Responses endpoint.

Only our own OAuth grant is used. No API-key fallback or alternate billing path.
The caller must choose a model from the account-specific catalog.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable

import requests

API_URL = "https://api.openai.com/v1"
MAX_STREAM_BYTES = 1_000_000


class ChatGPTError(RuntimeError):
    """Safe user-facing error with non-secret diagnostics kept separately."""

    def __init__(self, reason: str, *, status: int | None = None, request_id: str | None = None):
        self.reason = reason
        self.status = status
        self.request_id = request_id
        messages = {
            "not_signed_in": "Войдите через ChatGPT в настройках Seruen AI.",
            "plan_disabled": "Разрешите использование подписки в настройках ChatGPT в Seruen AI.",
            "auth_unavailable": "Не удалось обновить доступ ChatGPT. Попробуйте позже; подключение сохранено.",
            "model_required": "Выберите доступную модель ChatGPT в настройках.",
            "catalog_invalid": "ChatGPT не вернул список доступных моделей.",
            "network_error": "Не удалось связаться с ChatGPT. Попробуйте позже.",
            "authentication_failed": "Проверьте вход в ChatGPT и разрешение использовать подписку.",
            "permission_denied": "ChatGPT не разрешил запрос для этого аккаунта или региона.",
            "subscription_sharing_user_not_eligible": "Использование подписки недоступно этому аккаунту или рабочему пространству.",
            "subscription_sharing_usage_limit_exceeded": "Достигнут лимит ChatGPT для приложения. Проверьте Settings → Usage в ChatGPT.",
            "subscription_sharing_usage_unavailable": "ChatGPT временно не может проверить доступный лимит. Попробуйте позже.",
            "subscription_sharing_unsupported_capability": "Выбранная модель не поддерживает этот запрос. Выберите другую доступную модель.",
            "stream_incomplete": "Ответ ChatGPT не завершён; показан локальный результат.",
            "response_invalid": "Ответ ChatGPT не прошёл проверку формата.",
            "service_error": "ChatGPT временно недоступен. Попробуйте позже.",
        }
        super().__init__(messages.get(reason, messages["service_error"]))


def _token(*, force_refresh: bool = False) -> str:
    from seruen.core.chatgpt_auth import AuthError, get_access_token

    try:
        return get_access_token(force_refresh=force_refresh)
    except AuthError as exc:
        reason = "not_signed_in"
        if exc.code == "plan_disabled":
            reason = "plan_disabled"
        elif exc.code == "auth_error":
            reason = "auth_unavailable"
        raise ChatGPTError(reason) from exc
    except OSError as exc:
        raise ChatGPTError("auth_unavailable") from exc


def _error(payload: object, status: int | None = None, request_id: str | None = None) -> ChatGPTError:
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    if isinstance(payload, dict) and payload.get("type") == "error":
        error = payload  # Responses ErrorEvent places code at the top level.
    code = error.get("code") if isinstance(error, dict) else None
    if not isinstance(code, str) or len(code) > 100 or not code.replace("_", "").isalnum():
        code = {401: "authentication_failed", 403: "permission_denied", 429: "subscription_sharing_usage_limit_exceeded"}.get(status, "service_error")
    return ChatGPTError(code, status=status, request_id=request_id)


def _request(method: str, path: str, *, payload: dict | None = None, on_request: Callable | None = None):
    token = _token()
    for attempt in range(2):
        try:
            if on_request:
                on_request()
            response = requests.request(
                method, API_URL + path,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload, stream=method == "POST", timeout=(10, 60), allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise ChatGPTError("network_error") from exc
        if response.status_code == 401 and attempt == 0:
            response.close()
            token = _token(force_refresh=True)
            continue
        if not 200 <= response.status_code < 300:
            try:
                body = response.json()
            except (ValueError, requests.RequestException):
                body = {}
            error = _error(body, response.status_code, response.headers.get("x-request-id"))
            response.close()
            raise error
        return response
    raise ChatGPTError("authentication_failed")


def list_models() -> list[dict[str, str]]:
    """Visible models, preserving server order. Never invent a fallback model."""
    response = _request("GET", "/models")
    try:
        body = response.json()
        rows = body.get("models") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            raise ChatGPTError("catalog_invalid")
        models = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or row.get("visibility") != "list":
                continue
            slug, name = row.get("slug"), row.get("display_name")
            if isinstance(slug, str) and slug and slug not in seen and isinstance(name, str) and name:
                models.append({"id": slug, "name": name})
                seen.add(slug)
        return models
    except (ValueError, requests.RequestException) as exc:
        raise ChatGPTError("catalog_invalid") from exc
    finally:
        response.close()


def _strict_schema(schema: dict) -> dict:
    schema = copy.deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema


def _events(response):
    data: list[str] = []
    total = 0
    deadline = time.monotonic() + 120
    for line in response.iter_lines():
        if time.monotonic() > deadline:
            raise ChatGPTError("stream_incomplete")
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="strict")
        total += len(line.encode("utf-8"))
        if total > MAX_STREAM_BYTES:
            raise ChatGPTError("response_invalid")
        if not line:
            if data:
                raw = "\n".join(data)
                data = []
                if raw != "[DONE]":
                    yield json.loads(raw)
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data and "\n".join(data) != "[DONE]":
        yield json.loads("\n".join(data))


def generate_json(*, model: str | None, instructions: str, prompt: str, schema: dict,
                  on_request: Callable | None = None) -> dict:
    if not model or not model.strip():
        raise ChatGPTError("model_required")
    payload = {
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": prompt}],
        "store": False,
        "stream": True,
        "text": {"format": {"type": "json_schema", "name": "seruen_recommendations",
                            "strict": True, "schema": _strict_schema(schema)}},
    }
    response = _request("POST", "/responses", payload=payload, on_request=on_request)
    try:
        for event in _events(response):
            if not isinstance(event, dict):
                raise ChatGPTError("response_invalid")
            kind = event.get("type")
            if kind in {"response.failed", "error"}:
                raise _error(event.get("response", event), request_id=response.headers.get("x-request-id"))
            if kind == "response.incomplete":
                raise ChatGPTError("stream_incomplete")
            if kind != "response.completed":
                continue
            result = event.get("response", {})
            if result.get("status") != "completed":
                raise ChatGPTError("stream_incomplete")
            # Read terminal output only. Partial deltas never become a recommendation.
            parts = []
            for item in result.get("output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            parts.append(content["text"])
            value = json.loads("".join(parts))
            if not isinstance(value, dict):
                raise ChatGPTError("response_invalid")
            return value
        raise ChatGPTError("stream_incomplete")
    except requests.RequestException as exc:
        raise ChatGPTError("network_error") from exc
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ChatGPTError("response_invalid") from exc
    finally:
        response.close()
