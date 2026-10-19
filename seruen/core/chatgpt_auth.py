"""App-owned Sign in with ChatGPT for a trusted, locally hosted Seruen AI.

No external application or browser credentials are read. Calling begin_sign_in opens only a loopback
listener; the person must follow the returned URL and approve access themselves.
https://developers.openai.com/siwc/token-sharing-open-source/sign-in
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import os
import secrets
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import jwt
import requests

AUTH_DIR = Path(__file__).resolve().parents[2] / ".seruen"
ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
USAGE_URL = "https://chatgpt.com/settings/usage"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
_LOCK = threading.RLock()
_ATTEMPTS: list[SignInAttempt] = []
_TERMINAL_REFRESH = {
    "invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired",
    "refresh_token_invalidated", "refresh_token_reused",
}


class AuthError(RuntimeError):
    """A safe user-facing error; raw server bodies and tokens are never included."""

    def __init__(self, message: str, code: str = "auth_error"):
        super().__init__(message)
        self.code = code


@dataclass(repr=False)
class SignInAttempt:
    url: str = field(repr=False)
    status: str = "waiting"
    error: str = ""
    _state: str = field(default="", repr=False)
    _nonce: str = field(default="", repr=False)
    _verifier: str = field(default="", repr=False)
    _client_id: str = ""
    _redirect_uri: str = ""
    _expires_at: float = 0
    _stopped: threading.Event = field(default_factory=threading.Event, repr=False)
    _server: HTTPServer | None = field(default=None, repr=False)

    @property
    def done(self) -> bool:
        return self.status != "waiting"

    def __repr__(self) -> str:
        return f"SignInAttempt(status={self.status!r})"


def _private_dir() -> None:
    if AUTH_DIR.is_symlink():
        raise AuthError("Каталог авторизации не должен быть символической ссылкой.")
    AUTH_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    AUTH_DIR.chmod(0o700)


@contextlib.contextmanager
def _locked():
    """Serialize rotating refresh tokens across threads and local processes."""
    with _LOCK:
        _private_dir()
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(AUTH_DIR / "auth.lock", flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _read(name: str) -> dict:
    path = AUTH_DIR / name
    if AUTH_DIR.is_symlink() or path.is_symlink():
        raise AuthError("Файл авторизации не должен быть символической ссылкой.")
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (OSError, ValueError):
        raise AuthError("Не удалось прочитать локальные настройки ChatGPT.") from None


def _write(name: str, value: dict) -> None:
    _private_dir()
    path = AUTH_DIR / name
    if path.is_symlink():
        raise AuthError("Файл авторизации не должен быть символической ссылкой.")
    fd, tmp = tempfile.mkstemp(prefix=".auth-", dir=AUTH_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _load() -> dict:
    data = _read("chatgpt-auth.json")
    data.setdefault("profiles", {})
    if not isinstance(data["profiles"], dict):
        raise AuthError("Неверный формат локальных настроек ChatGPT.")
    return data


def _summary(client_id: str, profile: dict, index: int) -> dict:
    connected = bool(profile.get("subject") and profile.get("access_token"))
    email = profile.get("email", "")
    return {
        "client_id": client_id,
        "email": email,
        "label": f"{email or 'ChatGPT'} · подключение {index}",
        "authenticated": connected,
        "plan_enabled": connected and PLAN_SCOPE in profile.get("scopes", []),
    }


def get_status() -> dict:
    """Local, read-only status. Never returns credentials or starts a request."""
    with _LOCK:
        data = _load()
        profiles = [_summary(cid, p, i) for i, (cid, p) in enumerate(data["profiles"].items(), 1)]
        selected = next((p for p in profiles if p["client_id"] == data.get("active")), None)
        result = selected or {"authenticated": False, "plan_enabled": False,
                              "client_id": "", "email": "", "label": "ChatGPT не подключён"}
        return {**result, "profiles": profiles}


def _official_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.netloc != "auth.openai.com" or parsed.fragment:
        raise AuthError("Не удалось проверить адрес службы авторизации.")
    return value


def _json_get(url: str) -> dict:
    try:
        response = requests.get(_official_url(url), timeout=20, allow_redirects=False)
        if response.status_code != 200:
            raise AuthError("Служба авторизации временно недоступна. Попробуйте позже.")
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (requests.RequestException, ValueError):
        raise AuthError("Не удалось связаться со службой авторизации. Попробуйте позже.") from None


def _discovery() -> dict:
    doc = _json_get(ISSUER + "/.well-known/openid-configuration")
    if doc.get("issuer") != ISSUER:
        raise AuthError("Не удалось проверить службу авторизации.")
    return doc


def _verify_id_token(token: str, client_id: str, nonce: str | None) -> dict:
    try:
        doc = _discovery()
        keys = _json_get(doc["jwks_uri"])["keys"]
        header = jwt.get_unverified_header(token)
        if header.get("alg") not in {"RS256", "ES256"} or not header.get("kid"):
            raise ValueError
        matching = [key for key in keys if key.get("kid") == header["kid"] and key.get("use", "sig") == "sig"]
        if len(matching) != 1:
            raise ValueError
        key = jwt.PyJWK.from_dict(matching[0], algorithm=header["alg"])
        claims = jwt.decode(token, key.key, algorithms=[header["alg"]], audience=client_id, issuer=ISSUER,
                            options={"require": ["iss", "aud", "exp", "iat", "sub"]})
        if not isinstance(claims["sub"], str) or not claims["sub"]:
            raise ValueError
        if nonce is not None and not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
            raise ValueError
        if claims.get("azp", client_id) != client_id:
            raise ValueError
        if isinstance(claims["aud"], list) and len(claims["aud"]) > 1 and claims.get("azp") != client_id:
            raise ValueError
        return claims
    except (KeyError, TypeError, ValueError, jwt.PyJWTError):
        raise AuthError("Не удалось подтвердить личность ChatGPT. Выполните вход снова.", "invalid_identity") from None


def _token_request(form: dict) -> dict:
    try:
        response = requests.post(TOKEN_URL, data=form, timeout=25, allow_redirects=False)
        value = response.json()
        if response.status_code != 200:
            error = value.get("error", "") if isinstance(value, dict) else ""
            code = error if isinstance(error, str) else error.get("code", "") if isinstance(error, dict) else ""
            if code in _TERMINAL_REFRESH:
                raise AuthError("Сеанс ChatGPT истёк. Войдите снова.", code)
            if code == "invalid_client":
                raise AuthError("Регистрация приложения отклонена. Проверьте доступность входа ChatGPT.", code)
            raise AuthError("ChatGPT не завершил вход. Попробуйте позже.")
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (requests.RequestException, ValueError):
        raise AuthError("Не удалось связаться с ChatGPT. Попробуйте позже.") from None


def _token_fields(tokens: dict, previous: dict | None = None) -> dict:
    previous = previous or {}
    try:
        if tokens.get("token_type", "").lower() != "bearer" or not isinstance(tokens.get("access_token"), str):
            raise ValueError
        if not tokens["access_token"]:
            raise ValueError
        expires = float(tokens["expires_in"])
        if not 0 < expires <= 86400:
            raise ValueError
        scope = tokens.get("scope")
        if scope is None and previous:
            scopes = previous.get("scopes", [])
        elif isinstance(scope, str):
            scopes = scope.split()
        else:
            raise ValueError
        refresh = tokens.get("refresh_token", previous.get("refresh_token", ""))
        if not isinstance(refresh, str):
            raise ValueError
        return {"access_token": tokens["access_token"], "refresh_token": refresh, "scopes": scopes,
                "token_type": "Bearer", "expires_at": time.time() + expires,
                "earliest_refresh_at": tokens.get("earliest_refresh_at", 0)}
    except (KeyError, TypeError, ValueError):
        raise AuthError("ChatGPT вернул неполные данные входа. Войдите снова.") from None


def _finish_callback(attempt: SignInAttempt, query: dict[str, list[str]]) -> None:
    if attempt._stopped.is_set() or time.monotonic() >= attempt._expires_at:
        raise AuthError("Время входа истекло. Начните вход снова.")
    if any(len(v) != 1 for v in query.values()):
        raise AuthError("Ответ входа не прошёл проверку.")
    if not secrets.compare_digest(query.get("state", [""])[0], attempt._state):
        raise AuthError("Ответ входа не прошёл проверку.")
    if query.get("error"):
        raise AuthError("Вход отменён или доступ не предоставлен. Можно повторить вход.", "access_denied")
    client_id = query.get("client_id", [attempt._client_id])[0]
    if not client_id or client_id == "dynamic_agent_client" or len(client_id) > 512:
        raise AuthError("ChatGPT не завершил регистрацию приложения.")
    if attempt._client_id and client_id != attempt._client_id:
        raise AuthError("Ответ входа относится к другому подключению.")
    code = query.get("code", [""])[0]
    if not code:
        raise AuthError("ChatGPT не вернул подтверждение входа.")
    with _locked():
        data = _load()
        previous = data["profiles"].get(client_id, {})
        data["profiles"].setdefault(client_id, {"client_id": client_id})
        if not data.get("active"):
            data["active"] = client_id
        _write("chatgpt-auth.json", data)  # Retain issued registration even if code exchange fails.
        tokens = _token_request({"grant_type": "authorization_code", "client_id": client_id,
                                 "code": code, "code_verifier": attempt._verifier,
                                 "redirect_uri": attempt._redirect_uri, "resource": RESOURCE})
        claims = _verify_id_token(tokens.get("id_token", ""), client_id, attempt._nonce)
        if previous.get("subject") and previous["subject"] != claims["sub"]:
            raise AuthError("ChatGPT вернул другой аккаунт. Добавьте его отдельным подключением.")
        fields = _token_fields(tokens)
        if attempt._stopped.is_set():
            raise AuthError("Вход отменён.")
        data["profiles"][client_id] = {"client_id": client_id, "issuer": ISSUER,
                                       "subject": claims["sub"], "email": claims.get("email", ""),
                                       "id_token": tokens["id_token"], **fields}
        data["active"] = client_id
        _write("chatgpt-auth.json", data)


def begin_sign_in(client_id: str | None = None, *, new_account: bool = False,
                  request_consent: bool = False) -> SignInAttempt:
    """Start a ten-minute, one-use loopback callback, without opening a browser."""
    with _locked():
        for previous_attempt in _ATTEMPTS:
            cancel_sign_in(previous_attempt)
        _ATTEMPTS.clear()
        data = _load()
        selected = "" if new_account else client_id or data.get("active", "")
        if selected and selected not in data["profiles"]:
            raise AuthError("Подключение ChatGPT не найдено.")
        profile = data["profiles"].get(selected, {})
        host = _read("chatgpt-host.json")
        if not host.get("ext_agent_host_id"):
            host = {"ext_agent_host_id": "urn:uuid:" + str(uuid.uuid4())}
            _write("chatgpt-host.json", host)
        attempt = SignInAttempt(url="", _state=secrets.token_urlsafe(32), _nonce=secrets.token_urlsafe(32),
                                _verifier=secrets.token_urlsafe(64), _client_id=selected,
                                _expires_at=time.monotonic() + 600)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass  # Callback URLs contain authorization codes.

            def do_GET(self):
                parsed = urlsplit(self.path)
                if parsed.path != "/auth/callback" or len(self.path) > 16384:
                    self.send_error(404)
                    return
                if self.headers.get("Host") != urlsplit(attempt._redirect_uri).netloc:
                    self.send_error(400)
                    return
                try:
                    _finish_callback(attempt, parse_qs(parsed.query, keep_blank_values=True, max_num_fields=20))
                    attempt.status = "completed"
                    message = "Вход завершён. Вернитесь в Seruen AI."
                except (AuthError, ValueError, OSError) as exc:
                    attempt.status = "error"
                    attempt.error = str(exc) if isinstance(exc, AuthError) else "Не удалось завершить вход. Повторите попытку."
                    message = attempt.error
                finally:
                    attempt._stopped.set()
                body = message.encode("utf-8")
                self.send_response(200 if attempt.status == "completed" else 400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(body)

        class CallbackServer(HTTPServer):
            def get_request(self):
                connection, address = super().get_request()
                connection.settimeout(5)
                return connection, address

        server = CallbackServer(("127.0.0.1", 0), Handler)
        server.timeout = 0.25
        attempt._server = server
        attempt._redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
        challenge = base64.urlsafe_b64encode(hashlib.sha256(attempt._verifier.encode()).digest()).rstrip(b"=").decode()
        params = {"client_id": selected or "dynamic_agent_client", "ext_agent_host_id": host["ext_agent_host_id"],
                  "redirect_uri": attempt._redirect_uri, "response_type": "code", "scope": SCOPES,
                  "resource": RESOURCE, "state": attempt._state, "nonce": attempt._nonce,
                  "code_challenge_method": "S256", "code_challenge": challenge}
        if not selected:
            params["agent_name_hint"] = "Seruen AI"
        elif profile.get("id_token"):
            params["id_token_hint"] = profile["id_token"]
        if profile.get("email"):
            params["login_hint"] = profile["email"]
        if request_consent:
            params["prompt"] = "consent"
        attempt.url = AUTHORIZE_URL + "?" + urlencode(params)

        def listen():
            try:
                while not attempt._stopped.is_set() and time.monotonic() < attempt._expires_at:
                    server.handle_request()
                if not attempt.done:
                    attempt.status = "error"
                    attempt.error = "Время входа истекло. Начните вход снова."
            finally:
                server.server_close()
                attempt._state = attempt._nonce = attempt._verifier = ""

        threading.Thread(target=listen, name="seruen-chatgpt-signin", daemon=True).start()
        _ATTEMPTS.append(attempt)
        return attempt


def poll_sign_in(attempt: SignInAttempt) -> dict:
    return {"status": attempt.status, "error": attempt.error}


def cancel_sign_in(attempt: SignInAttempt) -> None:
    if not attempt.done:
        attempt.status = "cancelled"
        attempt._stopped.set()


def select_account(client_id: str) -> None:
    with _locked():
        data = _load()
        if client_id not in data["profiles"]:
            raise AuthError("Подключение ChatGPT не найдено.")
        data["active"] = client_id
        _write("chatgpt-auth.json", data)


def _clear_tokens(profile: dict) -> None:
    for field_name in ("access_token", "refresh_token", "id_token", "expires_at", "earliest_refresh_at", "scopes"):
        profile.pop(field_name, None)


def get_access_token(force_refresh: bool = False) -> str:
    """Return active credentials to backend code only; renew under a process lock."""
    with _locked():
        data = _load()
        cid = data.get("active", "")
        profile = data["profiles"].get(cid, {})
        if not profile.get("access_token") or not profile.get("subject"):
            raise AuthError("Подключите ChatGPT в настройках.", "not_connected")
        if PLAN_SCOPE not in profile.get("scopes", []):
            raise AuthError("Разрешите Seruen AI использовать подписку ChatGPT.", "plan_disabled")
        if not force_refresh and float(profile.get("expires_at", 0)) > time.time() + 60:
            return profile["access_token"]
        if not profile.get("refresh_token"):
            raise AuthError("Сеанс ChatGPT истёк. Войдите снова.", "not_connected")
        try:
            tokens = _token_request({"grant_type": "refresh_token", "client_id": cid,
                                     "refresh_token": profile["refresh_token"], "resource": RESOURCE})
        except AuthError as exc:
            if exc.code in _TERMINAL_REFRESH:
                _clear_tokens(profile)
                _write("chatgpt-auth.json", data)
            raise
        fields = _token_fields(tokens, profile)
        if tokens.get("id_token"):
            claims = _verify_id_token(tokens["id_token"], cid, None)
            if claims["sub"] != profile["subject"]:
                raise AuthError("Аккаунт ChatGPT изменился. Войдите снова.")
            profile["id_token"] = tokens["id_token"]
        profile.update(fields)
        _write("chatgpt-auth.json", data)
        if PLAN_SCOPE not in profile["scopes"]:
            raise AuthError("Разрешите Seruen AI использовать подписку ChatGPT.", "plan_disabled")
        return profile["access_token"]


def sign_out(client_id: str | None = None) -> dict:
    """Revoke selected renewable session and clear only its credentials locally."""
    with _locked():
        for attempt in _ATTEMPTS:
            cancel_sign_in(attempt)
        data = _load()
        cid = client_id or data.get("active", "")
        profile = data["profiles"].get(cid, {})
        revoked = not bool(profile.get("refresh_token"))
        if profile.get("refresh_token"):
            try:
                endpoint = _official_url(_discovery()["revocation_endpoint"])
                for retry in range(2):
                    try:
                        response = requests.post(endpoint, data={"token": profile["refresh_token"],
                                                  "token_type_hint": "refresh_token", "client_id": cid},
                                                 timeout=10, allow_redirects=False)
                        revoked = response.status_code == 200
                        if revoked or response.status_code < 500:
                            break
                    except requests.RequestException:
                        revoked = False
                    if retry == 0:
                        time.sleep(0.3)
            except (AuthError, requests.RequestException, KeyError):
                revoked = False
        _clear_tokens(profile)
        _write("chatgpt-auth.json", data)
        message = "Вы вышли из ChatGPT в Seruen AI."
        if not revoked:
            message += " Отзыв доступа на сервере не подтверждён. Отключите приложение в настройках ChatGPT."
        return {"remote_revoked": revoked, "message": message}
