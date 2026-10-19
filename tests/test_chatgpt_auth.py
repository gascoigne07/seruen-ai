"""Offline checks of the local OAuth boundary, using generated signing keys."""

import json
import os
import time
from urllib.parse import parse_qs, urlsplit

import jwt
import pytest
import requests
from cryptography.hazmat.primitives.asymmetric import rsa

from seruen.core import chatgpt_auth as auth


@pytest.fixture(autouse=True)
def isolated_auth(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, "AUTH_DIR", tmp_path / ".seruen")
    monkeypatch.setattr(auth, "_ATTEMPTS", [])
    yield
    for attempt in auth._ATTEMPTS:
        auth.cancel_sign_in(attempt)


@pytest.fixture
def signer(monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update(kid="test-key", use="sig")

    def get(url):
        if url.endswith("openid-configuration"):
            return {"issuer": auth.ISSUER, "jwks_uri": auth.ISSUER + "/jwks",
                    "revocation_endpoint": auth.ISSUER + "/revoke"}
        return {"keys": [jwk]}

    monkeypatch.setattr(auth, "_json_get", get)

    def sign(**overrides):
        claims = {"iss": auth.ISSUER, "aud": "oaiapp_test", "sub": "user-1",
                  "exp": int(time.time()) + 600, "iat": int(time.time()),
                  "nonce": "nonce", "email": "person@example.test"}
        claims.update(overrides)
        return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test-key"})

    return sign


def _profile(**overrides):
    profile = {"subject": "user-1", "email": "person@example.test", "access_token": "SECRET-ACCESS",
               "refresh_token": "SECRET-REFRESH", "id_token": "SECRET-ID", "scopes": [auth.PLAN_SCOPE],
               "expires_at": time.time() + 3600}
    profile.update(overrides)
    with auth._locked():
        auth._write("chatgpt-auth.json", {"active": "oaiapp_test", "profiles": {"oaiapp_test": profile}})


def _tokens(id_token="", **overrides):
    tokens = {"access_token": "NEW-ACCESS", "refresh_token": "NEW-REFRESH", "id_token": id_token,
              "token_type": "Bearer", "expires_in": 3600, "scope": auth.SCOPES}
    tokens.update(overrides)
    return tokens


def test_status_is_read_only_and_contains_no_tokens():
    assert not auth.get_status()["authenticated"]
    assert not auth.AUTH_DIR.exists()
    _profile()
    public = auth.get_status()
    assert public["plan_enabled"]
    assert "SECRET" not in json.dumps(public)
    assert os.stat(auth.AUTH_DIR).st_mode & 0o777 == 0o700
    assert os.stat(auth.AUTH_DIR / "chatgpt-auth.json").st_mode & 0o777 == 0o600


def test_begin_uses_pkce_and_stable_private_host():
    first = auth.begin_sign_in()
    one = parse_qs(urlsplit(first.url).query)
    assert one["client_id"] == ["dynamic_agent_client"]
    assert one["agent_name_hint"] == ["Seruen AI"]
    assert one["code_challenge_method"] == ["S256"]
    assert first._verifier not in first.url
    assert one["redirect_uri"][0].startswith("http://127.0.0.1:")
    assert "nonce" not in repr(first)
    second = auth.begin_sign_in()
    two = parse_qs(urlsplit(second.url).query)
    assert one["ext_agent_host_id"] == two["ext_agent_host_id"]
    assert one["state"] != two["state"]
    assert first.status == "cancelled"


@pytest.mark.parametrize("override", [
    {"iss": "https://evil.test"}, {"aud": "wrong"}, {"nonce": "wrong"},
    {"exp": 1}, {"iat": time.time() + 86400}, {"sub": ""}, {"azp": "wrong"},
])
def test_id_token_rejects_untrusted_claims(signer, override):
    with pytest.raises(auth.AuthError, match="личность"):
        auth._verify_id_token(signer(**override), "oaiapp_test", "nonce")


def test_real_signature_is_checked(signer):
    token = signer()
    assert auth._verify_id_token(token, "oaiapp_test", "nonce")["sub"] == "user-1"
    parts = token.split(".")
    parts[2] = ("A" if parts[2][0] != "A" else "B") + parts[2][1:]
    with pytest.raises(auth.AuthError):
        auth._verify_id_token(".".join(parts), "oaiapp_test", "nonce")
    bad = jwt.encode({"sub": "user-1"}, "password", algorithm="HS256", headers={"kid": "test-key"})
    with pytest.raises(auth.AuthError):
        auth._verify_id_token(bad, "oaiapp_test", "nonce")


def test_registration_saved_before_exchange_and_callback_completes(signer, monkeypatch):
    attempt = auth.begin_sign_in()

    def exchange(form):
        assert "oaiapp_test" in auth._load()["profiles"]
        assert form["redirect_uri"] == attempt._redirect_uri
        assert form["code_verifier"] == attempt._verifier
        return _tokens(signer(nonce=attempt._nonce))

    monkeypatch.setattr(auth, "_token_request", exchange)
    response = requests.get(attempt._redirect_uri, params={"state": attempt._state, "code": "code",
                                                          "client_id": "oaiapp_test"}, timeout=5)
    assert response.status_code == 200
    assert attempt.status == "completed"
    assert auth.get_status()["plan_enabled"]
    assert auth.get_access_token() == "NEW-ACCESS"


def test_invalid_state_and_duplicate_parameters_never_exchange(monkeypatch):
    attempt = auth.begin_sign_in()
    monkeypatch.setattr(auth, "_token_request", lambda form: pytest.fail("Must not exchange"))
    for query in [{"state": ["wrong"]}, {"state": [attempt._state, "extra"]}]:
        with pytest.raises(auth.AuthError):
            auth._finish_callback(attempt, query)
    assert not auth.get_status()["authenticated"]


def test_decline_does_not_exchange(monkeypatch):
    attempt = auth.begin_sign_in()
    monkeypatch.setattr(auth, "_token_request", lambda form: pytest.fail("Must not exchange"))
    with pytest.raises(auth.AuthError) as error:
        auth._finish_callback(attempt, {"state": [attempt._state], "error": ["access_denied"]})
    assert error.value.code == "access_denied"


def test_failed_exchange_retains_registration(monkeypatch):
    attempt = auth.begin_sign_in()

    def fail(form):
        raise auth.AuthError("expired", "invalid_grant")

    monkeypatch.setattr(auth, "_token_request", fail)
    with pytest.raises(auth.AuthError):
        auth._finish_callback(attempt, {"state": [attempt._state], "code": ["code"], "client_id": ["oaiapp_test"]})
    retry = auth.begin_sign_in()
    assert parse_qs(urlsplit(retry.url).query)["client_id"] == ["oaiapp_test"]
    assert not auth.get_status()["authenticated"]


def test_returning_client_mismatch_rejected(monkeypatch):
    _profile()
    attempt = auth.begin_sign_in()
    monkeypatch.setattr(auth, "_token_request", lambda form: pytest.fail("Must not exchange"))
    with pytest.raises(auth.AuthError, match="другому"):
        auth._finish_callback(attempt, {"state": [attempt._state], "code": ["code"], "client_id": ["other"]})
    assert auth.get_access_token() == "SECRET-ACCESS"


def test_no_plan_permission_retains_identity_but_blocks_inference(signer, monkeypatch):
    attempt = auth.begin_sign_in()
    monkeypatch.setattr(auth, "_token_request", lambda form: _tokens(signer(nonce=attempt._nonce), scope="openid"))
    auth._finish_callback(attempt, {"state": [attempt._state], "code": ["code"], "client_id": ["oaiapp_test"]})
    assert auth.get_status()["authenticated"]
    assert not auth.get_status()["plan_enabled"]
    with pytest.raises(auth.AuthError) as error:
        auth.get_access_token()
    assert error.value.code == "plan_disabled"


def test_refresh_rotates_and_is_not_repeated(monkeypatch):
    _profile(expires_at=0)
    calls = []

    def refresh(form):
        calls.append(form)
        return _tokens()

    monkeypatch.setattr(auth, "_token_request", refresh)
    assert auth.get_access_token() == "NEW-ACCESS"
    assert auth.get_access_token() == "NEW-ACCESS"
    assert len(calls) == 1
    assert "scope" not in calls[0]
    assert calls[0]["client_id"] == "oaiapp_test"
    assert auth._load()["profiles"]["oaiapp_test"]["refresh_token"] == "NEW-REFRESH"


@pytest.mark.parametrize("code,cleared", [("invalid_grant", True), ("network_error", False)])
def test_refresh_failure_clears_only_terminal_failure(monkeypatch, code, cleared):
    _profile(expires_at=0)

    def fail(form):
        raise auth.AuthError("safe message", code)

    monkeypatch.setattr(auth, "_token_request", fail)
    with pytest.raises(auth.AuthError):
        auth.get_access_token()
    assert auth.get_status()["authenticated"] is not cleared
    assert auth.get_status()["client_id"] == "oaiapp_test"


def test_force_refresh_and_revocation_failure_preserve_mapping(monkeypatch):
    _profile()
    monkeypatch.setattr(auth, "_token_request", lambda form: _tokens())
    assert auth.get_access_token(force_refresh=True) == "NEW-ACCESS"
    monkeypatch.setattr(auth, "_discovery", lambda: {"revocation_endpoint": auth.ISSUER + "/revoke"})

    def fail(*args, **kwargs):
        raise requests.ConnectionError("SENSITIVE-SERVER-TEXT")

    monkeypatch.setattr(auth.requests, "post", fail)
    result = auth.sign_out()
    assert not result["remote_revoked"]
    assert "SENSITIVE" not in result["message"]
    profile = auth._load()["profiles"]["oaiapp_test"]
    assert profile["subject"] == "user-1"
    assert not {"access_token", "refresh_token", "id_token"} & profile.keys()


def test_discovery_cannot_redirect_tokens_to_other_host():
    with pytest.raises(auth.AuthError):
        auth._official_url("https://auth.openai.com.evil.test/revoke")


def test_auth_files_cannot_follow_symlinks(tmp_path):
    auth.AUTH_DIR.mkdir()
    target = tmp_path / "unrelated"
    target.write_text("do not touch")
    (auth.AUTH_DIR / "chatgpt-auth.json").symlink_to(target)
    with pytest.raises(auth.AuthError):
        auth.get_status()
    assert target.read_text() == "do not touch"


def test_logout_preserves_other_account_and_issued_registration(monkeypatch):
    _profile()
    with auth._locked():
        data = auth._load()
        data["profiles"]["second"] = {"subject": "user-2", "access_token": "OTHER-ACCESS",
                                      "scopes": [auth.PLAN_SCOPE]}
        auth._write("chatgpt-auth.json", data)
    monkeypatch.setattr(auth, "_discovery", lambda: {"revocation_endpoint": auth.ISSUER + "/revoke"})

    class Revoked:
        status_code = 200

    monkeypatch.setattr(auth.requests, "post", lambda *args, **kwargs: Revoked())
    assert auth.sign_out()["remote_revoked"]
    assert auth._load()["profiles"]["second"]["access_token"] == "OTHER-ACCESS"
    attempt = auth.begin_sign_in()
    params = parse_qs(urlsplit(attempt.url).query)
    assert params["client_id"] == ["oaiapp_test"]
    assert "id_token_hint" not in params
    auth.select_account("second")
    assert auth.get_status()["client_id"] == "second"


def test_returning_identity_change_does_not_overwrite_active(signer, monkeypatch):
    _profile()
    attempt = auth.begin_sign_in()
    monkeypatch.setattr(auth, "_token_request",
                        lambda form: _tokens(signer(nonce=attempt._nonce, sub="different-user")))
    with pytest.raises(auth.AuthError, match="другой аккаунт"):
        auth._finish_callback(attempt, {"state": [attempt._state], "code": ["code"]})
    assert auth.get_access_token() == "SECRET-ACCESS"
