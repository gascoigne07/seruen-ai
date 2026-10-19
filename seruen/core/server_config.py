"""Owner-managed backend configuration, never browser/session credentials."""

import os


def get_server_config() -> dict:
    provider = os.environ.get("SERUEN_PROVIDER", "").strip().lower()
    if os.environ.get("SERUEN_DEMO", "0") == "1":
        provider = "local"
    if not provider:
        provider = "openai" if os.environ.get("OPENAI_API_KEY", "").strip() else (
            "gemini" if os.environ.get("GOOGLE_API_KEY", "").strip() else "local"
        )
    if provider not in {"local", "openai", "gemini"}:
        raise ValueError("SERUEN_PROVIDER должен быть local, openai или gemini.")
    model = os.environ.get({"openai": "OPENAI_MODEL", "gemini": "GEMINI_MODEL"}.get(provider, ""), "").strip() or None
    key_present = bool(os.environ.get({"openai": "OPENAI_API_KEY", "gemini": "GOOGLE_API_KEY"}.get(provider, ""), "").strip())
    return {"provider": provider, "model": model, "ready": provider == "local" or bool(key_present and model)}
