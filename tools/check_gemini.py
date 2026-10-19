"""Check Gemini configuration without printing credentials or error bodies."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seruen.cli import load_env  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    load_env()
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", "").strip()
    print("GOOGLE_API_KEY: " + ("настроен" if key else "не задан"))
    print("GEMINI_MODEL: " + (model if model else "не задан"))
    if not key:
        print("Доступен локальный шаблонный режим. Для Gemini заполните .env и выполните make models.")
        return 0
    if not args.list_models:
        print("Сетевой запрос не выполнялся. Список доступных моделей: make models.")
        return 0
    try:
        from google import genai

        with genai.Client(api_key=key, http_options={"timeout": 15000}) as client:
            for item in client.models.list():
                actions = getattr(item, "supported_actions", []) or []
                if "generateContent" in actions:
                    print(item.name.removeprefix("models/"))
    except Exception as exc:
        print(f"Не удалось получить список ({type(exc).__name__}). Проверьте ключ, сеть и квоту.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
