"""Owner-only configuration check. No secrets printed; no inference performed."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seruen.cli import load_env  # noqa: E402
from seruen.core.openai_provider import OpenAIError, list_models  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    load_env()
    key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    model = bool(os.environ.get("OPENAI_MODEL", "").strip())
    print("OPENAI_API_KEY: " + ("настроен" if key else "не задан"))
    print("OPENAI_MODEL: " + ("настроена" if model else "не задана"))
    if not key:
        print("Владелец задаёт ключ и модель в .env; посетителям подключение не требуется.")
        return 0
    if not args.list_models:
        print("Сетевые запросы не выполнялись. Каталог моделей: make models.")
        return 0
    try:
        for model_id in list_models():
            print(model_id)
        print("Выберите текстовую модель с поддержкой Responses API и structured outputs.")
    except OpenAIError as exc:
        print(f"Каталог недоступен: {exc.reason}. Проверьте серверный ключ, доступ и квоту API.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
