"""Seruen AI command line.

    python -m seruen.cli --preset mega --query "хочу погулять 45 минут" --demo
    python -m seruen.cli --lat 51.1283 --lon 71.4305 --query "тихое кафе с розетками"

`advise()` is the whole pipeline and is also what the Streamlit app calls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from seruen.core import retrieval
from seruen.core import weather as weather_mod
from seruen.core.generate import Answer, generate
from seruen.core.intent import Intent, parse_intent
from seruen.core.location_presets import DEFAULT_ORIGIN as DEFAULT_ORIGIN
from seruen.core.location_presets import PRESETS
from seruen.core.retrieval import RetrievalResult
from seruen.core.route import Route, build_route, describe_route

ASTANA_TZ = ZoneInfo("Asia/Almaty")


def local_now() -> datetime:
    return datetime.now(ASTANA_TZ)


def resolve_origin(
    query: str, preset: str | None = None, lat: float | None = None, lon: float | None = None,
) -> tuple[str, float, float]:
    """Explicit coordinates/preset win; textual landmarks use the local catalogue."""
    if (lat is None) != (lon is None):
        raise ValueError("Укажите вместе --lat и --lon.")
    if lat is not None:
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Координаты должны быть конечными: широта −90…90, долгота −180…180.")
        return "указанная точка", lat, lon
    if preset:
        return PRESETS[preset]
    from seruen.core.location import infer_origin

    inferred = infer_origin(query, PRESETS)
    if inferred is not None:
        return inferred
    raise ValueError("Укажите, где вы находитесь: название места, адрес или точку старта вручную.")

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    """Read KEY=VALUE lines from .env into the environment.

    Deliberately hand-rolled: one more dependency is not worth it, and the real
    environment always wins, so `SERUEN_DEMO=1 make cli` still overrides the file.
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# both entry points (CLI and the Streamlit app) import this module
load_env()


@dataclass
class Advice:
    """Everything one request produced — answer, evidence and the data trail."""

    intent: Intent
    answer: Answer
    retrieval: RetrievalResult
    route: Route | None
    weather: dict | None
    origin: tuple[float, float]
    origin_label: str
    when: datetime
    origin_resolution: dict | None = None

    def data_used(self) -> dict:
        """The "Какие данные использованы" panel. Location is rounded on purpose."""
        from seruen.core.geo import round_coords

        lat, lon = round_coords(*self.origin)
        return {
            "location_rounded": [lat, lon],
            "location_label": self.origin_label,
            "origin_resolution": {
                key: self.origin_resolution[key]
                for key in ("status", "place_query", "ai_used", "message")
                if key in (self.origin_resolution or {})
            },
            "origin_source": ((self.origin_resolution or {}).get("place") or {}).get("source", "manual"),
            "origin_source_url": ((self.origin_resolution or {}).get("place") or {}).get("source_url", ""),
            "local_time": self.when.strftime("%Y-%m-%d %H:%M"),
            "timezone": "Asia/Almaty",
            "time_budget_min": self.intent.time_budget_min,
            "filters": self.intent.filters_ru(),
            "dataset_size": self.retrieval.stats["dataset_size"],
            "search_radius_m": self.retrieval.stats["radius_m"],
            "candidates_after_filters": self.retrieval.stats["passed_filters"],
            "candidates_prepared": self.retrieval.stats.get("returned", 0),
            "candidates_shown_to_model": getattr(self.answer, "candidates_sent_to_model", 0),
            "excluded_closed": self.retrieval.stats["excluded_closed"],
            "excluded_too_far": self.retrieval.stats["excluded_too_far"],
            "excluded_no_time": self.retrieval.stats["excluded_no_time"],
            "excluded_by_reason": {key.removeprefix("excluded_"): value for key, value in self.retrieval.stats.items() if key.startswith("excluded_")},
            "unknown_hours_kept": self.retrieval.stats["unknown_hours_kept"],
            "radius_relaxed": self.retrieval.stats["relaxed"],
            "weather": weather_mod.describe(self.weather),
            "intent_source": self.intent.source,
            "answer_mode": self.answer.mode,
            "model": self.answer.model or "—",
            "dataset_license": "OpenStreetMap, ODbL — © OpenStreetMap contributors",
        }

    def as_dict(self) -> dict:
        return {
            "query": self.intent.raw_query,
            "intent": self.intent.as_dict(),
            "answer": self.answer.as_dict(),
            "route": self.route.as_dict() if self.route else None,
            "data_used": self.data_used(),
        }


def advise(
    query: str,
    lat: float,
    lon: float,
    budget_min: int | None = None,
    when: datetime | None = None,
    demo: bool | None = None,
    use_weather: bool = True,
    origin_label: str = "указанная точка",
    dataset_path: str | Path = retrieval.DEFAULT_DATASET,
    max_picks: int = 3,
    language: str | None = None,
    overrides: dict | None = None,
    provider: str | None = None,
    model: str | None = None,
    origin_resolution: dict | None = None,
    intent_query: str | None = None,
) -> Advice:
    """intent -> weather -> retrieval -> generation -> route.

    `language` forces the answer language (the UI switch); `overrides` lets the
    UI set intent fields the user ticked by hand, e.g. {"with_child": True}.
    """
    explicit_when = when is not None
    when = when or local_now()
    when = when.replace(tzinfo=ASTANA_TZ) if when.tzinfo is None else when.astimezone(ASTANA_TZ)
    resolve_origin(query, lat=lat, lon=lon)  # validate before any network request
    if not query.strip():
        raise ValueError("Напишите, куда хотите сходить.")
    if len(query) > 2000:
        raise ValueError("Сократите запрос до 2000 символов.")
    if budget_min is not None and not 10 <= budget_min <= 480:
        raise ValueError("Бюджет времени должен быть от 10 до 480 минут.")
    if not 1 <= max_picks <= 3:
        raise ValueError("Число рекомендаций должно быть от 1 до 3.")
    if provider not in {None, "local", "chatgpt", "gemini", "openai"}:
        raise ValueError("Неизвестный провайдер модели.")
    if provider is None:
        from seruen.core.server_config import get_server_config

        config = get_server_config()
        provider = config["provider"]
        model = model or config["model"]
    if demo is None:
        demo = os.environ.get("SERUEN_DEMO", "0") == "1" or provider == "local"

    # Origin resolution has already separated the starting place from wishes.
    # Time budgets still come from the original request, never an AI rewrite.
    intent = parse_intent(query, budget_min=budget_min, use_llm=not demo and provider == "gemini",
                          context_query=intent_query)
    if intent.night and not explicit_when:
        when = when.replace(hour=23, minute=30, second=0, microsecond=0)
    if language in {"ru", "kk", "en"}:
        intent.language = language
    for field_name, value in (overrides or {}).items():
        if hasattr(intent, field_name) and value not in (None, False, 0, ""):
            setattr(intent, field_name, value)
    # Current weather must not masquerade as weather for a simulated date/time.
    live_time = abs((when - local_now()).total_seconds()) < 3600
    current_weather = weather_mod.get_weather(lat, lon) if use_weather and live_time else None

    result = retrieval.retrieve(
        intent, lat, lon, when, weather=current_weather, dataset_path=dataset_path
    )
    answer = generate(
        intent,
        result.candidates,
        when,
        radius_m=result.radius_m,
        weather=current_weather,
        demo=demo,
        provider=provider,
        model=model,
    )
    answer.picks = answer.picks[:max_picks]
    if re.search(r"аэропорт|әуежай|airport|до (?:самол[её]та|рейса)|перед вылетом", query + " " + origin_label, re.IGNORECASE):
        airport_notes = {
            "ru": " Перед вылетом отдельно оставьте время на регистрацию, досмотр и посадку. "
                  "Расчёт прогулки не гарантирует, что вы успеете на рейс.",
            "en": " Before a flight, reserve time separately for check-in, security and boarding. "
                  "This walk estimate does not guarantee making your flight.",
            "kk": " Ұшар алдында тіркелуге, тексеруге және ұшаққа отыруға бөлек уақыт қалдырыңыз. "
                  "Серуен есебі рейске үлгеруге кепілдік бермейді.",
        }
        answer.note += airport_notes[intent.language]
    # Places to work are alternatives, not a sightseeing circuit between desks.
    stationary_work = getattr(intent, "wants_work", False) and not getattr(intent, "wants_route", False)
    route = None if stationary_work else build_route(
        answer.picks[:max_picks], lat, lon,
        return_to_start=intent.return_to_start,
        budget_min=intent.time_budget_min, when=when, wants_work=intent.wants_work,
    )

    return Advice(
        intent=intent,
        answer=answer,
        retrieval=result,
        route=route,
        weather=current_weather,
        origin=(lat, lon),
        origin_label=origin_label,
        when=when,
        origin_resolution=origin_resolution,
    )


# --- printing --------------------------------------------------------------

SOURCE_LABEL = {
    "poi_data": "источник: данные POI",
    "generated_idea": "идея сгенерирована",
}
CONFIDENCE_LABEL = {"high": "высокая", "medium": "средняя", "low": "низкая"}
HOURS_LABEL = {"open": "открыто", "closed": "закрыто", "unknown": "неизвестно"}
OUTDOOR_ACCESS_LABEL = "Под открытым небом"
OUTDOOR_ACCESS_NOTE = "Учитывайте правила доступа на территорию."


def format_advice(advice: Advice) -> str:
    lines: list[str] = []
    lines.append(f"Запрос: {advice.intent.raw_query}")
    lines.append(f"Старт: {advice.origin_label} · {advice.when.strftime('%H:%M')} · бюджет {advice.intent.time_budget_min} мин")
    lines.append("")

    if not advice.answer.picks:
        lines.append(advice.answer.note)

    for number, pick in enumerate(advice.answer.picks, 1):
        lines.append(f"{number}. {pick.name}  —  {pick.distance_m} м, {pick.walk_min} мин пешком")
        lines.append(f"   Почему: {pick.why} [{SOURCE_LABEL.get(pick.source, pick.source)}]")
        lines.append(f"   Что делать: {pick.activity} [идея]")
        if pick.hours.get("context") == "outdoor":
            lines.append(f"   На месте: ~{pick.visit_minutes} мин · уверенность: {CONFIDENCE_LABEL.get(pick.confidence, pick.confidence)}")
            lines.extend([f"   {OUTDOOR_ACCESS_LABEL}", f"   {OUTDOOR_ACCESS_NOTE}"])
        else:
            lines.append(
                f"   На месте: ~{pick.visit_minutes} мин · часы: {HOURS_LABEL.get(pick.hours.get('status'), '?')}"
                f" ({pick.hours.get('detail')}) · уверенность: {CONFIDENCE_LABEL.get(pick.confidence, pick.confidence)}"
            )
        if pick.uncertainties:
            lines.append(f"   Не проверено: {', '.join(pick.uncertainties)}")
        lines.append("")

    if advice.route:
        lines.append(describe_route(advice.route, advice.intent.language))
        names = {pick.poi_id: pick.name for pick in advice.answer.picks}
        for poi_id in advice.route.order:
            lines.append(f"  {names[poi_id]}: {advice.route.stop_visit_min[poi_id]} мин на остановке")
        lines.append("")

    lines.append(f"Примечание: {advice.answer.note}")
    if advice.answer.fallback_reason:
        lines.append(f"Режим ответа: {advice.answer.mode} ({advice.answer.fallback_reason})")
    else:
        lines.append(f"Режим ответа: {advice.answer.mode}")
    lines.append("")

    used = advice.data_used()
    lines.append("Какие данные использованы:")
    lines.append(f"  местоположение (округлено): {used['location_rounded'][0]}, {used['location_rounded'][1]} ({used['location_label']})")
    lines.append(f"  время: {used['local_time']} · фильтры: {', '.join(used['filters'])}")
    lines.append(f"  датасет: {used['dataset_size']} POI, OSM/ODbL · радиус поиска: {used['search_radius_m']} м"
                 + (" (расширен автоматически)" if used["radius_relaxed"] else ""))
    lines.append(f"  прошли фильтры: {used['candidates_after_filters']}, показано модели: {used['candidates_shown_to_model']}")
    lines.append(f"  отброшено: закрыто {used['excluded_closed']}, далеко {used['excluded_too_far']}, не влезает по времени {used['excluded_no_time']}")
    lines.append(f"  часы неизвестны у кандидатов: {used['unknown_hours_kept']} (помечены, не выдаются за «открыто»)")
    lines.append(f"  погода: {used['weather']}")
    lines.append(f"  разбор запроса: {used['intent_source']} · генерация: {used['answer_mode']} · модель: {used['model']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seruen AI — ИИ-помощник по прогулкам в Астане",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Пресеты точек старта: " + ", ".join(PRESETS),
    )
    parser.add_argument("query", nargs="?", help="запрос пользователя")
    parser.add_argument("--query", dest="query_opt", help="запрос пользователя (альтернатива)")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--preset", choices=sorted(PRESETS), help="готовая точка старта")
    parser.add_argument("--budget", type=int, help="бюджет времени в минутах")
    parser.add_argument("--demo", action="store_true", help="только шаблоны, без обращения к модели")
    parser.add_argument("--provider", choices=["local", "openai", "gemini"], help="провайдер сервера; по умолчанию из .env")
    parser.add_argument("--model", help="идентификатор модели из каталога аккаунта")
    parser.add_argument("--now", help="переопределить время, например '2026-10-19 23:30'")
    parser.add_argument("--no-weather", action="store_true", help="не запрашивать погоду")
    parser.add_argument("--json", action="store_true", help="вывести JSON вместо карточек")
    parser.add_argument("--dataset", default=str(retrieval.DEFAULT_DATASET))
    args = parser.parse_args(argv)

    query = args.query_opt or args.query
    if not query:
        parser.error("нужен запрос: python -m seruen.cli \"Я у Mega Silk Way, хочу погулять 45 минут\"")

    if not query.strip() or len(query) > 2000:
        parser.error("Напишите запрос длиной от 1 до 2000 символов.")
    if args.budget is not None and not 10 <= args.budget <= 480:
        parser.error("Бюджет времени должен быть от 10 до 480 минут.")

    when = None
    if args.now:
        for fmt in ("%Y-%m-%d %H:%M", "%d.%m.%Y %H:%M", "%H:%M"):
            try:
                parsed = datetime.strptime(args.now, fmt)
                when = (
                    local_now().replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
                    if fmt == "%H:%M"
                    else parsed
                )
                break
            except ValueError:
                continue
        else:
            parser.error(f"не понял время: {args.now}")

    try:
        from seruen.core.origin_resolution import resolve_query_origin
        from seruen.core.server_config import get_server_config

        config = get_server_config()
        provider = args.provider or config["provider"]
        provider_model = {"openai": "OPENAI_MODEL", "gemini": "GEMINI_MODEL"}.get(provider)
        model = args.model or (os.environ.get(provider_model) if provider_model else None)
        demo = args.demo or provider == "local" or os.environ.get("SERUEN_DEMO") == "1"
        resolution = None
        if args.preset or args.lat is not None or args.lon is not None:
            label, lat, lon = resolve_origin(query, args.preset, args.lat, args.lon)
        else:
            resolution = resolve_query_origin(
                query, use_ai=not demo and provider == "openai", allow_online=not demo,
                model=model, pois=retrieval.load_pois(args.dataset),
            )
            if resolution.status != "resolved" or resolution.place is None:
                if args.json:
                    print(json.dumps({"origin_resolution": resolution.as_dict()}, ensure_ascii=False, indent=2))
                    return 2
                choices = "\n".join(f"  {p.name} — {p.address}" for p in resolution.candidates)
                parser.error(resolution.message + ("\n" + choices if choices else ""))
            label, lat, lon = resolution.place.name, resolution.place.lat, resolution.place.lon
        advice = advise(
            query, lat, lon, budget_min=args.budget, when=when,
            demo=demo, use_weather=not args.no_weather,
            origin_label=label, dataset_path=args.dataset,
            provider=provider, model=model,
            origin_resolution=resolution.as_dict() if resolution else None,
            intent_query=resolution.context_query if resolution else None,
        )
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(advice.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_advice(advice))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
