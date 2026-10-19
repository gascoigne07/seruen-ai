"""Generate scenario reports from actual answers, with reproducible offline defaults.

    make scenarios
    .venv/bin/python tools/gen_test_scenarios.py --live --output-dir docs/live

--live explicitly enables the owner's configured API and can incur charges.
Default runs always use local templates, even when server credentials are configured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seruen.cli import ASTANA_TZ, PRESETS, advise, format_advice, local_now, resolve_origin  # noqa: E402
from seruen.core.server_config import get_server_config  # noqa: E402

SCENARIOS = [
    dict(n=1, title="Прогулка у Mega Silk Way на 45 минут",
         query="Я у Mega Silk Way, хочу погулять 45 минут", preset="mega", budget=45, clock=None,
         expect="1–3 места для прогулки в пешем бюджете; маршрут только если помещается",
         check="категории прогулки, дорога и посещение с запасом; маршрут необязателен",
         categories={"park", "landmark", "viewpoint"}),
    dict(n=2, title="Тихое кафе с розетками, поработать час",
         query="ищу тихое кафе с розетками поработать за ноутбуком, час", preset="baiterek", budget=60, clock=None,
         expect="кафе рядом; тишина и розетки остаются неподтверждёнными",
         check="явное кафе не заменяется библиотекой; отсутствие данных не превращается в обещание",
         categories={"cafe"}, no_route=True),
    dict(n=3, title="Прогулка с ребёнком и коляской",
         query="гуляю с ребёнком и коляской, есть час, куда пойти", preset="khan", budget=60, clock=None,
         expect="места для семейной прогулки с wheelchair=yes; допустим короткий или пустой ответ",
         check="доступность — жёсткое условие, бары/пабы исключены; тип объекта не доказывает наличие интерактивов",
         child=True, wheelchair=True),
    dict(n=4, title="Турист: 2 часа до самолёта",
         query="я турист, до самолёта 2 часа, хочу посмотреть что-то рядом", preset="airport", budget=120, clock=None,
         expect="варианты с возвратом в пешем бюджете либо честный отказ; без гарантии успеть на рейс",
         check="обязателен возврат; при отсутствии кандидатов модель не вызывается; регистрация и досмотр отдельно",
         return_to_start=True),
    dict(n=5, title="Необычная активность ночью",
         query="хочу чего-то необычного ночью, 90 минут", preset="baiterek", budget=90, clock="23:30",
         expect="места без известного закрытия в 23:30, неизвестные часы явно помечены",
         check="учитывается ночь; идея не доказывает безопасность, доступ или работу заведения",
         night=True, unusual=True),
    dict(n=6, title="Ответ на английском языке",
         query="I have 40 minutes near Baiterek, show me something interesting", preset="baiterek", budget=40, clock=None,
         expect="объяснения и действия на английском", check="язык определяется по запросу; названия берутся из POI",
         language="en"),
    dict(n=7, title="Казахский запрос: парк за 40 минут",
         query="Байтерек жанындамын, 40 минут уақытым бар, саябаққа барғым келеді", preset="baiterek", budget=40, clock=None,
         expect="парк в пределах бюджета или честный ответ на казахском, что вариантов нет",
         check="явный парк не заменяется кафе; нехватка времени не скрывается",
         language="kk", categories={"park"}),
    dict(n=8, title="Казахский запрос: парк за 60 минут",
         query="Байтерек жанындамын, 60 минут уақытым бар, саябаққа барғым келеді", preset="baiterek", budget=60, clock=None,
         expect="1–3 подходящих парка с объяснением и действием на казахском",
         check="увеличение бюджета допускает подходящий парк, часы по-прежнему не выдумываются",
         language="kk", categories={"park"}),
    dict(n=9, title="Запрос трёх остановок с возвратом к Байтереку",
         query="Я у Байтерека, хочу пеший маршрут из 3 мест за 90 минут и вернуться к началу",
         preset="baiterek", budget=90, clock=None,
         expect="маршрут из 2–3 остановок с возвращением и запасом; три остановки, если помещаются",
         check="явный запрос маршрута, разные остановки, обратный путь включён в бюджет",
         return_to_start=True, route_stops=2),
    dict(n=10, title="Старт возле Ботанического сада из текста",
         query="я возле ботанического сада куда могу пойти", preset=None, budget=60, clock=None,
         expect="места рядом с Ботаническим садом; старт не подменяется Байтереком",
         check="локальный поиск названия с падежом, явный старт из текста и видимая исходная точка",
         origin_contains="ботан"),
]


def run(scenario: dict, demo: bool, reference: datetime, use_weather: bool = False,
        provider: str = "local", model: str | None = None):
    when = reference
    if scenario["clock"]:
        hour, minute = (int(part) for part in scenario["clock"].split(":"))
        when = when.replace(hour=hour, minute=minute, second=0, microsecond=0)
    label, lat, lon = PRESETS[scenario["preset"]] if scenario["preset"] else resolve_origin(scenario["query"])
    advice = advise(
        scenario["query"], lat, lon, budget_min=scenario["budget"], when=when,
        demo=demo, provider=provider, model=model,
        use_weather=use_weather, origin_label=label,
    )
    return format_advice(advice), advice


def check_answer(scenario: dict, advice, *, live: bool) -> dict[str, bool]:
    """Check observable contracts; these are not a semantic score of generated prose."""
    candidates = {candidate.id: candidate for candidate in advice.retrieval.candidates}
    picks = advice.answer.picks
    ids = [pick.poi_id for pick in picks]
    returned = all(poi_id in candidates for poi_id in ids)
    checks = {
        "count_matches_candidates": 1 <= len(picks) <= 3 if candidates else len(picks) == 0,
        "unique_retrieved_ids": returned and len(ids) == len(set(ids)),
        "cards_have_explanation_action_source": all(
            pick.why.strip() and pick.activity.strip() and pick.visit_minutes > 0
            and pick.source == "poi_data" and pick.activity_source == "generated_idea"
            and bool(pick.source_url) for pick in picks
        ),
        "known_closed_not_recommended": all(pick.hours.get("status") in {"open", "unknown"} for pick in picks),
        "unknown_hours_are_explained": all(
            pick.hours.get("context") == "outdoor" or any(
                text in pick.uncertainties for text in (
                    "часы работы не подтверждены", "opening hours not confirmed", "жұмыс уақыты расталмаған"
                )
            ) for pick in picks if pick.hours.get("status") == "unknown"
        ),
        "each_visit_fits_budget": returned and all(
            ceil(candidates[pick.poi_id].walk_min) * (2 if advice.intent.return_to_start else 1)
            + pick.visit_minutes + 5 <= advice.intent.time_budget_min for pick in picks
        ),
        "route_fits_budget": advice.route is None or advice.route.fits(advice.intent.time_budget_min),
        "route_uses_selected_ids": advice.route is None or set(advice.route.order).issubset(ids),
        "empty_candidates_no_model_call": bool(candidates) or (
            advice.answer.candidates_sent_to_model == 0 and advice.answer.mode == "retrieval-template"
            and bool(advice.answer.note.strip())
        ),
        "offline_no_model_call": live or (
            advice.answer.mode == "retrieval-template" and advice.answer.candidates_sent_to_model == 0
        ),
    }
    if scenario.get("origin_contains"):
        checks["named_origin_resolved"] = scenario["origin_contains"] in advice.origin_label.lower()
    if scenario.get("no_route"):
        checks["single_place_activity_no_route"] = advice.route is None
    if scenario.get("route_stops"):
        checks["explicit_route_stops"] = advice.route is not None and scenario["route_stops"] <= len(advice.route.order) <= 3
    if "categories" in scenario:
        checks["requested_category_preserved"] = all(pick.category in scenario["categories"] for pick in picks)
    if scenario.get("wheelchair"):
        checks["confirmed_wheelchair_only"] = advice.intent.needs_wheelchair and returned and all(
            candidates[pick.poi_id].poi.get("wheelchair") == "yes" for pick in picks
        )
    if scenario.get("child"):
        checks["child_context_preserved"] = advice.intent.with_child
        checks["no_child_bars"] = returned and all(
            candidates[pick.poi_id].poi.get("osm_subtype") not in {"bar", "pub", "nightclub"} for pick in picks
        )
    if scenario.get("return_to_start"):
        checks["return_preserved"] = advice.intent.return_to_start and (
            advice.route is None or advice.route.return_leg is not None
        )
    for field in ("night", "unusual"):
        if scenario.get(field):
            checks[f"{field}_preserved"] = bool(getattr(advice.intent, "wants_unusual" if field == "unusual" else field))
    if "language" in scenario:
        checks["response_language"] = advice.answer.language == scenario["language"]
    return {key: bool(value) for key, value in checks.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--at", default="2026-10-19 12:00", help="фиксированное местное время")
    parser.add_argument("--weather", action="store_true", help="погода только для текущего времени; требуется сеть")
    parser.add_argument("--live", action="store_true", help="использовать серверную модель; запросы могут быть платными")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs", help="папка для Markdown и scenario_results.json")
    args = parser.parse_args()
    try:
        reference = datetime.strptime(args.at, "%Y-%m-%d %H:%M").replace(tzinfo=ASTANA_TZ)
    except ValueError:
        parser.error("--at должен иметь формат YYYY-MM-DD HH:MM")
    config = get_server_config() if args.live else {"provider": "local", "model": None, "ready": True}
    if args.live and (config["provider"] == "local" or not config["ready"]):
        parser.error("Для --live владелец должен настроить серверный провайдер и модель; SERUEN_DEMO=0.")

    dataset = ROOT / "data/pois.json"
    pois = json.loads(dataset.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    command = [".venv/bin/python", "tools/gen_test_scenarios.py", "--at", args.at]
    if args.live:
        command.append("--live")
    if args.weather:
        command.append("--weather")
    try:
        display_output_dir = args.output_dir.relative_to(ROOT)
    except ValueError:
        display_output_dir = args.output_dir
    command.extend(["--output-dir", str(display_output_dir)])
    out = ["# Тестовые сценарии\n",
           f"Прогон: {local_now():%Y-%m-%d %H:%M} (Asia/Almaty). Датасет: {len(pois)} POI.\n",
           f"SHA-256 `data/pois.json`: `{dataset_hash}`.\n",
           f"Время: {reference:%Y-%m-%d %H:%M} Asia/Almaty; ночной сценарий — 23:30.\n",
           f"Повтор из папки проекта: `{shlex.join(command)}`.\n",
           "Все блоки вывода получены этим скриптом из настоящих ответов приложения.\n"]
    if args.live:
        out.append(f"Запрошен серверный провайдер `{config['provider']}`, модель `{config['model']}`. "
                   "Это сетевой прогон с возможными расходами. Фактический режим каждого ответа указан ниже. "
                   "Локальный fallback не считается успешной генерацией модели.\n")
    else:
        out.append("Режим `retrieval-template`: локальный поиск и шаблоны, без вызова модели, независимо от ключей "
                   "в окружении. Платный прогон запускается отдельно: "
                   "`.venv/bin/python tools/gen_test_scenarios.py --live --output-dir docs/live`.\n")
    out.append("Погода: " + ("запрошена; фактическое использование видно в каждом ответе.\n" if args.weather
                               else "отключена, исторические погодные данные не подставляются.\n"))
    out.append("Проверки ниже контролируют структуру, источники, ограничения и бюджет. Они не оценивают "
               "истинность каждой фразы модели и не заменяют проверку заведений на месте. "
               "Ноль мест при пустом наборе кандидатов — честное ограничение покрытия, а не успешная выдача 1–3 мест.\n")

    records, summaries, detail = [], [], []
    for scenario in SCENARIOS:
        text, advice = run(scenario, not args.live, reference, args.weather, config["provider"], config["model"])
        checks = check_answer(scenario, advice, live=args.live)
        failed = [name for name, passed in checks.items() if not passed]
        records.append({"scenario": scenario["title"], **advice.as_dict(), "checks": checks})
        summaries.append(f"| {scenario['n']} | {len(advice.answer.picks)} | {advice.answer.mode} | "
                         f"{'есть' if advice.route else 'нет'} | {'FAIL: ' + ', '.join(failed) if failed else 'PASS'} |")
        detail.extend([
            f"## {scenario['n']}. {scenario['title']}\n",
            f"**Вход:** точка `{scenario['preset'] or 'ориентир из текста'}`, бюджет {scenario['budget']} мин, время {advice.when:%H:%M}\n\n"
            f"**Запрос:** «{scenario['query']}»\n",
            f"**Ожидаемый тип ответа:** {scenario['expect']}\n",
            f"**Что проверяем:** {scenario['check']}\n",
            "**Фактический вывод:**\n", "```\n" + text.strip() + "\n```\n",
            f"**Результат:** мест — {len(advice.answer.picks)}; кандидатов после фильтров — "
            f"{advice.retrieval.stats['passed_filters']}; неизвестные часы у "
            f"{sum(p.hours.get('status') == 'unknown' for p in advice.answer.picks)} из показанных; "
            f"режим — `{advice.answer.mode}`; проверки — **{'FAIL' if failed else 'PASS'}**.\n",
        ])
        if failed:
            detail.append("Нарушены проверки: " + ", ".join(failed) + ".\n")

    out.extend(["## Сводка\n", "| № | Мест | Фактический режим | Маршрут | Проверки |",
                "|---|---:|---|---|---|", *summaries, "", *detail])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "TEST_SCENARIOS.md").write_text("\n".join(out), encoding="utf-8")
    (args.output_dir / "scenario_results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = sum(not all(record["checks"].values()) for record in records)
    print(f"[scenarios] {args.output_dir}: {len(SCENARIOS)} сценариев, live={args.live}, нарушений={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
