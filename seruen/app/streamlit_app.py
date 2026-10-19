"""Seruen AI — an explainable walking advisor with session-only history.

    streamlit run seruen/app/streamlit_app.py
"""

from __future__ import annotations

import math
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seruen.cli import (  # noqa: E402
    OUTDOOR_ACCESS_LABEL,
    OUTDOOR_ACCESS_NOTE,
    PRESETS,
    advise,
    local_now,
    resolve_origin,
)
from seruen.core import weather as weather_mod  # noqa: E402
from seruen.core.geo import BUFFER_MIN  # noqa: E402
from seruen.core.origin_resolution import resolve_query_origin  # noqa: E402
from seruen.core.retrieval import DEFAULT_DATASET  # noqa: E402
from seruen.core.route import describe_route  # noqa: E402
from seruen.core.server_config import get_server_config  # noqa: E402

DATASET = ROOT / DEFAULT_DATASET
SOURCE_LABEL = {"poi_data": "Обоснование: сведения о месте", "generated_idea": "Обоснование: сгенерированная идея"}
CONFIDENCE_LABEL = {"high": "высокая", "medium": "средняя", "low": "низкая"}
HOURS_LABEL = {
    "open": "Открыто по расписанию OSM",
    "closed": "Закрыто по расписанию OSM",
    "unknown": "Часы работы неизвестны",
}
CATEGORY_RU = {
    "park": "Парк", "cafe": "Кафе", "restaurant": "Ресторан", "museum": "Музей",
    "landmark": "Достопримечательность", "viewpoint": "Смотровая", "playground": "Детская площадка",
    "mall": "Торговый центр", "theatre": "Театр / кино", "library": "Библиотека", "sport": "Спорт",
    "other": "Место",
}
SCENARIOS = [
    {"label": "45 минут у MEGA", "icon": "🛍️", "query": "Я у Mega Silk Way, хочу погулять 45 минут",
     "preset": "mega", "budget": 45},
    {"label": "Кафе для работы", "icon": "☕", "query": "Я у Байтерека, ищу тихое кафе с розетками поработать за ноутбуком, час",
     "preset": "baiterek", "budget": 60},
    {"label": "С ребёнком", "icon": "🧸", "query": "Я у Хан Шатыра, гуляю с ребёнком и коляской, есть час, куда пойти",
     "preset": "khan", "budget": 60},
    {"label": "Перед вылетом", "icon": "✈️", "query": "Я в аэропорту, я турист, до самолёта 2 часа, хочу посмотреть что-то рядом",
     "preset": "airport", "budget": 120},
    {"label": "Ночная прогулка", "icon": "🌙", "query": "Я у Байтерека, хочу чего-то необычного ночью, 90 минут",
     "preset": "baiterek", "budget": 90, "time": "23:30"},
]


def safe_text(value: object) -> str:
    """POI, model and user text must not become Markdown links or HTML."""
    text = str(value or "").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*_{}\[\]()#+.!|~])", r"\\\1", text)


def init_state() -> None:
    defaults = {
        "history": [], "pending": None, "preset": "baiterek", "budget": 60, "clock": "",
        "manual_coordinates": False, "query_location": True, "manual_budget": False,
        "with_child": False, "wheelchair": False, "return_to_start": False, "group_size": 1,
        "indoor": "Из запроса", "language": "Авто", "weather": False,
        "history_selection": 0, "next_result_id": 1,
        "origin_clarification": None, "pending_origin": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_scenario(scenario: dict) -> None:
    """Callbacks run before widgets, so an example resets all relevant controls."""
    st.session_state.update({
        "preset": scenario["preset"], "budget": scenario["budget"], "clock": scenario.get("time", ""),
        "manual_coordinates": False, "query_location": True, "manual_budget": False,
        "with_child": False, "wheelchair": False, "return_to_start": False, "group_size": 1,
        "indoor": "Из запроса", "language": "Авто", "pending": scenario["query"],
        "origin_clarification": None, "pending_origin": None,
    })


def clear_history() -> None:
    st.session_state["history"] = []
    st.session_state["pending"] = None
    st.session_state["history_selection"] = 0
    st.session_state["origin_clarification"] = None
    st.session_state["pending_origin"] = None


def repeat_query(query: str) -> None:
    """An explicit action applies current controls to the displayed request."""
    st.session_state["pending"] = query


def choose_start() -> None:
    """An explicit landmark choice takes priority over the request text."""
    st.session_state["query_location"] = False


def choose_origin(index: int) -> None:
    """Continue a pending request from a confirmed candidate without resolving again."""
    clarification = st.session_state.get("origin_clarification")
    if not clarification:
        return
    resolution = clarification["resolution"]
    selected = resolution["candidates"][index]
    st.session_state["pending_origin"] = {
        **clarification,
        "resolution": {**resolution, "status": "resolved", "place": selected, "message": "", "user_confirmed": True},
    }
    st.session_state["origin_clarification"] = None


def retry_with_manual_start() -> None:
    clarification = st.session_state.get("origin_clarification")
    if clarification:
        st.session_state["pending"] = clarification["query"]


def visit_plan(pick, return_to_start: bool) -> tuple[int, str]:
    """Use the same conservative minute counts as the recommendation pipeline."""
    walking = pick.walk_min * (2 if return_to_start else 1)
    total = walking + pick.visit_minutes + BUFFER_MIN
    if return_to_start:
        walking_text = f"{pick.walk_min} мин туда + {pick.walk_min} мин обратно"
    else:
        walking_text = f"{pick.walk_min} мин пешком"
    return total, f"{walking_text} + {pick.visit_minutes} мин на месте + {BUFFER_MIN} мин запаса"


def resolve_when(clock: str) -> datetime | None:
    """Empty input leaves time inference to the pipeline; invalid input is explicit."""
    if not clock.strip():
        return None
    if not re.fullmatch(r"\d{1,2}:\d{2}", clock.strip()):
        raise ValueError("Введите время в формате ЧЧ:ММ, например 18:30.")
    hour, minute = (int(part) for part in clock.strip().split(":"))
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("Часы должны быть от 00 до 23, минуты — от 00 до 59.")
    return local_now().replace(hour=hour, minute=minute, second=0, microsecond=0)


def map_deck(advice) -> pdk.Deck:
    """One ordering is used by both route text and map. Other picks are alternatives."""
    lat, lon = advice.origin
    picks = advice.answer.picks
    route_order = advice.route.order if advice.route else []
    route_indices = {poi_id: number for number, poi_id in enumerate(route_order, 1)}
    picked_ids = {pick.poi_id for pick in picks}
    considered = [
        {"lat": candidate.poi["lat"], "lon": candidate.poi["lon"], "name": candidate.name(),
         "kind": "Прошло фильтры"}
        for candidate in advice.retrieval.candidates if candidate.poi["id"] not in picked_ids
    ]
    chosen = [
        {"lat": pick.lat, "lon": pick.lon, "name": pick.name,
         "label": str(route_indices[pick.poi_id]) if pick.poi_id in route_indices else chr(64 + number),
         "kind": f"Остановка {route_indices[pick.poi_id]}" if pick.poi_id in route_indices else "Вариант места",
         "color": [20, 116, 92, 240] if pick.poi_id in route_indices else [181, 115, 40, 240]}
        for number, pick in enumerate(picks, 1)
    ]
    me = [{"lat": lat, "lon": lon, "name": advice.origin_label, "kind": "Точка старта"}]
    layers = [
        pdk.Layer("ScatterplotLayer", data=considered, get_position="[lon, lat]",
                  get_fill_color=[143, 155, 160, 140], get_radius=25, radius_min_pixels=3, pickable=True),
    ]
    if route_order:
        by_id = {pick.poi_id: pick for pick in picks}
        path = [[lon, lat]] + [[by_id[poi_id].lon, by_id[poi_id].lat] for poi_id in route_order if poi_id in by_id]
        if advice.route.return_leg:
            path.append([lon, lat])
        layers.append(pdk.Layer("PathLayer", data=[{"path": path}], get_path="path",
                                get_color=[20, 116, 92, 190], get_width=5, width_min_pixels=3))
    layers.extend([
        pdk.Layer("ScatterplotLayer", data=chosen, get_position="[lon, lat]", get_fill_color="color",
                  get_radius=55, radius_min_pixels=12, radius_max_pixels=18, pickable=True),
        pdk.Layer("TextLayer", data=chosen, get_position="[lon, lat]", get_text="label",
                  get_size=14, get_color=[255, 255, 255], get_text_anchor="'middle'", get_alignment_baseline="'center'"),
        pdk.Layer("ScatterplotLayer", data=me, get_position="[lon, lat]", get_fill_color=[44, 102, 180, 255],
                  get_radius=45, radius_min_pixels=9, radius_max_pixels=14, pickable=True),
    ])
    points = chosen + me
    latitudes, longitudes = [p["lat"] for p in points], [p["lon"] for p in points]
    span = max(max(longitudes) - min(longitudes), (max(latitudes) - min(latitudes)) / math.cos(math.radians(lat)), .012)
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=(min(latitudes) + max(latitudes)) / 2,
            longitude=(min(longitudes) + max(longitudes)) / 2,
            zoom=max(10, min(14.5, math.log2(360 / span) - .9)), pitch=0,
        ),
        tooltip={"text": "{name}\n{kind}"}, map_provider="carto", map_style="light",
    )


def render_map(advice) -> None:
    st.pydeck_chart(map_deck(advice), width="stretch", height=400)
    st.caption("Синий — старт · зелёный — маршрут · оранжевый — варианты · серый — другие места. Цифры — порядок маршрута, буквы — варианты.")
    st.caption("Линии соединяют точки по прямой. Время ходьбы оценочное: переходы и препятствия не проверены. Подложке карты нужен интернет.")


def source_url(advice, poi_id: str) -> str | None:
    candidate = next((item for item in advice.retrieval.candidates if item.poi["id"] == poi_id), None)
    url = (candidate.poi.get("source") or {}).get("url", "") if candidate else ""
    parsed = urlparse(url)
    return url if parsed.scheme == "https" and parsed.hostname in {"openstreetmap.org", "www.openstreetmap.org"} else None


def render_description_source(advice, poi_id: str) -> None:
    candidate = next((item for item in advice.retrieval.candidates if item.poi["id"] == poi_id), None)
    if not candidate or candidate.poi.get("description_source") != "wikipedia":
        return
    for evidence in candidate.poi.get("evidence", []):
        url = evidence.get("source_url", "")
        parsed = urlparse(url)
        if evidence.get("field") == "description" and parsed.scheme == "https" and (parsed.hostname or "").endswith(".wikipedia.org"):
            st.link_button("Описание: Wikipedia ↗", url, key=f"wiki_source_{poi_id}")
            st.caption("Описание сокращено; [лицензия CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).")
            return


def render_cards(advice) -> None:
    if not advice.answer.picks:
        st.warning(safe_text(advice.answer.note))
        st.caption("Попробуйте другую точку старта или больше времени. Ограничения доступности и закрытые места не обходятся.")
        return
    st.subheader("Варианты рядом")
    st.caption("Каждая карточка — отдельный вариант посещения. Время и расстояние рассчитаны от точки старта.")
    for number, pick in enumerate(advice.answer.picks, 1):
        with st.container(border=True):
            st.markdown(f"### {chr(64 + number)} · {safe_text(pick.name)}")
            st.caption(f"{CATEGORY_RU.get(pick.category, pick.category)} · {pick.distance_m} м по прямой · ≈{pick.walk_min} мин пешком")
            st.markdown("**Почему подходит**")
            st.markdown(safe_text(pick.why))
            st.caption(SOURCE_LABEL.get(pick.source, "Источник обоснования не указан"))
            st.markdown("**Что делать**")
            st.markdown(safe_text(pick.activity))
            st.caption("Сгенерированная идея · условия на месте могут отличаться")
            total, breakdown = visit_plan(pick, advice.intent.return_to_start)
            st.markdown(f"**На этот вариант: ≈{total} из {advice.intent.time_budget_min} мин**")
            st.caption(breakdown)
            if not advice.intent.return_to_start:
                st.caption("Возвращение к старту не включено.")
            if advice.route and pick.poi_id in advice.route.order:
                route_number = advice.route.order.index(pick.poi_id) + 1
                st.caption(f"В готовом маршруте: остановка {route_number}, {advice.route.stop_visit_min[pick.poi_id]} мин на месте.")
            outdoor_access = pick.hours.get("context") == "outdoor"
            if outdoor_access:
                st.markdown(OUTDOOR_ACCESS_LABEL)
                st.caption(OUTDOOR_ACCESS_NOTE)
            else:
                status = pick.hours.get("status", "unknown")
                st.markdown(safe_text(HOURS_LABEL.get(status, HOURS_LABEL["unknown"])))
                st.caption(f"Часы: {safe_text(pick.hours.get('detail', 'нет данных'))}")
            hours_confidence = pick.hours.get("confidence")
            confidence = f"Уверенность рекомендации: {CONFIDENCE_LABEL.get(pick.confidence, pick.confidence)}"
            if hours_confidence and not outdoor_access:
                confidence += f" · расписания: {CONFIDENCE_LABEL.get(hours_confidence, hours_confidence)}"
            st.caption(safe_text(confidence))
            if pick.uncertainties:
                st.caption("Не проверено: " + safe_text("; ".join(pick.uncertainties)))
            url = source_url(advice, pick.poi_id)
            if url:
                st.link_button("Источник: OpenStreetMap ↗", url, key=f"osm_source_{pick.poi_id}")
            render_description_source(advice, pick.poi_id)
    st.caption(safe_text(advice.answer.note))


def render_route(advice) -> None:
    if not advice.route:
        st.subheader("Места на карте" if advice.answer.picks else "Район поиска")
        if advice.answer.picks and getattr(advice.intent, "wants_work", False):
            st.caption("Для работы выберите одно место и проведите время там. Возможность спокойно работать и наличие розеток уточните на месте.")
        elif advice.answer.picks:
            st.caption("Выберите один из вариантов. Карточки уже учитывают дорогу и время на месте; объединять их в один маршрут не требуется.")
        else:
            st.caption("Подходящих вариантов пока нет. Можно изменить точку старта или условия и обновить подборку.")
        return
    route = advice.route
    st.subheader("Готовый маршрут")
    st.caption("Пройдите его целиком или выберите отдельное место из карточек.")
    planned = route.as_dict()["planned_min"]
    budget = advice.intent.time_budget_min
    st.progress(min(1., planned / budget), text=f"≈{planned} из {budget} мин · остаётся {max(0, budget - planned)} мин")
    st.caption(f"{route.total_walk_min} мин пешком + {route.total_visit_min} мин на остановках + {route.as_dict()['buffer_min']} мин запаса")
    by_id = {pick.poi_id: pick for pick in advice.answer.picks}
    for number, (poi_id, leg) in enumerate(zip(route.order, route.legs, strict=True), 1):
        if poi_id in by_id:
            st.markdown(f"**{number}. {safe_text(by_id[poi_id].name)}** · {route.stop_visit_min[poi_id]} мин на месте")
            st.caption(f"Переход: {leg.walk_min} мин от {safe_text(advice.origin_label if number == 1 else leg.from_name)}")
    if route.return_leg:
        st.markdown(f"**Возвращение к старту** · {route.return_leg.walk_min} мин пешком")
    else:
        st.caption("Маршрут заканчивается у последней остановки. Возвращение к старту не включено.")
    if route.trimmed:
        st.caption("В этом маршруте визиты короче, чем при посещении одного места отдельно.")


def selection_text(advice) -> str:
    """A portable plan, without the raw request or precise starting coordinates."""
    lines = ["Seruen AI · Астана", advice.when.strftime("%d.%m.%Y %H:%M (Астана)"),
             f"Старт: {advice.origin_label}", f"Время: {advice.intent.time_budget_min} мин", ""]
    if advice.route:
        lines.extend(["ГОТОВЫЙ МАРШРУТ", describe_route(advice.route, advice.answer.language)])
        by_id = {pick.poi_id: pick for pick in advice.answer.picks}
        for number, (poi_id, leg) in enumerate(zip(advice.route.order, advice.route.legs, strict=True), 1):
            lines.append(f"{number}. {by_id[poi_id].name}: переход {leg.walk_min} мин, на месте {advice.route.stop_visit_min[poi_id]} мин.")
        if advice.route.return_leg:
            lines.append(f"Возвращение к старту: {advice.route.return_leg.walk_min} мин.")
        else:
            lines.append("Возвращение к старту не включено.")
        lines.append("")
    lines.append("ВАРИАНТЫ ДЛЯ ОТДЕЛЬНОГО ПОСЕЩЕНИЯ")
    for number, pick in enumerate(advice.answer.picks, 1):
        total, breakdown = visit_plan(pick, advice.intent.return_to_start)
        lines.extend([f"{chr(64 + number)}. {pick.name}", f"Почему: {pick.why}",
                      SOURCE_LABEL.get(pick.source, "Источник не указан"),
                      f"Что делать (сгенерированная идея): {pick.activity}",
                      f"Время: ≈{total} мин — {breakdown}."])
        if pick.hours.get("context") == "outdoor":
            lines.extend([OUTDOOR_ACCESS_LABEL, OUTDOOR_ACCESS_NOTE])
        else:
            lines.append(HOURS_LABEL.get(pick.hours.get("status"), HOURS_LABEL["unknown"]))
        lines.append(f"Уверенность: {CONFIDENCE_LABEL.get(pick.confidence, pick.confidence)}.")
        if not advice.intent.return_to_start:
            lines.append("Возвращение к старту не включено.")
        if pick.uncertainties:
            lines.append("Не проверено: " + "; ".join(pick.uncertainties))
        url = source_url(advice, pick.poi_id)
        if url:
            lines.append("Источник места: " + url)
        for evidence in getattr(pick, "evidence", []):
            url = evidence.get("source_url", "")
            parsed = urlparse(url)
            if evidence.get("field") == "description" and parsed.scheme == "https" and (parsed.hostname or "").endswith(".wikipedia.org"):
                lines.append("Описание: " + url + " · CC BY-SA 4.0; описание сокращено.")
        lines.append("")
    lines.extend([advice.answer.note, "", "Время ходьбы оценочное; переходы и препятствия не проверены.",
                  "© OpenStreetMap contributors · ODbL. Часы работы взяты из снимка данных и требуют проверки."])
    return "\n".join(lines)


def render_data_panel(advice) -> None:
    used = advice.data_used()
    with st.expander("Какие данные использованы", expanded=False):
        first, second, third = st.columns(3)
        first.metric("Кандидатов после фильтров", used["candidates_after_filters"])
        second.metric("Передано модели", used["candidates_shown_to_model"])
        third.metric("Радиус поиска", f"{used['search_radius_m']} м")
        st.caption(f"Подготовлено для ответа: {used.get('candidates_prepared', len(advice.retrieval.candidates))} кандидатов. В локальном режиме модель не вызывается.")
        rows = {
            "Точка старта (округлено)": f"{used['location_rounded'][0]}, {used['location_rounded'][1]} · {used['location_label']}",
            "Определение старта": ("ИИ разобрал место; точка проверена по источнику" if used.get("origin_resolution", {}).get("ai_used")
                                  else "Ручной выбор или локальный каталог"),
            "Источник точки": {"visitastana": "Visit Astana", "photon": "Photon / OpenStreetMap",
                              "local": "Каталог OpenStreetMap", "manual": "Выбрано вами",
                              "preset": "Сохранённый ориентир"}.get(used.get("origin_source"), "Каталог мест"),
            "Время проверки": f"{used['local_time']} · Астана (UTC+5)",
            "Время на прогулку": f"{used['time_budget_min']} мин",
            "Фильтры": ", ".join(used["filters"]) or "Дополнительных фильтров нет",
            "Погода": used["weather"],
            "Датасет": f"{used['dataset_size']} мест · {used['dataset_license']}",
            "Не прошли фильтры": f"закрыто: {used['excluded_closed']}; далеко: {used['excluded_too_far']}; не хватает времени: {used['excluded_no_time']}",
            "Часы неизвестны": f"{used['unknown_hours_kept']} кандидатов; их статус остаётся «неизвестно»",
            "Разбор запроса": used["intent_source"],
            "Генерация": f"{used['answer_mode']} · модель: {used['model']}",
        }
        for label, value in rows.items():
            st.markdown(f"**{label}:** {safe_text(value)}")
        if used["radius_relaxed"]:
            st.caption("Радиус поиска расширен: в исходном было мало вариантов.")
        if advice.answer.fallback_reason:
            st.caption("Рекомендации собраны по данным мест и шаблонам. Ответ модели для рекомендаций отсутствует или не прошёл проверку.")


def sidebar() -> dict:
    with st.sidebar:
        st.markdown("### Контекст прогулки")
        st.caption("Измените условия и нажмите «Обновить подборку» или отправьте новый запрос.")
        st.checkbox("Определить старт из сообщения", key="query_location",
                    help="Напишите, где вы сейчас. Если место не найдено или есть несколько совпадений, попросим уточнить.")
        if st.session_state["query_location"]:
            st.caption("Автоматический выбор: ищем место из вашего сообщения.")
        else:
            st.caption("Используем выбранную вами точку; место в сообщении её не меняет.")
        origin_display = st.empty()
        lat, lon = None, None
        with st.expander("Указать старт вручную", expanded=not st.session_state["query_location"] or st.session_state["manual_coordinates"]):
            preset = st.selectbox("Точка старта", options=list(PRESETS), format_func=lambda key: PRESETS[key][0], key="preset",
                                  on_change=choose_start, help="Явный выбор включает ручной старт.")
            manual = st.checkbox("Ввести координаты вручную", key="manual_coordinates", on_change=choose_start)
            if manual:
                lat = st.number_input("Широта", min_value=-90., max_value=90., value=float(PRESETS[preset][1]), format="%.5f", key="latitude")
                lon = st.number_input("Долгота", min_value=-180., max_value=180., value=float(PRESETS[preset][2]), format="%.5f", key="longitude")
                st.caption("Ручные координаты имеют приоритет над текстом и ориентиром.")
        st.checkbox("Задать время вручную", key="manual_budget", help="Без этой настройки бюджет берётся из запроса. Если время не указано — 60 минут.")
        if st.session_state["manual_budget"]:
            st.slider("Сколько времени, мин", 15, 240, step=5, key="budget")
        else:
            st.caption("Время — из запроса; по умолчанию 60 минут.")
        with st.expander("Дополнительные условия"):
            st.checkbox("С ребёнком", key="with_child")
            st.checkbox("Подтверждённый доступ без ступеней", key="wheelchair", help="Строгий фильтр по данным OSM. Без подтверждения место не предлагается.")
            st.checkbox("Вернуться к старту", key="return_to_start")
            st.number_input("Размер группы", 1, 30, key="group_size", help="1 означает: определить размер группы из запроса.")
            st.radio("Где провести время", ["Из запроса", "В помещении", "На улице"], key="indoor")
            st.radio("Язык ответа", ["Авто", "ru", "kk", "en"], horizontal=True, key="language")
            st.text_input("Время проверки, ЧЧ:ММ", key="clock", placeholder=local_now().strftime("%H:%M"), help="Пусто — время из запроса или текущее время Астаны. Для демонстрации можно задать 23:30.")
        st.divider()
        st.checkbox("Учитывать погоду", key="weather", help="Получить текущую погоду Open-Meteo. Сервису передаются координаты, округлённые до 2 знаков.")
        st.caption("История хранится только в памяти этой сессии. Автопоиск места может использовать онлайн-поиск; карта загружает подложку CARTO. © OpenStreetMap contributors · ODbL.")
        st.button("Очистить историю", key="clear_history", on_click=clear_history, width="stretch")
    return {"preset": preset, "lat": lat, "lon": lon, "origin_display": origin_display}


def process_query(query: str, context: dict) -> None:
    if not query.strip():
        return
    st.session_state["origin_clarification"] = None
    try:
        when = resolve_when(st.session_state["clock"])
        overrides = {
            "with_child": True if st.session_state["with_child"] else None,
            "needs_wheelchair": True if st.session_state["wheelchair"] else None,
            "return_to_start": True if st.session_state["return_to_start"] else None,
            "group_size": int(st.session_state["group_size"]) if st.session_state["group_size"] > 1 else None,
            "indoor_pref": {"В помещении": "indoor", "На улице": "outdoor"}.get(st.session_state["indoor"]),
        }
        config = get_server_config()
        settings = {
            "budget_min": st.session_state["budget"] if st.session_state["manual_budget"] else None,
            "when": when, "demo": config["provider"] == "local", "provider": config["provider"], "model": config["model"],
            "use_weather": st.session_state["weather"], "dataset_path": DATASET,
            "language": None if st.session_state["language"] == "Авто" else st.session_state["language"], "overrides": overrides,
        }
        if context["lat"] is not None:
            label, lat, lon = resolve_origin(query, lat=context["lat"], lon=context["lon"])
            resolution = manual_resolution(label, lat, lon, query, "coordinates")
        elif not st.session_state["query_location"]:
            label, lat, lon = PRESETS[context["preset"]]
            resolution = manual_resolution(label, lat, lon, query, "preset")
        else:
            with st.spinner("Определяю вашу точку старта…"):
                resolution = resolve_query_origin(
                    query, use_ai=config["provider"] == "openai", allow_online=config["provider"] == "openai", model=config["model"],
                ).as_dict()
        request = {"query": query, "settings": settings, "resolution": resolution}
        if resolution["status"] != "resolved" or not resolution.get("place"):
            st.session_state["origin_clarification"] = request
            return
        complete_query(request)
    except (OSError, ValueError) as exc:
        st.error("Не удалось выполнить запрос: " + safe_text(exc))


def manual_resolution(label: str, lat: float, lon: float, query: str, kind: str) -> dict:
    return {"status": "resolved", "place": {"id": "manual", "name": label, "lat": lat, "lon": lon,
            "address": "", "source_url": "", "source": "manual", "kind": kind},
            "candidates": [], "message": "", "place_query": "", "context_query": query, "ai_used": False}


def complete_query(request: dict) -> None:
    """A confirmed origin resumes only recommendation generation, never extraction."""
    query, resolution = request["query"], request["resolution"]
    place = resolution["place"]
    try:
        with st.spinner("Сопоставляю запрос, время и места поблизости…"):
            advice = advise(
                query, place["lat"], place["lon"], origin_label=place["name"],
                origin_resolution=resolution, intent_query=resolution.get("context_query", query),
                **request["settings"],
            )
    except (OSError, ValueError) as exc:
        st.error("Не удалось выполнить запрос: " + safe_text(exc))
        return
    result_id = st.session_state["next_result_id"]
    st.session_state["next_result_id"] += 1
    st.session_state["history"] = (st.session_state["history"] + [{"id": result_id, "query": query, "advice": advice, "origin_resolution": resolution}])[-5:]
    st.session_state["history_selection"] = result_id
    st.session_state["origin_clarification"] = None


def render_origin_source(place: dict) -> None:
    if place.get("address"):
        st.caption(safe_text(place["address"]))
    label = {"manual": "Выбрано вами", "visitastana": "Источник: Visit Astana"}.get(place.get("source"), "Источник: OpenStreetMap")
    url = place.get("source_url") or ""
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.hostname in {"openstreetmap.org", "www.openstreetmap.org", "visitastana.kz", "www.visitastana.kz"}:
        st.caption(f"[{label}]({url})")
    else:
        st.caption(label if place.get("source") == "manual" else "Источник: каталог мест Астаны")


def render_origin_clarification() -> None:
    request = st.session_state["origin_clarification"]
    resolution = request["resolution"]
    st.subheader("Уточним точку старта")
    st.markdown(safe_text(request["query"]))
    messages = {
        "missing": "Напишите, где вы сейчас: название места или адрес в Астане. Можно также указать старт вручную в боковой панели.",
        "ambiguous": "Нашли несколько возможных совпадений. Проверьте название и адрес, затем выберите точку старта.",
        "unresolved": "Не нашли указанное место в Астане. Уточните название или адрес в новом сообщении либо укажите старт вручную.",
        "unavailable": "Сейчас не удалось проверить место. Укажите старт вручную или попробуйте ещё раз позже.",
    }
    candidates = resolution.get("candidates", [])
    if resolution["status"] == "ambiguous" and len(candidates) == 1:
        messages["ambiguous"] = "Проверьте найденное место и его адрес, затем подтвердите точку старта."
    st.warning(messages.get(resolution["status"], messages["unresolved"]))
    if candidates:
        st.caption("После выбора продолжим этот запрос с его исходными условиями.")
    if resolution.get("place_query"):
        st.caption("Искали: " + safe_text(resolution["place_query"]))
    for index, place in enumerate(candidates):
        with st.container(border=True):
            st.markdown("**" + safe_text(place["name"]) + "**")
            render_origin_source(place)
            st.button("Начать отсюда", key=f"origin_candidate_{index}", on_click=choose_origin, args=(index,))
    if not st.session_state["query_location"] or st.session_state["manual_coordinates"]:
        st.button("Продолжить с выбранным стартом", key="retry_manual_origin", on_click=retry_with_manual_start)


def main() -> None:
    st.set_page_config(page_title="Seruen AI · Прогулки по Астане", page_icon="🌿", layout="wide")
    init_state()
    st.html("""<style>
        .block-container {max-width: 1320px; padding-top: 2.2rem;}
        [data-testid="stAppViewContainer"] {background: #fbfcf9;}
        [data-testid="stSidebar"] {background: #f1f4ef;}
        [data-testid="stChatInput"] {border-radius: 14px;}
        h1 {letter-spacing: -.04em;}
        h3 {letter-spacing: -.02em;}
        [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px;}
        @media (max-width: 1100px) {
            .block-container {padding: 1.4rem 1.2rem 6rem;}
            .st-key-result_columns [data-testid="stHorizontalBlock"] {flex-direction: column-reverse;}
            .st-key-result_columns [data-testid="stColumn"] {width: 100%; flex: 1 1 auto; min-width: 100%;}
            .st-key-scenario_buttons [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
            .st-key-scenario_buttons [data-testid="stColumn"] {flex: 1 1 calc(50% - 1rem); min-width: calc(50% - 1rem);}
        }
        </style>""")
    st.caption("АСТАНА · МЕСТА И КОРОТКИЕ МАРШРУТЫ")
    st.title("Seruen AI")
    st.markdown("Найдите место под настроение и свободное время.")
    st.caption("1–3 места с источниками, объяснением выбора и расчётом времени на прогулку.")
    if not DATASET.exists():
        st.error("Не удалось загрузить данные о местах. Попробуйте позже.")
        st.stop()
    context = sidebar()
    st.markdown("**Попробуйте готовый сценарий**")
    with st.container(key="scenario_buttons"):
        for index, (column, scenario) in enumerate(zip(st.columns(5), SCENARIOS, strict=True)):
            column.button(scenario["label"], icon=scenario["icon"], key=f"scenario_{index}", width="stretch",
                          on_click=apply_scenario, args=(scenario,))
    typed = st.chat_input("Например: «Я у Байтерека, 40 минут, хочу тихое кафе»", key="query_input", max_chars=2000)
    pending = st.session_state.pop("pending", None)
    query = typed or pending
    continuation = st.session_state.pop("pending_origin", None)
    if query:
        process_query(query, context)
    elif continuation:
        complete_query(continuation)
    if st.session_state["origin_clarification"]:
        render_origin_clarification()
        return
    history = st.session_state["history"]
    if not history:
        with st.container(border=True):
            st.subheader("Куда отправимся?")
            st.markdown("Укажите ориентир, время и желание в одном сообщении. Например: «У MEGA Silk Way, 45 минут, прогулка с ребёнком».")
            st.caption("Если место не указано, попросим уточнить старт. Если время не указано — предложим варианты на 60 минут. Часы работы и доступность проверяем по имеющимся данным.")
        return
    st.divider()
    if len(history) > 1:
        by_id = {entry["id"]: entry for entry in history}
        st.selectbox("Запросы этой сессии", options=[entry["id"] for entry in reversed(history)],
                     format_func=lambda result_id: by_id[result_id]["query"], key="history_selection")
        entry = by_id[st.session_state["history_selection"]]
    else:
        entry = history[0]
    advice = entry["advice"]
    context["origin_display"].markdown("**Старт подборки:** " + safe_text(advice.origin_label))
    st.markdown("**Ваш запрос**")
    st.markdown(safe_text(entry["query"]))
    st.markdown("**Старт определён:** " + safe_text(advice.origin_label))
    origin_metadata = entry.get("origin_resolution") or getattr(advice, "origin_resolution", None)
    if origin_metadata and origin_metadata.get("place"):
        render_origin_source(origin_metadata["place"])
    if advice.answer.mode in {"openai", "gemini"}:
        mode = "Ответ ИИ"
    elif advice.data_used()["candidates_shown_to_model"]:
        mode = "Резервный ответ по шаблонам"
    else:
        mode = "Локальный ответ по шаблонам"
    st.caption(f"{safe_text(advice.origin_label)} · {advice.intent.time_budget_min} мин · {advice.when.strftime('%d.%m %H:%M')} (Астана) · {mode}")
    actions = st.columns(2)
    actions[0].button("Обновить подборку", key="repeat_query", on_click=repeat_query, args=(entry["query"],),
                      help="Повторить этот запрос с текущими условиями из боковой панели.", width="stretch")
    if advice.answer.picks:
        actions[1].download_button(
            "Скачать подборку", data=selection_text(advice),
            file_name=advice.when.strftime("seruen-%Y%m%d-%H%M.txt"), mime="text/plain; charset=utf-8",
            key="download_selection", on_click="ignore", width="stretch",
            help="Текстовый файл для прогулки: места, время и источники. Исходный запрос и точные координаты старта в файл не входят.",
        )
    if advice.weather and advice.weather.get("used"):
        st.caption(safe_text(weather_mod.describe(advice.weather)))
        st.caption("Погода: [Open-Meteo](https://open-meteo.com/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).")
    with st.container(key="result_columns"):
        answer_column, map_column = st.columns([1.12, 1], gap="large")
        with answer_column:
            render_cards(advice)
        with map_column:
            render_route(advice)
            render_map(advice)
    render_data_panel(advice)


main()
