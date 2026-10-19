"""Offline regression cases for multilingual requests and untrusted provider JSON."""

import pytest

from seruen.core import intent


@pytest.mark.parametrize(("query", "minutes"), [
    ("на два часа", 120),
    ("погулять полтора часа", 90),
    ("1 час 30 минут", 90),
    ("1h 30min", 90),
    ("30–60 минут", 60),
    ("1-2 hours", 120),
    ("екі сағат", 120),
    ("жарты сағат", 30),
    ("5 минут", 5),
    ("1,5 часа", 90),
    ("час 30 минут", 90),
    ("an hour and 30 minutes", 90),
    ("бір сағат 20 минут", 80),
    ("сейчас", None),
    ("в 12:30", None),
    ("детям 7-10 лет", None),
])
def test_duration_parsing(query, minutes):
    assert intent.parse_budget(query) == minutes


def test_english_words_do_not_match_inside_other_words():
    parsed = intent.parse_intent_regex("A theatre for an hour, please")
    assert parsed.categories_wanted == ["theatre"]
    assert not parsed.night
    assert intent.parse_intent_regex("restart the search").categories_wanted == []
    assert intent.parse_intent_regex("show me something interesting").categories_wanted == []


def test_negated_categories_and_child_context():
    parsed = intent.parse_intent_regex("без кафе и ресторанов, хочу парк без детей")
    assert parsed.categories_wanted == ["park"]
    assert set(parsed.categories_avoided) == {"cafe", "restaurant"}
    assert parsed.with_child is False
    assert intent.parse_intent_regex("без музеев").categories_avoided == ["museum"]
    assert intent.parse_intent_regex("no museums or cafes").categories_avoided == ["cafe", "museum"]


def test_explicit_language_wins_over_place_name_script():
    assert intent.detect_language("Бәйтерек, ответь на русском") == "ru"
    assert intent.detect_language("Байтерек, answer in English") == "en"
    assert intent.detect_language("Байтерек, қазақша") == "kk"


def test_kazakh_verbs_do_not_mean_bar():
    assert "restaurant" not in intent.parse_intent_regex("саябаққа барғым келеді").categories_wanted


def test_indoor_negation_and_step_free_requirement():
    assert intent.parse_intent_regex("не на улице").indoor_pref == "indoor"
    assert intent.parse_intent_regex("not indoors").indoor_pref == "outdoor"
    assert intent.parse_intent_regex("без ступеней").needs_wheelchair
    assert not intent.parse_intent_regex("без коляски").needs_wheelchair


def test_model_cannot_erase_explicit_constraints():
    parsed = intent.merge_intent("С ребёнком, без ступеней, без кафе, 45 минут и обратно", {
        "categories_wanted": ["cafe", "park"], "categories_avoided": [],
        "time_budget_min": 200, "with_child": False, "needs_wheelchair": False,
        "return_to_start": False, "language": "en",
    })
    assert parsed.time_budget_min == 45
    assert parsed.with_child and parsed.needs_wheelchair and parsed.return_to_start
    assert "cafe" in parsed.categories_avoided
    assert "cafe" not in parsed.categories_wanted
    assert parsed.language == "ru"


def test_model_malformed_types_do_not_crash_or_enable_false_strings():
    parsed = intent.merge_intent("кафе на час", {
        "categories_wanted": None, "categories_avoided": {}, "keywords": None,
        "indoor_pref": [], "with_child": "false", "needs_wheelchair": "false",
        "return_to_start": 1, "group_size": True, "time_budget_min": True,
    })
    assert not parsed.with_child and not parsed.needs_wheelchair and not parsed.return_to_start
    assert parsed.group_size == 1 and parsed.time_budget_min == 60
    assert parsed.categories_wanted == ["cafe"]
    assert intent.merge_intent("парк", []).source == "regex"


def test_demo_mode_skips_intent_provider(monkeypatch):
    monkeypatch.setenv("SERUEN_DEMO", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("demo must never call the intent provider")

    monkeypatch.setattr(intent, "parse_intent_llm", forbidden)
    assert intent.parse_intent("парк").source == "regex"


def test_precise_coordinates_are_redacted_before_provider():
    value = intent.redact_coordinates("Я у 51.128345, 71.430567, хочу кафе на 1.5 часа")
    assert "51.128345" not in value and "71.430567" not in value
    assert "1.5 часа" in value


def test_starting_landmark_name_does_not_override_kazakh_park_request():
    parsed = intent.parse_intent_regex("Байтерек жанындамын, 40 минут уақытым бар, саябаққа барғым келеді")
    assert parsed.categories_wanted == ["park"]
    assert parsed.language == "kk"
    assert parsed.time_budget_min == 40


def test_explicit_negative_context_wins_over_provider_boolean():
    parsed = intent.merge_intent("кафе без детей, не нужно возвращаться", {"with_child": True, "return_to_start": True})
    assert not parsed.with_child and not parsed.return_to_start


def test_general_walk_allows_scenic_categories_but_explicit_park_stays_strict():
    assert intent.parse_intent_regex("погулять 45 минут").categories_wanted == ["park", "landmark", "viewpoint"]
    assert intent.parse_intent_regex("I want to walk for 45 minutes").categories_wanted == ["park", "landmark", "viewpoint"]
    assert intent.parse_intent_regex("погулять в парке 45 минут").categories_wanted == ["park"]
    assert intent.parse_intent_regex("walk to a cafe for an hour").categories_wanted == ["cafe"]


@pytest.mark.parametrize("query", [
    "Ищу тихое кафе с розетками поработать за ноутбуком, час",
    "A quiet cafe to work on my laptop for an hour",
    "Тыныш дәмханада ноутбукпен жұмыс істегім келеді, бір сағат",
])
def test_work_does_not_broaden_explicit_cafe_request(query):
    parsed = intent.parse_intent_regex(query)
    assert parsed.categories_wanted == ["cafe"]
    assert parsed.wants_work and parsed.wants_quiet
    assert not parsed.wants_route


@pytest.mark.parametrize("query", [
    "Где можно поработать за ноутбуком час?",
    "A place to work with power sockets for an hour",
    "Ноутбукпен жұмыс істеу үшін орын керек",
])
def test_unspecified_work_venue_uses_cafes_and_libraries(query):
    parsed = intent.parse_intent_regex(query)
    assert parsed.categories_wanted == ["cafe", "library"]
    assert parsed.wants_work


@pytest.mark.parametrize("query", [
    "Поработать в кафе и затем маршрут по нескольким местам",
    "Work in a cafe, then a sightseeing route",
    "Дәмханада жұмыс істеу, сосын бірнеше орынға бару",
])
def test_explicit_route_can_be_requested_alongside_work(query):
    parsed = intent.parse_intent_regex(query)
    assert parsed.wants_work and parsed.wants_route


def test_work_and_route_negations_remain_negative():
    parsed = intent.merge_intent("кафе, не хочу работать, без маршрута", {"wants_work": True, "wants_route": True})
    assert not parsed.wants_work and not parsed.wants_route


def test_open_venue_is_not_an_outdoor_requirement():
    for query in ("кафе открыто сейчас на час", "хочу открытое кафе", "библиотека открыта?"):
        assert intent.parse_intent_regex(query).indoor_pref is None
    assert intent.parse_intent_regex("под открытым небом").indoor_pref == "outdoor"


def test_positive_child_context_does_not_erase_negated_pram():
    parsed = intent.parse_intent_regex("С ребёнком без коляски на час")
    assert parsed.with_child and not parsed.needs_wheelchair
    parsed = intent.parse_intent_regex("кафе без ступеней с ребёнком")
    assert parsed.with_child and parsed.needs_wheelchair
    assert not intent.parse_intent_regex("не хочу с детьми").with_child
    assert not intent.parse_intent_regex("not with children").with_child


def test_quiet_and_socket_requirements_can_follow_without_noise():
    parsed = intent.parse_intent_regex("Кафе без шума и с розетками")
    assert parsed.wants_quiet and parsed.wants_sockets


@pytest.mark.parametrize("query", [
    "я возле ботанического сада, хочу кафе на час",
    "я возле Национального музея, хочу кафе на час",
    "near the Botanical Garden, I want a cafe for an hour",
    "Ботаникалық бақтың жанындамын, дәмханада бір сағат",
])
def test_starting_landmark_does_not_broaden_explicit_destination_category(query):
    parsed = intent.parse_intent_regex(query)
    assert parsed.categories_wanted == ["cafe"]
    assert parsed.raw_query == query
    assert parsed.time_budget_min == 60


def test_starting_landmark_removal_keeps_following_walk_and_work_preferences():
    walk = intent.parse_intent_regex("я возле ботанического сада, хочу погулять 45 минут")
    assert walk.categories_wanted == ["park", "landmark", "viewpoint"]
    assert walk.time_budget_min == 45
    work = intent.parse_intent_regex("я возле Национального музея, хочу тихое кафе поработать 90 минут")
    assert work.categories_wanted == ["cafe"]
    assert work.wants_work and work.wants_quiet
    assert work.time_budget_min == 90
