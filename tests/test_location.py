"""Starting-point resolution stays local and never invents a known location."""

import pytest

from seruen.core.location import infer_origin
from seruen.core.retrieval import load_pois

PRESETS = {
    "mega": ("Mega Silk Way", 51.0891, 71.4073),
    "baiterek": ("Байтерек", 51.1283, 71.4305),
    "khan": ("Хан Шатыр", 51.1325, 71.4039),
    "airport": ("Аэропорт Нурсултан Назарбаев", 51.0222, 71.4669),
}


@pytest.mark.parametrize("query", [
    "я возле ботанического сада куда могу пойти",
    "Я рядом с Ботаническим садом, куда сходить?",
    "Я в Ботаническом саду, у меня час",
    "I am near the Botanical Garden, where can I go?",
    "Ботаникалық бақтың жанындамын, қайда баруға болады?",
])
def test_botanical_garden_resolves_from_dataset_without_a_special_preset(query):
    garden = next(poi for poi in load_pois() if poi["id"] == "way/1196402246")
    result = infer_origin(query, PRESETS)
    assert result == ("Ботанический сад", garden["lat"], garden["lon"])


def test_regular_russian_name_inflection_uses_local_dataset():
    dataset = [{"id": "library", "name": {"ru": "Национальная академическая библиотека"},
                "lat": 51.123, "lon": 71.456}]
    assert infer_origin("около Национальной академической библиотеки, хочу кофе", {}, pois=dataset) == (
        "Национальная академическая библиотека", 51.123, 71.456,
    )


def test_unique_two_word_landmark_prefix_can_omit_official_title_suffix():
    dataset = [{"id": "museum", "name": {"ru": "Национальный музей Республики Казахстан"},
                "lat": 51.123, "lon": 71.456}]
    assert infer_origin("я возле Национального музея, куда пойти", {}, pois=dataset) == (
        "Национальный музей Республики Казахстан", 51.123, 71.456,
    )
    with pytest.raises(ValueError):
        infer_origin("я возле музея", {}, pois=dataset)
    with pytest.raises(ValueError):
        infer_origin("я возле Национального музея Республики Таджикистан", {}, pois=dataset)


@pytest.mark.parametrize("query", [
    "Я у Неизвестного маяка, хочу Байтерек",
    "I am at Unknown Tower and want to visit Mega",
    "Белгісіз мұнараның жанындамын, Байтерекке барамын",
])
def test_unknown_explicit_origin_cannot_be_replaced_by_destination(query):
    with pytest.raises(ValueError, match="Выберите точку старта"):
        infer_origin(query, PRESETS, pois=())


def test_repeated_chain_name_requires_branch_selection():
    dataset = [
        {"id": "a", "name": {"ru": "Degirmen", "en": "Degirmen"}, "lat": 51.1, "lon": 71.4},
        {"id": "b", "name": {"ru": "Degirmen", "en": "Degirmen"}, "lat": 51.2, "lon": 71.5},
    ]
    with pytest.raises(ValueError, match="несколько мест"):
        infer_origin("я возле кафе Degirmen, куда пойти", {}, pois=dataset)


def test_same_poi_in_several_name_languages_is_not_ambiguous():
    dataset = [{"id": "a", "name": {"ru": "Degirmen", "en": "Degirmen"}, "lat": 51.1, "lon": 71.4}]
    assert infer_origin("я возле кафе Degirmen", {}, pois=dataset) == ("Degirmen", 51.1, 71.4)


@pytest.mark.parametrize("query", [
    "хочу погулять 45 минут", "у меня час, хочу кафе", "есть около часа",
    "what can I do at night?", "find a cafe near me", "кафе рядом со мной",
    "хочу у воды", "хочу погулять у воды", "тихое кафе рядом с водой",
    "I have at least 40 minutes", "I want to go by foot", "one hour from now",
    "у меня около двух часов, хочу погулять", "есть около полутора часов",
    "около двадцати минут", "I have about two hours", "at about two hours from now",
])
def test_non_location_context_keeps_default_selection_available(query):
    assert infer_origin(query, PRESETS, pois=()) is None


@pytest.mark.parametrize("query,preset", [
    ("Я у Байтерека, хочу дойти до Mega Silk Way", "baiterek"),
    ("Хочу в Mega, сейчас у Байтерека", "baiterek"),
    ("Байтерек жанындамын, Мегаға барғым келеді", "baiterek"),
    ("Я у Хан Шатыра, до рейса 2 часа", "khan"),
    ("I am at the airport", "airport"),
])
def test_known_starting_point_still_wins(query, preset):
    assert infer_origin(query, PRESETS, pois=()) == PRESETS[preset]


def test_longest_named_place_wins_over_shared_prefix():
    dataset = [
        {"id": "a", "name": {"ru": "Керуен"}, "lat": 51.1, "lon": 71.4},
        {"id": "b", "name": {"ru": "Керуен City"}, "lat": 51.2, "lon": 71.5},
    ]
    assert infer_origin("я у Керуен City", {}, pois=dataset) == ("Керуен City", 51.2, 71.5)


def test_typo_or_partial_name_is_not_fuzzily_assigned_to_other_place():
    dataset = [{"id": "a", "name": {"ru": "Маяк"}, "lat": 51.1, "lon": 71.4}]
    with pytest.raises(ValueError, match="Выберите точку старта"):
        infer_origin("я возле Неизвестного маяка", {}, pois=dataset)
