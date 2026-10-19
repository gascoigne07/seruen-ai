"""Shared, verified starting points; never a substitute for an unknown origin."""

PRESETS: dict[str, tuple[str, float, float]] = {
    "mega": ("Mega Silk Way", 51.0891, 71.4073),
    "expo": ("EXPO / Нур Алем", 51.0893, 71.4178),
    "baiterek": ("Байтерек", 51.1283, 71.4305),
    "khan": ("Хан Шатыр", 51.1325, 71.4039),
    "airport": ("Аэропорт Нурсултан Назарбаев", 51.0222, 71.4669),
    "nu": ("Nazarbayev University", 51.0905, 71.3978),
    "botanical": ("Ботанический сад", 51.106165, 71.416571),
}

# Initial values for the explicit manual controls only.
DEFAULT_ORIGIN = PRESETS["baiterek"]
