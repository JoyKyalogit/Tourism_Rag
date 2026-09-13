"""Load and filter hotel records."""

from __future__ import annotations

import json
import re
from typing import Any

from .config import HOTELS_PATH


def slugify(text: str) -> str:
    return re.sub(r"(^-|-$)", "", re.sub(r"[^a-z0-9]+", "-", text.lower().strip()))


def load_hotels() -> list[dict[str, Any]]:
    if not HOTELS_PATH.exists():
        return []
    with open(HOTELS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_hotels(hotels: list[dict[str, Any]]) -> None:
    HOTELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HOTELS_PATH, "w", encoding="utf-8") as f:
        json.dump(hotels, f, indent=2, ensure_ascii=False)


def filter_by_location(location: str) -> list[dict[str, Any]]:
    key = location.lower().strip().replace("-", " ")
    hotels = load_hotels()
    return [
        h
        for h in hotels
        if key in h.get("location", "").lower()
        or key in h.get("county_or_region", "").lower()
        or key in h.get("country", "").lower()
        or key in " ".join(h.get("destination_tags", [])).lower()
    ]


def hotel_to_source(hotel: dict[str, Any]) -> dict[str, str]:
    return {
        "hotel_name": hotel.get("hotel_name", ""),
        "location": hotel.get("location", ""),
        "country": hotel.get("country", ""),
        "source_url": hotel.get("source_url", ""),
        "website_url": hotel.get("website_url", ""),
        "price_range": hotel.get("price_range", ""),
    }
