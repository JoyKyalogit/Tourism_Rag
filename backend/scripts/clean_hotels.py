"""
Clean and normalize raw hotel JSON into processed/hotels.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import HOTELS_PATH, HOTELS_RAW_PATH  # noqa: E402
from app.hotels import save_hotels  # noqa: E402

VALID_PRICE = {"budget", "midrange", "luxury"}


def normalize_price(value: str | None) -> str:
    if not value:
        return "midrange"
    v = str(value).lower().strip()
    if v in VALID_PRICE:
        return v
    if v in {"low", "cheap", "affordable"}:
        return "budget"
    if v in {"high", "premium"}:
        return "luxury"
    return "midrange"


def clean_row(row: dict) -> dict | None:
    name = (row.get("hotel_name") or "").strip()
    if not name or len(name) < 2:
        return None
    source = (row.get("source_url") or "").strip()
    if not source:
        return None

    amenities = row.get("amenities") or []
    if isinstance(amenities, str):
        amenities = [a.strip() for a in amenities.split(",") if a.strip()]

    room_types = row.get("room_types") or []
    if isinstance(room_types, str):
        room_types = [r.strip() for r in room_types.split(",") if r.strip()]

    nearby = row.get("nearby_attractions") or []
    if isinstance(nearby, str):
        nearby = [n.strip() for n in nearby.split(",") if n.strip()]

    dest_tags = row.get("destination_tags") or []
    if isinstance(dest_tags, str):
        dest_tags = [d.strip() for d in dest_tags.split(",") if d.strip()]

    rating = row.get("rating")
    try:
        rating = float(rating) if rating not in (None, "", "null") else None
    except (TypeError, ValueError):
        rating = None

    location = re.sub(r"\s+", " ", (row.get("location") or "Unknown").strip())
    country = (row.get("country") or "Kenya").strip()

    return {
        "hotel_name": name,
        "location": location,
        "county_or_region": (row.get("county_or_region") or location).strip(),
        "country": country,
        "description": (row.get("description") or f"{name} in {location}, {country}.").strip(),
        "price_range": normalize_price(row.get("price_range")),
        "amenities": amenities,
        "room_types": room_types,
        "rating": rating,
        "review_summary": (row.get("review_summary") or "").strip(),
        "nearby_attractions": nearby,
        "hotel_category": (row.get("hotel_category") or "hotel").strip(),
        "contact": (row.get("contact") or "").strip(),
        "website_url": (row.get("website_url") or "").strip(),
        "image_url": (row.get("image_url") or "").strip(),
        "source_url": source,
        "lat": row.get("lat"),
        "lng": row.get("lng"),
        "destination_tags": dest_tags or [location.lower()],
        "data_source": row.get("data_source", "openstreetmap"),
    }


def main() -> None:
    if not HOTELS_RAW_PATH.exists():
        raise SystemExit(f"Missing raw file: {HOTELS_RAW_PATH}. Run collect_hotels_osm.py first.")

    with open(HOTELS_RAW_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    cleaned = [c for c in (clean_row(r) for r in raw) if c]
    df = pd.DataFrame(cleaned)
    df.drop_duplicates(subset=["hotel_name", "location", "country"], keep="first", inplace=True)
    df = df[df["hotel_name"].str.len() > 1]
    hotels = df.to_dict(orient="records")

    save_hotels(hotels)
    print(f"Cleaned {len(hotels)} hotels -> {HOTELS_PATH}")


if __name__ == "__main__":
    main()
