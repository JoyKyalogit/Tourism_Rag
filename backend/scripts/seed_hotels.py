"""
Bootstrap hotel records from curated destination lists + East Africa extras.
Use when OSM is unavailable or to enrich processed hotels.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import DESTINATIONS_PATH, HOTELS_PATH  # noqa: E402
from app.hotels import save_hotels  # noqa: E402

EXTRA_HOTELS = [
    {
        "hotel_name": "Park Hyatt Zanzibar",
        "location": "Stone Town",
        "county_or_region": "Zanzibar",
        "country": "Tanzania",
        "description": "Luxury beachfront hotel in historic Stone Town with ocean views.",
        "price_range": "luxury",
        "amenities": ["pool", "spa", "restaurant", "wifi"],
        "hotel_category": "resort",
        "source_url": "https://www.hyatt.com/en-US/hotel/tanzania/park-hyatt-zanzibar/zanzp",
        "destination_tags": ["zanzibar", "stone town"],
    },
    {
        "hotel_name": "Four Seasons Safari Lodge Serengeti",
        "location": "Serengeti",
        "county_or_region": "Mara Region",
        "country": "Tanzania",
        "description": "Safari lodge in the Serengeti with wildlife viewing decks.",
        "price_range": "luxury",
        "amenities": ["pool", "restaurant", "safari desk"],
        "hotel_category": "safari_lodge",
        "source_url": "https://www.fourseasons.com/serengeti/",
        "destination_tags": ["serengeti", "safari"],
    },
    {
        "hotel_name": "Arusha Coffee Lodge",
        "location": "Arusha",
        "county_or_region": "Arusha",
        "country": "Tanzania",
        "description": "Boutique lodge on a working coffee plantation near safari circuits.",
        "price_range": "luxury",
        "amenities": ["restaurant", "garden", "wifi"],
        "hotel_category": "safari_lodge",
        "source_url": "https://www.elewanacollection.com/arusha-coffee-lodge/at-a-glance",
        "destination_tags": ["arusha"],
    },
    {
        "hotel_name": "Hyatt Regency Dar es Salaam",
        "location": "Dar es Salaam",
        "county_or_region": "Dar es Salaam",
        "country": "Tanzania",
        "description": "City hotel on the Indian Ocean waterfront in Tanzania's commercial capital.",
        "price_range": "luxury",
        "amenities": ["pool", "restaurant", "wifi"],
        "hotel_category": "hotel",
        "source_url": "https://www.hyatt.com/hyatt-regency/en-US/darrt-hyatt-regency-dar-es-salaam-the-kilimanjaro",
        "destination_tags": ["dar es salaam"],
    },
    {
        "hotel_name": "Serena Hotel Kampala",
        "location": "Kampala",
        "county_or_region": "Central Region",
        "country": "Uganda",
        "description": "Landmark city hotel near Kampala's business and cultural district.",
        "price_range": "luxury",
        "amenities": ["pool", "restaurant", "wifi"],
        "hotel_category": "hotel",
        "source_url": "https://www.serenahotels.com/kampala",
        "destination_tags": ["kampala"],
    },
    {
        "hotel_name": "Protea Hotel by Marriott Entebbe",
        "location": "Entebbe",
        "county_or_region": "Central Region",
        "country": "Uganda",
        "description": "Lakeside hotel convenient for Entebbe International Airport.",
        "price_range": "midrange",
        "amenities": ["pool", "restaurant", "wifi"],
        "hotel_category": "hotel",
        "source_url": "https://www.marriott.com/en-us/hotels/ebbsi-protea-hotel-entebbe/overview/",
        "destination_tags": ["entebbe"],
    },
]


def hotels_from_destinations() -> list[dict]:
    if not DESTINATIONS_PATH.exists():
        return []

    with open(DESTINATIONS_PATH, encoding="utf-8") as f:
        destinations = json.load(f)

    rows: list[dict] = []
    for dest in destinations:
        location = dest.get("destination", "Kenya")
        region = dest.get("region", location)
        source = dest.get("source_url", "https://www.magicalkenya.com/")
        for tier, names in (dest.get("hotels") or {}).items():
            for name in names:
                rows.append(
                    {
                        "hotel_name": name,
                        "location": location,
                        "county_or_region": region,
                        "country": "Kenya",
                        "description": f"{name} is a {tier} stay in {location}, Kenya.",
                        "price_range": tier if tier in {"budget", "midrange", "luxury"} else "midrange",
                        "amenities": [],
                        "room_types": [],
                        "rating": None,
                        "review_summary": "",
                        "nearby_attractions": dest.get("activities", [])[:3],
                        "hotel_category": "safari_lodge" if "lodge" in name.lower() or "camp" in name.lower() else "hotel",
                        "contact": "",
                        "website_url": "",
                        "image_url": "",
                        "source_url": source,
                        "destination_tags": [location.lower()],
                        "data_source": "curated_destinations",
                    }
                )
    return rows


def main() -> None:
    seeded = hotels_from_destinations()
    existing = []
    if HOTELS_PATH.exists():
        with open(HOTELS_PATH, encoding="utf-8") as f:
            existing = json.load(f)

    merged: dict[str, dict] = {}
    for row in [*seeded, *existing, *EXTRA_HOTELS]:
        key = f"{row['hotel_name'].lower()}|{row.get('location', '').lower()}"
        merged[key] = row

    hotels = list(merged.values())
    save_hotels(hotels)
    print(f"Seeded {len(hotels)} hotels -> {HOTELS_PATH}")


if __name__ == "__main__":
    main()
