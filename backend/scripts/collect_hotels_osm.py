"""
Collect hotel records from OpenStreetMap via Overpass API.
Public data only — stores OSM node/way URL as source_url.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import HOTELS_RAW_PATH, OVERPASS_URL, REQUEST_DELAY_SEC  # noqa: E402

# (location label, country, south, west, north, east)
SEARCH_AREAS = [
    ("Nairobi", "Kenya", -1.45, 36.65, -1.15, 37.05),
    ("Mombasa", "Kenya", -4.15, 39.55, -3.95, 39.75),
    ("Diani", "Kenya", -4.45, 39.45, -4.20, 39.75),
    ("Watamu", "Kenya", -3.45, 39.95, -3.25, 40.15),
    ("Malindi", "Kenya", -3.35, 39.95, -3.10, 40.20),
    ("Lamu", "Kenya", -2.35, 40.75, -2.20, 41.00),
    ("Naivasha", "Kenya", -0.95, 36.30, -0.70, 36.55),
    ("Nakuru", "Kenya", -0.40, 36.00, -0.20, 36.20),
    ("Kisumu", "Kenya", -0.15, 34.65, 0.00, 34.85),
    ("Nanyuki", "Kenya", -0.10, 37.00, 0.10, 37.25),
    ("Narok", "Kenya", -1.20, 35.75, -0.95, 36.00),
    ("Amboseli", "Kenya", -2.80, 37.15, -2.55, 37.45),
    ("Maasai Mara", "Kenya", -1.60, 34.85, -1.30, 35.35),
    ("Zanzibar", "Tanzania", -6.25, 39.10, -5.65, 39.65),
    ("Arusha", "Tanzania", -3.45, 36.60, -3.30, 36.80),
    ("Dar es Salaam", "Tanzania", -6.95, 39.15, -6.70, 39.40),
    ("Kampala", "Uganda", 0.25, 32.50, 0.40, 32.70),
    ("Entebbe", "Uganda", 0.03, 32.42, 0.10, 32.52),
]


def overpass_hotels(south: float, west: float, north: float, east: float) -> list[dict]:
    query = f"""
[out:json][timeout:90];
(
  node["tourism"~"hotel|guest_house|hostel|motel|resort"]({south},{west},{north},{east});
  way["tourism"~"hotel|guest_house|hostel|motel|resort"]({south},{west},{north},{east});
);
out center tags;
""".strip()
    headers = {
        "User-Agent": "TravelAfricaRAG/1.0 (class project; contact: local)",
        "Accept": "application/json",
    }
    resp = requests.post(
        OVERPASS_URL,
        data=query.encode("utf-8"),
        headers=headers,
        timeout=120,
    )
    if resp.status_code == 406:
        resp = requests.get(
            OVERPASS_URL,
            params={"data": query},
            headers=headers,
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json().get("elements", [])


def element_to_hotel(el: dict, location: str, country: str) -> dict | None:
    tags = el.get("tags", {})
    name = tags.get("name") or tags.get("brand")
    if not name:
        return None

    osm_type = el.get("type", "node")
    osm_id = el.get("id")
    source_url = f"https://www.openstreetmap.org/{osm_type}/{osm_id}"

    lat = el.get("lat") or (el.get("center") or {}).get("lat")
    lng = el.get("lon") or (el.get("center") or {}).get("lon")

    tourism = tags.get("tourism", "hotel")
    stars = tags.get("stars")
    category = "safari_lodge" if "safari" in name.lower() or "lodge" in name.lower() else tourism

    description = tags.get("description") or (
        f"{name} is a {tourism.replace('_', ' ')} in {location}, {country}."
    )

    amenities = []
    for key in ("internet_access", "swimming_pool", "air_conditioning", "parking", "restaurant"):
        if tags.get(key) == "yes":
            amenities.append(key.replace("_", " "))

    return {
        "hotel_name": name.strip(),
        "location": location,
        "county_or_region": tags.get("addr:county") or tags.get("is_in:county") or location,
        "country": country,
        "description": description,
        "price_range": "luxury" if stars and int(stars) >= 4 else "midrange",
        "amenities": amenities,
        "room_types": [],
        "rating": float(stars) if stars and stars.isdigit() else None,
        "review_summary": "",
        "nearby_attractions": [],
        "hotel_category": category,
        "contact": tags.get("phone") or tags.get("contact:phone") or "",
        "website_url": tags.get("website") or tags.get("contact:website") or "",
        "image_url": "",
        "source_url": source_url,
        "lat": lat,
        "lng": lng,
        "destination_tags": [location.lower()],
        "data_source": "openstreetmap",
    }


def main() -> None:
    all_hotels: list[dict] = []
    seen: set[str] = set()

    for location, country, south, west, north, east in SEARCH_AREAS:
        print(f"Fetching {location}, {country}...")
        try:
            elements = overpass_hotels(south, west, north, east)
            for el in elements:
                row = element_to_hotel(el, location, country)
                if not row:
                    continue
                key = f"{row['hotel_name'].lower()}|{row['location'].lower()}|{row['country'].lower()}"
                if key in seen:
                    continue
                seen.add(key)
                all_hotels.append(row)
            print(f"  +{len(elements)} elements, total unique {len(all_hotels)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERR {location}: {exc}")
        time.sleep(REQUEST_DELAY_SEC)

    HOTELS_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HOTELS_RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(all_hotels, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(all_hotels)} hotels to {HOTELS_RAW_PATH}")


if __name__ == "__main__":
    main()
