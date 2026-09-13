import json
import re
from typing import Any

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .config import DESTINATIONS_PATH, EA_DESTINATIONS_PATH, EMBED_MODEL, HOTELS_PATH, INDEX_DIR
from .destination_meta import lookup_meta
from .llm import apply_polish
from .hotels import hotel_to_source, load_hotels
DESTINATION_ALIASES: dict[str, set[str]] = {
    "maasai mara": {"maasai mara", "mara"},
    "amboseli": {"amboseli"},
    "diani beach": {"diani", "diani beach", "dlani", "dlani beach"},
    "watamu": {"watamu"},
    "naivasha": {"naivasha", "lake naivasha"},
    "nairobi": {"nairobi"},
    "tsavo": {"tsavo"},
    "nanyuki and mount kenya": {"nanyuki", "mount kenya", "nanyuki and mount kenya"},
    "zanzibar": {"zanzibar", "stone town"},
    "pemba island": {"pemba", "pemba island"},
    "serengeti": {"serengeti"},
    "ngorongoro crater": {"ngorongoro", "ngorongoro crater"},
    "tarangire national park": {"tarangire", "tarangire national park"},
    "arusha": {"arusha"},
    "kampala": {"kampala"},
    "dar es salaam": {"dar es salaam", "dar"},
    "kigali": {"kigali"},
    "mount kilimanjaro": {"kilimanjaro", "mount kilimanjaro"},
    "bwindi impenetrable forest": {"bwindi", "bwindi impenetrable forest"},
    "volcanoes national park": {"volcanoes", "volcanoes national park", "virunga"},
    "queen elizabeth national park": {"queen elizabeth", "queen elizabeth national park"},
}
COUNTRY_ALIASES: dict[str, set[str]] = {
    "kenya": {"kenya", "kenyan"},
    "tanzania": {"tanzania", "tanzanian", "tz"},
    "uganda": {"uganda", "ugandan"},
    "rwanda": {"rwanda", "rwandan"},
}
BEACH_TERMS = {"beach", "beaches", "coast", "coastal", "ocean", "island", "marine", "snorkeling"}
SAFARI_TERMS = {
    "safari",
    "safaris",
    "wildlife",
    "game",
    "reserve",
    "migration",
    "national park",
}
KNOWN_BEACH_DESTINATIONS = {
    "diani beach",
    "watamu",
    "malindi",
    "lamu",
    "zanzibar",
    "pemba island",
}
KNOWN_SAFARI_DESTINATIONS = {
    "maasai mara",
    "amboseli",
    "tsavo",
    "samburu",
    "nairobi national park",
    "ol pejeta conservancy",
    "meru national park",
    "serengeti",
    "ngorongoro crater",
    "tarangire national park",
    "queen elizabeth national park",
    "bwindi impenetrable forest",
    "volcanoes national park",
    "taita hills",
}
INLAND_EXCLUDE_FROM_BEACH = {
    "naivasha",
    "nakuru",
    "nairobi",
    "kisumu",
    "nanyuki and mount kenya",
    "hells gate",
    "aberdare",
    "kakamega forest",
    "mount elgon",
    "maasai mara",
    "amboseli",
    "tsavo",
    "samburu",
    "arusha",
    "kampala",
    "kigali",
    "mombasa",
    "taita hills",
    "dar es salaam",
}
REGION_KEYWORDS: dict[str, set[str]] = {
    "coast": {"coast", "coastal", "beach", "ocean", "indian ocean", "mombasa", "diani", "watamu", "malindi", "lamu"},
    "western": {"western", "west", "kisumu", "kakamega", "bungoma", "lake victoria", "mount elgon"},
    "nairobi": {"nairobi", "capital", "city"},
    "northern": {"northern", "north", "turkana", "marsabit", "isiolo", "samburu"},
    "rift valley": {"rift valley", "nakuru", "naivasha", "hells gate", "maasai mara"},
    "central highlands": {"central", "highland", "nyeri", "nyandarua", "aberdare", "nanyuki", "mount kenya", "laikipia"},
    "eastern": {"eastern", "meru", "tsavo", "makueni", "kitui", "kajiado", "chyulu"},
}
CURATED_PRIORITY: dict[str, int] = {
    "maasai mara": 10,
    "diani beach": 10,
    "watamu": 9,
    "mombasa": 9,
    "nairobi": 8,
    "nairobi national park": 8,
    "amboseli": 9,
    "tsavo": 8,
    "naivasha": 8,
    "nanyuki and mount kenya": 5,
    "hells gate": 7,
    "samburu": 7,
    "lamu": 7,
    "malindi": 7,
    "meru national park": 6,
    "kakamega forest": 6,
    "mount elgon": 6,
    "aberdare": 6,
    "zanzibar": 10,
    "serengeti": 10,
    "ngorongoro crater": 9,
    "tarangire national park": 8,
    "volcanoes national park": 9,
    "bwindi impenetrable forest": 9,
    "queen elizabeth national park": 8,
    "kigali": 7,
    "pemba island": 8,
    "mount kilimanjaro": 9,
    "arusha": 6,
}

# Iconic picks for country-level questions — not every town in the dataset.
TOP_TOURIST_PICKS: dict[str, list[str]] = {
    "kenya": [
        "maasai mara",
        "diani beach",
        "watamu",
        "amboseli",
        "mombasa",
        "lamu",
        "malindi",
        "tsavo",
        "naivasha",
        "nairobi",
        "samburu",
    ],
    "tanzania": [
        "serengeti",
        "ngorongoro crater",
        "zanzibar",
        "mount kilimanjaro",
        "tarangire national park",
        "arusha",
    ],
    "uganda": [
        "bwindi impenetrable forest",
        "queen elizabeth national park",
        "kampala",
    ],
    "rwanda": [
        "volcanoes national park",
        "kigali",
    ],
}

# County towns and regional hubs — fine if asked directly, not for generic country lists.
NON_TOURIST_DESTINATIONS = {
    "meru town",
    "embu",
    "eldoret",
    "machakos",
    "thika",
    "nyeri",
    "kericho",
    "kitale",
    "narok",
    "kisumu",
    "nakuru",
    "marsabit",
    "isiolo and shaba",
    "dar es salaam",
}

# Balanced highlights when the user asks about the whole region.
TOP_EAST_AFRICA_PICKS = [
    "maasai mara",
    "serengeti",
    "diani beach",
    "zanzibar",
    "bwindi impenetrable forest",
    "volcanoes national park",
    "amboseli",
    "ngorongoro crater",
]

EAST_AFRICA_COUNTRIES = frozenset({"kenya", "tanzania", "uganda", "rwanda"})


def _create_embeddings() -> HuggingFaceEmbeddings:
    try:
        # Avoid network lookups on every request once model files are cached.
        return HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"local_files_only": True},
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Embeddings model is not available locally. Run `python scripts/ingest.py` once with internet access."
        ) from exc


def _hotel_to_document(row: dict[str, Any]) -> Document:
    amenities = ", ".join(row.get("amenities") or [])
    room_types = ", ".join(row.get("room_types") or [])
    nearby = ", ".join(row.get("nearby_attractions") or [])
    content = (
        f"Hotel: {row['hotel_name']}\n"
        f"Location: {row['location']}\n"
        f"Country: {row['country']}\n"
        f"Region: {row.get('county_or_region', row['location'])}\n"
        f"Price range: {row.get('price_range', 'midrange')}\n"
        f"Category: {row.get('hotel_category', 'hotel')}\n"
        f"Rating: {row.get('rating') or 'N/A'}\n"
        f"Amenities: {amenities or 'N/A'}\n"
        f"Room types: {room_types or 'N/A'}\n"
        f"Nearby attractions: {nearby or 'N/A'}\n"
        f"Description: {row.get('description', '')}\n"
        f"Contact: {row.get('contact', '')}\n"
        f"Website: {row.get('website_url', '')}\n"
        f"Source: {row['source_url']}"
    )
    return Document(
        page_content=content,
        metadata={
            "type": "hotel",
            "hotel_name": row["hotel_name"],
            "location": row["location"],
            "country": row.get("country", "Kenya"),
            "source_url": row["source_url"],
            "price_range": row.get("price_range", "midrange"),
        },
    )


def _to_document(row: dict[str, Any]) -> Document:
    country = row.get("country", "Kenya")
    content = (
        f"Destination: {row['destination']}\n"
        f"Region: {row['region']}\n"
        f"Country: {country}\n"
        f"Best months: {', '.join(row['best_months'])}\n"
        f"Best time to visit: {row['best_time']}\n"
        f"Budget hotels: {', '.join(row['hotels']['budget'])}\n"
        f"Midrange hotels: {', '.join(row['hotels']['midrange'])}\n"
        f"Luxury hotels: {', '.join(row['hotels']['luxury'])}\n"
        f"Activities: {', '.join(row['activities'])}\n"
        f"Summary: {row['summary']}\n"
        f"Source: {row['source_url']}"
    )
    return Document(
        page_content=content,
        metadata={
            "destination": row["destination"],
            "country": country,
            "source_url": row["source_url"],
            "type": "tourism_destination",
        },
    )


def _parse_sources(results: list[Any]) -> list[str]:
    unique: list[str] = []
    for doc in results:
        url = doc.metadata.get("source_url")
        if url and url not in unique:
            unique.append(url)
    return unique


def _parse_structured_sources(results: list[Any]) -> list[dict[str, str]]:
    structured: list[dict[str, str]] = []
    seen: set[str] = set()
    for doc in results:
        meta = doc.metadata or {}
        url = meta.get("source_url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        if meta.get("type") == "hotel":
            structured.append(
                {
                    "hotel_name": meta.get("hotel_name", ""),
                    "location": meta.get("location", ""),
                    "country": meta.get("country", ""),
                    "source_url": url,
                    "price_range": meta.get("price_range", ""),
                }
            )
        else:
            fields = _to_fields(doc)
            structured.append(
                {
                    "hotel_name": fields.get("destination", ""),
                    "location": fields.get("region", ""),
                    "country": "Kenya",
                    "source_url": url,
                    "price_range": "",
                }
            )
    return structured


def _extract_hotels(results: list[Any], limit: int = 6) -> list[dict[str, Any]]:
    hotels: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in results:
        if doc.metadata.get("type") != "hotel":
            continue
        fields = _to_fields(doc)
        name = fields.get("hotel", doc.metadata.get("hotel_name", "Unknown"))
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        hotels.append(
            {
                "hotel_name": name,
                "location": fields.get("location", doc.metadata.get("location", "")),
                "country": fields.get("country", doc.metadata.get("country", "Kenya")),
                "description": fields.get("description", ""),
                "price_range": fields.get("price range", doc.metadata.get("price_range", "")),
                "amenities": fields.get("amenities", ""),
                "rating": fields.get("rating", ""),
                "source_url": fields.get("source", doc.metadata.get("source_url", "")),
                "website_url": fields.get("website", ""),
            }
        )
        if len(hotels) >= limit:
            break
    return hotels


def _hotel_docs(results: list[Any]) -> list[Any]:
    return [doc for doc in results if doc.metadata.get("type") == "hotel"]


def _destination_docs(results: list[Any]) -> list[Any]:
    return [doc for doc in results if doc.metadata.get("type") == "tourism_destination"]


def _is_east_africa_query(query: str) -> bool:
    q = query.lower()
    return "east africa" in q or "east african" in q


def _mentioned_countries(query: str) -> set[str]:
    query_lower = query.lower()
    matches: set[str] = set()
    for canonical, aliases in COUNTRY_ALIASES.items():
        for alias in aliases:
            if len(alias) <= 2:
                if re.search(rf"\b{re.escape(alias)}\b", query_lower):
                    matches.add(canonical)
            elif alias in query_lower:
                matches.add(canonical)
    if matches:
        return matches
    if _is_east_africa_query(query):
        return set(EAST_AFRICA_COUNTRIES)
    return matches


def _fields_match_country(fields: dict[str, str], country: str) -> bool:
    country_lower = country.lower()
    region = fields.get("region", "").lower()
    doc_country = fields.get("country", "").lower()
    return country_lower in doc_country or country_lower in region


def _row_countries(row: dict[str, Any]) -> set[str]:
    raw = (row.get("country") or "Kenya").lower()
    parts = {p.strip() for p in re.split(r"[/,]| and ", raw) if p.strip()}
    return parts or {"kenya"}


def _row_matches_countries(row: dict[str, Any], countries: set[str]) -> bool:
    if not countries:
        return True
    row_countries = _row_countries(row)
    return bool(row_countries.intersection(countries))


def _query_theme(query: str) -> str | None:
    query_lower = query.lower()
    terms = _tokenize(query)
    if terms.intersection(BEACH_TERMS) or any(t in query_lower for t in ("beach", "beaches")):
        return "beach"
    if terms.intersection({"safari", "wildlife", "migration"}) or "safari" in query_lower or "national park" in query_lower:
        # Don't treat alone "park visit nairobi" city as safari unless wildlife cue — keep simple.
        if terms.intersection({"safari", "safaris", "wildlife", "game", "migration", "reserve"}) or "safari" in query_lower:
            return "safari"
        if "best" in terms and "park" in terms:
            return "safari"
    return None


def _row_matches_theme(row: dict[str, Any], theme: str) -> bool:
    name = str(row.get("destination", "")).lower().strip()
    blob = " ".join(
        [
            name,
            str(row.get("summary", "")),
            str(row.get("type", "")),
            " ".join(row.get("activities") or []),
            str(row.get("region", "")),
        ]
    ).lower()
    tokens = _tokenize(blob)
    if theme == "beach":
        if name in KNOWN_BEACH_DESTINATIONS:
            return True
        if name in INLAND_EXCLUDE_FROM_BEACH or name in KNOWN_SAFARI_DESTINATIONS:
            return False
        return bool(tokens.intersection({"beach", "beaches", "marine", "snorkeling"})) or "beach" in blob
    if theme == "safari":
        if name in KNOWN_SAFARI_DESTINATIONS:
            return True
        if name in KNOWN_BEACH_DESTINATIONS or name in {
            "arusha",
            "kampala",
            "kigali",
            "dar es salaam",
            "nairobi",
            "mombasa",
        }:
            return False
        return bool(
            tokens.intersection({"safari", "wildlife", "game", "reserve", "migration", "gorilla", "lion", "elephant"})
        ) or "national park" in blob
    return False


def _format_destination_list(title: str, rows: list[dict[str, Any]], limit: int = 6) -> str:
    lines = [title]
    ranked = sorted(
        rows,
        key=lambda r: CURATED_PRIORITY.get(str(r.get("destination", "")).lower(), 0),
        reverse=True,
    )
    for row in ranked[:limit]:
        name = row.get("destination", "Unknown")
        region = row.get("region", "")
        country = row.get("country", "Kenya")
        place = region if country.lower() in region.lower() else f"{region}, {country}".strip(", ")
        summary = row.get("summary", "").strip()
        activities = ", ".join(row.get("activities") or []) or "N/A"
        lines.append(f"- {name} ({place}): {summary}")
        lines.append(f"  Activities: {activities}")
    return "\n".join(lines)


def _is_broad_country_query(query: str) -> bool:
    """Country-wide or regional question with no specific destination named."""
    if _mentioned_destinations(query):
        return False
    if _mentioned_regions(query):
        return False
    terms = _tokenize(query)
    if terms.intersection({"hotel", "hotels", "resort", "stay", "accommodation", "lodge"}) and not _query_theme(query):
        return False
    if _mentioned_countries(query):
        return True
    return False


def _tourist_catalog_rows(countries: set[str], limit: int = 6) -> list[dict[str, Any]]:
    """Curated iconic destinations for country or regional overviews."""
    all_rows = _load_destination_rows()
    by_name = {str(r.get("destination", "")).lower(): r for r in all_rows}
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()

    if countries >= EAST_AFRICA_COUNTRIES:
        for pick in TOP_EAST_AFRICA_PICKS:
            row = by_name.get(pick)
            if row and pick not in seen:
                picked.append(row)
                seen.add(pick)
            if len(picked) >= limit:
                return picked[:limit]

    country_list = sorted(countries)
    if len(country_list) > 1:
        per_country = max(2, limit // len(country_list))
        for country in country_list:
            count = 0
            for pick in TOP_TOURIST_PICKS.get(country, []):
                if count >= per_country or len(picked) >= limit:
                    break
                row = by_name.get(pick)
                if row and _row_matches_countries(row, {country}) and pick not in seen:
                    picked.append(row)
                    seen.add(pick)
                    count += 1
        if len(picked) >= limit:
            return picked[:limit]

    for country in country_list:
        for pick in TOP_TOURIST_PICKS.get(country, []):
            row = by_name.get(pick)
            if row and _row_matches_countries(row, {country}) and pick not in seen:
                picked.append(row)
                seen.add(pick)
            if len(picked) >= limit:
                return picked[:limit]

    if len(picked) < limit:
        fallback = sorted(
            [
                row
                for row in all_rows
                if _row_matches_countries(row, countries)
                and str(row.get("destination", "")).lower() not in NON_TOURIST_DESTINATIONS
                and str(row.get("destination", "")).lower() not in seen
            ],
            key=lambda r: CURATED_PRIORITY.get(str(r.get("destination", "")).lower(), 0),
            reverse=True,
        )
        for row in fallback:
            key = str(row.get("destination", "")).lower()
            picked.append(row)
            seen.add(key)
            if len(picked) >= limit:
                break
    return picked[:limit]


def _catalog_country_answer(query: str) -> dict[str, Any] | None:
    """Fast curated list when someone asks about a whole country."""
    if not _is_broad_country_query(query):
        return None

    countries = _mentioned_countries(query)
    rows = _tourist_catalog_rows(countries, limit=8 if countries >= EAST_AFRICA_COUNTRIES else 6)
    if not rows:
        return None

    if countries >= EAST_AFRICA_COUNTRIES:
        label = "East Africa"
    else:
        label = ", ".join(c.title() for c in sorted(countries))
    title = f"Top places to visit in {label}:"
    limit = 8 if countries >= EAST_AFRICA_COUNTRIES else 6
    destinations = [_row_to_destination(row) for row in rows]
    answer = _format_destination_list(title, rows, limit=limit)
    sources = [d["source_url"] for d in destinations if d.get("source_url")]
    return {
        "answer": answer,
        "sources": sources,
        "structured_sources": [],
        "hotels": [],
        "destinations": destinations,
    }


def _catalog_theme_answer(query: str) -> dict[str, Any] | None:
    """Fast, country-scoped lists for beach/safari questions (skip noisy hotel retrieval)."""
    theme = _query_theme(query)
    if not theme:
        return None

    # Skip if user clearly wants hotels only.
    terms = _tokenize(query)
    if terms.intersection({"hotel", "hotels", "resort", "stay", "accommodation"}) and not terms.intersection(
        {"beach", "beaches", "safari", "safaris"}
    ):
        return None

    countries = _mentioned_countries(query)
    if not countries:
        countries = {"kenya"}

    rows = [
        row
        for row in _load_destination_rows()
        if _row_matches_countries(row, countries) and _row_matches_theme(row, theme)
    ]
    if not rows:
        return None

    label = ", ".join(c.title() for c in sorted(countries))
    if theme == "beach":
        title = f"Best beaches in {label}:"
    else:
        title = f"Best safari destinations in {label}:"

    destinations = [_row_to_destination(row) for row in sorted(
        rows,
        key=lambda r: CURATED_PRIORITY.get(str(r.get("destination", "")).lower(), 0),
        reverse=True,
    )[:6]]
    answer = _format_destination_list(title, rows, limit=6)
    sources = [d["source_url"] for d in destinations if d.get("source_url")]
    return {
        "answer": answer,
        "sources": sources,
        "structured_sources": [],
        "hotels": [],
        "destinations": destinations,
    }


# Ambiguous short aliases that must NOT match hotel names alone (e.g. "mara" → Marathon/Marafiki).
AMBIGUOUS_PLACE_TOKENS = {"mara", "dar", "ark"}

# Strong location hints for hotel filtering (location/tags preferred over hotel name).
PLACE_LOCATION_HINTS: dict[str, set[str]] = {
    "maasai mara": {"maasai mara", "masai mara", "narok"},
    "diani beach": {"diani", "diani beach", "ukunda", "kwale"},
    "watamu": {"watamu", "kilifi"},
    "amboseli": {"amboseli", "kajiado"},
    "tsavo": {"tsavo"},
    "samburu": {"samburu"},
    "nairobi": {"nairobi"},
    "mombasa": {"mombasa", "nyali", "bamburi"},
    "naivasha": {"naivasha"},
    "zanzibar": {"zanzibar", "stone town", "nungwi"},
    "serengeti": {"serengeti"},
    "arusha": {"arusha"},
    "kampala": {"kampala"},
    "kigali": {"kigali"},
    "lamu": {"lamu", "shela"},
    "malindi": {"malindi"},
}


def _hotel_location_keys(place: str) -> list[str]:
    """Search keys for a destination, including aliases and short forms."""
    place = place.lower().strip()
    keys = {place, place.replace(" beach", ""), place.replace(" national park", "")}
    aliases = DESTINATION_ALIASES.get(place, set())
    keys.update(aliases)
    for canonical, alias_set in DESTINATION_ALIASES.items():
        if place in alias_set or place == canonical:
            keys.add(canonical)
            keys.update(alias_set)
    hints = PLACE_LOCATION_HINTS.get(place, set())
    keys.update(hints)
    return [k for k in keys if k]


def _hotel_place_blob(hotel: dict[str, Any], include_name: bool = False) -> str:
    parts = [
        str(hotel.get("location", "")),
        str(hotel.get("county_or_region", "")),
        " ".join(hotel.get("destination_tags") or []),
    ]
    if include_name:
        parts.append(str(hotel.get("hotel_name", "")))
        parts.append(str(hotel.get("description", "")))
    return " ".join(parts).lower()


def _hotel_matches_place(hotel: dict[str, Any], places: set[str]) -> bool:
    """Match hotels to a place by location/tags first — avoid 'mara' hitting Marathon/Marafiki."""
    loc_blob = _hotel_place_blob(hotel, include_name=False)
    name = str(hotel.get("hotel_name", "")).lower()

    for place in places:
        hints = PLACE_LOCATION_HINTS.get(place) or set(_hotel_location_keys(place))
        strong = {h for h in hints if h not in AMBIGUOUS_PLACE_TOKENS and len(h) >= 4}
        strong.add(place)
        if any(h in loc_blob for h in strong):
            return True
        # Name match only for full place phrases (e.g. "Maasai Mara Camp"), not bare "mara"
        if place in name or place.replace(" ", "") in name.replace(" ", ""):
            return True
        if place == "maasai mara" and ("maasai mara" in name or "masai mara" in name):
            return True
    return False


def _price_rank(value: str) -> int:
    order = {"luxury": 0, "midrange": 1, "budget": 2}
    return order.get((value or "midrange").lower(), 3)


def _hotels_from_guides(places: set[str]) -> list[dict[str, Any]]:
    """Curated hotel names from destination guides when OSM has no place-tagged rows."""
    rows: list[dict[str, Any]] = []
    for row in _load_destination_rows():
        name = row["destination"].lower().strip()
        aliases = DESTINATION_ALIASES.get(name, {name})
        if not (name in places or places.intersection(aliases)):
            continue
        source = row.get("source_url", "")
        location = row["destination"]
        country = row.get("country", "Kenya")
        region = row.get("region", location)
        for tier, names in (row.get("hotels") or {}).items():
            for hotel_name in names:
                rows.append(
                    {
                        "hotel_name": hotel_name,
                        "location": location,
                        "country": country,
                        "description": f"{tier.title()} stay in {region}.",
                        "price_range": tier if tier in {"budget", "midrange", "luxury"} else "midrange",
                        "amenities": "",
                        "rating": "",
                        "source_url": source,
                        "website_url": "",
                        "data_source": "destination_guide",
                    }
                )
    return rows


def _filter_hotels_for_places(places: set[str], limit: int = 8) -> list[dict[str, Any]]:
    if not places:
        return []

    # Destination-guide lodges first (Maasai Mara safari camps, Diani resorts, etc.)
    guided = _hotels_from_guides(places)
    matched = [h for h in load_hotels() if _hotel_matches_place(h, places)]
    matched.sort(key=lambda h: (_price_rank(h.get("price_range", "")), h.get("hotel_name", "")))

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hotel in [*guided, *matched]:
        key = str(hotel.get("hotel_name", "")).lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(hotel)
        if len(merged) >= limit:
            break
    return merged


def _normalize_hotel_payload(hotels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for h in hotels:
        out.append(
            {
                "hotel_name": h.get("hotel_name", ""),
                "location": h.get("location", ""),
                "country": h.get("country", "Kenya"),
                "description": h.get("description", ""),
                "price_range": h.get("price_range", "midrange"),
                "amenities": ", ".join(h.get("amenities") or []) if isinstance(h.get("amenities"), list) else (h.get("amenities") or ""),
                "rating": str(h.get("rating") or ""),
                "source_url": h.get("source_url", ""),
                "website_url": h.get("website_url", ""),
            }
        )
    return out


def _detect_time_of_day(query: str) -> str | None:
    q = query.lower()
    if any(w in q for w in ("morning", "sunrise", "dawn")):
        return "morning"
    if any(w in q for w in ("afternoon", "midday", "lunch")):
        return "afternoon"
    if any(w in q for w in ("evening", "sunset", "dusk")):
        return "evening"
    if any(w in q for w in ("night", "nightlife")):
        return "night"
    return None


def _activity_time_note(activity: str, time_of_day: str | None) -> str:
    text = activity.lower()
    if time_of_day == "morning":
        if any(k in text for k in ("dolphin", "snorkel", "dive", "marine", "fish", "boat", "game drive", "safari", "trek", "hike")):
            return "Best early — calmer seas/wildlife and cooler temperatures."
        if "beach" in text or "leisure" in text:
            return "Nice after an early marine outing, while the sand is still cool."
        return "Works well in the morning before the heat builds."
    if time_of_day == "afternoon":
        if "beach" in text or "leisure" in text or "relax" in text:
            return "Ideal for swimming, shade, and a slower pace."
        return "Doable in the afternoon; bring sun protection."
    if time_of_day == "evening":
        if any(k in text for k in ("beach", "dining", "food", "culture", "walk", "town")):
            return "Lovely around sunset for cooler air and atmosphere."
        return "Better toward late afternoon/evening when possible."
    if time_of_day == "night":
        return "Check locally — many outdoor activities slow down after dark."
    return ""


def _split_activity_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(a).strip() for a in raw if str(a).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _format_rich_activity_answer(
    destination: str,
    region: str,
    summary: str,
    activities_raw: Any,
    best_time: str = "",
    time_of_day: str | None = None,
) -> str:
    activities = _split_activity_list(activities_raw)
    place = f"{destination}" + (f" ({region})" if region else "")

    if time_of_day == "morning":
        header = f"Morning plan for {place}"
    elif time_of_day:
        header = f"{time_of_day.title()} ideas in {place}"
    else:
        header = f"What to do in {place}"

    lines = [f"{header}:"]
    if summary:
        lines.append(summary.rstrip(".") + ".")

    if best_time:
        lines.append(f"Local tip: {best_time.rstrip('.')}.")

    if not activities:
        lines.append("Try asking about beaches, snorkeling, or wildlife for a more specific plan.")
        return "\n".join(lines)

    if time_of_day == "morning":
        priority_keys = ("dolphin", "snorkel", "dive", "marine", "fish", "boat", "game", "trek", "hike", "walk")
        activities = sorted(
            activities,
            key=lambda a: (0 if any(k in a.lower() for k in priority_keys) else 1, a.lower()),
        )

    lines.append("")
    lines.append("Recommended:" if time_of_day else "Top activities:")
    for activity in activities[:5]:
        note = _activity_time_note(activity, time_of_day)
        if note:
            lines.append(f"- {activity} — {note}")
        else:
            lines.append(f"- {activity}")

    if time_of_day == "morning":
        lines.append("")
        lines.append(
            "Practical tip: book boat or marine-park slots the day before, and head out early "
            "(about 6–9am) when conditions are usually best."
        )

    return "\n".join(lines)


def _catalog_activity_answer(query: str) -> dict[str, Any] | None:
    terms = _tokenize(query)
    q = query.lower()
    activity_intent = bool(
        terms.intersection({"activity", "activities", "things", "todo", "do", "adventure", "fun"})
    ) or "what to do" in q or "things to do" in q
    if not activity_intent:
        return None

    theme = _query_theme(query)
    if theme in {"beach", "safari"} and "activit" not in q and "things to do" not in q and "what to do" not in q:
        return None
    if terms.intersection({"hotel", "hotels", "resort", "stay", "lodge"}) and "activit" not in q:
        return None

    places = _mentioned_destinations(query)
    if not places:
        return None

    time_of_day = _detect_time_of_day(query)
    destinations = _destinations_for_mentioned(places)
    if not destinations:
        for row in _load_destination_rows():
            key = row["destination"].lower().strip()
            if key in places or places.intersection(DESTINATION_ALIASES.get(key, {key})):
                destinations = [_row_to_destination(row)]
                break
    if not destinations:
        return None

    primary = destinations[0]
    best_visit = ""
    for row in _load_destination_rows():
        if row["destination"].lower().strip() == primary["name"].lower().strip():
            best_visit = row.get("best_time", "")
            break

    answer = _format_rich_activity_answer(
        destination=primary["name"],
        region=primary.get("location", ""),
        summary=primary.get("description", ""),
        activities_raw=primary.get("activities", ""),
        best_time=best_visit,
        time_of_day=time_of_day,
    )
    return {
        "answer": answer,
        "sources": [primary["source_url"]] if primary.get("source_url") else [],
        "structured_sources": [],
        "hotels": [],
        "destinations": destinations[:3],
    }


def _is_chitchat_query(query: str) -> bool:
    """True for greetings / thanks with no travel ask — must not invent destinations."""
    q = re.sub(r"[^\w\s]", " ", query.lower()).strip()
    q = re.sub(r"\s+", " ", q)
    if len(q) < 2:
        return True

    chitchat = {
        "hi",
        "hello",
        "hey",
        "hola",
        "karibu",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "how are you",
        "how are u",
        "whats up",
        "what s up",
        "sup",
        "yo",
        "thanks",
        "thank you",
        "thx",
        "ty",
        "ok",
        "okay",
        "bye",
        "goodbye",
        "see you",
        "hello there",
        "hi there",
        "hey there",
        "test",
        "testing",
    }
    if q in chitchat:
        return True
    if q.startswith(("hello ", "hi ", "hey ", "thanks ", "thank you")) and len(q.split()) <= 4:
        # "hello there", "hi friend" — no place/country/theme
        if not _mentioned_destinations(query) and not _mentioned_countries(query) and not _query_theme(query):
            travel_cues = {
                "hotel",
                "hotels",
                "safari",
                "beach",
                "trip",
                "plan",
                "visit",
                "go",
                "kenya",
                "tanzania",
                "uganda",
                "rwanda",
                "where",
                "best",
                "activity",
                "activities",
            }
            if not _tokenize(q).intersection(travel_cues):
                return True
    return False


def _chitchat_answer(query: str) -> dict[str, Any]:
    q = query.lower()
    if any(w in q for w in ("thank", "thx", "ty")):
        text = "You're welcome! How else can I help?"
    elif any(w in q for w in ("bye", "goodbye")):
        text = "Goodbye — safe travels!"
    else:
        text = "Hello! How may I help you today?"
    return {
        "answer": text,
        "sources": [],
        "structured_sources": [],
        "hotels": [],
        "destinations": [],
    }


def _catalog_hotel_answer(query: str) -> dict[str, Any] | None:
    """
    Constraint-first hotel answers:
    if the user names a place, only return hotels for that place (OSM or destination guide).
    Never silently substitute a nearby coast town.
    """
    terms = _tokenize(query)
    hotel_intent = bool(
        terms.intersection({"hotel", "hotels", "resort", "stay", "stays", "accommodation", "lodge", "lodges"})
    ) or "where to stay" in query.lower()
    if not hotel_intent:
        return None

    places = _mentioned_destinations(query)
    if not places:
        return None

    # Prefer primary place label for the heading.
    place_label = ", ".join(p.title() for p in sorted(places))
    hotels = _normalize_hotel_payload(_filter_hotels_for_places(places, limit=8))
    destinations = _destinations_for_mentioned(places)

    if hotels:
        lines = [f"Hotels in {place_label}:"]
        for h in hotels[:6]:
            price = (h.get("price_range") or "midrange").title()
            lines.append(f"- {h['hotel_name']} — {price} · {h['location']}, {h['country']}")
        return {
            "answer": "\n".join(lines),
            "sources": [h["source_url"] for h in hotels if h.get("source_url")],
            "structured_sources": [hotel_to_source(h) for h in hotels[:6]],
            "hotels": hotels[:6],
            "destinations": destinations,
        }

    return {
        "answer": (
            f"I could not find hotels specifically listed for {place_label} yet. "
            f"Try asking about things to do in {place_label}, or hotels in a nearby town."
        ),
        "sources": [],
        "structured_sources": [],
        "hotels": [],
        "destinations": destinations,
    }


def _build_country_answer(countries: set[str], results: list[Any]) -> str | None:
    if not countries:
        return None

    catalog_rows = _tourist_catalog_rows(countries, limit=8 if countries >= EAST_AFRICA_COUNTRIES else 6)
    if catalog_rows:
        if countries >= EAST_AFRICA_COUNTRIES:
            label = "East Africa"
        else:
            label = ", ".join(c.title() for c in sorted(countries))
        limit = 8 if countries >= EAST_AFRICA_COUNTRIES else 6
        return _format_destination_list(f"Top places to visit in {label}:", catalog_rows, limit=limit)

    dest_docs = _destination_docs(results)
    picks: list[dict[str, str]] = []
    seen: set[str] = set()
    for doc in dest_docs:
        fields = _to_fields(doc)
        name = fields.get("destination", "").strip()
        if not name or name.lower() in seen:
            continue
        if any(_fields_match_country(fields, country) for country in countries):
            picks.append(fields)
            seen.add(name.lower())
        if len(picks) >= 6:
            break

    if not picks:
        return None

    label = ", ".join(c.title() for c in sorted(countries))
    lines = [f"Top places to explore in {label}:"]
    for fields in picks:
        destination = fields.get("destination", "Unknown")
        region = fields.get("region", "")
        summary = fields.get("summary", "No summary available.")
        activities = fields.get("activities", "N/A")
        lines.append(f"- {destination} ({region}): {summary}")
        lines.append(f"  Activities: {activities}")
    return "\n".join(lines)


def _to_fields(doc: Any) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in doc.page_content.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    return fields


def _build_answer(query: str, results: list[Any]) -> str:
    if not results:
        return "I could not find enough context for that question. Try asking about a specific destination, month, or activity."

    query_terms = _tokenize(query)
    intent_hotel = bool(query_terms.intersection({"hotel", "resort", "stay", "accommodation", "luxury", "budget"}))
    intent_time = bool(query_terms.intersection({"when", "month", "season", "time"})) or ("best time" in query.lower())
    intent_beach = _query_theme(query) == "beach"
    intent_safari = _query_theme(query) == "safari"
    intent_hike = bool(query_terms.intersection({"hike", "hiking", "trek", "trekking", "climb", "mountain"}))
    intent_activity = bool(query_terms.intersection({"activity", "activities", "things", "todo", "do", "adventure"}))
    mentioned_countries = _mentioned_countries(query)

    # Country overview — use curated picks, even when the user says "things to do" or "places to visit".
    if _is_broad_country_query(query):
        country_answer = _build_country_answer(mentioned_countries, results)
        if country_answer:
            return country_answer

    # Country-only overview when the user asks about a country without beach/safari/hotel intent.
    if mentioned_countries and not (intent_hotel or intent_time or intent_beach or intent_safari or intent_hike or intent_activity):
        country_answer = _build_country_answer(mentioned_countries, results)
        if country_answer:
            return country_answer

    broad_intent = bool(
        query_terms.intersection(
            {"kenya", "tanzania", "uganda", "rwanda", "east", "africa", "country", "overall", "general", "around", "across"}
        )
    ) and not bool(query_terms.intersection({"nanyuki", "diani", "watamu", "amboseli", "mara", "nairobi", "tsavo", "kigali"}))
    mentioned_destinations = _mentioned_destinations(query)
    query_lower = query.lower()

    top_fields = _to_fields(results[0])

    def destination_in_query(destination_name: str) -> bool:
        return destination_name.lower() in query_lower

    def matches_mentioned_destination(destination_name: str) -> bool:
        canonical = destination_name.lower().strip()
        if not canonical or canonical == "unknown":
            return False
        aliases = DESTINATION_ALIASES.get(canonical, {canonical})
        return any(alias and alias in query_lower for alias in aliases)

    def collect_unique(docs: list[Any], limit: int = 6) -> list[dict[str, str]]:
        ordered = _destination_docs(docs) + _hotel_docs(docs)
        unique: list[dict[str, str]] = []
        seen: set[str] = set()
        for doc in ordered:
            fields = _to_fields(doc)
            name = fields.get("destination", "").strip()
            if not name or name.lower() == "unknown" or name in seen:
                continue
            if mentioned_countries and not any(_fields_match_country(fields, c) for c in mentioned_countries):
                continue
            unique.append(fields)
            seen.add(name)
            if len(unique) >= limit:
                break
        return unique

    destination_results = collect_unique(_destination_docs(results), limit=12)
    unique_results = destination_results if destination_results else collect_unique(results, limit=12)

    def with_location(fields: dict[str, str]) -> str:
        destination = fields.get("destination", "Unknown")
        region = fields.get("region", "Unknown location")
        country = fields.get("country", "")
        if country:
            return f"{destination} ({region}, {country})"
        return f"{destination} ({region})"

    def priority(fields: dict[str, str]) -> int:
        return CURATED_PRIORITY.get(fields.get("destination", "").lower(), 0)

    curated_sorted = sorted(unique_results, key=priority, reverse=True)
    country_label = ", ".join(c.title() for c in sorted(mentioned_countries)) if mentioned_countries else "Kenya"

    direct_destination = next(
        (f for f in unique_results if matches_mentioned_destination(f.get("destination", ""))),
        None,
    )
    if not direct_destination:
        direct_destination = next(
            (f for f in unique_results if destination_in_query(f.get("destination", ""))),
            None,
        )
    if direct_destination and not (intent_hotel or intent_time or intent_activity or intent_beach or intent_safari):
        return (
            f"{with_location(direct_destination)}:\n"
            f"{direct_destination.get('summary', 'No summary available.')}\n"
            f"Top activities: {direct_destination.get('activities', 'N/A')}"
        )

    if intent_hotel:
        places = mentioned_destinations
        if places:
            constrained = _normalize_hotel_payload(_filter_hotels_for_places(places, limit=8))
            place_label = ", ".join(p.title() for p in sorted(places))
            if constrained:
                lines = [f"Hotels in {place_label}:"]
                for h in constrained[:6]:
                    price = (h.get("price_range") or "midrange").title()
                    lines.append(f"- {h['hotel_name']} — {price} · {h['location']}, {h['country']}")
                return "\n".join(lines)
            return (
                f"I could not find hotels specifically listed for {place_label}. "
                "I will not substitute hotels from other towns."
            )

        hotel_results = _extract_hotels(results, limit=8)
        if mentioned_countries:
            hotel_results = [
                h for h in hotel_results if any(c in (h.get("country") or "").lower() for c in mentioned_countries)
            ] or hotel_results
        if hotel_results:
            lines = [f"Hotels matching your search in {country_label if mentioned_countries else 'East Africa'}:"]
            for h in hotel_results[:6]:
                price = h.get("price_range", "midrange")
                lines.append(
                    f"- {h['hotel_name']} ({h['location']}, {h['country']}) — {price}. "
                    f"{h.get('description', '')[:120]}"
                )
            return "\n".join(lines)

        target = next((f for f in unique_results if destination_in_query(f.get("destination", ""))), None)
        if mentioned_destinations or target:
            chosen = target if target else top_fields
            return (
                f"Hotels in {with_location(chosen)}:\n"
                f"Luxury: {chosen.get('luxury hotels', 'N/A')}\n"
                f"Midrange: {chosen.get('midrange hotels', 'N/A')}\n"
                f"Budget: {chosen.get('budget hotels', 'N/A')}"
            )

        picks: list[str] = []
        for fields in curated_sorted[:4]:
            picks.append(f"{with_location(fields)}: {fields.get('luxury hotels', 'N/A')}")
        return "Top hotel destinations:\n- " + "\n- ".join(picks)

    if intent_time:
        return (
            f"Best time to visit {with_location(top_fields)}: "
            f"{top_fields.get('best months', 'N/A')} ({top_fields.get('best time to visit', 'N/A')})"
        )

    if intent_beach:
        beaches = [
            f
            for f in curated_sorted
            if _tokenize(f.get("summary", "") + " " + f.get("activities", "") + " " + f.get("destination", "")).intersection(
                BEACH_TERMS
            )
        ]
        picks = beaches[:6] if beaches else curated_sorted[:4]
        return f"Best beaches in {country_label}:\n" + "\n".join(
            [f"- {with_location(f)}: {f.get('summary', '')}" for f in picks]
        )

    if intent_safari:
        safaris = [
            f
            for f in curated_sorted
            if _tokenize(f.get("summary", "") + " " + f.get("activities", "") + " " + f.get("destination", "")).intersection(
                {"safari", "wildlife", "game", "park", "reserve", "migration", "gorilla"}
            )
        ]
        picks = safaris[:6] if safaris else curated_sorted[:4]
        return f"Best safari destinations in {country_label}:\n" + "\n".join(
            [f"- {with_location(f)}: {f.get('summary', '')}" for f in picks]
        )

    if intent_hike:
        hikes = [
            f
            for f in curated_sorted
            if _tokenize(f.get("activities", "") + " " + f.get("summary", "")).intersection(
                {"hiking", "trekking", "climbing", "mountain", "hike", "forest"}
            )
        ]
        picks = hikes[:6] if hikes else unique_results[:4]
        return f"Best places to go on hikes in {country_label}:\n- " + "\n- ".join(
            [f"{f.get('destination', 'Unknown')} - {f.get('activities', 'N/A')}" for f in picks]
        )

    if intent_activity:
        target = next((f for f in unique_results if matches_mentioned_destination(f.get("destination", ""))), None)
        if not target:
            target = next((f for f in unique_results if destination_in_query(f.get("destination", ""))), None)
        if target:
            return _format_rich_activity_answer(
                destination=target.get("destination", "Unknown"),
                region=target.get("region", ""),
                summary=target.get("summary", ""),
                activities_raw=target.get("activities", ""),
                best_time=target.get("best time to visit", ""),
                time_of_day=_detect_time_of_day(query),
            )
        activity_picks: list[str] = []
        for fields in curated_sorted[:5]:
            activity_picks.append(f"{with_location(fields)}: {fields.get('activities', 'N/A')}")
        return f"Best activities to do in {country_label}:\n- " + "\n- ".join(activity_picks)
    if mentioned_destinations and curated_sorted:
        chosen = next(
            (f for f in curated_sorted if matches_mentioned_destination(f.get("destination", ""))),
            curated_sorted[0],
        )
        return f"{with_location(chosen)}: {chosen.get('summary', 'No summary available.')}"

    if broad_intent and not mentioned_destinations:
        beach_set = {"diani beach", "watamu", "malindi", "lamu", "zanzibar", "pemba island"}
        safari_set = {
            "maasai mara",
            "amboseli",
            "tsavo",
            "samburu",
            "nairobi national park",
            "serengeti",
            "ngorongoro crater",
            "tarangire national park",
        }
        beach_picks: list[str] = []
        safari_picks: list[str] = []
        other_picks: list[str] = []

        for fields in curated_sorted:
            destination = fields.get("destination", "Unknown")
            key = destination.lower()
            if key in beach_set and len(beach_picks) < 2:
                beach_picks.append(destination)
            elif key in safari_set and len(safari_picks) < 2:
                safari_picks.append(destination)
            elif len(other_picks) < 1:
                other_picks.append(destination)

        mixed = beach_picks + safari_picks + other_picks
        if len(mixed) < 5:
            for fields in curated_sorted:
                destination = fields.get("destination", "Unknown")
                if destination not in mixed:
                    mixed.append(destination)
                if len(mixed) >= 5:
                    break

        mixed_with_location: list[str] = []
        for destination in mixed[:5]:
            fields = next((f for f in curated_sorted if f.get("destination", "") == destination), None)
            if fields:
                mixed_with_location.append(with_location(fields))
            else:
                mixed_with_location.append(destination)
        return f"Best places to visit in {country_label}: {', '.join(mixed_with_location)}"

    top_three = [with_location(f) for f in curated_sorted[:4]]
    return f"Top matches: {', '.join(top_three)}" if top_three else "I could not find a clear match. Try naming a country or destination."


def _normalize_term(term: str) -> str:
    if term.endswith("ies") and len(term) > 4:
        return f"{term[:-3]}y"
    if term.endswith(("ches", "shes", "xes", "zes")) and len(term) > 5:
        return term[:-2]
    if term.endswith("s") and len(term) > 3 and not term.endswith("ss"):
        return term[:-1]
    return term


def _tokenize(text: str) -> set[str]:
    raw = re.findall(r"[a-zA-Z]{3,}", text.lower())
    normalized = {_normalize_term(token) for token in raw}
    return normalized


def _intent_keywords(query: str) -> set[str]:
    keywords = _tokenize(query)
    intent_map: dict[str, set[str]] = {
        "beach": {"beach", "coast", "coastal", "ocean", "island", "snorkeling", "marine", "sea"},
        "wildlife": {"safari", "wildlife", "game", "park", "reserve", "animal"},
        "city": {"city", "urban", "nightlife", "museum", "shopping"},
        "mountain": {"mountain", "hiking", "trekking", "climb", "highland"},
        "lake": {"lake", "boat", "fishing", "birdwatching"},
        "hotel": {"hotel", "resort", "budget", "midrange", "luxury"},
    }
    expanded = set(keywords)
    for trigger, related in intent_map.items():
        if trigger in keywords or keywords.intersection(related):
            expanded.update(related)
    return expanded


def _mentioned_destinations(query: str) -> set[str]:
    query_lower = query.lower()
    matches: set[str] = set()
    for canonical, aliases in DESTINATION_ALIASES.items():
        if any(alias and alias in query_lower for alias in aliases):
            matches.add(canonical)
    return matches


def _mentioned_regions(query: str) -> set[str]:
    query_lower = query.lower()
    matches: set[str] = set()
    for canonical, keywords in REGION_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            matches.add(canonical)
    return matches


def _destination_region(destination: str, region_text: str) -> str:
    text = f"{destination.lower()} {region_text.lower()}"
    for canonical, keywords in REGION_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return canonical
    return "other"


def _filter_results_by_region(results: list[Any], regions: set[str]) -> list[Any]:
    if not regions:
        return results
    filtered: list[Any] = []
    for doc in results:
        fields = _to_fields(doc)
        destination = fields.get("destination", "").lower()
        region_text = fields.get("region", "")
        doc_region = _destination_region(destination, region_text)
        if doc_region in regions:
            filtered.append(doc)
    return filtered


def _is_beach_doc(doc: Any) -> bool:
    fields = _to_fields(doc)
    terms = _tokenize(f"{fields.get('summary', '')} {fields.get('activities', '')} {fields.get('destination', '')}")
    return bool(terms.intersection({"beach", "coast", "coastal", "ocean", "marine", "island", "snorkeling"}))


def _merge_unique_docs(primary: list[Any], secondary: list[Any], limit: int = 20) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for doc in [*primary, *secondary]:
        destination = _to_fields(doc).get("destination", "unknown").lower()
        if destination in seen:
            continue
        merged.append(doc)
        seen.add(destination)
        if len(merged) >= limit:
            break
    return merged


def _rerank_results(query: str, results: list[Any]) -> list[Any]:
    if not results:
        return results

    query_lower = query.lower()
    query_terms = _intent_keywords(query)
    mentioned_destinations = _mentioned_destinations(query)
    mentioned_regions = _mentioned_regions(query)
    mentioned_countries = _mentioned_countries(query)
    hotel_intent = bool(query_terms.intersection({"hotel", "resort", "stay", "accommodation", "luxury", "budget"}))
    season_intent = bool(query_terms.intersection({"when", "month", "season", "time", "visit"}))
    activity_intent = bool(query_terms.intersection({"activity", "things", "do", "adventure", "safari", "beach"}))

    # Weighted field matching makes ranking adaptive for every query type.
    field_weights: dict[str, int] = {
        "destination": 6,
        "region": 4,
        "activities": 5,
        "summary": 3,
        "best months": 2,
        "best time to visit": 2,
        "budget hotels": 2,
        "midrange hotels": 2,
        "luxury hotels": 2,
    }
    if hotel_intent:
        field_weights["budget hotels"] = 8
        field_weights["midrange hotels"] = 8
        field_weights["luxury hotels"] = 8
    if season_intent:
        field_weights["best months"] = 8
        field_weights["best time to visit"] = 8
    if activity_intent:
        field_weights["activities"] = 8

    def score(doc: Any) -> tuple[int, int, int]:
        fields = _to_fields(doc)
        destination = fields.get("destination", "").lower()
        region_value = fields.get("region", "")
        content_terms = _tokenize(doc.page_content)
        base_overlap = len(query_terms.intersection(content_terms))
        destination_bonus = 8 if destination and destination in query_lower else 0
        destination_match_bonus = 0
        if mentioned_destinations and destination in mentioned_destinations:
            destination_match_bonus = 30
        country_bonus = 0
        if mentioned_countries:
            doc_country = fields.get("country", doc.metadata.get("country", "")).lower()
            if any(country in doc_country or country in region_value.lower() for country in mentioned_countries):
                country_bonus = 28
            elif doc.metadata.get("type") == "hotel":
                hotel_country = doc.metadata.get("country", "").lower()
                if any(country in hotel_country for country in mentioned_countries):
                    country_bonus = 20
                else:
                    country_bonus = -10
        region_bonus = 0
        if mentioned_regions:
            doc_region = _destination_region(destination, region_value)
            if doc_region in mentioned_regions:
                region_bonus = 26
            else:
                region_bonus = -6

        weighted_overlap = 0
        for field_name, weight in field_weights.items():
            value = fields.get(field_name, "")
            if not value:
                continue
            weighted_overlap += len(query_terms.intersection(_tokenize(value))) * weight

        return (
            weighted_overlap + base_overlap + destination_bonus + destination_match_bonus + region_bonus + country_bonus,
            destination_match_bonus + destination_bonus + region_bonus + country_bonus,
            base_overlap,
        )

    return sorted(results, key=score, reverse=True)


def _enrich_query(
    query: str,
    budget: str | None = None,
    travel_type: str | None = None,
    season: str | None = None,
) -> str:
    parts = [query.strip()]
    if budget:
        parts.append(f"{budget} budget hotels and stays")
    if travel_type:
        parts.append(f"{travel_type} travel style")
    if season:
        parts.append(f"best in {season}")
    return ". ".join(parts)


def _extract_destinations(results: list[Any], limit: int = 6) -> list[dict[str, Any]]:
    destinations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in _destination_docs(results):
        fields = _to_fields(doc)
        name = fields.get("destination", "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        meta = lookup_meta(name)
        destinations.append(
            {
                "name": name,
                "description": fields.get("summary", ""),
                "location": fields.get("region", fields.get("country", "Kenya")),
                "activities": fields.get("activities", ""),
                "best_months": fields.get("best months", ""),
                "source_url": fields.get("source", doc.metadata.get("source_url", "")),
                "lat": meta["lat"],
                "lng": meta["lng"],
                "image": meta["image"],
            }
        )
        if len(destinations) >= limit:
            break
    return destinations


def _parse_trip_days(query: str) -> int | None:
    query_lower = query.lower()
    word_nums = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}

    numeric = re.search(r"\b(\d{1,2})\s*-?\s*da[yts]?\b", query_lower)
    if numeric:
        days = int(numeric.group(1))
        if 2 <= days <= 14:
            return days

    word = re.search(r"\b(one|two|three|four|five|six|seven|eight)\s*-?\s*da[yts]?\b", query_lower)
    if word:
        return word_nums[word.group(1)]

    trip_phrase = re.search(r"\b(\d{1,2})\s*-?\s*day\s+trip\b", query_lower)
    if trip_phrase:
        days = int(trip_phrase.group(1))
        if 2 <= days <= 14:
            return days

    return None


def _split_activities(raw: str) -> list[str]:
    if not raw:
        return ["Explore local highlights"]
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load_destination_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in (DESTINATIONS_PATH, EA_DESTINATIONS_PATH):
        if path.exists():
            with open(path, encoding="utf-8") as file_obj:
                rows.extend(json.load(file_obj))
    return rows


def _row_to_destination(row: dict[str, Any]) -> dict[str, Any]:
    name = row["destination"]
    meta = lookup_meta(name)
    return {
        "name": name,
        "description": row.get("summary", ""),
        "location": row.get("region", row.get("country", "Kenya")),
        "activities": ", ".join(row.get("activities", [])),
        "best_months": ", ".join(row.get("best_months", [])),
        "source_url": row.get("source_url", ""),
        "lat": meta["lat"],
        "lng": meta["lng"],
        "image": meta["image"],
    }


def _destinations_for_mentioned(mentioned: set[str]) -> list[dict[str, Any]]:
    if not mentioned:
        return []

    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _load_destination_rows():
        key = row["destination"].lower().strip()
        matched = key in mentioned or any(
            key in DESTINATION_ALIASES.get(canonical, {canonical}) for canonical in mentioned
        )
        if matched and key not in seen:
            matches.append(_row_to_destination(row))
            seen.add(key)
    return matches


def _generate_itinerary(query: str, destinations: list[dict[str, Any]], days: int = 5) -> list[dict[str, str]]:
    if not destinations:
        return [
            {
                "day": "Day 1",
                "title": "Explore Nairobi",
                "description": "Start in Nairobi with a city tour and Nairobi National Park safari.",
            }
        ]

    # Single-destination trip: spread activities across each day.
    primary = destinations[0]
    mentioned = _mentioned_destinations(query)
    primary_key = primary["name"].lower().strip()
    single_place = len(destinations) == 1 or (
        bool(mentioned)
        and destinations
        and (primary_key in mentioned or any(primary_key in DESTINATION_ALIASES.get(c, {c}) for c in mentioned))
    )

    if single_place:
        activities = _split_activities(primary.get("activities", ""))
        itinerary: list[dict[str, str]] = []
        for index in range(days):
            activity = activities[index % len(activities)]
            day_num = index + 1
            if day_num == 1:
                title = f"Arrival in {primary['name']}"
                description = (
                    f"Check in around {primary['location']}. "
                    f"Start with {activity.lower()} and an easy orientation of the area."
                )
            else:
                title = f"{primary['name']} — explore"
                description = (
                    f"Day {day_num} in {primary['name']}: {activity}. "
                    f"Mix beach time with local dining and relaxed coastal pace."
                )
            itinerary.append({"day": f"Day {day_num}", "title": title, "description": description})
        return itinerary

    itinerary = []
    for index in range(min(days, len(destinations) + 1)):
        if index == 0:
            itinerary.append(
                {
                    "day": "Day 1",
                    "title": "Arrival in Nairobi",
                    "description": "Arrive, settle in, and enjoy a light city orientation with local cuisine.",
                }
            )
            continue

        dest = destinations[index - 1]
        activities = dest.get("activities", "Explore local highlights")
        activity_list = [a.strip() for a in activities.split(",")][:2]
        activity_text = " and ".join(activity_list) if activity_list else "local sightseeing"
        itinerary.append(
            {
                "day": f"Day {index + 1}",
                "title": dest["name"],
                "description": (
                    f"Travel to {dest['name']} ({dest['location']}). "
                    f"Enjoy {activity_text}. "
                    f"Best months: {dest.get('best_months', 'year-round')}."
                ),
            }
        )
    return itinerary


class TravelAfricaRAG:
    def __init__(self) -> None:
        self.embeddings = _create_embeddings()
        if not INDEX_DIR.exists():
            raise RuntimeError("Vector index not found. Run `python scripts/ingest.py` after collecting hotel data.")
        self.vector_store = FAISS.load_local(
            str(INDEX_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 12})
        # Warm the model once so the first user question is not stuck loading weights.
        self.embeddings.embed_query("east africa travel")

    def _filter_docs_by_country(self, results: list[Any], countries: set[str]) -> list[Any]:
        if not countries:
            return results
        filtered: list[Any] = []
        for doc in results:
            fields = _to_fields(doc)
            meta_country = (doc.metadata or {}).get("country", "")
            if any(
                c in (fields.get("country") or meta_country).lower()
                or c in fields.get("region", "").lower()
                for c in countries
            ):
                filtered.append(doc)
        return filtered or results

    def _retrieve(self, query: str) -> list[Any]:
        results = self.retriever.invoke(query)
        theme = _query_theme(query)
        countries = _mentioned_countries(query)
        country_label = " ".join(sorted(countries)) if countries else "kenya"

        if theme == "beach":
            beach_probe = self.retriever.invoke(
                f"best beach destinations {country_label} coast ocean sand snorkeling"
            )
            results = _merge_unique_docs(_destination_docs(results) + _destination_docs(beach_probe), results, limit=24)
        elif theme == "safari":
            safari_probe = self.retriever.invoke(
                f"best safari wildlife parks reserves game drives {country_label}"
            )
            results = _merge_unique_docs(_destination_docs(results) + _destination_docs(safari_probe), results, limit=24)

        ranked_results = _rerank_results(query, results)

        if theme == "beach":
            beach_ranked = [doc for doc in ranked_results if _is_beach_doc(doc) or doc.metadata.get("type") == "tourism_destination"]
            beach_ranked = [d for d in beach_ranked if d.metadata.get("type") == "tourism_destination" and _is_beach_doc(d)] or beach_ranked
            if beach_ranked:
                ranked_results = _merge_unique_docs(beach_ranked, _destination_docs(ranked_results), limit=24)
        elif theme == "safari":
            safari_docs = [
                doc
                for doc in _destination_docs(ranked_results)
                if _tokenize(_to_fields(doc).get("summary", "") + " " + _to_fields(doc).get("activities", "")).intersection(
                    {"safari", "wildlife", "game", "park", "reserve", "migration", "gorilla"}
                )
            ]
            if safari_docs:
                ranked_results = _merge_unique_docs(safari_docs, _destination_docs(ranked_results), limit=24)

        ranked_results = self._filter_docs_by_country(ranked_results, countries)

        explicit_regions = _mentioned_regions(query)
        region_filtered = _filter_results_by_region(ranked_results, explicit_regions)
        return region_filtered if explicit_regions and region_filtered else ranked_results

    def ask(
        self,
        query: str,
        budget: str | None = None,
        travel_type: str | None = None,
        season: str | None = None,
    ) -> dict[str, Any]:
        if _is_chitchat_query(query):
            return _chitchat_answer(query)

        catalog = _catalog_hotel_answer(query)
        if catalog:
            return apply_polish(catalog, query)

        catalog = _catalog_activity_answer(query)
        if catalog:
            return apply_polish(catalog, query)

        catalog = _catalog_country_answer(query)
        if catalog:
            return catalog

        catalog = _catalog_theme_answer(query)
        if catalog:
            return apply_polish(catalog, query)

        enriched_query = _enrich_query(query, budget=budget, travel_type=travel_type, season=season)
        # Still no place/country/theme after enrichment? Don't invent "top matches".
        if (
            not _mentioned_destinations(enriched_query)
            and not _mentioned_countries(enriched_query)
            and not _query_theme(enriched_query)
            and not _tokenize(enriched_query).intersection(
                {
                    "hotel",
                    "hotels",
                    "resort",
                    "safari",
                    "beach",
                    "trip",
                    "plan",
                    "visit",
                    "itinerary",
                    "activity",
                    "activities",
                    "hike",
                    "best",
                    "kenya",
                    "tanzania",
                    "uganda",
                    "rwanda",
                    "travel",
                    "where",
                    "east",
                    "africa",
                }
            )
        ):
            return apply_polish(
                {
                    "answer": (
                        "I need a bit more detail to help.\n\n"
                        "Try asking about a place, country, or type of trip — for example hotels in Diani, "
                        "beaches in Kenya, or a safari in Tanzania."
                    ),
                    "sources": [],
                    "structured_sources": [],
                    "hotels": [],
                    "destinations": [],
                },
                query,
            )

        final_results = self._retrieve(enriched_query)
        countries = _mentioned_countries(query)
        destinations = _extract_destinations(final_results)
        if countries:
            destinations = [
                d
                for d in destinations
                if any(c in (d.get("location") or "").lower() or c in (d.get("name") or "").lower() for c in countries)
            ] or destinations
        hotels = _extract_hotels(final_results)
        if countries:
            hotels = [h for h in hotels if any(c in (h.get("country") or "").lower() for c in countries)]
        # Destination listing questions should not dump unrelated hotels.
        if _query_theme(query) in {"beach", "safari"} and not _tokenize(query).intersection(
            {"hotel", "hotels", "resort", "stay"}
        ):
            hotels = []
        hotel_results = _hotel_docs(final_results)
        structured = _parse_structured_sources(hotel_results or final_results) if hotels else []
        return apply_polish(
            {
                "answer": _build_answer(enriched_query, final_results),
                "sources": _parse_sources(final_results),
                "structured_sources": structured,
                "hotels": hotels,
                "destinations": destinations,
            },
            query,
        )

    def plan_itinerary(
        self,
        query: str,
        budget: str | None = None,
        travel_type: str | None = None,
        season: str | None = None,
        days: int = 5,
    ) -> dict[str, Any]:
        enriched_query = _enrich_query(query, budget=budget, travel_type=travel_type, season=season)
        trip_days = _parse_trip_days(query) or days
        mentioned = _mentioned_destinations(query)
        final_results = self._retrieve(enriched_query)
        destinations = _destinations_for_mentioned(mentioned)
        if not destinations:
            destinations = _extract_destinations(final_results, limit=max(trip_days, 3))
        itinerary = _generate_itinerary(enriched_query, destinations, days=trip_days)
        if len(destinations) == 1:
            place_names = destinations[0]["name"]
            answer = f"Here is your {trip_days}-day trip in {place_names} with suggested activities for each day."
        else:
            place_names = ", ".join([d["name"] for d in destinations[:4]]) or "East Africa highlights"
            answer = f"Here is your {trip_days}-day East Africa itinerary covering {place_names}."
        return apply_polish(
            {
                "answer": answer,
                "sources": _parse_sources(final_results),
                "structured_sources": _parse_structured_sources(final_results),
                "hotels": _extract_hotels(final_results),
                "destinations": destinations,
                "itinerary": itinerary,
            },
            query,
        )


KenyaTourismRAG = TravelAfricaRAG


def build_vector_index() -> int:
    docs: list[Document] = []

    for path in (DESTINATIONS_PATH, EA_DESTINATIONS_PATH):
        if path.exists():
            with open(path, encoding="utf-8") as file_obj:
                rows = json.load(file_obj)
            docs.extend(_to_document(row) for row in rows)

    if HOTELS_PATH.exists():
        hotels = load_hotels()
        docs.extend(_hotel_to_document(row) for row in hotels)

    if not docs:
        raise RuntimeError("No data found. Run collect_hotels_osm.py and ensure destinations.json exists.")

    embeddings = _create_embeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))
    return len(docs)
