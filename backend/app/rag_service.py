import json
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "processed" / "kenya_tourism_data.json"
INDEX_DIR = REPO_ROOT / "vectorstore" / "kenya_faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DESTINATION_ALIASES: dict[str, set[str]] = {
    "maasai mara": {"maasai mara", "mara"},
    "amboseli": {"amboseli"},
    "diani beach": {"diani", "diani beach"},
    "watamu": {"watamu"},
    "naivasha": {"naivasha", "lake naivasha"},
    "nairobi": {"nairobi"},
    "tsavo": {"tsavo"},
    "nanyuki and mount kenya": {"nanyuki", "mount kenya", "nanyuki and mount kenya"},
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
    "nanyuki and mount kenya": 8,
    "hells gate": 7,
    "samburu": 7,
    "lamu": 7,
    "malindi": 7,
    "kakamega forest": 6,
    "mount elgon": 6,
    "aberdare": 6,
}


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


def _to_document(row: dict[str, Any]) -> Document:
    content = (
        f"Destination: {row['destination']}\n"
        f"Region: {row['region']}\n"
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
    intent_beach = bool(query_terms.intersection({"beach", "coast", "coastal", "ocean", "island", "marine"}))
    intent_hike = bool(query_terms.intersection({"hike", "hiking", "trek", "trekking", "climb", "mountain"}))
    intent_activity = bool(query_terms.intersection({"activity", "activities", "things", "todo", "do", "adventure"}))
    broad_intent = bool(
        query_terms.intersection({"kenya", "country", "overall", "general", "around", "across"})
    ) and not bool(query_terms.intersection({"nanyuki", "diani", "watamu", "amboseli", "mara", "nairobi", "tsavo"}))
    mentioned_destinations = _mentioned_destinations(query)
    query_lower = query.lower()

    top_fields = _to_fields(results[0])
    top_destination = top_fields.get("destination", "Unknown")

    def destination_in_query(destination_name: str) -> bool:
        return destination_name.lower() in query_lower

    def matches_mentioned_destination(destination_name: str) -> bool:
        canonical = destination_name.lower().strip()
        aliases = DESTINATION_ALIASES.get(canonical, {canonical})
        return any(alias in query_lower for alias in aliases)

    def collect_unique(docs: list[Any], limit: int = 6) -> list[dict[str, str]]:
        unique: list[dict[str, str]] = []
        seen: set[str] = set()
        for doc in docs:
            fields = _to_fields(doc)
            name = fields.get("destination", "Unknown")
            if name in seen:
                continue
            unique.append(fields)
            seen.add(name)
            if len(unique) >= limit:
                break
        return unique

    unique_results = collect_unique(results, limit=12)

    def with_location(fields: dict[str, str]) -> str:
        destination = fields.get("destination", "Unknown")
        region = fields.get("region", "Unknown location")
        return f"{destination} ({region})"

    def priority(fields: dict[str, str]) -> int:
        return CURATED_PRIORITY.get(fields.get("destination", "").lower(), 0)

    # Prefer globally recognized tourism spots for generic "best places" style queries.
    curated_sorted = sorted(unique_results, key=priority, reverse=True)

    direct_destination = next(
        (f for f in unique_results if matches_mentioned_destination(f.get("destination", ""))),
        None,
    )
    # If the user names a destination directly (e.g. "Diani beach"), answer only for that place.
    if direct_destination and not (intent_hotel or intent_time or intent_activity):
        return (
            f"{with_location(direct_destination)}:\n"
            f"{direct_destination.get('summary', 'No summary available.')}\n"
            f"Top activities: {direct_destination.get('activities', 'N/A')}"
        )

    if intent_hotel:
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
            picks.append(
                f"{with_location(fields)}: {fields.get('luxury hotels', 'N/A')}"
            )
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
            if _tokenize(f.get("summary", "") + " " + f.get("activities", "")).intersection({"beach", "coast", "ocean", "marine"})
        ]
        picks = beaches[:6] if beaches else unique_results[:4]
        return "Best beaches in Kenya:\n- " + "\n- ".join([with_location(f) for f in picks])

    if intent_hike:
        hikes = [
            f
            for f in curated_sorted
            if _tokenize(f.get("activities", "") + " " + f.get("summary", "")).intersection({"hiking", "trekking", "climbing", "mountain", "hike", "forest"})
        ]
        picks = hikes[:6] if hikes else unique_results[:4]
        return "Best places to go on hikes in Kenya:\n- " + "\n- ".join(
            [f"{f.get('destination', 'Unknown')} - {f.get('activities', 'N/A')}" for f in picks]
        )

    if intent_activity:
        target = next((f for f in unique_results if matches_mentioned_destination(f.get("destination", ""))), None)
        if not target:
            target = next((f for f in unique_results if destination_in_query(f.get("destination", ""))), None)
        if target:
            return (
                f"Best activities in {with_location(target)}:\n"
                f"{target.get('activities', 'N/A')}"
            )
        activity_picks: list[str] = []
        for fields in curated_sorted[:5]:
            destination = with_location(fields)
            activities = fields.get("activities", "N/A")
            activity_picks.append(f"{destination}: {activities}")
        return "Best activities to do in Kenya:\n- " + "\n- ".join(activity_picks)
    if mentioned_destinations:
        return f"{with_location(top_fields)}: {top_fields.get('summary', 'No summary available.')}"

    if broad_intent and not mentioned_destinations:
        beach_set = {"diani beach", "watamu", "mombasa", "malindi", "lamu"}
        safari_set = {"maasai mara", "amboseli", "tsavo", "samburu", "nairobi national park"}
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
        return f"Best places to visit in Kenya: {', '.join(mixed_with_location)}"

    top_three = [with_location(f) for f in curated_sorted[:4]]
    return f"Top matches: {', '.join(top_three)}"


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
        if any(alias in query_lower for alias in aliases):
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
            weighted_overlap + base_overlap + destination_bonus + destination_match_bonus + region_bonus,
            destination_match_bonus + destination_bonus + region_bonus,
            base_overlap,
        )

    return sorted(results, key=score, reverse=True)


class KenyaTourismRAG:
    def __init__(self) -> None:
        self.embeddings = _create_embeddings()
        if not INDEX_DIR.exists():
            raise RuntimeError("Vector index not found. Run `python backend/scripts/ingest.py` first.")
        self.vector_store = FAISS.load_local(
            str(INDEX_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 12})

    def ask(self, query: str) -> dict[str, Any]:
        results = self.retriever.invoke(query)
        query_terms = _tokenize(query)
        beach_intent = bool(query_terms.intersection({"beach", "coast", "coastal", "ocean", "marine", "island"}))
        if beach_intent:
            beach_probe = self.retriever.invoke("best beach destinations in Kenya coast Diani Watamu Malindi Mombasa Lamu")
            results = _merge_unique_docs(results, beach_probe, limit=24)

        ranked_results = _rerank_results(query, results)

        if beach_intent:
            beach_ranked = [doc for doc in ranked_results if _is_beach_doc(doc)]
            if beach_ranked:
                ranked_results = _merge_unique_docs(beach_ranked, ranked_results, limit=24)

        explicit_regions = _mentioned_regions(query)
        region_filtered = _filter_results_by_region(ranked_results, explicit_regions)
        final_results = region_filtered if explicit_regions and region_filtered else ranked_results
        return {
            "answer": _build_answer(query, final_results),
            "sources": _parse_sources(final_results),
        }


def build_vector_index() -> int:
    with open(DATA_PATH, "r", encoding="utf-8") as file_obj:
        rows = json.load(file_obj)

    docs = [_to_document(row) for row in rows]
    embeddings = _create_embeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))
    return len(docs)
