from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_BACKEND_ROOT / ".env", override=True)
load_dotenv(_REPO_ROOT / ".env", override=True)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import HOTELS_PATH
from .hotels import filter_by_location, hotel_to_source, load_hotels, save_hotels
from .llm import groq_enabled
from .rag_service import TravelAfricaRAG, build_vector_index, try_catalog_ask

BACKEND_ROOT = Path(__file__).resolve().parents[1]


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class AskRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    budget: Literal["low", "medium", "luxury"] | None = None
    travel_type: Literal["solo", "family", "adventure"] | None = None
    season: str | None = Field(default=None, max_length=50)


class SourceItem(BaseModel):
    hotel_name: str = ""
    location: str = ""
    country: str = ""
    source_url: str = ""
    price_range: str = ""


class HotelItem(BaseModel):
    hotel_name: str
    location: str = ""
    country: str = ""
    description: str = ""
    price_range: str = ""
    amenities: str = ""
    rating: str = ""
    source_url: str = ""
    website_url: str = ""


class DestinationItem(BaseModel):
    name: str
    description: str
    location: str
    activities: str = ""
    best_months: str = ""
    source_url: str = ""
    lat: float = 0.0
    lng: float = 0.0
    image: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    structured_sources: list[SourceItem] = []
    hotels: list[HotelItem] = []
    destinations: list[DestinationItem] = []


class ItineraryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    question: str | None = Field(default=None, max_length=500)
    budget: Literal["low", "medium", "luxury"] | None = None
    travel_type: Literal["solo", "family", "adventure"] | None = None
    season: str | None = Field(default=None, max_length=50)
    days: int = Field(default=5, ge=2, le=14)


class ItineraryDay(BaseModel):
    day: str
    title: str
    description: str


class ItineraryResponse(BaseModel):
    answer: str
    sources: list[str]
    structured_sources: list[SourceItem] = []
    hotels: list[HotelItem] = []
    destinations: list[DestinationItem] = []
    itinerary: list[ItineraryDay] = []


app = FastAPI(title="Travel Africa API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_rag: TravelAfricaRAG | None = None


def get_rag() -> TravelAfricaRAG:
    global _rag
    if _rag is None:
        _rag = TravelAfricaRAG()
    return _rag


@app.on_event("startup")
def warmup_rag() -> None:
    """Optionally warm embeddings in the background after the port is open.

    Skip on free Render (SKIP_RAG_WARMUP=1): loading torch there often OOMs
    and returns 502. Catalog answers still work via try_catalog_ask.
    """
    if os.getenv("SKIP_RAG_WARMUP", "").strip().lower() in {"1", "true", "yes"}:
        print("RAG warmup skipped (SKIP_RAG_WARMUP)")
        return

    def _load() -> None:
        try:
            get_rag()
            print("RAG warmup complete")
        except Exception as exc:  # noqa: BLE001
            print(f"RAG warmup skipped: {exc}")

    threading.Thread(target=_load, daemon=True, name="rag-warmup").start()


def _run_script(script_name: str) -> str:
    script = BACKEND_ROOT / "scripts" / script_name
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(BACKEND_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"{script_name} failed")
    return result.stdout.strip()


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Travel Africa RAG", "status": "ok"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": _rag is not None, "groq_enabled": groq_enabled()}


@app.post("/scrape")
def scrape() -> dict[str, Any]:
    try:
        collect_out = ""
        try:
            collect_out = _run_script("collect_hotels_osm.py")
            _run_script("clean_hotels.py")
        except Exception:
            _run_script("seed_hotels.py")
        hotels = load_hotels()
        return {
            "message": "Scrape and clean completed.",
            "hotel_count": len(hotels),
            "collect_log": collect_out,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        raw = await file.read()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON must be an array of hotel records.")
        save_hotels(payload)
        return {"message": f"Uploaded {len(payload)} hotel records.", "hotel_count": len(payload)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/create-embeddings")
def create_embeddings() -> dict[str, str]:
    global _rag
    try:
        count = build_vector_index()
        _rag = None
        return {"message": f"Embeddings created for {count} documents."}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _ask_safe(
    query: str,
    *,
    budget: str | None = None,
    travel_type: str | None = None,
    season: str | None = None,
) -> dict[str, Any]:
    # Prefer catalog so demos work without loading the heavy embedding model.
    catalog = try_catalog_ask(query)
    if catalog is not None:
        return catalog
    try:
        return get_rag().ask(
            query,
            budget=budget,
            travel_type=travel_type,
            season=season,
        )
    except Exception as exc:  # noqa: BLE001
        # Avoid 502/crash loops when torch/embeddings OOM on free Render.
        raise HTTPException(
            status_code=503,
            detail=(
                "The search index is still starting (or ran out of memory on this plan). "
                "Try a country or destination question like 'places to visit in Kenya', "
                "or wait a minute and try again."
            ),
        ) from exc


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: QuestionRequest) -> AskResponse:
    try:
        return AskResponse(**_ask_safe(payload.question))
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc


@app.post("/api/ask", response_model=AskResponse)
def ask_legacy(payload: AskRequest) -> AskResponse:
    try:
        return AskResponse(
            **_ask_safe(
                payload.query,
                budget=payload.budget,
                travel_type=payload.travel_type,
                season=payload.season,
            )
        )
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc


@app.get("/hotels")
def list_hotels() -> dict[str, Any]:
    hotels = load_hotels()
    return {"count": len(hotels), "hotels": hotels}


@app.get("/hotels/{location}")
def hotels_by_location(location: str) -> dict[str, Any]:
    hotels = filter_by_location(location)
    return {
        "location": location,
        "count": len(hotels),
        "hotels": hotels,
        "sources": [hotel_to_source(h) for h in hotels[:20]],
    }


@app.post("/plan-trip", response_model=ItineraryResponse)
@app.post("/api/itinerary", response_model=ItineraryResponse)
def plan_trip(payload: ItineraryRequest) -> ItineraryResponse:
    query = payload.question or payload.query
    try:
        result = get_rag().plan_itinerary(
            query,
            budget=payload.budget,
            travel_type=payload.travel_type,
            season=payload.season,
            days=payload.days,
        )
        return ItineraryResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc


@app.post("/api/reindex")
def reindex() -> dict[str, str]:
    return create_embeddings()
