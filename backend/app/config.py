"""Paths and constants for Travel Africa RAG."""

from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

DESTINATIONS_PATH = DATA_PROCESSED / "destinations.json"
EA_DESTINATIONS_PATH = DATA_PROCESSED / "ea_destinations.json"
HOTELS_RAW_PATH = DATA_RAW / "hotels_osm_raw.json"
HOTELS_PATH = DATA_PROCESSED / "hotels.json"

INDEX_DIR = REPO_ROOT / "vectorstore" / "ea_travel_faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
REQUEST_DELAY_SEC = 2.5

# Optional Groq answer writer — set GROQ_API_KEY in backend/.env or the environment.
# llama-3.3-70b-versatile was retired on Groq free/dev tier (Aug 2026).
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()