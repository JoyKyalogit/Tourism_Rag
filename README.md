# Travel Africa — Hotel Discovery & Trip Planning RAG

East Africa travel assistant covering **Kenya, Tanzania, Uganda, and Rwanda**. Users ask in natural language; the app retrieves hotels and destinations from a local FAISS index and returns recommendations or a day-by-day itinerary.

No Booking.com / Tripadvisor scraping — only public OSM data and curated tourism sources.

## Live Demo
https://safiri-kenya-web.onrender.com

---

## Stack (what each tool does)

| Tool | Role in this project |
|------|----------------------|
| **OpenStreetMap + Overpass API** | Source of hotel POIs (`tourism=hotel\|resort\|…`) by bounding box |
| **Requests** | HTTP client for Overpass queries (`collect_hotels_osm.py`) |
| **Pandas** | Dedupe/normalize raw hotel JSON → `hotels.json` |
| **Curated JSON** | Destination guides + seed hotels with tourism-board / brand `source_url`s |
| **sentence-transformers** (`all-MiniLM-L6-v2`) | Local embeddings of hotel + destination text |
| **FAISS + LangChain** | Vector store + retrieval for RAG |
| **Groq (optional)** | Free-tier LLM that rewrites grounded answers for clearer wording |
| **FastAPI + Uvicorn + Pydantic** | REST API: ask, itinerary, hotels, scrape, reindex |
| **React + Vite + Tailwind + Axios** | UI; search, destinations, trip plans |
| **Leaflet** | Map markers for destinations |
| **Render** | Optional cloud deploy (`render.yaml`) |

---

## Where the data comes from

### 1. Hotels — scraped from OpenStreetMap

| Item | Detail |
|------|--------|
| **API** | [Overpass API](https://overpass-api.de/api/interpreter) |
| **Script** | `backend/scripts/collect_hotels_osm.py` |
| **What** | OSM nodes/ways tagged `tourism` ∈ hotel, guest_house, hostel, motel, resort |
| **Where** | Bounding boxes: Nairobi, Mombasa, Diani, Watamu, Malindi, Lamu, Naivasha, Nakuru, Kisumu, Nanyuki, Narok, Amboseli, Maasai Mara, Zanzibar, Arusha, Dar es Salaam, Kampala, Entebbe |
| **Provenance** | Each row gets `source_url` like `https://www.openstreetmap.org/node/<id>` |
| **Ethics** | Rate-limited (`REQUEST_DELAY_SEC` ≈ 2.5s), custom User-Agent, public data only |

Raw output → `data/raw/hotels_osm_raw.json`.

### 2. Hotels — curated seed (fallback / enrichment)

| Item | Detail |
|------|--------|
| **Script** | `backend/scripts/seed_hotels.py` |
| **From destinations** | Budget / midrange / luxury names listed under each place in `destinations.json` (Kenya guides) |
| **Extras** | Named East Africa hotels (e.g. Park Hyatt Zanzibar, Serena Kampala) with official brand URLs |
| **When** | OSM down, sparse areas, or to enrich processed hotels |

### 3. Destinations — curated (not scraped)

| File | Content | Typical `source_url` |
|------|---------|----------------------|
| `data/processed/destinations.json` | Kenya parks, beaches, cities (summary, activities, best months, hotel tiers) | Magical Kenya, KWS, etc. |
| `data/processed/ea_destinations.json` | Tanzania / Uganda / Rwanda (Serengeti, Zanzibar, Kigali, Volcanoes NP, …) | Tanzania Tourism, Visit Rwanda, Uganda Wildlife |

Frontend catalog mirrors these in `frontend/src/data/`.

---

## End-to-end data & request flow

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA PIPELINE (offline / once)                                 │
│                                                                 │
│  Overpass (OSM) ──► hotels_osm_raw.json                         │
│       │                                                         │
│       ▼  Pandas clean (require name + source_url, dedupe)       │
│  destinations.json ──┐                                          │
│  seed_hotels.py ─────┼──► hotels.json                           │
│  ea_destinations.json┘                                          │
│       │                                                         │
│       ▼  ingest.py (MiniLM embeddings)                          │
│  vectorstore/ea_travel_faiss/   (hotels + destinations)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  RUNTIME                                                        │
│                                                                 │
│  Browser (Vite :3000) ──Axios──► FastAPI (:8000)                │
│       │                              │                          │
│       │                    TravelAfricaRAG                      │
│       │                    FAISS similarity search              │
│       │                    intent / country / trip routing        │
│       │                              │                          │
│       ◄──── answer, hotels, destinations, itinerary ────────────┘
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline steps (in order)

1. **`collect_hotels_osm.py`** — Overpass POST per region → tags → hotel records → `data/raw/hotels_osm_raw.json`
2. **`seed_hotels.py`** (optional) — merge destination hotel names + EA extras into `hotels.json`
3. **`clean_hotels.py`** — Pandas: drop nameless / sourceless rows, normalize `price_range`, dedupe `(name, location, country)` → `data/processed/hotels.json`
4. **`ingest.py`** — load hotels + Kenya + EA destinations → embed → save FAISS at `vectorstore/ea_travel_faiss/`

### Runtime (user question)

1. UI detects intent (`queryIntent.js`): plain ask vs trip/itinerary (“plan”, “3 day”, typos like “Dlani”).
2. `POST /api/ask` or `POST /api/itinerary` → `TravelAfricaRAG` embeds the query, retrieves top docs from FAISS.
3. Rerank by destination / country / hotel intent; build a short answer (or day-by-day plan).
4. Response includes `answer`, `hotels` (with OSM/brand `source_url`), `destinations`, and optionally `itinerary`.

---

## Project layout

```
Tourism_Rag/
├── backend/
│   ├── app/main.py          # FastAPI routes
│   ├── app/rag_service.py   # FAISS RAG, answers, itineraries
│   ├── app/hotels.py        # Load/filter hotels
│   ├── app/config.py        # Paths, Overpass URL, embed model
│   └── scripts/             # collect → seed → clean → ingest
├── data/
│   ├── raw/hotels_osm_raw.json
│   └── processed/           # hotels.json, destinations.json, ea_destinations.json
├── vectorstore/ea_travel_faiss/
├── frontend/src/            # React UI
└── render.yaml              # Deploy blueprints
```

---

## Run locally

**Backend** (API → http://localhost:8000)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# First time / refresh data only:
# python scripts/collect_hotels_osm.py
# python scripts/seed_hotels.py
# python scripts/clean_hotels.py
# python scripts/ingest.py
python -m uvicorn app.main:app --reload --port 8000
```

Optional **Groq** answer polish (clearer wording; facts still come from your filtered data):

1. Create a free key at [console.groq.com/keys](https://console.groq.com/keys)
2. In `backend/`, copy `.env.example` → `.env` and set `GROQ_API_KEY=...`
3. `pip install groq python-dotenv` (or reinstall `requirements.txt`)
4. Restart uvicorn — `/health` should show `"groq_enabled": true`

Without a key, the API still works using template answers.

**Frontend** (UI → http://localhost:3000)

```powershell
cd frontend
copy .env.example .env
npm install
npm run dev
```

Processed data + FAISS index are usually already in the repo; re-run collect/clean/ingest only when refreshing OSM data.

First `/api/ask` after startup can take 30–60s while the embedding model loads.

---

## API (short)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + model loaded |
| POST | `/api/ask` | RAG Q&A |
| POST | `/api/itinerary` | Multi-day trip plan |
| GET | `/hotels`, `/hotels/{location}` | Hotel list / filter |
| POST | `/scrape` | Trigger OSM collect + clean |
| POST | `/create-embeddings` | Rebuild FAISS |

```json
POST /api/ask
{ "query": "Find midrange hotels in Diani" }
```

---

## Ethics

- OSM only for scraped hotels; each record keeps an OSM node/way URL.
- Destinations cite public tourism-board pages (Magical Kenya, Visit Rwanda, etc.).
- Overpass calls are spaced; no OTA scraping.
- Seed hotels point at official brand / property pages.

---

## Deploy

See `render.yaml`. For the frontend production build, set `VITE_API_BASE_URL` to the deployed API URL.

---

## Example questions

- Find hotels in Nairobi  
- Luxury lodges in Maasai Mara  
- Rwanda / Kigali highlights  
- Plan a 3-day trip in Diani with activities each day  
- Best safari parks for families in July  
