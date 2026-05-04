# Safiri Kenya - Tourism RAG Assistant

Safiri Kenya is a tourism recommendation app that answers travel questions about Kenya using a Retrieval-Augmented Generation (RAG) workflow.

It combines:
- a curated tourism dataset,
- vector search for relevant context,
- and a lightweight UI for asking natural language questions.

## What the App Does

Safiri Kenya helps users ask questions like:
- "Best beaches to visit in Kenya"
- "Best activities to do in Diani Beach"
- "Best places to visit in Kenya in July"
- "Best hotels in Maasai Mara"

The app returns focused answers that can include:
- destination recommendations,
- activities,
- best visiting months/time,
- hotel options by budget tier,
- and destination locations (county/region style labels).

## How It Works

1. User asks a question in the Next.js frontend.
2. Frontend sends the query to the FastAPI backend (`POST /api/ask`).
3. Backend retrieves relevant destination documents from FAISS.
4. Retrieved results are re-ranked with query-intent logic (beach, hotel, time, activity, destination, region).
5. A concise answer is composed from structured fields in the retrieved documents.

## How This Project Was Created

This project was built in layers:

1. **Knowledge curation**
   - Tourism data was prepared in JSON format (`data/processed/kenya_tourism_data.json`).
   - Each record includes destination, region, best months/time, hotels, activities, summary, and source URL.

2. **Indexing pipeline**
   - Records are converted into documents and embedded using `sentence-transformers/all-MiniLM-L6-v2`.
   - Embeddings are stored in a local FAISS vector index.
   - Index build script: `backend/scripts/ingest.py`.

3. **Backend API**
   - FastAPI exposes `/api/ask`, `/api/reindex`, and `/health`.
   - RAG and ranking logic lives in `backend/app/rag_service.py`.
   - API entrypoint: `backend/app/main.py`.

4. **Frontend UI**
   - Built with Next.js (App Router + TypeScript).
   - Provides a simple question box, Enter-to-submit behavior, and styled response display.
   - Frontend calls backend using `NEXT_PUBLIC_API_BASE_URL`.

5. **Deployment setup**
   - `render.yaml` defines both backend and frontend services for Render.
   - Both services are configured for free plan deployment.

## Tech Stack

- **Frontend:** Next.js, React, TypeScript
- **Backend:** FastAPI, Pydantic
- **RAG/Vector:** LangChain Community, FAISS
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Hosting:** Render (Blueprint via `render.yaml`)

## Project Structure

```text
Tourism_Rag/
  backend/
    app/
      main.py
      rag_service.py
    scripts/
      ingest.py
    requirements.txt
  frontend/
    app/
      globals.css
      layout.tsx
      page.tsx
    .env.local.example
    package.json
  data/
    processed/
      kenya_tourism_data.json
  render.yaml
```

## Run Locally

### 1) Backend (FastAPI)

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/ingest.py
uvicorn app.main:app --reload --port 8000
```

Health check:

```text
http://localhost:8000/health
```

### 2) Frontend (Next.js)

Open a second terminal:

```bash
cd frontend
copy .env.local.example .env.local
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

## API Endpoints

- `POST /api/ask`
  - Body: `{ "query": "Best places to visit in Kenya in July?" }`
- `POST /api/reindex`
- `GET /health`

## Deploy on Render

This repo includes a Render Blueprint file: `render.yaml`.

### Steps

1. Push this repository to GitHub.
2. In Render, create a new **Blueprint** and select the repo.
3. Render creates:
   - `safiri-kenya-api` (backend)
   - `safiri-kenya-web` (frontend)
4. Set frontend environment variable:
   - `NEXT_PUBLIC_API_BASE_URL=https://<your-backend-service>.onrender.com`
5. Redeploy the frontend service after setting the variable.

### Verify

- Backend: `https://<your-backend-service>.onrender.com/health`
- Frontend: `https://<your-frontend-service>.onrender.com`

## Example Questions

- Best places to visit in Kenya
- Best beaches to visit in Kenya
- Best activities to do in Diani Beach
- Best hotels in Maasai Mara
- Best time to visit Amboseli
