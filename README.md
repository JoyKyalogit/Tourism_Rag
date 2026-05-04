# Kenya Tourism RAG

Tourism recommendation RAG app with a Next.js frontend and FastAPI backend.

## Features

- Curated Kenya tourism knowledge base (8 destinations)
- Local vector search with FAISS
- Free local embeddings using sentence-transformers
- Source citations in every response
- Split architecture: Next.js frontend + Python API backend

## Stack

- Frontend: Next.js (TypeScript)
- Backend: FastAPI + LangChain + FAISS
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

## Project Structure

```text
Tourism_Rag/
  backend/
    app/
      main.py
      rag_service.py
    scripts/ingest.py
    requirements.txt
  frontend/
    app/
      layout.tsx
      page.tsx
      globals.css
    package.json
    .env.local.example
  data/processed/kenya_tourism_data.json
```

## Run Locally

### 1) Backend (FastAPI)

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python scripts/ingest.py
uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
http://localhost:8000/health
```

### 2) Frontend (Next.js)

Open a second terminal:

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Open:

```bash
http://localhost:3000
```

## API Endpoints

- `POST /api/ask`
  - Body: `{ "query": "Best places to visit in Kenya in July?" }`
- `POST /api/reindex`
- `GET /health`

## Deploy on Render

This repo includes a Render Blueprint at `render.yaml` to deploy both backend and frontend.

### 1) Push code to GitHub

Render deploys from your Git repository, so make sure this project is pushed.

### 2) Create Blueprint in Render

- In Render dashboard: **New +** -> **Blueprint**
- Connect your GitHub repo
- Render will detect `render.yaml` and create:
  - `safiri-kenya-api` (FastAPI backend)
  - `safiri-kenya-web` (Next.js frontend)

### 3) Set frontend API URL

After first deploy, open `safiri-kenya-api` and copy its public URL, then set this env var on `safiri-kenya-web`:

- `NEXT_PUBLIC_API_BASE_URL=https://<your-backend-service>.onrender.com`

Then redeploy the frontend service.

### 4) Verify

- Backend health: `https://<your-backend-service>.onrender.com/health`
- Frontend app: `https://<your-frontend-service>.onrender.com`

## Example Questions

- Best places to visit in Kenya in July
- Best beach destination with luxury hotels
- Where should I go for wildlife and adventure in January?
