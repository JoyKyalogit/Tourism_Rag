from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .rag_service import KenyaTourismRAG, build_vector_index


class AskRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


app = FastAPI(title="Kenya Tourism RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_rag: KenyaTourismRAG | None = None


def get_rag() -> KenyaTourismRAG:
    global _rag
    if _rag is None:
        _rag = KenyaTourismRAG()
    return _rag


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    try:
        result = get_rag().ask(payload.query)
        return AskResponse(answer=result["answer"], sources=result["sources"])
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc


@app.post("/api/reindex")
def reindex() -> dict[str, str]:
    global _rag
    try:
        count = build_vector_index()
        _rag = None
        return {"message": f"Index rebuilt successfully with {count} documents."}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to build index: {exc}") from exc
