from pathlib import Path
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "vectorstore" / "kenya_faiss"


def _format_context(results: list[Any]) -> str:
    blocks = []
    for i, doc in enumerate(results, 1):
        blocks.append(
            f"[{i}] {doc.page_content}\n"
            f"[{i}] source_url: {doc.metadata.get('source_url', 'unknown')}"
        )
    return "\n\n".join(blocks)


def _parse_sources(results: list[Any]) -> list[str]:
    unique = []
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

    top = _to_fields(results[0])
    rec_places = ", ".join([_to_fields(doc).get("destination", "Unknown") for doc in results[:3]])
    return (
        f"Recommended Place(s): {rec_places}\n"
        f"Best Season/Time: {top.get('best months', 'See source context')} | {top.get('best time to visit', 'Varies by destination')}\n"
        f"Hotel Suggestions: Budget - {top.get('budget hotels', 'N/A')} | Midrange - {top.get('midrange hotels', 'N/A')} | Luxury - {top.get('luxury hotels', 'N/A')}\n"
        f"Activities: {top.get('activities', 'See destination details')}\n"
        f"Why: Based on the closest matches to your question ('{query}') from the tourism dataset."
    )


class KenyaTourismRAG:
    def __init__(self) -> None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if not INDEX_DIR.exists():
            raise RuntimeError(
                "Vector index not found. Run `python scripts/ingest.py` first."
            )
        self.vector_store = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

    def ask(self, query: str) -> dict[str, Any]:
        results = self.retriever.invoke(query)
        sources = _parse_sources(results)
        return {
            "answer": _build_answer(query, results),
            "sources": sources,
        }
