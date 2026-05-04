import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "kenya_tourism_data.json"
INDEX_DIR = ROOT / "vectorstore" / "kenya_faiss"


def build_documents(rows: list[dict]) -> list[Document]:
    docs: list[Document] = []
    for row in rows:
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
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "destination": row["destination"],
                    "source_url": row["source_url"],
                    "type": "tourism_destination",
                },
            )
        )
    return docs


def main() -> None:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)

    docs = build_documents(rows)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))
    print(f"Built index with {len(docs)} destination documents at: {INDEX_DIR}")


if __name__ == "__main__":
    main()
