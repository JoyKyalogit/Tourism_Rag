"""Optional Groq LLM for rewriting grounded answers (not inventing facts)."""

from __future__ import annotations

import json
import os
from typing import Any

from .config import GROQ_MODEL

_SYSTEM = """You are Travel Africa, a helpful East Africa travel assistant.
Rewrite the draft answer so it is clear, warm, and well structured.
Rules:
- Use ONLY facts from the provided context and draft. Do not invent hotels, places, prices, or activities.
- If the draft says data is missing, keep that honesty.
- Prefer short paragraphs plus bullet points when listing places or hotels.
- Keep place/country names exactly as given.
- Do not mention OpenStreetMap, FAISS, RAG, embeddings, or that you are an AI model.
- Keep the answer under 220 words unless the draft is longer for a multi-day itinerary.
- For greetings / thanks / goodbye: keep it short and welcoming. You may rephrase the draft, but keep the same intent and any example prompts from the draft (as bullets). Do NOT invent destinations or hotels. Do NOT say the user must ask a "specific" question — invite them to ask anything about East Africa travel.
- Reply with the polished answer only — no preamble, no chain-of-thought.
"""


def _api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    # Fallback: load .env if the server started before the key was added
    try:
        from pathlib import Path
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env", override=True)
        load_dotenv(root / "backend" / ".env", override=True)
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "").strip()


def groq_enabled() -> bool:
    return bool(_api_key())


def polish_answer(
    query: str,
    draft_answer: str,
    *,
    hotels: list[dict[str, Any]] | None = None,
    destinations: list[dict[str, Any]] | None = None,
    itinerary: list[dict[str, Any]] | None = None,
) -> str:
    """Return a Groq-polished answer, or the draft if Groq is off / fails."""
    api_key = _api_key()
    if not api_key or not draft_answer.strip():
        return draft_answer

    context: dict[str, Any] = {
        "user_question": query,
        "draft_answer": draft_answer,
    }
    if destinations:
        context["destinations"] = [
            {
                "name": d.get("name") or d.get("destination"),
                "location": d.get("location") or d.get("region"),
                "description": d.get("description") or d.get("summary"),
                "activities": d.get("activities"),
            }
            for d in destinations[:8]
        ]
    if hotels:
        context["hotels"] = [
            {
                "name": h.get("hotel_name") or h.get("name"),
                "location": h.get("location"),
                "country": h.get("country"),
                "price_range": h.get("price_range") or h.get("tier"),
            }
            for h in hotels[:8]
        ]
    if itinerary:
        context["itinerary"] = itinerary[:14]

    try:
        from groq import Groq

        model = os.getenv("GROQ_MODEL", GROQ_MODEL).strip() or "openai/gpt-oss-20b"
        is_chitchat = not hotels and not destinations and not itinerary
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            temperature=0.45 if is_chitchat else 0.3,
            max_tokens=500 if is_chitchat else 700,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "Polish this travel answer using only the JSON context.\n\n"
                        f"{json.dumps(context, ensure_ascii=False)}"
                    ),
                },
            ],
        )
        text = (completion.choices[0].message.content or "").strip()
        return text or draft_answer
    except Exception as exc:  # noqa: BLE001
        print(f"Groq polish skipped: {exc}")
        return draft_answer


def apply_polish(result: dict[str, Any], query: str) -> dict[str, Any]:
    """In-place polish of a standard ask/itinerary response dict."""
    if not result.get("answer"):
        return result
    result["answer"] = polish_answer(
        query,
        result["answer"],
        hotels=result.get("hotels") or [],
        destinations=result.get("destinations") or [],
        itinerary=result.get("itinerary") or [],
    )
    return result
