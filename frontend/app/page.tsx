"use client";

import { FormEvent, useMemo, useState } from "react";

type AskResponse = {
  answer: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const SAMPLE_QUERY = "best beaches to visit in Kenya";

export default function Page() {
  const [query, setQuery] = useState(SAMPLE_QUERY);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);

  const disabled = useMemo(() => loading || query.trim().length < 2, [loading, query]);

  async function submitQuery() {
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || "Request failed.");
      }

      const body = (await response.json()) as AskResponse;
      setResult(body);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Something went wrong.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await submitQuery();
  }

  return (
    <main className="page-shell">
      <h1 className="app-name">Safiri Kenya</h1>
      <p className="intro-message">
        Welcome to your Kenya travel assistant. Ask about beaches, safaris, hotels, seasons, or city breaks to
        get quick recommendations.
      </p>
      <form className="form-section" onSubmit={onSubmit}>
        <label htmlFor="query">Your question</label>
        <textarea
          id="query"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={async (event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              if (!disabled) {
                await submitQuery();
              }
            }
          }}
          placeholder={`Example: ${SAMPLE_QUERY}`}
        />
        <button type="submit" disabled={disabled}>
          {loading ? "Searching..." : "Get Recommendation"}
        </button>
      </form>

      {error ? <p className="error">{error}</p> : null}

      {result ? (
        <section className="result-section">
          <h2>Recommendation</h2>
          <pre className="answer-block">{result.answer}</pre>
        </section>
      ) : null}
    </main>
  );
}
