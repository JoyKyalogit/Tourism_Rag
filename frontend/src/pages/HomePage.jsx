import { useCallback, useState } from "react";
import { askQuestion, planItinerary } from "../api/client";
import { getFriendlyError } from "../utils/errors";
import { processApiResponse } from "../utils/parseResponse";
import { detectQueryIntent } from "../utils/queryIntent";
import AppLayout from "../components/AppLayout";
import Sidebar from "../components/Sidebar";
import ItineraryTimeline from "../components/ItineraryTimeline";
import LoadingSpinner from "../components/LoadingSpinner";
import HotelCard from "../components/HotelCard";

export default function HomePage() {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [answer, setAnswer] = useState("");
  const [activeNav, setActiveNav] = useState("explore");
  const [itinerary, setItinerary] = useState([]);
  const [hotels, setHotels] = useState([]);

  const sendMessage = useCallback(
    async (text, mode = "ask", days = null) => {
      const query = text.trim();
      if (!query || loading) return;

      setError("");
      setAnswer("");
      setItinerary([]);
      setHotels([]);
      setInput(query);
      setLoading(true);
      setActiveNav(mode === "itinerary" ? "itineraries" : "explore");

      try {
        const raw =
          mode === "itinerary"
            ? await planItinerary({ query, ...(days ? { days } : {}) })
            : await askQuestion({ query });
        const data = processApiResponse(raw);
        setAnswer(data.answer);
        setHotels(data.hotels || []);
        if (data.itinerary.length) setItinerary(data.itinerary);
      } catch (err) {
        setError(getFriendlyError(err));
      } finally {
        setLoading(false);
      }
    },
    [loading]
  );

  function handleSubmit(e) {
    e.preventDefault();
    const intent = detectQueryIntent(input);
    sendMessage(input, intent.mode, intent.days);
  }

  function handlePlanTrip() {
    sendMessage(input.trim() || "Plan a 6-day trip across Kenya and East Africa", "itinerary");
  }

  return (
    <AppLayout sidebar={<Sidebar activeNav={activeNav} onNavChange={setActiveNav} onPlanTrip={handlePlanTrip} />}>
      <main className="flex min-w-0 flex-1 flex-col overflow-y-auto">
        <div className="page-content-scrim mx-auto w-full max-w-3xl px-6 py-14 sm:px-10 sm:py-20 lg:py-24">
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-emerald-900">
            Karibu — welcome
          </p>
          <h1 className="editorial-title mt-3 font-display text-4xl font-semibold leading-tight text-slate-900 sm:text-5xl lg:text-[3.4rem]">
            Welcome to <span className="text-emerald-800">Travel Africa</span>
          </h1>
          <p className="mt-4 max-w-lg text-base leading-relaxed text-slate-800 sm:text-lg">
            Your guide to safaris, Indian Ocean beaches, city breaks, and hotels across Kenya, Tanzania,
            Uganda and Rwanda. Just ask — we&apos;ll help you plan.
          </p>

          <form onSubmit={handleSubmit} className="mt-8 max-w-xl">
            <div className="flex items-center gap-2 rounded-full bg-white py-2 pl-5 pr-2 shadow-lg ring-1 ring-black/5">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Where would you like to go?"
                disabled={loading}
                className="min-w-0 flex-1 bg-transparent text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none sm:text-base"
              />
              <button
                type="submit"
                disabled={loading || input.trim().length < 2}
                className="btn-gradient flex h-11 w-11 shrink-0 items-center justify-center rounded-full disabled:cursor-not-allowed"
                aria-label="Search"
              >
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          </form>
        </div>

        <div className="mx-auto w-full max-w-3xl flex-1 px-6 pb-12 sm:px-10">
          {loading && <LoadingSpinner />}

          {error && <p className="mb-6 text-sm text-red-800">{error}</p>}

          {answer && !loading && (
            <article className="page-content-scrim mb-8 rounded-2xl p-5">
              <p className="text-xs font-semibold uppercase tracking-wider text-emerald-900">Travel Africa</p>
              <p className="mt-2 whitespace-pre-line text-sm leading-relaxed text-slate-800 sm:text-base">
                {answer}
              </p>
            </article>
          )}

          {itinerary.length > 0 && !loading && (
            <div className="page-content-scrim mb-8 rounded-2xl p-5">
              <ItineraryTimeline itinerary={itinerary} />
            </div>
          )}

          {hotels.length > 0 && !loading && (
            <section className="page-content-scrim mb-8 rounded-2xl p-5">
              <h2 className="mb-3 font-display text-base font-semibold text-slate-900">Hotels</h2>
              <div className="grid gap-3 sm:grid-cols-2">
                {hotels.map((hotel) => (
                  <HotelCard key={`${hotel.name}-${hotel.location}`} hotel={hotel} />
                ))}
              </div>
            </section>
          )}
        </div>
      </main>
    </AppLayout>
  );
}
