import { useCallback, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { askQuestion, planItinerary } from "../api/client";
import { getFriendlyError } from "../utils/errors";
import { processApiResponse } from "../utils/parseResponse";
import { detectQueryIntent, parseTripDays } from "../utils/queryIntent";
import AppLayout from "../components/AppLayout";
import Sidebar from "../components/Sidebar";
import ItineraryTimeline from "../components/ItineraryTimeline";
import LoadingSpinner from "../components/LoadingSpinner";
import HotelCard from "../components/HotelCard";

const EMPTY_VIEW = { input: "", answer: "", error: "", itinerary: [], hotels: [] };

const TAB_COPY = {
  explore: {
    eyebrow: "Karibu — welcome",
    title: (
      <>
        Welcome to <span className="text-emerald-800">Travel Africa</span>
      </>
    ),
    intro:
      "Your guide to safaris, Indian Ocean beaches, city breaks, and hotels across Kenya, Tanzania, Uganda and Rwanda. Just ask — we'll help you plan.",
    placeholder: "Where would you like to go?",
  },
  itineraries: {
    eyebrow: "Itineraries",
    title: (
      <>
        Plan your <span className="text-emerald-800">trip</span>
      </>
    ),
    intro:
      "Tell us where you want to go and for how long — we'll build a day-by-day plan with activities and hotels.",
    placeholder: "e.g. Plan a 4-day trip to Diani and Tsavo",
  },
};

export default function HomePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const tab = searchParams.get("tab") === "itineraries" ? "itineraries" : "explore";
  const [views, setViews] = useState({ explore: EMPTY_VIEW, itineraries: EMPTY_VIEW });
  const [loadingTab, setLoadingTab] = useState(null);

  const view = views[tab];
  const copy = TAB_COPY[tab];
  const loading = loadingTab === tab;

  const updateView = useCallback((target, patch) => {
    setViews((prev) => ({ ...prev, [target]: { ...prev[target], ...patch } }));
  }, []);

  const sendMessage = useCallback(
    async (text, target, days = null) => {
      const query = text.trim();
      if (!query || loadingTab) return;

      setViews((prev) => ({ ...prev, [target]: { ...EMPTY_VIEW, input: query } }));
      setLoadingTab(target);
      setSearchParams(target === "itineraries" ? { tab: "itineraries" } : {});

      try {
        const raw =
          target === "itineraries"
            ? await planItinerary({ query, ...(days ? { days } : {}) })
            : await askQuestion({ query });
        const data = processApiResponse(raw);
        updateView(target, {
          answer: data.answer,
          hotels: data.hotels || [],
          itinerary: data.itinerary || [],
        });
      } catch (err) {
        updateView(target, { error: getFriendlyError(err) });
      } finally {
        setLoadingTab(null);
      }
    },
    [loadingTab, setSearchParams, updateView]
  );

  function handleSubmit(e) {
    e.preventDefault();
    if (tab === "itineraries") {
      sendMessage(view.input, "itineraries", parseTripDays(view.input));
      return;
    }
    const intent = detectQueryIntent(view.input);
    sendMessage(view.input, intent.mode === "itinerary" ? "itineraries" : "explore", intent.days);
  }

  function handlePlanTrip() {
    const query = view.input.trim() || "Plan a 6-day trip across Kenya and East Africa";
    sendMessage(query, "itineraries", parseTripDays(query));
  }

  const { answer, error, itinerary, hotels } = view;

  return (
    <AppLayout sidebar={<Sidebar onPlanTrip={handlePlanTrip} />}>
      <main className="flex min-w-0 flex-1 flex-col overflow-y-auto">
        <div className="page-content-scrim mx-auto w-full max-w-3xl px-6 py-14 sm:px-10 sm:py-20 lg:py-24">
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-emerald-900">
            {copy.eyebrow}
          </p>
          <h1 className="editorial-title mt-3 font-display text-4xl font-semibold leading-tight text-slate-900 sm:text-5xl lg:text-[3.4rem]">
            {copy.title}
          </h1>
          <p className="mt-4 max-w-lg text-base leading-relaxed text-slate-800 sm:text-lg">{copy.intro}</p>

          <form onSubmit={handleSubmit} className="mt-8 max-w-xl">
            <div className="flex items-center gap-2 rounded-full bg-white py-2 pl-5 pr-2 shadow-lg ring-1 ring-black/5">
              <input
                type="text"
                value={view.input}
                onChange={(e) => updateView(tab, { input: e.target.value })}
                placeholder={copy.placeholder}
                disabled={loading}
                className="min-w-0 flex-1 bg-transparent text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none sm:text-base"
              />
              <button
                type="submit"
                disabled={loadingTab !== null || view.input.trim().length < 2}
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
