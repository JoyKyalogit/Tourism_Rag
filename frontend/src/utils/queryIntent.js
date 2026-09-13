const WORD_DAYS = {
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
};

const TRIP_HINT =
  /\b(plan|itinerary|schedule|trip|days?\b|day\s*trip|weekend|getaway)\b/i;

export function parseTripDays(text) {
  const lower = text.toLowerCase();

  const numeric = lower.match(/\b(\d{1,2})\s*-?\s*da[yts]?\b/);
  if (numeric) {
    const days = Number(numeric[1]);
    if (days >= 2 && days <= 14) return days;
  }

  const word = lower.match(/\b(one|two|three|four|five|six|seven)\s*-?\s*da[yts]?\b/);
  if (word) return WORD_DAYS[word[1]];

  const tripPhrase = lower.match(/\b(\d{1,2})\s*-?\s*day\s+trip\b/);
  if (tripPhrase) {
    const days = Number(tripPhrase[1]);
    if (days >= 2 && days <= 14) return days;
  }

  return null;
}

/** Route trip-planning questions to the itinerary API. */
export function detectQueryIntent(query) {
  const text = query.trim();
  if (!text) return { mode: "ask", days: null };

  const days = parseTripDays(text);
  const looksLikeTrip = TRIP_HINT.test(text) || days !== null;

  if (looksLikeTrip && (days !== null || /\b(plan|itinerary|trip)\b/i.test(text))) {
    return { mode: "itinerary", days };
  }

  return { mode: "ask", days: null };
}
